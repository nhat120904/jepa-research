"""Train the action-conditioned representation adapter φ = A(z) — Track 2, the
heavy fix (docs/plans/2026-07-01-adversarial-cost-and-repr-design.md;
models/heads/action_repr_adapter.py has the full mechanism writeup).

Four losses, one combined step per batch:

  1. grounding      — MSE: φ[:, :obj_dim] -> true object xyz (held-out cache).
  2. cf-contrastive  — margin: φ_obj of the factual z_t1 vs φ_obj of an IN-BATCH
                       hard-effect negative's z_t1 (metrics.negative_samplers,
                       return_indices=True) must be >= their TRUE object gap.
  3. temporal        — RANKING (not regression — see the note below): within a
                       trajectory, a frame closer in time to the goal must have
                       smaller φ_extra-distance to the goal than one further away.
                       Deliberately NOT a Huber regression to the raw step-gap
                       (scripts/33's recipe): that would require φ_extra's raw
                       magnitude to sit on a ~1-12 step-count scale, fighting term
                       1's metres-scale grounding once both enter the same phi
                       vector. A pairwise ranking loss is scale-free — CEM only
                       needs the cost to DESCEND toward the goal, not match a
                       literal step count — and needs no rescaling machinery.
  4. adversarial     — margin: φ_obj of a mined CEM-EXPLOITED elite
                       (scripts/_cem_mining, `--mine-adv`/`--adv-buffer`) vs φ_obj
                       of that episode's goal latent must be >= their TRUE object
                       gap. Same mechanism as term 2 (models.heads.action_repr_
                       adapter.margin_loss), applied to the adversarially-mined
                       negative source instead of an in-batch one — this is what
                       directly closes the pockets that broke Phase-3 3b.

Gates (printed; do not spend the scripts/30 --cost phi GPU-hour unless these look
reasonable): held-out grounding decode (median cm, <5cm/<7cm, same table as
scripts/21/22); held-out cf-margin satisfaction rate; held-out temporal ranking
accuracy + monotone-to-goal Spearman (scripts/33-style, on φ_extra).

    python scripts/37_train_repr_adapter.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --mine-adv --adv-cost l2 --adv-episodes 4 \
        --out checkpoints/action_repr_adapter_dino_wm_metaworld.pt
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from metrics.negative_samplers import hard_effect_negative  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.heads.action_repr_adapter import ActionReprAdapter, margin_loss  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402

PUSH_RADIUS, PICK_RADIUS = 0.05, 0.07


def split_by_trajectory(records, val_frac, seed):
    tids = sorted({r["tid"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(tids)
    val_tids = set(tids[: max(1, int(len(tids) * val_frac))])
    return ([r for r in records if r["tid"] not in val_tids],
            [r for r in records if r["tid"] in val_tids])


def iter_chunks(records, chunk, rng):
    order = np.arange(len(records))
    rng.shuffle(order)
    for lo in range(0, len(order), chunk):
        sel = [records[int(i)] for i in order[lo: lo + chunk]]
        sel.sort(key=lambda r: r["tid"])
        yield sel


def sample_temporal_ranking(trajs, batch, rng, device):
    """One ranking triplet per sample: (near, far, goal) from the same trajectory,
    `near` temporally closer to the goal than `far`. See the module docstring."""
    near, far, goal = [], [], []
    n = len(trajs)
    for _ in range(batch):
        tr = trajs[rng.integers(n)]
        T = tr.shape[0]
        i, j = (int(x) for x in rng.integers(0, T - 1, size=2))
        while j == i:
            j = int(rng.integers(0, T - 1))
        ni, fi = (i, j) if (T - 1 - i) < (T - 1 - j) else (j, i)
        near.append(tr[ni]); far.append(tr[fi]); goal.append(tr[-1])
    return (torch.stack(near).to(device), torch.stack(far).to(device),
            torch.stack(goal).to(device))


def sample_cross_traj(trajs, batch, rng, device):
    a, g = [], []
    n = len(trajs)
    for _ in range(batch):
        tr_a = trajs[rng.integers(n)]
        tr_g = trajs[rng.integers(n)]
        a.append(tr_a[int(rng.integers(0, tr_a.shape[0]))])
        g.append(tr_g[-1])
    return torch.stack(a).to(device), torch.stack(g).to(device)


def _spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--phi-dim", type=int, default=64)
    ap.add_argument("--obj-dim", type=int, default=3)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    # term 2: cf-contrastive margin (in-batch hard-effect negatives)
    ap.add_argument("--cf-k", type=int, default=6)
    ap.add_argument("--cf-similarity-radius", type=float, default=0.5)
    ap.add_argument("--cf-action-penalty", type=float, default=0.5)
    ap.add_argument("--cf-margin-cap", type=float, default=0.15,
                    help="cap the object-gap margin (m) so outliers don't dominate")
    ap.add_argument("--lambda-cf", type=float, default=1.0)
    # term 3: temporal ranking (on phi_extra)
    ap.add_argument("--rank-margin", type=float, default=1.0)
    ap.add_argument("--cross-margin", type=float, default=6.0)
    ap.add_argument("--lambda-temporal", type=float, default=0.3)
    ap.add_argument("--lambda-cross", type=float, default=0.3)
    # term 4: adversarial (mined CEM-elite pockets)
    ap.add_argument("--mine-adv", action="store_true",
                    help="mine CEM-exploited frames (scripts/_cem_mining) live before training")
    ap.add_argument("--adv-buffer", default=None,
                    help="load a pre-mined buffer (.pt, scripts/35 --save-buffer) instead "
                         "of mining live")
    ap.add_argument("--adv-cost", default="l2", choices=["l2", "gobj", "stateprobe", "metric"])
    ap.add_argument("--adv-probe", default=None)
    ap.add_argument("--adv-ee-probe", default=None)
    ap.add_argument("--adv-w-hand", type=float, default=0.5)
    ap.add_argument("--adv-episodes", type=int, default=4)
    ap.add_argument("--adv-tasks", nargs="+", default=["mw-push", "mw-pick-place"])
    ap.add_argument("--lambda-adv", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/action_repr_adapter.pt")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, cfg["dataset"]["name"])
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    regime_by_traj = read_regimes(cache_path)
    rng = np.random.default_rng(args.seed)

    # ---- term 4 buffer: mined CEM-exploited elites (optional) ----
    adv_z = adv_obj = adv_zgoal = adv_goalobj = None
    if args.adv_buffer:
        b = torch.load(args.adv_buffer, map_location="cpu", weights_only=False)
        adv_z, adv_obj, adv_zgoal, adv_goalobj = b["z"], b["obj"], b["z_goal"], b["goal_obj"]
        print(f"adversarial buffer (loaded): n={adv_z.shape[0]}", flush=True)
    elif args.mine_adv:
        from scripts._cem_mining import mine_cem_frames
        cost_kw = dict(cost=args.adv_cost)
        if args.adv_cost in ("gobj", "stateprobe"):
            from models.probes import load_probe
            p, _ = load_probe(args.adv_probe, device)
            cost_kw["probe"] = p
        if args.adv_cost == "stateprobe":
            from models.probes import load_probe
            ep, _ = load_probe(args.adv_ee_probe, device)
            cost_kw["ee_probe"] = ep; cost_kw["w_hand"] = args.adv_w_hand
        buf = mine_cem_frames(adapter, device, cost_spec_kwargs=cost_kw, tasks=args.adv_tasks,
                              episodes=args.adv_episodes)
        adv_z, adv_obj, adv_zgoal, adv_goalobj = (buf["z"], buf["obj"], buf["z_goal"],
                                                   buf["goal_obj"])
        print(f"adversarial buffer (mined): n={adv_z.shape[0]} cost={args.adv_cost}", flush=True)

    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=True)
        train_recs, val_recs = split_by_trajectory(records, args.val_frac, args.seed)
        print(f"transitions: train={len(train_recs)} val={len(val_recs)}", flush=True)

        probe_d = helpers.materialize_records(cache, train_recs[:2], step,
                                              want_proprio=False, want_state=True)
        latent_dim = int(probe_d["z_t"].shape[-1])
        del probe_d

        # ---- term 3 buffer: full trajectories for temporal ranking ----
        def _load_trajs(tids_):
            out = []
            for tid in tids_:
                z = cache.read_trajectory(tid)["z"]
                if len(z) >= 3:
                    out.append(torch.tensor(np.asarray(z)).float())
            return out
        train_trajs = _load_trajs(sorted({r["tid"] for r in train_recs}))
        val_trajs = _load_trajs(sorted({r["tid"] for r in val_recs}))
        print(f"temporal trajs: train={len(train_trajs)} val={len(val_trajs)}", flush=True)

        model = ActionReprAdapter(latent_dim=latent_dim, phi_dim=args.phi_dim,
                                  obj_dim=args.obj_dim, n_layers=args.n_layers,
                                  hidden=args.hidden).to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"repr adapter params: {n_params/1e6:.2f}M  phi_dim={args.phi_dim} "
              f"obj_dim={args.obj_dim}", flush=True)

        def train_epoch():
            tot = dict(g=0.0, cf=0.0, t=0.0, cross=0.0, a=0.0, n=0)
            for sel in iter_chunks(train_recs, args.chunk, rng):
                d = helpers.materialize_records(cache, sel, step, want_proprio=False, want_state=True)
                obj_t1_all = d["state_t1"][:, OBJECT_SLICE].float()
                m = d["z_t"].shape[0]
                order = np.arange(m); rng.shuffle(order)
                for lo in range(0, m, args.batch_size):
                    idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
                    if len(idx) < 4:
                        continue
                    zt = d["z_t"][idx].to(device)
                    zt1 = d["z_t1"][idx].to(device)
                    at = d["a_t"][idx].to(device)
                    objt1 = obj_t1_all[idx].to(device)
                    B = len(idx)

                    phi_t1 = model(zt1)
                    obj_est = phi_t1[:, :args.obj_dim]
                    loss_g = ((obj_est - objt1) ** 2).mean()

                    # term 2: in-batch hard-effect negatives
                    k = min(args.cf_k, B - 1)
                    with torch.no_grad():
                        _, neg_idx = hard_effect_negative(
                            zt, zt1, at, pool_z=zt, pool_z1=zt1, pool_a=at, K=k,
                            similarity_radius=args.cf_similarity_radius,
                            action_penalty=args.cf_action_penalty, return_indices=True)
                    obj_neg_true = objt1[neg_idx.reshape(-1)].reshape(B, k, -1)
                    phi_neg = model(zt1[neg_idx.reshape(-1)])[:, :args.obj_dim].reshape(B, k, -1)
                    true_gap = (objt1.unsqueeze(1) - obj_neg_true).norm(dim=-1).reshape(-1)
                    obj_est_rep = obj_est.unsqueeze(1).expand(-1, k, -1).reshape(-1, args.obj_dim)
                    loss_cf = margin_loss(obj_est_rep, phi_neg.reshape(-1, args.obj_dim),
                                          true_gap, args.cf_margin_cap)

                    # term 3: temporal ranking on phi_extra
                    near, far, goal = sample_temporal_ranking(train_trajs, B, rng, device)
                    d_near = (model(near)[:, args.obj_dim:] - model(goal)[:, args.obj_dim:]).norm(dim=-1)
                    d_far = (model(far)[:, args.obj_dim:] - model(goal)[:, args.obj_dim:]).norm(dim=-1)
                    loss_t = torch.relu(args.rank_margin + d_near - d_far).mean()
                    ca, cg = sample_cross_traj(train_trajs, B, rng, device)
                    d_cross = (model(ca)[:, args.obj_dim:] - model(cg)[:, args.obj_dim:]).norm(dim=-1)
                    loss_cross = torch.relu(args.cross_margin - d_cross).mean()

                    # term 4: adversarial (mined CEM-exploited pockets)
                    loss_a = zt.new_tensor(0.0)
                    if adv_z is not None:
                        ai = rng.integers(0, adv_z.shape[0], size=B)
                        za = adv_z[ai].to(device); zga = adv_zgoal[ai].to(device)
                        oa = adv_obj[ai].to(device); ga = adv_goalobj[ai].to(device)
                        phi_a = model(za)[:, :args.obj_dim]
                        phi_g = model(zga)[:, :args.obj_dim]
                        true_gap_a = (oa - ga).norm(dim=-1)
                        loss_a = margin_loss(phi_a, phi_g, true_gap_a, args.cf_margin_cap)

                    loss = (loss_g + args.lambda_cf * loss_cf + args.lambda_temporal * loss_t
                           + args.lambda_cross * loss_cross + args.lambda_adv * loss_a)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot["g"] += loss_g.item() * B; tot["cf"] += loss_cf.item() * B
                    tot["t"] += loss_t.item() * B; tot["cross"] += loss_cross.item() * B
                    tot["a"] += loss_a.item() * B; tot["n"] += B
                del d; gc.collect()
            n = max(tot["n"], 1)
            return {k: v / n for k, v in tot.items() if k != "n"}

        for ep in range(args.epochs):
            L = train_epoch()
            print(f"epoch {ep+1}/{args.epochs}: grnd={L['g']:.5f} cf={L['cf']:.4f} "
                  f"temporal={L['t']:.4f} cross={L['cross']:.4f} adv={L['a']:.4f}", flush=True)
        model.eval()

        # ================= held-out gates =================
        with torch.no_grad():
            # (a) grounding decode — same table format as scripts/21/22.
            errs, extras = [], []
            for sel in iter_chunks(val_recs, args.chunk, np.random.default_rng(2)):
                d = helpers.materialize_records(cache, sel, step, want_proprio=False, want_state=True)
                obj_true = d["state_t1"][:, OBJECT_SLICE].float()
                for lo in range(0, d["z_t1"].shape[0], args.batch_size):
                    s = slice(lo, lo + args.batch_size)
                    phi = model(d["z_t1"][s].to(device))
                    e = (phi[:, :args.obj_dim].cpu() - obj_true[s]).norm(dim=-1)
                    errs.append(e.numpy()); extras.append(phi[:, args.obj_dim:].cpu().numpy())
                del d; gc.collect()
            err = np.concatenate(errs)
            extra_scale = float(np.concatenate(extras).std())
            print(f"\nphi grounding held-out decode: median={100*np.median(err):.2f}cm "
                  f"<5cm={(err < PUSH_RADIUS).mean()*100:.1f}% "
                  f"<7cm={(err < PICK_RADIUS).mean()*100:.1f}%  extra_scale={extra_scale:.4f}",
                  flush=True)

            # (b) cf-margin satisfaction on held-out hard-effect pairs.
            viol, sat = [], []
            for sel in iter_chunks(val_recs, args.chunk, np.random.default_rng(3)):
                d = helpers.materialize_records(cache, sel, step, want_proprio=False, want_state=True)
                obj_t1_all = d["state_t1"][:, OBJECT_SLICE].float()
                m = d["z_t"].shape[0]
                for lo in range(0, m, args.batch_size):
                    s = slice(lo, lo + args.batch_size)
                    zt, zt1, at = d["z_t"][s].to(device), d["z_t1"][s].to(device), d["a_t"][s].to(device)
                    objt1 = obj_t1_all[s].to(device)
                    B = zt.shape[0]
                    if B < 4:
                        continue
                    k = min(args.cf_k, B - 1)
                    _, neg_idx = hard_effect_negative(zt, zt1, at, pool_z=zt, pool_z1=zt1, pool_a=at,
                                                      K=k, similarity_radius=args.cf_similarity_radius,
                                                      action_penalty=args.cf_action_penalty,
                                                      return_indices=True)
                    obj_neg_true = objt1[neg_idx.reshape(-1)].reshape(B, k, -1)
                    obj_est = model(zt1)[:, :args.obj_dim]
                    phi_neg = model(zt1[neg_idx.reshape(-1)])[:, :args.obj_dim].reshape(B, k, -1)
                    true_gap = (objt1.unsqueeze(1) - obj_neg_true).norm(dim=-1)
                    d_cf = (obj_est.unsqueeze(1) - phi_neg).norm(dim=-1)
                    margin = true_gap.clamp(max=args.cf_margin_cap)
                    viol.append(torch.relu(margin - d_cf).cpu().numpy().reshape(-1))
                    sat.append((d_cf >= margin).float().cpu().numpy().reshape(-1))
                del d; gc.collect()
            viol = np.concatenate(viol) if viol else np.array([np.nan])
            sat = np.concatenate(sat) if sat else np.array([np.nan])
            print(f"cf-margin held-out: mean_violation={viol.mean():.4f}m "
                  f"satisfied_frac={sat.mean()*100:.1f}%", flush=True)

            # (c) temporal ranking accuracy + monotone-to-goal Spearman (scripts/33-style).
            r2 = np.random.default_rng(4)
            near, far, goal = sample_temporal_ranking(val_trajs, 2000, r2, device)
            d_near = (model(near)[:, args.obj_dim:] - model(goal)[:, args.obj_dim:]).norm(dim=-1)
            d_far = (model(far)[:, args.obj_dim:] - model(goal)[:, args.obj_dim:]).norm(dim=-1)
            rank_acc = float((d_near < d_far).float().mean().item())
            mono = []
            for tr in val_trajs:
                T = tr.shape[0]
                zt_ = tr.to(device)
                zg_ = tr[-1:].to(device).expand(T, *([-1] * (tr.dim() - 1)))
                d = (model(zt_)[:, args.obj_dim:] - model(zg_)[:, args.obj_dim:]).norm(dim=-1).cpu().numpy()
                steps_to_go = np.arange(T)[::-1].astype(np.float64)
                mono.append(_spearman(d, steps_to_go))
            mono_sp = float(np.nanmean(mono)) if mono else float("nan")
            print(f"temporal held-out: ranking_acc={rank_acc:.3f} "
                  f"monotone_to_goal_spearman={mono_sp:.3f}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model, "latent_dim": latent_dim, "phi_dim": args.phi_dim,
        "obj_dim": args.obj_dim, "n_layers": args.n_layers, "n_heads": 6, "hidden": args.hidden,
        "state_dict": model.state_dict(), "extra_scale": extra_scale,
        "val_obj_median_cm": float(100 * np.median(err)),
        "val_obj_frac_5cm": float((err < PUSH_RADIUS).mean()),
        "val_cf_satisfied_frac": float(sat.mean()),
        "val_rank_acc": rank_acc, "val_mono_spearman": mono_sp,
        "adv_n": int(adv_z.shape[0]) if adv_z is not None else 0,
    }, out)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
