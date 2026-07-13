"""Phase D — LoRA-finetune the ENCODER with the Phase-C objective
(docs/plans/2026-07-02-encoder-lora-action-grounding-design.md).

Phase 4 exhausted every lever that is a function of the frozen ``z`` (Phase C v2:
relearned φ with healthy grounding still gates push 1/16 — see
``results/latent_oracle_phi_v2.csv``). This script moves the same losses INSIDE
the encoder: zero-init LoRA adapters on the encoder's transformer blocks
(``models/heads/lora_encoder.py``) train jointly with a fresh φ head
(``ActionReprAdapter``, unchanged), so gradients can separate latents the frozen
encoder had merged — the pockets themselves — instead of reading around them.

Because the encoder moves, the latent cache is INVALID as training input: this
script consumes RAW FRAMES (expert trajectories via ``data.loaders``; off-policy
frames via ``scripts/_offpolicy_frames`` ``return_frames=True``; optionally
CEM-mined elite frames from ``scripts/_cem_mining`` ``keep_frames=True`` buffers)
and encodes them live through ``adapter.encpred.encode`` — the exact upstream
pipeline ``adapter.encode`` wraps, minus its ``@torch.no_grad()``.

Losses per batch (terms 1-3 = scripts/37, now with encoder gradients):
  1. grounding      — MSE φ_obj(z̃_{t+1}) -> true object xyz.
  2. cf-margin      — in-batch hard-effect negatives, φ_obj separation >= true gap.
  3. temporal       — ranking hinge on φ_extra distance-to-goal (+ cross floor).
  4. adversarial    — (--adv-buffer, frames-carrying) mined CEM elites re-encoded
                      live; OFF in v0, ON in DAgger round >= 1.
  5. preservation   — ‖z̃ − z_frozen‖²/‖z_frozen‖², z_frozen = same batch with LoRA
                      toggled off (free via set_lora_enabled). Keeps the reshape in
                      the predictor's neighborhood; --lambda-preserve 0 ablates.

Gate afterwards (do not skip the held-out prints below first):
    python scripts/30_latent_oracle.py --config ... --model dino_wm_metaworld \
        --cost phi --repr-adapter <out-phi> --encoder-lora <out-lora> --strict-success

    python scripts/38_train_encoder_lora.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --offpolicy-frac 0.5 \
        --out-lora checkpoints/encoder_lora_dino_wm_metaworld.pt \
        --out-phi checkpoints/phi_enclora_dino_wm_metaworld.pt
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

from data.loaders import iterate_metaworld_trajectories  # noqa: E402
from metrics.negative_samplers import hard_effect_negative  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.heads.action_repr_adapter import ActionReprAdapter, margin_loss  # noqa: E402
from models.heads.lora_encoder import (  # noqa: E402
    encoder_lora_state_dict, inject_encoder_lora)
from models.heads.lora_predictor import set_lora_enabled  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402

PUSH_RADIUS, PICK_RADIUS = 0.05, 0.07


# ---------------------------------------------------------------------------
# grad-enabled encoding (adapter.encode is @torch.no_grad — same pipeline)
# ---------------------------------------------------------------------------

def _undecorated_encode(encpred):
    """``EncPredWM.encode`` is ``@torch.no_grad()`` upstream (vit_enc_preds.py) —
    calling it directly would SILENTLY train phi on grad-less latents while the
    LoRA never moves. ``torch.no_grad``'s decorator wraps via ``functools.wraps``,
    so ``__wrapped__`` is the exact same pipeline minus the grad guard (verified
    on torch 2.7.0)."""
    fn = type(encpred).encode
    if not hasattr(fn, "__wrapped__"):
        raise RuntimeError(
            "EncPredWM.encode has no __wrapped__ — torch changed its no_grad "
            "decorator; encoder gradients would silently be dropped. STOP.")
    return fn.__wrapped__


def encode_grad(adapter, vis: torch.Tensor, prop: torch.Tensor | None):
    """vis: (B, C, H, W) float [0,255] -> single-frame latent (B, V, Hp, Wp, D),
    with gradients flowing into the (LoRA-injected) encoder."""
    from tensordict.tensordict import TensorDict

    enc = _undecorated_encode(adapter.encpred)
    vis = vis.to(adapter.device, dtype=torch.float32).unsqueeze(1)   # (B,1,C,H,W)
    if adapter.spec.uses_proprio and prop is not None:
        obs = TensorDict(
            {"visual": vis, "proprio": prop.to(adapter.device, dtype=torch.float32).unsqueeze(1)},
            batch_size=[])
        z = enc(adapter.encpred, obs)["visual"]
    else:
        z = enc(adapter.encpred, vis)
        z = z["visual"] if hasattr(z, "keys") else z
    if torch.is_grad_enabled() and not z.requires_grad:
        raise RuntimeError("encoded latent has no grad — the encoder gradient path "
                           "is broken; LoRA would silently never train. STOP.")
    return z[:, 0]


# ---------------------------------------------------------------------------
# data: expert trajectories as raw frames in RAM (uint8)
# ---------------------------------------------------------------------------

def load_expert_trajs(cfg, max_trajs_per_task: int):
    ds_cfg = cfg["dataset"]
    tasks = ds_cfg["tasks"]
    all_tasks = tasks["easy"] + tasks["medium"] + tasks["hard"]
    root = os.environ.get("CAI_JEPA_DATA_ROOT_METAWORLD", ds_cfg["root"])
    trajs = []
    for tb in iterate_metaworld_trajectories(
            root=root, tasks=all_tasks,
            max_trajectories_per_task=max_trajs_per_task,
            external_root=ds_cfg.get("external_root", "external/jepa-wms")):
        trajs.append({
            "frames": tb.obs_visual.to(torch.uint8),            # (T,C,H,W)
            "prop": tb.proprio[:, :4].float(),                   # (T,4)
            "obj": tb.state[:, OBJECT_SLICE].float(),            # (T,3)
            "action": tb.action.float(),                         # (T-1,A)
            "tid": tb.traj_id, "task": tb.task,
        })
    return trajs


def build_records(trajs, step):
    recs = []
    for ti, tr in enumerate(trajs):
        T = tr["frames"].shape[0]
        for t in range(0, T - step):
            recs.append((ti, t))
    return recs


def split_by_trajectory(recs, n_trajs, val_frac, seed):
    rng = np.random.default_rng(seed)
    tids = np.arange(n_trajs)
    rng.shuffle(tids)
    val = set(tids[: max(1, int(n_trajs * val_frac))].tolist())
    return [r for r in recs if r[0] not in val], [r for r in recs if r[0] in val], val


def sample_temporal(trajs, tids, batch, rng):
    """(near, far, goal) frame triplets + props, `near` temporally closer to goal."""
    fr_n, fr_f, fr_g, pr_n, pr_f, pr_g = [], [], [], [], [], []
    tids = list(tids)
    for _ in range(batch):
        tr = trajs[tids[rng.integers(len(tids))]]
        T = tr["frames"].shape[0]
        if T < 3:
            continue
        i, j = (int(x) for x in rng.integers(0, T - 1, size=2))
        while j == i:
            j = int(rng.integers(0, T - 1))
        ni, fi = (i, j) if (T - 1 - i) < (T - 1 - j) else (j, i)
        for buf_f, buf_p, k in ((fr_n, pr_n, ni), (fr_f, pr_f, fi), (fr_g, pr_g, T - 1)):
            buf_f.append(tr["frames"][k].float()); buf_p.append(tr["prop"][k])
    if not fr_n:
        return None
    st = lambda x: torch.stack(x)  # noqa: E731
    return st(fr_n), st(fr_f), st(fr_g), st(pr_n), st(pr_f), st(pr_g)


def materialize(trajs, recs, step):
    """recs -> stacked (frames_t, frames_t1, prop_t, prop_t1, a_t, obj_t1)."""
    ft, ft1, pt, pt1, at, ot1 = [], [], [], [], [], []
    for ti, t in recs:
        tr = trajs[ti]
        ft.append(tr["frames"][t].float()); ft1.append(tr["frames"][t + step].float())
        pt.append(tr["prop"][t]); pt1.append(tr["prop"][t + step])
        at.append(tr["action"][t: t + step].reshape(-1))
        ot1.append(tr["obj"][t + step])
    st = torch.stack
    return st(ft), st(ft1), st(pt), st(pt1), st(at), st(ot1)


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
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--lambda-preserve", type=float, default=0.05,
                    help="0 = unconstrained-reshape ablation")
    # φ head (scripts/37 defaults; ckpt saved in the scripts/37 format)
    ap.add_argument("--phi-dim", type=int, default=64)
    ap.add_argument("--obj-dim", type=int, default=3)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--phi-lr", type=float, default=5e-4)
    # data
    ap.add_argument("--max-trajs-per-task", type=int, default=60)
    ap.add_argument("--offpolicy-frac", type=float, default=0.5,
                    help="grounding rows from random-rollout frames (3b recipe)")
    ap.add_argument("--offpolicy-episodes", type=int, default=8)
    ap.add_argument("--offpolicy-tasks", nargs="+",
                    default=["mw-reach", "mw-push", "mw-pick-place"])
    ap.add_argument("--val-frac", type=float, default=0.1)
    # optimization
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--temporal-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    # term 2 (scripts/37 v2 lambdas)
    ap.add_argument("--cf-k", type=int, default=6)
    ap.add_argument("--cf-similarity-radius", type=float, default=0.5)
    ap.add_argument("--cf-action-penalty", type=float, default=0.5)
    ap.add_argument("--cf-margin-cap", type=float, default=0.15)
    ap.add_argument("--lambda-cf", type=float, default=1.0)
    # term 3
    ap.add_argument("--rank-margin", type=float, default=1.0)
    ap.add_argument("--cross-margin", type=float, default=6.0)
    ap.add_argument("--lambda-temporal", type=float, default=0.03)
    ap.add_argument("--lambda-cross", type=float, default=0.03)
    # term 4 (round >= 1): frames-carrying buffer from scripts/35 --keep-frames
    ap.add_argument("--adv-buffer", default=None)
    ap.add_argument("--lambda-adv", type=float, default=1.0)
    ap.add_argument("--out-lora", default="checkpoints/encoder_lora.pt")
    ap.add_argument("--out-phi", default="checkpoints/phi_enclora.pt")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    injected = inject_encoder_lora(adapter, r=args.lora_r, alpha=args.lora_alpha)
    lora_params = [p for m in injected for p in (m.A, m.B)]
    n_lora = sum(p.numel() for p in lora_params)
    print(f"encoder LoRA: {len(injected)} modules, {n_lora/1e6:.2f}M params "
          f"(r={args.lora_r} alpha={args.lora_alpha})", flush=True)

    # ---- data ----
    trajs = load_expert_trajs(cfg, args.max_trajs_per_task)
    recs = build_records(trajs, step)
    train_recs, val_recs, val_tids = split_by_trajectory(recs, len(trajs), args.val_frac, args.seed)
    train_tids = [i for i in range(len(trajs)) if i not in val_tids]
    print(f"expert: {len(trajs)} trajs, transitions train={len(train_recs)} "
          f"val={len(val_recs)}", flush=True)

    op = None
    if args.offpolicy_frac > 0:
        from scripts._offpolicy_frames import collect_offpolicy_frames
        op = collect_offpolicy_frames(
            adapter, device, tasks=args.offpolicy_tasks, episodes=args.offpolicy_episodes,
            seed=args.seed + 20000, return_frames=True, verbose=True)
        # Shuffle before the val split — collection is task-sequential, so an
        # unshuffled head slice would hold out a single task's frames only.
        perm = torch.from_numpy(np.random.default_rng(args.seed + 1).permutation(
            len(op["frames"])))
        for key in ("frames", "prop", "obj"):
            op[key] = op[key][perm]
        n_val_op = max(1, int(len(op["frames"]) * args.val_frac))
        print(f"off-policy frames: {len(op['frames'])} ({n_val_op} held out)", flush=True)

    adv = None
    if args.adv_buffer:
        adv = torch.load(args.adv_buffer, map_location="cpu", weights_only=False)
        if "frames" not in adv:
            raise SystemExit(f"--adv-buffer {args.adv_buffer} has no 'frames' key — "
                             "re-mine with scripts/35 --keep-frames (frozen-z buffers are "
                             "stale once the encoder moves)")
        print(f"adversarial buffer: n={adv['frames'].shape[0]}", flush=True)

    # ---- φ head + probe latent_dim ----
    with torch.no_grad():
        z0 = encode_grad(adapter, trajs[0]["frames"][:1].float(), trajs[0]["prop"][:1])
    latent_dim = int(z0.shape[-1])
    phi = ActionReprAdapter(latent_dim=latent_dim, phi_dim=args.phi_dim, obj_dim=args.obj_dim,
                            n_layers=args.n_layers, hidden=args.hidden).to(device).train()
    opt = torch.optim.Adam([
        {"params": lora_params, "lr": args.lora_lr},
        {"params": phi.parameters(), "lr": args.phi_lr},
    ])

    def train_epoch():
        tot = dict(g=0.0, cf=0.0, t=0.0, cross=0.0, a=0.0, p=0.0, n=0)
        order = np.arange(len(train_recs)); rng.shuffle(order)
        for lo in range(0, len(order), args.batch_size):
            sel = [train_recs[int(i)] for i in order[lo: lo + args.batch_size]]
            if len(sel) < 4:
                continue
            ft, ft1, pt, pt1, at, ot1 = materialize(trajs, sel, step)
            B = ft.shape[0]
            z_t = encode_grad(adapter, ft, pt)
            z_t1 = encode_grad(adapter, ft1, pt1)
            objt1 = ot1.to(device)
            at_d = at.to(device)

            phi_t1 = phi(z_t1)
            obj_est = phi_t1[:, :args.obj_dim]
            loss_g = ((obj_est - objt1) ** 2).mean()

            # off-policy grounding rows (term 1 on the distribution CEM scores)
            if op is not None and args.offpolicy_frac > 0:
                n_op = max(1, int(B * args.offpolicy_frac))
                oi = rng.integers(n_val_op, len(op["frames"]), size=n_op)
                fo = torch.stack([op["frames"][int(i)] for i in oi]).float()
                po = torch.stack([op["prop"][int(i)] for i in oi])
                oo = torch.stack([op["obj"][int(i)] for i in oi]).to(device)
                z_op = encode_grad(adapter, fo, po)
                loss_g = loss_g + ((phi(z_op)[:, :args.obj_dim] - oo) ** 2).mean()

            # term 2: in-batch hard-effect negatives (reuse z_t1 — no extra encode)
            k = min(args.cf_k, B - 1)
            with torch.no_grad():
                _, neg_idx = hard_effect_negative(
                    z_t.detach(), z_t1.detach(), at_d, pool_z=z_t.detach(),
                    pool_z1=z_t1.detach(), pool_a=at_d, K=k,
                    similarity_radius=args.cf_similarity_radius,
                    action_penalty=args.cf_action_penalty, return_indices=True)
            obj_neg_true = objt1[neg_idx.reshape(-1)].reshape(B, k, -1)
            phi_neg = phi_t1[neg_idx.reshape(-1), : args.obj_dim].reshape(B, k, -1)
            true_gap = (objt1.unsqueeze(1) - obj_neg_true).norm(dim=-1).reshape(-1)
            obj_est_rep = obj_est.unsqueeze(1).expand(-1, k, -1).reshape(-1, args.obj_dim)
            loss_cf = margin_loss(obj_est_rep, phi_neg.reshape(-1, args.obj_dim),
                                  true_gap, args.cf_margin_cap)

            # term 3: temporal ranking on φ_extra
            loss_t = z_t.new_tensor(0.0); loss_cross = z_t.new_tensor(0.0)
            trip = sample_temporal(trajs, train_tids, args.temporal_batch, rng)
            if trip is not None:
                fn, ff, fg, pn, pf, pg = trip
                zn = encode_grad(adapter, fn, pn); zf = encode_grad(adapter, ff, pf)
                zg = encode_grad(adapter, fg, pg)
                en = phi(zn)[:, args.obj_dim:]; ef = phi(zf)[:, args.obj_dim:]
                eg = phi(zg)[:, args.obj_dim:]
                loss_t = torch.relu(args.rank_margin + (en - eg).norm(dim=-1)
                                    - (ef - eg).norm(dim=-1)).mean()
                perm = torch.randperm(eg.shape[0], device=device)
                loss_cross = torch.relu(args.cross_margin
                                        - (en - eg[perm]).norm(dim=-1)).mean()

            # term 4: mined CEM elites, re-encoded live through the current encoder
            loss_a = z_t.new_tensor(0.0)
            if adv is not None:
                ai = rng.integers(0, adv["frames"].shape[0], size=min(B, 16))
                fa = adv["frames"][torch.as_tensor(ai, dtype=torch.long)].float().permute(0, 3, 1, 2)
                pa = adv["prop"][torch.as_tensor(ai, dtype=torch.long)]
                ep = adv["ep_idx"][torch.as_tensor(ai, dtype=torch.long)].long()
                fg_ = adv["ep_goal_frames"][ep].float().permute(0, 3, 1, 2)
                pg_ = adv["ep_goal_prop"][ep]
                z_a = encode_grad(adapter, fa, pa)
                z_ag = encode_grad(adapter, fg_, pg_)
                oa = adv["obj"][torch.as_tensor(ai, dtype=torch.long)].to(device)
                ga = adv["goal_obj"][torch.as_tensor(ai, dtype=torch.long)].to(device)
                loss_a = margin_loss(phi(z_a)[:, :args.obj_dim], phi(z_ag)[:, :args.obj_dim],
                                     (oa - ga).norm(dim=-1), args.cf_margin_cap)

            # term 5: preservation vs the frozen encoder (LoRA toggled off)
            loss_p = z_t.new_tensor(0.0)
            if args.lambda_preserve > 0:
                set_lora_enabled(injected, False)
                with torch.no_grad():
                    z_ref = encode_grad(adapter, ft1, pt1)
                set_lora_enabled(injected, True)
                loss_p = ((z_t1 - z_ref) ** 2).mean() / ((z_ref ** 2).mean() + 1e-8)

            loss = (loss_g + args.lambda_cf * loss_cf + args.lambda_temporal * loss_t
                    + args.lambda_cross * loss_cross + args.lambda_adv * loss_a
                    + args.lambda_preserve * loss_p)
            opt.zero_grad(); loss.backward(); opt.step()
            for key, v in (("g", loss_g), ("cf", loss_cf), ("t", loss_t),
                           ("cross", loss_cross), ("a", loss_a), ("p", loss_p)):
                tot[key] += v.item() * B
            tot["n"] += B
            del z_t, z_t1
        gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
        n = max(tot["n"], 1)
        return {key: v / n for key, v in tot.items() if key != "n"}

    for ep in range(args.epochs):
        L = train_epoch()
        print(f"epoch {ep+1}/{args.epochs}: grnd={L['g']:.5f} cf={L['cf']:.4f} "
              f"temporal={L['t']:.4f} cross={L['cross']:.4f} adv={L['a']:.4f} "
              f"preserve={L['p']:.5f}", flush=True)
    phi.eval()

    # ================= held-out gates =================
    with torch.no_grad():
        errs, extras, drifts = [], [], []
        r2 = np.random.default_rng(2)
        vsel = list(val_recs); r2.shuffle(vsel)
        for lo in range(0, len(vsel), args.batch_size):
            sel = vsel[lo: lo + args.batch_size]
            if not sel:
                break
            ft, ft1, pt, pt1, at, ot1 = materialize(trajs, sel, step)
            z1 = encode_grad(adapter, ft1, pt1)
            p1 = phi(z1)
            errs.append((p1[:, :args.obj_dim].cpu() - ot1).norm(dim=-1).numpy())
            extras.append(p1[:, args.obj_dim:].cpu().numpy())
            set_lora_enabled(injected, False)
            z_ref = encode_grad(adapter, ft1, pt1)
            set_lora_enabled(injected, True)
            drifts.append(float(((z1 - z_ref) ** 2).mean() / ((z_ref ** 2).mean() + 1e-8)))
        err = np.concatenate(errs)
        extra_scale = float(np.concatenate(extras).std())
        print(f"\nphi grounding held-out (expert): median={100*np.median(err):.2f}cm "
              f"<5cm={(err < PUSH_RADIUS).mean()*100:.1f}% "
              f"<7cm={(err < PICK_RADIUS).mean()*100:.1f}%  extra_scale={extra_scale:.4f} "
              f"preserve_drift={np.mean(drifts):.5f}", flush=True)

        err_op = None
        if op is not None:
            e = []
            for lo in range(0, n_val_op, args.batch_size):
                oi = list(range(lo, min(lo + args.batch_size, n_val_op)))
                fo = torch.stack([op["frames"][i] for i in oi]).float()
                po = torch.stack([op["prop"][i] for i in oi])
                oo = torch.stack([op["obj"][i] for i in oi])
                z_op = encode_grad(adapter, fo, po)
                e.append((phi(z_op)[:, :args.obj_dim].cpu() - oo).norm(dim=-1).numpy())
            err_op = np.concatenate(e)
            print(f"phi grounding held-out (OFF-POLICY): median={100*np.median(err_op):.2f}cm "
                  f"<5cm={(err_op < PUSH_RADIUS).mean()*100:.1f}%", flush=True)

        # temporal ranking + monotone Spearman on held-out trajectories
        r3 = np.random.default_rng(4)
        accs, mono = [], []
        for _ in range(20):
            trip = sample_temporal(trajs, list(val_tids), args.temporal_batch, r3)
            if trip is None:
                continue
            fn, ff, fg, pn, pf, pg = trip
            en = phi(encode_grad(adapter, fn, pn))[:, args.obj_dim:]
            ef = phi(encode_grad(adapter, ff, pf))[:, args.obj_dim:]
            eg = phi(encode_grad(adapter, fg, pg))[:, args.obj_dim:]
            accs.append(((en - eg).norm(dim=-1) < (ef - eg).norm(dim=-1))
                        .float().mean().item())
        for ti in list(val_tids)[:16]:
            tr = trajs[ti]
            T = tr["frames"].shape[0]
            ds = []
            zgoal = encode_grad(adapter, tr["frames"][-1:].float(), tr["prop"][-1:])
            eg = phi(zgoal)[:, args.obj_dim:]
            for lo in range(0, T, args.batch_size):
                zz = encode_grad(adapter, tr["frames"][lo: lo + args.batch_size].float(),
                                 tr["prop"][lo: lo + args.batch_size])
                ds.append((phi(zz)[:, args.obj_dim:] - eg).norm(dim=-1).cpu().numpy())
            d = np.concatenate(ds)
            mono.append(_spearman(d, np.arange(T)[::-1].astype(np.float64)))
        rank_acc = float(np.mean(accs)) if accs else float("nan")
        mono_sp = float(np.nanmean(mono)) if mono else float("nan")
        print(f"temporal held-out: ranking_acc={rank_acc:.3f} "
              f"monotone_to_goal_spearman={mono_sp:.3f}", flush=True)

    # ---- save: LoRA ckpt + φ ckpt (scripts/37 format -> load_repr_adapter works) ----
    out_l = Path(args.out_lora); out_l.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model, "r": args.lora_r, "alpha": args.lora_alpha,
        "target_substrs": ("blocks", "layers"),
        "lora": encoder_lora_state_dict(injected, adapter),
        "lambda_preserve": args.lambda_preserve,
        "val_obj_median_cm": float(100 * np.median(err)),
        "val_preserve_drift": float(np.mean(drifts)),
    }, out_l)
    out_p = Path(args.out_phi); out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model, "latent_dim": latent_dim, "phi_dim": args.phi_dim,
        "obj_dim": args.obj_dim, "n_layers": args.n_layers, "n_heads": 6,
        "hidden": args.hidden, "state_dict": phi.state_dict(), "extra_scale": extra_scale,
        "val_obj_median_cm": float(100 * np.median(err)),
        "val_obj_frac_5cm": float((err < PUSH_RADIUS).mean()),
        "val_obj_frac_5cm_offpolicy": (float((err_op < PUSH_RADIUS).mean())
                                       if err_op is not None else None),
        "val_rank_acc": rank_acc, "val_mono_spearman": mono_sp,
        "encoder_lora": str(out_l),
    }, out_p)
    print(f"\nwrote {out_l}\nwrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
