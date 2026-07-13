"""Train the OBJECT-GROUNDED residual corrector Δ(z,a) so the corrected object
channel is usable INSIDE the planner's rollout — the post-oracle-ladder fix
(docs/plans/2026-06-25-grounded-corrector-in-loop-design.md).

    ẑ_{t+1} = F_frozen(z_t, a_t) + Δ_θ(z_t, a_t)

The oracle ladder (scripts/29 state-oracle solves contact; scripts/30 latent-
oracle with perfect dynamics + the upstream L2-latent cost is still 0/16) localised
the contact wall to the **L2-in-DINO-latent cost geometry**, not the predictor or
the planner. The earlier residual corrector (scripts/20) was trained single-step
and only ever scored by L2 (the `l2c`/`hdync` arms) — exactly the cost the latent-
oracle proves cannot work. This trainer keeps the residual but adds the two pieces
the closed-loop null needs:

  recon   ‖ẑ_{t+1} − z_{t+1}‖²                          keep the latent sane
  λ_obj   ‖g(ẑ_{t+1}) − obj_{t+1}‖²                     1-step object grounding (frozen probe g)
  λ_cf    ‖g(ẑ(z,a')) − (g(z)+h(z,a'))‖²               distil the cf-corr-0.68 dyn-head h
                                                          into the corrected object channel
  λ_ms    Σ_{k=1..K} ‖g(ẑ_k) − obj_{t+k}‖²             K-STEP recursive grounding so the
                                                          object channel survives the unroll
                                                          ("readout-fixed ≠ rollout-fixed",
                                                          docs/plans/2026-06-24-object-aware-predictor-design.md)

Everything frozen except Δ (zero-init output → starts at the frozen predictor).
Pairs with the `gobjc` arm in scripts/18 (cost reads g(corrected ẑ_t) along the
horizon). Gates before it is worth a closed-loop spend: cf-corr ≥ --min-cf-corr,
and corrected object-MSE (1-step AND K-step) below the frozen baseline.

    .venv/bin/python scripts/31_train_grounded_corrector.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
        --probe checkpoints/object_probe_dino_wm_metaworld.pt \
        --dyn-head checkpoints/object_dynamics_dino_wm_metaworld.pt \
        --epochs 6 --lambda-obj 10 --lambda-cf 5 --lambda-ms 5 --K 3
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
from models.adapters import build_adapter  # noqa: E402
from models.heads import ResidualPredictorHead  # noqa: E402
from models.probes import load_probe, load_dynamics_head  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers, compute_true_outcome  # noqa: E402
from stratification import state_neighbours, boundary_score_per_transition  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


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


def build_sequences(cache, step, H, tids, stride, max_seqs, seed):
    """K-step rollout tensors from contiguous slices (mirrors scripts/23):
    (z0 (n,*frame), prop0 (n,P), actions (n,H,A_stack), obj (n,H+1,3))."""
    Z0, P0, ACTS, OBJ = [], [], [], []
    for tid in tids:
        traj = cache.read_trajectory(tid)
        z, act = traj["z"], traj["action"]
        state = traj.get("state"); prop = traj.get("proprio")
        if state is None:
            continue
        La = len(act)
        for i in range(0, La - H * step + 1, stride * step):
            acts = np.stack([act[i + h * step: i + (h + 1) * step].reshape(-1)
                             for h in range(H)])                    # (H, A_stack)
            obj = np.stack([state[i + h * step][OBJECT_SLICE] for h in range(H + 1)])
            Z0.append(z[i]); ACTS.append(acts); OBJ.append(obj)
            P0.append(prop[i] if prop is not None else np.zeros(1, np.float32))
    if not Z0:
        return (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
    z0 = torch.tensor(np.stack(Z0)).float()
    p0 = torch.tensor(np.stack(P0)).float()
    acts = torch.tensor(np.stack(ACTS)).float()
    obj = torch.tensor(np.stack(OBJ)).float()
    if len(z0) > max_seqs:
        sel = np.random.default_rng(seed).choice(len(z0), max_seqs, replace=False)
        z0, p0, acts, obj = z0[sel], p0[sel], acts[sel], obj[sel]
    return z0, p0, acts, obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True, help="frozen object probe (scripts/14)")
    ap.add_argument("--dyn-head", required=True, help="frozen cf teacher h (scripts/17)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-obj", type=float, default=10.0)
    ap.add_argument("--lambda-cf", type=float, default=5.0)
    ap.add_argument("--lambda-ms", type=float, default=5.0)
    ap.add_argument("--K", type=int, default=3, help="multistep horizon for the rollout grounding")
    ap.add_argument("--ms-batch", type=int, default=16, help="K-step sequences per optimiser step")
    ap.add_argument("--stride", type=int, default=4, help="K-step start stride (in model-steps)")
    ap.add_argument("--max-seqs", type=int, default=2000)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-cf-corr", type=float, default=0.5,
                    help="gate: corr(spread of corrected object readout, true outcome spread)")
    ap.add_argument("--out-dir", default="checkpoints")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    dataset_name = cfg["dataset"]["name"]
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, dataset_name)
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    a_dim = adapter.action_dim()
    ma_dim = adapter._model_action_dim
    probe, _ = load_probe(args.probe, device)
    dyn_head, dyn_meta = load_dynamics_head(args.dyn_head, device)
    print(f"teacher h: cf_corr={dyn_meta.get('cf_corr', float('nan')):+.3f} "
          f"out_dim={dyn_head.out_dim}", flush=True)
    regime_by_traj = read_regimes(cache_path)
    rng = np.random.default_rng(args.seed)

    def to_model_action(a_raw: torch.Tensor) -> torch.Tensor:
        """Raw stacked action (B, *) -> normalised model-space action (B, ma_dim)."""
        B = a_raw.shape[0]
        a = adapter.normalize_action(a_raw.to(device).float().reshape(B, -1, a_dim))
        return a.reshape(B, ma_dim)

    def frozen_step(z_cur, a_raw_step, prop_feat):
        """One frozen predictor step (no grad). Returns (z_next_visual, next_prop_feat)."""
        from einops import rearrange
        from tensordict.tensordict import TensorDict
        b = adapter
        a_t = to_model_action(a_raw_step)                              # (B, ma_dim)
        act_suffix = a_t.unsqueeze(0)                                  # (1, B, A)
        with torch.no_grad():
            if b.spec.uses_proprio and prop_feat is not None:
                ctxt = TensorDict({"visual": z_cur.unsqueeze(1), "proprio": prop_feat},
                                  batch_size=[])
                pred = b.encpred.unroll(ctxt, act_suffix=act_suffix)
                z_next = pred["visual"][-1]
                nxt = rearrange(pred["proprio"], "t b ... -> b t ...")[:, -1:]
            else:
                z_next = b.encpred.unroll(z_cur.unsqueeze(1), act_suffix=act_suffix)[-1]
                nxt = None
        return z_next.detach(), (nxt.detach() if nxt is not None else None), a_t

    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=True)
        train_recs, val_recs = split_by_trajectory(records, args.val_frac, args.seed)
        train_tids = sorted({r["tid"] for r in train_recs})
        print(f"transitions: train={len(train_recs)} val={len(val_recs)}", flush=True)

        d0 = helpers.materialize_records(cache, train_recs[:2], step,
                                         want_proprio=adapter.uses_proprio(), want_state=True)
        latent_dim = int(d0["z_t"].shape[-1])
        del d0
        head = ResidualPredictorHead(latent_dim=latent_dim, action_dim=ma_dim,
                                     n_layers=args.n_layers).to(device).train()
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"residual head params: {n_params/1e6:.2f}M (latent_dim={latent_dim} "
              f"action_dim={ma_dim} layers={args.n_layers})", flush=True)

        # K-step rollout tensors (grounding the corrected object channel through the unroll).
        ms_z0, ms_p0, ms_acts, ms_obj = build_sequences(
            cache, step, args.K, train_tids, args.stride, args.max_seqs, args.seed)
        print(f"K-step sequences: {len(ms_z0)} (K={args.K})", flush=True)

        def ms_loss(idx):
            """Recursive corrected rollout grounded to the true object at each step.
            Δ has grad; the frozen predictor and its proprio context do not."""
            z_cur = ms_z0[idx].to(device).float()
            acts = ms_acts[idx].to(device).float()                    # (b, K, A_stack)
            prop0 = ms_p0[idx].to(device).float()
            uses_p = adapter.spec.uses_proprio
            prop_feat = (adapter.encode_proprio_features(prop0.reshape(len(idx), 1, -1))
                         if uses_p else None)
            obj = ms_obj[idx].to(device).float()                      # (b, K+1, 3)
            loss = 0.0
            for t in range(args.K):
                z_next, prop_feat, a_t = frozen_step(z_cur, acts[:, t], prop_feat)
                z_cur = z_next + head(z_cur, a_t)                     # corrected (Δ has grad)
                loss = loss + ((probe(z_cur) - obj[:, t + 1]) ** 2).mean()
            return loss / args.K

        def run_batch(z_t, a_raw, prop_t, z_t1, obj_t1, train):
            base = adapter.predict(z_t, a_raw, proprio_t=prop_t)      # frozen, detached
            a_norm = to_model_action(a_raw)
            z_hat = base + head(z_t.to(device).float(), a_norm)
            recon = ((z_hat - z_t1.to(device).float()) ** 2).mean()
            g_hat = probe(z_hat)
            obj = ((g_hat - obj_t1.to(device).float()) ** 2).mean()
            # Counterfactual distillation: shuffle actions within the batch so each
            # anchor sees a *different* action a', and match the corrected object
            # readout to the teacher's g(z)+h(z,a') (h carries the cf-corr-0.68 signal).
            B = z_t.shape[0]
            cf = torch.zeros((), device=device)
            if B > 1 and args.lambda_cf > 0:
                perm = torch.randperm(B)
                a_cf_raw = a_raw[perm]
                a_cf = to_model_action(a_cf_raw)
                base_cf = adapter.predict(z_t, a_cf_raw, proprio_t=prop_t)
                z_cf = base_cf + head(z_t.to(device).float(), a_cf)
                with torch.no_grad():
                    g_z = probe(z_t.to(device).float())
                    target = g_z + dyn_head(z_t.to(device).float(), a_cf)
                cf = ((probe(z_cf) - target) ** 2).mean()
            loss = recon + args.lambda_obj * obj + args.lambda_cf * cf
            if train and len(ms_z0) and args.lambda_ms > 0:
                midx = torch.as_tensor(
                    np.random.default_rng().choice(len(ms_z0),
                                                   min(args.ms_batch, len(ms_z0)),
                                                   replace=False), dtype=torch.long)
                loss = loss + args.lambda_ms * ms_loss(midx)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            return float(recon), float(obj), float(cf)

        def epoch_pass(recs, train):
            sr = so = sc = n = 0.0
            for sel in iter_chunks(recs, args.chunk, rng if train else np.random.default_rng(1)):
                d = helpers.materialize_records(cache, sel, step,
                                                want_proprio=adapter.uses_proprio(),
                                                want_state=True)
                m = d["z_t"].shape[0]
                order = np.arange(m)
                if train:
                    rng.shuffle(order)
                obj1 = d["state_t1"][:, OBJECT_SLICE].float()
                for lo in range(0, m, args.batch_size):
                    idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
                    prop = (d["proprio_t"][idx].to(device)
                            if d.get("proprio_t") is not None else None)
                    r, o, c = run_batch(d["z_t"][idx], d["a_t"][idx], prop,
                                        d["z_t1"][idx], obj1[idx], train)
                    sr += r * len(idx); so += o * len(idx); sc += c * len(idx); n += len(idx)
                del d
                gc.collect()
            return sr / max(n, 1), so / max(n, 1), sc / max(n, 1)

        # Frozen baseline (Δ≡0) object error — the headline comparison.
        head.eval()
        with torch.no_grad():
            base_r, base_o, _ = epoch_pass(val_recs, False)
        head.train()
        print(f"frozen baseline (Δ=0): val recon={base_r:.5f} obj_MSE={base_o:.6f}", flush=True)

        vr = vo = vc = base_o
        for ep in range(args.epochs):
            tr, to, tc = epoch_pass(train_recs, True)
            head.eval()
            with torch.no_grad():
                vr, vo, vc = epoch_pass(val_recs, False)
            head.train()
            print(f"epoch {ep+1}/{args.epochs}: train recon={tr:.5f} obj={to:.6f} cf={tc:.6f} | "
                  f"val recon={vr:.5f} obj={vo:.6f} cf={vc:.6f} "
                  f"(obj vs frozen: {vo/max(base_o,1e-9):.3f}x)", flush=True)
        head.eval()

        # ---- K-step corrected object error (gate 2: rollout-grounded, not just readout) ----
        ms_obj_mse = float("nan")
        if len(ms_z0):
            with torch.no_grad():
                tot = 0.0
                vsel = np.random.default_rng(7).choice(
                    len(ms_z0), min(256, len(ms_z0)), replace=False)
                tot = float(ms_loss(torch.as_tensor(vsel, dtype=torch.long)))
                ms_obj_mse = tot
            print(f"K-step corrected object MSE (val sample): {ms_obj_mse:.6f}", flush=True)

        # ---- counterfactual gate: does the CORRECTED object channel track the bifurcation? ----
        pool_idx = np.random.default_rng(0).choice(
            np.arange(len(records)), size=min(cfg["hard_nn"]["pool_size"], len(records)),
            replace=False)
        pool = helpers.materialize_records(cache, [records[int(i)] for i in pool_idx],
                                           step, want_proprio=False, want_state=True)
        pool_out = compute_true_outcome(dataset_name, pool["z_t"], pool["z_t1"],
                                        pool.get("state_t"), pool.get("state_t1"))
        sub = [val_recs[int(i)] for i in np.random.default_rng(3).choice(
            len(val_recs), size=min(512, len(val_recs)), replace=False)]
        d = helpers.materialize_records(cache, sub, step, want_proprio=False, want_state=True)
        idx, mask, _ = state_neighbours(d["z_t"], cfg["hard_nn"]["similarity_radius"],
                                        max_neighbours=16, pool_z=pool["z_t"])
        bscore = boundary_score_per_transition(
            d["a_t"], pool["a_t"][idx], torch.as_tensor(pool_out)[idx], mask)
        thr = np.nanquantile(bscore, cfg.get("boundary", {}).get("quantile", 0.75))
        bsel = np.where(np.isfinite(bscore) & (bscore > thr))[0][:128]
        spreads_h, spreads_true = [], []
        with torch.no_grad():
            for i in bsel.tolist():
                acts = pool["a_t"][idx[i]][mask[i]]
                m = acts.shape[0]
                if m < 2:
                    continue
                z_rep = d["z_t"][i: i + 1].expand(m, *d["z_t"].shape[1:]).to(device).float()
                base = adapter.predict(z_rep, acts)                   # frozen one-step
                z_corr = base + head(z_rep, to_model_action(acts))
                gv = probe(z_corr)                                    # corrected object readout
                spreads_h.append(float((gv - gv.mean(0, keepdim=True))
                                       .norm(dim=-1).pow(2).mean().sqrt()))
                spreads_true.append(float(torch.as_tensor(pool_out)[idx[i]][mask[i]].std()))
        sh, st = np.asarray(spreads_h), np.asarray(spreads_true)
        cf_corr = float(np.corrcoef(sh, st)[0, 1]) if len(sh) > 2 else float("nan")
        print(f"cf gate: n={len(sh)} median spread_h={np.median(sh):.4f} "
              f"median spread_true={np.median(st):.4f} corr={cf_corr:+.3f} "
              f"(min required {args.min_cf_corr})", flush=True)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"grounded_corrector_{args.model}.pt"
    torch.save({
        "model": args.model, "latent_dim": latent_dim, "action_dim": ma_dim,
        "n_layers": args.n_layers, "n_heads": 6, "ff_mult": 2,
        "state_dict": head.state_dict(),
        "lambda_obj": args.lambda_obj, "lambda_cf": args.lambda_cf,
        "lambda_ms": args.lambda_ms, "K": args.K,
        "frozen_obj_mse": base_o, "corrected_obj_mse": vo,
        "frozen_recon": base_r, "corrected_recon": vr,
        "multistep_obj_mse": ms_obj_mse, "cf_corr": cf_corr,
    }, path)
    gate_obj = vo < base_o
    gate_cf = np.isfinite(cf_corr) and cf_corr >= args.min_cf_corr
    print(f"\nwrote {path}\n  obj corrected/frozen = {vo/max(base_o,1e-9):.3f}x "
          f"({base_o:.6f} -> {vo:.6f})  K-step obj_MSE={ms_obj_mse:.6f}\n"
          f"  GATES: obj<frozen={gate_obj}  cf_corr>={args.min_cf_corr}={gate_cf} "
          f"(corr={cf_corr:+.3f})", flush=True)
    if not (gate_obj and gate_cf):
        print("  WARNING: a gate failed — do NOT spend closed-loop budget on this ckpt "
              "until the corrected object channel both beats the frozen baseline and "
              "tracks the bifurcation (cf-corr).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
