"""Train the COUNTERFACTUAL corrective predictor Δ(z,a) — option D
(docs/plans/2026-06-23-predictor-cf-channel-design.md).

    ẑ_{t+1} = F_frozen(z_t, a_t) + Δ_θ(z_t, a_t)

Option C (scripts/20) trained Δ with `recon + λ_obj·‖g_pooled(ẑ)−obj_{t+1}‖²` and
was an illusory null: the POOLED probe was gameable, and a purely FACTUAL object
term never supervised the dead channel — that different actions from the same
state give different objects. Option D fixes both:

  * decode with the SPATIAL probe g (scripts/22, 2.1 cm, ungameable);
  * add the decisive COUNTERFACTUAL distillation term: for a counterfactual action
    a' applied to the SAME state, force the corrected rollout's decoded object to
    match the dyn-head teacher h(z,a') (scripts/17, cf-corr +0.68 — the proven
    action-discriminative object model). obj_t is cached, so the teacher's absolute
    next-object under a' is  obj_t + h(z_t, a').

  L = ‖ẑ(z,a) − z_{t+1}‖²                            (recon: keep the latent sane)
    + λ_obj · ‖g(ẑ(z,a))  − obj_{t+1}‖²              (FACTUAL object, TRUE target)
    + λ_cf  · ‖g(ẑ(z,a')) − (obj_t + h(z,a'))‖²      (COUNTERFACTUAL, dyn-head target)

Everything frozen except Δ (zero-init output → starts at the frozen predictor).
The saved checkpoint is the residual-head format scripts/18 `--residual-head`
(and scripts/23/24 `--residual-head`) already load, so the gates run unchanged.

    .venv/bin/python scripts/25_train_cf_predictor.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
        --spatial-probe checkpoints/spatial_object_probe_dino_wm_metaworld.pt \
        --dyn-head checkpoints/object_dynamics_dino_wm_metaworld.pt \
        --epochs 8 --lambda-obj 10 --lambda-cf 10
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
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--spatial-probe", required=True, help="frozen SPATIAL object probe (scripts/22)")
    ap.add_argument("--dyn-head", required=True, help="frozen dyn-head h(z,a)→Δobj teacher (scripts/17)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-obj", type=float, default=10.0)
    ap.add_argument("--lambda-cf", type=float, default=10.0)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--out", default=None,
                    help="explicit checkpoint path (default checkpoints/cf_predictor_<model>.pt)")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model,
                                   cfg["dataset"]["name"])
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    a_dim = adapter.action_dim()
    ma_dim = adapter._model_action_dim
    probe, probe_meta = load_probe(args.spatial_probe, device)
    if probe_meta.get("probe_type") != "spatial":
        print("WARNING: --spatial-probe is not a spatial probe; option D needs the "
              "ungameable spatial readout (the pooled probe is what made option C illusory).",
              flush=True)
    dyn_head, dyn_meta = load_dynamics_head(args.dyn_head, device)
    for p in dyn_head.parameters():
        p.requires_grad_(False)
    print(f"teacher dyn-head: cf_corr={dyn_meta.get('cf_corr'):.3f} "
          f"cf_spread_h={100*dyn_meta.get('cf_spread_h', float('nan')):.2f}cm "
          f"cf_spread_true={100*dyn_meta.get('cf_spread_true', float('nan')):.2f}cm", flush=True)
    regime_by_traj = read_regimes(cache_path)
    rng = np.random.default_rng(args.seed)

    def to_model_action(a_raw: torch.Tensor) -> torch.Tensor:
        """Raw stacked action (B, ma_dim) -> normalised model-space action."""
        B = a_raw.shape[0]
        a = adapter.normalize_action(a_raw.to(device).float().reshape(B, -1, a_dim))
        return a.reshape(B, ma_dim)

    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=True)
        train_recs, val_recs = split_by_trajectory(records, args.val_frac, args.seed)
        print(f"transitions: train={len(train_recs)} val={len(val_recs)}", flush=True)

        d0 = helpers.materialize_records(cache, train_recs[:2], step,
                                         want_proprio=adapter.uses_proprio(), want_state=True)
        latent_dim = int(d0["z_t"].shape[-1])
        del d0
        head = ResidualPredictorHead(latent_dim=latent_dim, action_dim=ma_dim,
                                     n_layers=args.n_layers).to(device).train()
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"residual head params: {n_params/1e6:.2f}M  (latent_dim={latent_dim} "
              f"action_dim={ma_dim} layers={args.n_layers})  "
              f"lambda_obj={args.lambda_obj} lambda_cf={args.lambda_cf}", flush=True)

        def run_batch(z_t, a_raw, prop_t, z_t1, obj_t, obj_t1, train):
            z_t = z_t.to(device).float()
            B = z_t.shape[0]
            # --- factual: corrected rollout must decode to the TRUE next object ---
            base = adapter.predict(z_t, a_raw, proprio_t=prop_t)           # frozen, detached
            a_norm = to_model_action(a_raw)
            z_hat = base + head(z_t, a_norm)
            recon = ((z_hat - z_t1.to(device).float()) ** 2).mean()
            obj_fac = ((probe(z_hat) - obj_t1.to(device).float()) ** 2).mean()

            # --- counterfactual: a' (in-batch shuffle) applied to the SAME state z_t;
            #     corrected rollout must decode to the dyn-head teacher's object ---
            perm = torch.randperm(B, device=device) if train else torch.roll(
                torch.arange(B, device=device), 1)
            a_cf_raw = a_raw[perm.cpu()]
            a_cf_norm = to_model_action(a_cf_raw)
            base_cf = adapter.predict(z_t, a_cf_raw, proprio_t=prop_t)     # frozen
            z_hat_cf = base_cf + head(z_t, a_cf_norm)
            with torch.no_grad():
                teacher_cf = obj_t.to(device).float() + dyn_head(z_t, a_cf_norm)
            obj_cf = ((probe(z_hat_cf) - teacher_cf) ** 2).mean()

            loss = recon + args.lambda_obj * obj_fac + args.lambda_cf * obj_cf
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            return float(recon), float(obj_fac), float(obj_cf)

        def epoch_pass(recs, train):
            sr = sf = sc = n = 0.0
            for sel in iter_chunks(recs, args.chunk, rng if train else np.random.default_rng(1)):
                d = helpers.materialize_records(cache, sel, step,
                                                want_proprio=adapter.uses_proprio(),
                                                want_state=True)
                m = d["z_t"].shape[0]
                order = np.arange(m)
                if train:
                    rng.shuffle(order)
                obj0 = d["state_t"][:, OBJECT_SLICE].float()
                obj1 = d["state_t1"][:, OBJECT_SLICE].float()
                for lo in range(0, m, args.batch_size):
                    idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
                    if len(idx) < 2:
                        continue
                    prop = (d["proprio_t"][idx].to(device)
                            if d.get("proprio_t") is not None else None)
                    r, f, c = run_batch(d["z_t"][idx], d["a_t"][idx], prop,
                                        d["z_t1"][idx], obj0[idx], obj1[idx], train)
                    sr += r * len(idx); sf += f * len(idx); sc += c * len(idx); n += len(idx)
                del d
                gc.collect()
            return sr / max(n, 1), sf / max(n, 1), sc / max(n, 1)

        # Frozen baseline (Δ≡0): factual object error AND the counterfactual gap to
        # the teacher — the two numbers option D must move.
        head.eval()
        with torch.no_grad():
            base_r, base_f, base_c = epoch_pass(val_recs, False)
        head.train()
        print(f"frozen baseline (Δ=0): val recon={base_r:.5f} obj_fac={base_f:.6f} "
              f"obj_cf(vs teacher)={base_c:.6f}", flush=True)

        vr = vf = vc = float("nan")
        for ep in range(args.epochs):
            tr, tf, tc = epoch_pass(train_recs, True)
            head.eval()
            with torch.no_grad():
                vr, vf, vc = epoch_pass(val_recs, False)
            head.train()
            print(f"epoch {ep+1}/{args.epochs}: train recon={tr:.5f} fac={tf:.6f} cf={tc:.6f} | "
                  f"val recon={vr:.5f} fac={vf:.6f} cf={vc:.6f} "
                  f"(fac {vf/max(base_f,1e-9):.3f}x  cf {vc/max(base_c,1e-9):.3f}x)", flush=True)
        head.eval()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else out_dir / f"cf_predictor_{args.model}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model, "latent_dim": latent_dim, "action_dim": ma_dim,
        "n_layers": args.n_layers, "n_heads": 6, "ff_mult": 2,
        "state_dict": head.state_dict(),
        "lambda_obj": args.lambda_obj, "lambda_cf": args.lambda_cf,
        "frozen_obj_fac": base_f, "corrected_obj_fac": vf,
        "frozen_obj_cf": base_c, "corrected_obj_cf": vc,
        "frozen_recon": base_r, "corrected_recon": vr,
        "teacher_cf_corr": dyn_meta.get("cf_corr"),
    }, path)
    print(f"\nwrote {path}\n  factual object error corrected/frozen = "
          f"{vf/max(base_f,1e-9):.3f}x  ({base_f:.6f} -> {vf:.6f})\n"
          f"  counterfactual-to-teacher gap corrected/frozen = "
          f"{vc/max(base_c,1e-9):.3f}x  ({base_c:.6f} -> {vc:.6f})\n"
          f"  -> gate on scripts/24 --residual-head (cf-corr) + scripts/23 (factual drift) "
          f"BEFORE closed-loop.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
