"""Train the residual corrective predictor Δ(z,a) on cached Metaworld latents —
the predictor-side fix (option C, docs/plans/2026-06-18-residual-predictor-design.md).

    ẑ_{t+1} = F_frozen(z_t, a_t) + Δ_θ(z_t, a_t)

Loss = latent recon + λ_obj · object-grounded:
    ‖ẑ_{t+1} − z_{t+1}‖²  +  λ_obj · ‖g(ẑ_{t+1}) − obj_{t+1}‖²
g = the frozen object probe (scripts/14), obj_{t+1} = cached sim state. The
recon term keeps the corrected latent near the true next latent; the object term
forces it to DECODE to the right object position — baking the cf-corr-0.682
action→object signal into the latent itself, where the planner's L2 cost sees it.

Everything frozen except Δ (zero-init output → starts at the frozen predictor).
Reports the headline number: corrected vs frozen one-step object-decode error.

    .venv/Scripts/python.exe scripts/20_train_residual_predictor.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
        --probe checkpoints/object_probe_dino_wm_metaworld.pt --epochs 6 --lambda-obj 10
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
from models.probes import load_probe  # noqa: E402
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
    ap.add_argument("--probe", required=True, help="frozen object probe (scripts/14)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-obj", type=float, default=10.0)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints")
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
    probe, _ = load_probe(args.probe, device)
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
              f"action_dim={ma_dim} layers={args.n_layers})", flush=True)

        def run_batch(z_t, a_raw, prop_t, z_t1, obj_t1, train):
            base = adapter.predict(z_t, a_raw, proprio_t=prop_t)        # frozen, detached
            a_norm = to_model_action(a_raw)
            z_hat = base + head(z_t.to(device).float(), a_norm)
            recon = ((z_hat - z_t1.to(device).float()) ** 2).mean()
            obj = ((probe(z_hat) - obj_t1.to(device).float()) ** 2).mean()
            loss = recon + args.lambda_obj * obj
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            return float(recon), float(obj)

        def epoch_pass(recs, train):
            sr = so = n = 0.0
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
                    r, o = run_batch(d["z_t"][idx], d["a_t"][idx], prop,
                                     d["z_t1"][idx], obj1[idx], train)
                    sr += r * len(idx); so += o * len(idx); n += len(idx)
                del d
                gc.collect()
            return sr / max(n, 1), so / max(n, 1)

        # Frozen baseline object error (Δ≡0) for the headline comparison.
        head.eval()
        with torch.no_grad():
            base_r, base_o = epoch_pass(val_recs, False)
        head.train()
        print(f"frozen baseline (Δ=0): val recon={base_r:.5f} val obj_MSE={base_o:.6f}", flush=True)

        for ep in range(args.epochs):
            tr, to = epoch_pass(train_recs, True)
            head.eval()
            with torch.no_grad():
                vr, vo = epoch_pass(val_recs, False)
            head.train()
            print(f"epoch {ep+1}/{args.epochs}: train recon={tr:.5f} obj={to:.6f} | "
                  f"val recon={vr:.5f} obj={vo:.6f} "
                  f"(obj vs frozen: {vo/max(base_o,1e-9):.3f}x)", flush=True)
        head.eval()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"residual_predictor_{args.model}.pt"
    torch.save({
        "model": args.model, "latent_dim": latent_dim, "action_dim": ma_dim,
        "n_layers": args.n_layers, "n_heads": 6, "ff_mult": 2,
        "state_dict": head.state_dict(), "lambda_obj": args.lambda_obj,
        "frozen_obj_mse": base_o, "corrected_obj_mse": vo,
        "frozen_recon": base_r, "corrected_recon": vr,
    }, path)
    print(f"\nwrote {path}\n  object error corrected/frozen = {vo/max(base_o,1e-9):.3f}x "
          f"({base_o:.6f} -> {vo:.6f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
