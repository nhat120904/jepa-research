"""Representation precision diagnostic — is the object decodable to the PRECISION
the contact task needs? (substantiates the option-C bottleneck claim;
docs/plans/2026-06-18-residual-predictor-design.md §5, decision rule branch 2).

The object probe proved object position is PRESENT in the frozen latent (V1
passes). But "present" != "precise enough to plan contact". This measures the
held-out object-decode error distribution and compares it to the Metaworld
success radii:
    push:        success when ‖obj - target‖ <= 0.05 m   (TARGET_RADIUS)
    pick-place:  success when ‖obj - target‖ <= 0.07 m

If the representation cannot even localise a STATIC object to within the success
radius, then no predictor or planner can place it there — the bottleneck is the
representation's precision, not the absence of information.

CPU + cache + the 0.5M object probe only — no model load, no VRAM (safe to run
alongside a GPU sweep). Reproduces script 14's held-out trajectory split.

    CUDA_VISIBLE_DEVICES= .venv/Scripts/python.exe scripts/21_representation_precision.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
        --probe checkpoints/object_probe_dino_wm_metaworld.pt
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from data.latent_cache import ID_TO_REGIME  # noqa: E402
from models.probes import load_probe  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402

PUSH_RADIUS = 0.05
PICK_RADIUS = 0.07


def split_by_trajectory(records, val_frac, seed):
    tids = sorted({r["tid"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(tids)
    val_tids = set(tids[: max(1, int(len(tids) * val_frac))])
    return [r for r in records if r["tid"] in val_tids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--step", type=int, default=5,
                    help="frames_per_step (dino_wm_metaworld=5); avoids loading the model")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/representation_precision.csv")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    import yaml
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cpu")          # CPU only — no VRAM contention with a sweep
    helpers = _load_runner_helpers()
    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model,
                                   cfg["dataset"]["name"])
    probe, pmeta = load_probe(args.probe, device)
    regime_by_traj = read_regimes(cache_path)

    errs, regimes = [], []
    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, args.step, per_task=True)
        val = split_by_trajectory(records, args.val_frac, args.seed)
        print(f"held-out transitions: {len(val)} (of {len(records)})", flush=True)
        for lo in range(0, len(val), args.chunk):
            sel = sorted(val[lo: lo + args.chunk], key=lambda r: r["tid"])
            d = helpers.materialize_records(cache, sel, args.step,
                                            want_proprio=False, want_state=True)
            obj_true = d["state_t1"][:, OBJECT_SLICE].float()
            with torch.no_grad():
                obj_pred = probe(d["z_t1"].to(device).float())
            e = (obj_pred - obj_true).norm(dim=-1).cpu().numpy()      # (n,) metres
            errs.append(e)
            regimes.extend(int(r["regime"]) for r in sel)
            del d
            gc.collect()
    err = np.concatenate(errs)
    reg = np.asarray(regimes)

    def line(name, e):
        return {"group": name, "n": len(e),
                "median_cm": 100 * float(np.median(e)),
                "mean_cm": 100 * float(e.mean()),
                "p90_cm": 100 * float(np.percentile(e, 90)),
                "frac_within_push_5cm": float((e < PUSH_RADIUS).mean()),
                "frac_within_pick_7cm": float((e < PICK_RADIUS).mean())}

    rows = [line("ALL", err)]
    for rid in sorted(set(reg.tolist())):
        rows.append(line(ID_TO_REGIME.get(rid, str(rid)), err[reg == rid]))

    print(f"\nobject per-dim sd (workspace spread) ≈ {100*pmeta.get('object_sd', float('nan')):.1f} cm")
    print(f"success radius: push 5.0 cm, pick-place 7.0 cm\n")
    print(f"{'group':16s} {'n':>5s} {'med':>6s} {'mean':>6s} {'p90':>6s} "
          f"{'<5cm':>6s} {'<7cm':>6s}")
    for r in rows:
        print(f"{r['group']:16s} {r['n']:5d} {r['median_cm']:6.2f} {r['mean_cm']:6.2f} "
              f"{r['p90_cm']:6.2f} {r['frac_within_push_5cm']*100:5.1f}% "
              f"{r['frac_within_pick_7cm']*100:5.1f}%", flush=True)

    import pandas as pd
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
