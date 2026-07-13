"""Collate the regime-threshold robustness sweep into one comparison table.

slurm_regime_robustness.sh re-runs scripts/04 (classify) + scripts/05 (diagnostic,
hard_nn only) under perturbed Metaworld regime thresholds, writing one
`results/regime_robust_<tag>.csv` per config. This script pools each config's
effect-conditioned CRA (nearest-neighbour distractors) per regime — exactly the
quantity in the paper's Table `tab:cra` — so the reader can see the chance-floor
collapse at pre-grasp/contact is stable to the threshold choice, not an artifact
of the 5mm / 10cm / 0.10 cut.

Pooling mirrors the paper: effect-conditioned CRA is n_effect-weighted across the
12 tasks (mw-door-close excluded as the documented proxy anomaly).

Run (CPU, after the sweep): .venv/bin/python scripts/46_analyze_regime_robustness.py
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
REGIMES = ["free_space", "pre_grasp", "gripper_actuation", "contact_manipulation"]
EXCLUDE_TASKS = {"mw-door-close"}  # documented proxy anomaly (paper §limits)


def pooled_eff_cra(df: pd.DataFrame) -> pd.DataFrame:
    """n_effect-weighted effect-CRA per (model, regime), pooled over tasks."""
    df = df[(df["strategy"] == "hard_nn") & (~df["task"].isin(EXCLUDE_TASKS))].copy()
    df = df.dropna(subset=["cra_top1_eff", "n_effect"])
    df = df[df["n_effect"] > 0]
    rows = []
    for (model, regime), g in df.groupby(["model", "regime"]):
        w = g["n_effect"].to_numpy(float)
        v = g["cra_top1_eff"].to_numpy(float)
        rows.append({"model": model, "regime": regime,
                     "eff_cra": float(np.average(v, weights=w)),
                     "n_effect": int(w.sum())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(RES / "regime_robust_*.csv"))
    ap.add_argument("--out", default=str(RES / "regime_robustness_summary.csv"))
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")

    long = []
    for f in files:
        tag = re.sub(r"^regime_robust_|\.csv$", "", Path(f).name)
        p = pooled_eff_cra(pd.read_csv(f))
        p.insert(0, "config", tag)
        long.append(p)
    allp = pd.concat(long, ignore_index=True)
    allp.to_csv(args.out, index=False)
    print(f"wrote {args.out}\n")

    # Pretty per-model pivot: config (rows) x regime (cols) of effect-CRA.
    for model in sorted(allp["model"].unique()):
        piv = (allp[allp["model"] == model]
               .pivot(index="config", columns="regime", values="eff_cra")
               .reindex(columns=[r for r in REGIMES if r in allp["regime"].unique()]))
        print(f"===== {model}: effect-CRA (nearest-neighbour), chance = 1/17 ≈ 0.059 =====")
        print(piv.round(3).to_string())
        # Stability: spread of pre_grasp across configs (the headline cell).
        if "pre_grasp" in piv.columns:
            pg = piv["pre_grasp"].dropna()
            print(f"  pre_grasp across configs: min={pg.min():.3f} max={pg.max():.3f} "
                  f"range={pg.max()-pg.min():.3f} (baseline paper DINO 0.467 / JEPA 0.541)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
