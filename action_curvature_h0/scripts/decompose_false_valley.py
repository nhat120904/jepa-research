#!/usr/bin/env python3
"""Which half of the curvature drives false valleys: angular or radial?

The AS cosine loss only touches the ANGULAR half of the exact identity

    ||D2||^2 = (||v+|| - ||v-||)^2  +  2 ||v+|| ||v-|| (1 - cos)
               \____radial_____/       \______angular______/

so confirming that *total* curvature predicts false valleys does not yet
license it.  If the effect lives in the radial half, a cosine regularizer aims
at the wrong component.

Radial and angular are parts of one total and are therefore mutually
correlated; a univariate test on each would show an effect for both simply
because both track the total.  The design here is conditional: bin on one
component and read the contrast along the other WITHIN those bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-bins", type=int, default=3)
    p.add_argument("--min-physical-spread-m", type=float, default=1e-4)
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260825)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.shard_root.glob("*/snapshot_*/records.json")):
        summary = json.loads((path.parent / "summary.json").read_text())
        order = int(summary["snapshot"]["order"])
        for r in json.loads(path.read_text()):
            if (r.get("valid_unclipped") and "ordinal_false_valley" in r
                    and r["ordinal_physical_cost_spread"] >= args.min_physical_spread_m
                    and np.isfinite(r.get("k_model_self", np.nan))
                    and np.isfinite(r.get("model_radial_fraction", np.nan))):
                r["snapshot"] = order
                rows.append(r)
    if not rows:
        raise RuntimeError("no usable records")

    k = np.array([r["k_model_self"] for r in rows])
    frac_r = np.array([r["model_radial_fraction"] for r in rows])
    frac_a = np.array([r["model_angular_fraction"] for r in rows])
    # k^2 = k_radial^2 + k_angular^2 by the identity, so each component is the
    # total scaled by the square root of its energy fraction.
    k_rad = k * np.sqrt(np.clip(frac_r, 0.0, 1.0))
    k_ang = k * np.sqrt(np.clip(frac_a, 0.0, 1.0))
    fv = np.array([bool(r["ordinal_false_valley"]) for r in rows], dtype=float)
    groups = np.array([r["snapshot"] for r in rows])

    def qbin(x: np.ndarray, n: int) -> np.ndarray:
        edges = np.quantile(x, np.linspace(0, 1, n + 1))
        edges[-1] = np.inf
        return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)

    b_rad, b_ang = qbin(k_rad, args.n_bins), qbin(k_ang, args.n_bins)

    grid = []
    for i in range(args.n_bins):
        for j in range(args.n_bins):
            sel = (b_rad == i) & (b_ang == j)
            if sel.sum():
                grid.append({"radial_bin": i, "angular_bin": j, "n": int(sel.sum()),
                             "false_valley_rate": float(fv[sel].mean())})

    rng = np.random.default_rng(args.seed)
    uniq = np.unique(groups)

    def conditional_contrast(bin_hold: np.ndarray, bin_vary: np.ndarray,
                             idx: np.ndarray | None = None) -> float:
        """Top-minus-bottom along `bin_vary`, pooled within levels of `bin_hold`."""
        bh = bin_hold if idx is None else bin_hold[idx]
        bv = bin_vary if idx is None else bin_vary[idx]
        y = fv if idx is None else fv[idx]
        diffs, weights = [], []
        for level in range(args.n_bins):
            lo = (bh == level) & (bv == 0)
            hi = (bh == level) & (bv == args.n_bins - 1)
            if lo.sum() and hi.sum():
                diffs.append(y[hi].mean() - y[lo].mean())
                weights.append(lo.sum() + hi.sum())
        if not diffs:
            return float("nan")
        return float(np.average(diffs, weights=weights))

    out: dict[str, Any] = {}
    for name, hold, vary in (("angular_holding_radial", b_rad, b_ang),
                             ("radial_holding_angular", b_ang, b_rad)):
        point = conditional_contrast(hold, vary)
        draws = np.empty(args.n_resamples)
        for t in range(args.n_resamples):
            pick = rng.choice(uniq, size=uniq.size, replace=True)
            idx = np.concatenate([np.nonzero(groups == g)[0] for g in pick])
            draws[t] = conditional_contrast(hold, vary, idx)
        fin = draws[np.isfinite(draws)]
        out[name] = {
            "conditional_top_minus_bottom": point,
            "ci_low": float(np.percentile(fin, 2.5)) if fin.size else None,
            "ci_high": float(np.percentile(fin, 97.5)) if fin.size else None,
            "excludes_zero": bool(fin.size and (np.percentile(fin, 2.5) > 0
                                                or np.percentile(fin, 97.5) < 0)),
        }

    ang = out["angular_holding_radial"]
    rad = out["radial_holding_angular"]
    if ang["excludes_zero"] and not rad["excludes_zero"]:
        verdict = "ANGULAR_DOMINATES_COSINE_AS_IS_ON_TARGET"
    elif rad["excludes_zero"] and not ang["excludes_zero"]:
        verdict = "RADIAL_DOMINATES_COSINE_AS_IS_OFF_TARGET"
    elif ang["excludes_zero"] and rad["excludes_zero"]:
        verdict = "BOTH_COMPONENTS_USE_COMBINED_CORRECTION"
    else:
        verdict = "NEITHER_COMPONENT_RESOLVED_INSUFFICIENT_POWER"

    report = {
        "counts": {"records": len(rows), "snapshots": int(uniq.size)},
        "energy_fraction": {"median_radial": float(np.median(frac_r)),
                            "median_angular": float(np.median(frac_a))},
        "grid_radial_x_angular": grid,
        "conditional_contrasts": out,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
