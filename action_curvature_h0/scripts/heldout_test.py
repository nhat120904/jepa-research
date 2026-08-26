#!/usr/bin/env python3
"""Held-out confirmatory test for continuation, per the thirteenth amendment.

Gate 1 (manipulation check): continuation's angular curvature below original's.
Gate 2 (primary, mechanism) : per snapshot, false_valley rate for the original
    versus the mean across the three continuation seeds on the identical
    triplets; snapshot-clustered bootstrap on the paired difference.
    Confirmed if continuation - original < 0 and the 95% CI excludes zero.

Gate 2 is not read unless Gate 1 passes.

Angular curvature per record is

    k_angular = sqrt(angular_energy) / (span + EPS)

which equals ``k_model_self * sqrt(model_angular_fraction)`` whenever the
fraction is defined.  ``model_angular_fraction`` is 0/0 when the total
second-difference energy ``||D2||^2`` falls to or below ``EPS = 1e-12``, i.e.
when the map is locally straight to numerical precision.  The fraction is
undefined there but ``k_angular`` is not: ``angular_energy <= ||D2||^2 <= EPS``
forces ``k_angular <= sqrt(EPS) / span``, so the zero-curvature limit is
``k_angular = 0``.  ``--degenerate-policy`` selects that limit (``zero``, the
locked default), its worst-case upper bound (``upper``, all of the vanishing
energy attributed to the angular part, ``k_angular = k_model_self``), or the
earlier behaviour of dropping the record (``drop``).  The three agree to well
under the reported effect; ``upper`` and ``drop`` are reported as sensitivity
checks alongside the locked ``zero`` run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = ("lam0_seed0", "lam0_seed1", "lam0_seed2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-physical-spread-m", type=float, default=1e-4)
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--degenerate-policy", choices=("zero", "upper", "drop"),
                   default="zero",
                   help="k_angular when the radial/angular split is 0/0")
    return p.parse_args()


def k_angular(record: dict[str, Any], policy: str) -> float | None:
    """sqrt(angular_energy) / (span + EPS) for one record; None to drop it."""
    k_self = record.get("k_model_self", float("nan"))
    if not np.isfinite(k_self):
        return None
    frac = record.get("model_angular_fraction", float("nan"))
    if np.isfinite(frac):
        return float(k_self * np.sqrt(max(frac, 0.0)))
    # 0/0: ||D2||^2 <= EPS, the map is straight to numerical precision.
    if policy == "zero":
        return 0.0
    if policy == "upper":
        return float(k_self)
    return None


def per_snapshot(root: Path, arm: str, min_spread: float,
                 policy: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for path in sorted((root / arm).glob("snapshot_*/records.json")):
        order = int(path.parent.name.split("_")[1])
        rows = [r for r in json.loads(path.read_text())
                if r.get("valid_unclipped")
                and r.get("ordinal_physical_cost_spread", 0.0) >= min_spread]
        if not rows:
            continue
        ang = [k for k in (k_angular(r, policy) for r in rows) if k is not None]
        n_degenerate = sum(
            1 for r in rows
            if not np.isfinite(r.get("model_angular_fraction", float("nan"))))
        if not ang:
            continue
        out[order] = {
            "n": len(rows), "n_angular": len(ang), "n_degenerate": n_degenerate,
            "angular": float(np.median(ang)),
            "false_valley": float(np.mean([bool(r["ordinal_false_valley"]) for r in rows])),
        }
    return out


def main() -> None:
    args = parse_args()
    pol = args.degenerate_policy
    orig = per_snapshot(args.root, "original", args.min_physical_spread_m, pol)
    seeds = {s: per_snapshot(args.root, s, args.min_physical_spread_m, pol)
             for s in SEEDS}
    shared = sorted(set(orig) & set.intersection(*(set(v) for v in seeds.values())))
    if not shared:
        raise RuntimeError("no snapshots shared across all arms")

    ang_o = np.array([orig[s]["angular"] for s in shared])
    ang_c = np.array([np.mean([seeds[k][s]["angular"] for k in SEEDS]) for s in shared])
    gate1_delta = float(np.median(ang_c - ang_o))
    gate1_pass = bool(gate1_delta < 0)

    report: dict[str, Any] = {
        "n_snapshots": len(shared),
        "degenerate_policy": pol,
        "n_degenerate_records": int(
            sum(orig[s]["n_degenerate"] for s in shared)
            + sum(seeds[k][s]["n_degenerate"] for k in SEEDS for s in shared)),
        "n_records": int(
            sum(orig[s]["n"] for s in shared)
            + sum(seeds[k][s]["n"] for k in SEEDS for s in shared)),
        "gate1_manipulation_check": {
            "median_angular_original": float(np.median(ang_o)),
            "median_angular_continuation": float(np.median(ang_c)),
            "median_paired_delta": gate1_delta,
            "n_snapshots_lower": int(np.sum(ang_c < ang_o)),
            "pass": gate1_pass,
        },
    }

    if not gate1_pass:
        report["gate2_primary"] = "NOT READ: Gate 1 failed"
        report["verdict"] = "GATE1_FAILED_CONTINUATION_DOES_NOT_LOWER_ANGULAR"
    else:
        fv_o = np.array([orig[s]["false_valley"] for s in shared])
        fv_c = np.array([np.mean([seeds[k][s]["false_valley"] for k in SEEDS])
                         for s in shared])
        diff = fv_c - fv_o
        rng = np.random.default_rng(args.seed)
        draws = np.empty(args.n_resamples)
        idx = np.arange(len(shared))
        for i in range(args.n_resamples):
            pick = rng.choice(idx, size=idx.size, replace=True)
            draws[i] = float(np.mean(diff[pick]))
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        point = float(np.mean(diff))
        confirmed = bool(point < 0 and hi < 0)
        report["gate2_primary"] = {
            "mean_false_valley_original": float(np.mean(fv_o)),
            "mean_false_valley_continuation": float(np.mean(fv_c)),
            "paired_mean_difference": point,
            "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_snapshots_lower": int(np.sum(diff < 0)),
            "pass": confirmed,
        }
        report["verdict"] = ("CONFIRMED_CONTINUATION_LOWERS_FALSE_VALLEYS"
                             if confirmed else "NOT_CONFIRMED")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
