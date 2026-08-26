#!/usr/bin/env python3
"""Does model-side curvature predict where the model misranks physical outcomes?

This is the kill test for the reformulated hypothesis.  Both quantities are
computed without any bridge:

  predictor : model-side curvature of the action -> predicted-outcome map,
              measured in float64 (verified exactly smooth, alpha = 2.000)
  outcome   : ordinal disagreement between the model's preference over the
              triplet and the simulator's physical preference

If high-curvature regions do not misrank more than low-curvature ones, curvature
does not explain the planning failures and the AS intervention loses its
empirical justification.

Near-tied triplets are excluded before anything is read: when the three costs
are indistinguishable, the ordinal shape is noise and an inversion costs a
planner nothing.
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
    p.add_argument("--curvature-key", default="model_ratio",
                   help="model-side curvature statistic used as the predictor")
    p.add_argument("--n-quantiles", type=int, default=4)
    p.add_argument("--min-physical-spread-m", type=float, default=1e-4,
                   help="triplets whose physical costs differ by less than this "
                        "carry no decision-relevant ordering")
    p.add_argument("--min-model-spread", type=float, default=0.0)
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260825)
    return p.parse_args()


def load(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/snapshot_*/records.json")):
        summary = json.loads((path.parent / "summary.json").read_text())
        order = int(summary["snapshot"]["order"])
        source = path.parent.parent.name
        for r in json.loads(path.read_text()):
            r["snapshot"] = order
            r["action_source"] = source
            rows.append(r)
    return rows


def main() -> None:
    args = parse_args()
    rows = load(args.shard_root)
    if not rows:
        raise RuntimeError(f"no records under {args.shard_root}")

    # Report the spread distribution BEFORE filtering: the physical spread is
    # controlled by sigma, so the right cut is an empirical question and must
    # not be guessed.  Also stratify by sigma, which is more informative than
    # any single threshold.
    with_ordinal = [r for r in rows if r.get("valid_unclipped")
                    and "ordinal_shape_agree" in r]
    spreads = np.array([r["ordinal_physical_cost_spread"] for r in with_ordinal])
    spread_report = {
        "n": int(spreads.size),
        "percentiles_m": {
            str(q): float(np.percentile(spreads, q)) for q in (5, 25, 50, 75, 95)
        } if spreads.size else {},
    }
    by_sigma: dict[str, Any] = {}
    for sig in sorted({r["sigma"] for r in with_ordinal}):
        sel = [r for r in with_ordinal if r["sigma"] == sig]
        by_sigma[str(sig)] = {
            "n": len(sel),
            "median_physical_spread_m": float(
                np.median([r["ordinal_physical_cost_spread"] for r in sel])
            ),
            "shape_disagree_rate": float(
                np.mean([not r["ordinal_shape_agree"] for r in sel])
            ),
            "false_valley_rate": float(
                np.mean([bool(r["ordinal_false_valley"]) for r in sel])
            ),
        }

    usable = [
        r for r in rows
        if r.get("valid_unclipped")
        and "ordinal_shape_agree" in r
        and np.isfinite(r.get(args.curvature_key, np.nan))
        and r["ordinal_physical_cost_spread"] >= args.min_physical_spread_m
        and r["ordinal_model_cost_spread"] >= args.min_model_spread
    ]
    if len(usable) < 4 * args.n_quantiles:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"error": "too few usable triplets after the spread filter",
             "n_usable": len(usable), "physical_spread": spread_report,
             "by_sigma": by_sigma}, indent=2))
        print(json.dumps({"n_usable": len(usable),
                          "physical_spread": spread_report,
                          "by_sigma": by_sigma}, indent=2))
        return

    curv = np.array([abs(float(r[args.curvature_key])) for r in usable])
    disagree = np.array([not bool(r["ordinal_shape_agree"]) for r in usable], dtype=float)
    false_valley = np.array([bool(r["ordinal_false_valley"]) for r in usable], dtype=float)
    argmin_wrong = np.array([not bool(r["ordinal_argmin_agree"]) for r in usable], dtype=float)
    groups = np.array([r["snapshot"] for r in usable])

    edges = np.quantile(curv, np.linspace(0, 1, args.n_quantiles + 1))
    edges[-1] = np.inf
    bins: list[dict[str, Any]] = []
    for i in range(args.n_quantiles):
        sel = (curv >= edges[i]) & (curv < edges[i + 1])
        if sel.sum() == 0:
            continue
        bins.append({
            "quantile": i,
            "curvature_range": [float(edges[i]),
                                float(edges[i + 1]) if np.isfinite(edges[i + 1])
                                else float(curv.max())],
            "n": int(sel.sum()),
            "median_curvature": float(np.median(curv[sel])),
            "shape_disagree_rate": float(disagree[sel].mean()),
            "false_valley_rate": float(false_valley[sel].mean()),
            "argmin_wrong_rate": float(argmin_wrong[sel].mean()),
        })

    # Snapshot-clustered bootstrap on the top-minus-bottom quantile contrast:
    # the effect the hypothesis actually predicts.
    rng = np.random.default_rng(args.seed)
    uniq = np.unique(groups)
    lo_sel = curv < edges[1]
    hi_sel = curv >= edges[args.n_quantiles - 1]

    def contrast(mask_lo: np.ndarray, mask_hi: np.ndarray, y: np.ndarray) -> float:
        if mask_lo.sum() == 0 or mask_hi.sum() == 0:
            return float("nan")
        return float(y[mask_hi].mean() - y[mask_lo].mean())

    results: dict[str, Any] = {}
    for name, y in (("shape_disagree", disagree), ("false_valley", false_valley),
                    ("argmin_wrong", argmin_wrong)):
        point = contrast(lo_sel, hi_sel, y)
        draws = np.empty(args.n_resamples)
        for k in range(args.n_resamples):
            pick = rng.choice(uniq, size=uniq.size, replace=True)
            idx = np.concatenate([np.nonzero(groups == g)[0] for g in pick])
            draws[k] = contrast(lo_sel[idx], hi_sel[idx], y[idx])
        finite = draws[np.isfinite(draws)]
        results[name] = {
            "top_minus_bottom": point,
            "ci_low": float(np.percentile(finite, 2.5)) if finite.size else None,
            "ci_high": float(np.percentile(finite, 97.5)) if finite.size else None,
            "excludes_zero": bool(finite.size and
                                  (np.percentile(finite, 2.5) > 0
                                   or np.percentile(finite, 97.5) < 0)),
        }

    spearman = float("nan")
    if len(usable) > 2:
        cr = np.argsort(np.argsort(curv)).astype(float)
        dr = np.argsort(np.argsort(disagree)).astype(float)
        if cr.std() > 0 and dr.std() > 0:
            spearman = float(np.corrcoef(cr, dr)[0, 1])

    report = {
        "curvature_key": args.curvature_key,
        "filters": {"min_physical_spread_m": args.min_physical_spread_m,
                    "min_model_spread": args.min_model_spread},
        "physical_spread_before_filter": spread_report,
        "by_sigma_unfiltered": by_sigma,
        "counts": {"records": len(rows), "usable": len(usable),
                   "snapshots": int(uniq.size)},
        "overall_rates": {
            "shape_disagree": float(disagree.mean()),
            "false_valley": float(false_valley.mean()),
            "argmin_wrong": float(argmin_wrong.mean()),
        },
        "by_curvature_quantile": bins,
        "top_vs_bottom_quantile": results,
        "spearman_curvature_vs_disagreement": spearman,
        "verdict": (
            "CURVATURE_PREDICTS_MISRANKING"
            if results["shape_disagree"]["excludes_zero"]
            and results["shape_disagree"]["top_minus_bottom"] > 0
            else "NO_CURVATURE_MISRANKING_LINK_KILL_AS"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("counts", "overall_rates", "by_curvature_quantile",
                       "top_vs_bottom_quantile", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
