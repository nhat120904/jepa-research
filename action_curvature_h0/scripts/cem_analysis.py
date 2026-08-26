#!/usr/bin/env python3
"""Analysis for the CEM-interaction test (fifteenth and sixteenth amendments).

Primary: arena 0 (initial proposal, conditioned on no arm), outcome = physical
goal distance of the top-30 elite mean, contrast = continuation (mean of the
three seeds) minus original, paired by snapshot, snapshot-clustered bootstrap.
PASS iff the point estimate is < 0 and the 95% CI excludes zero.

Arena 1 (the original CEM's own final population) is a secondary robustness
arena and is never pooled with the primary.  Failing it while the primary passes
supports only the basin-conditioned reading, not a general null.

Analysis population: snapshots whose pre-action start state is not already
successful (sixteenth amendment).  Snapshot 064 is retained with its partial
unblinding disclosed; the sensitivity excluding it is reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = ("lam0_seed0", "lam0_seed1", "lam0_seed2")
UNBLINDED = 64
PRIMARY_ARENA = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--viability", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260826)
    return p.parse_args()


def load_scores(root: Path) -> dict[int, dict[str, list[dict[str, Any]]]]:
    out: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for path in sorted(root.glob("snapshot_*/cem_score.json")):
        blob = json.loads(path.read_text())
        arms = {a["arm"]: a["arenas"] for a in blob["arms"]}
        if len(set(blob["candidate_sha256"])) != len(blob["candidate_sha256"]):
            raise RuntimeError(f"{path}: arenas share a candidate hash")
        out[int(blob["snapshot"])] = arms
    return out


def paired_bootstrap(diff: np.ndarray, n_resamples: int, seed: int
                     ) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(diff))
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        draws[i] = float(np.mean(diff[rng.choice(idx, size=idx.size, replace=True)]))
    return (float(np.mean(diff)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def contrast(scores: dict[int, Any], snapshots: list[int], arena: int, field: str,
             n_resamples: int, seed: int) -> dict[str, Any]:
    orig_all = np.array([scores[s]["original"][arena][field] for s in snapshots],
                        dtype=np.float64)
    cont_all = np.array([np.mean([scores[s][k][arena][field] for k in SEEDS])
                         for s in snapshots], dtype=np.float64)
    # Rank correlation is undefined where every candidate has an identical
    # physical distance (a fully converged population), which happens on a
    # couple of arena-1 snapshots.  Drop those pairs explicitly and report the
    # count rather than letting one nan propagate through a mean.
    keep = np.isfinite(orig_all) & np.isfinite(cont_all)
    orig, cont = orig_all[keep], cont_all[keep]
    if orig.size == 0:
        raise RuntimeError(f"{field} arena {arena}: no finite pairs")
    point, lo, hi = paired_bootstrap(cont - orig, n_resamples, seed)
    return {
        "field": field, "arena": arena, "n_snapshots": int(orig.size),
        "n_dropped_non_finite": int((~keep).sum()),
        "mean_original": float(orig.mean()),
        "mean_continuation": float(cont.mean()),
        "paired_mean_difference": point, "ci_low": lo, "ci_high": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_snapshots_lower": int(np.sum(cont < orig)),
        "n_snapshots_tied": int(np.sum(cont == orig)),
        "per_seed_mean": {k: float(np.mean([scores[s][k][arena][field]
                                            for s in snapshots])) for k in SEEDS},
        "pass": bool(point < 0 and hi < 0),
    }


def main() -> None:
    args = parse_args()
    viability = json.loads(args.viability.read_text())
    scores = load_scores(args.scores)
    viable = [s for s in sorted(scores) if s in set(viability["viable_snapshots"])]
    if not viable:
        raise RuntimeError("no viable snapshots with scores")

    report: dict[str, Any] = {
        "n_scored": len(scores),
        "n_viable": len(viable),
        "n_excluded_start_state_already_successful": len(scores) - len(viable),
        "viability_corroboration": viability["corroboration_max_executed_steps_gt_1"],
        "unblinded_snapshot_retained": UNBLINDED in viable,
    }

    report["primary"] = contrast(scores, viable, PRIMARY_ARENA,
                                 "elite_mean_physical_distance_m",
                                 args.n_resamples, args.seed)
    without = [s for s in viable if s != UNBLINDED]
    report["primary_sensitivity_excluding_064"] = contrast(
        scores, without, PRIMARY_ARENA, "elite_mean_physical_distance_m",
        args.n_resamples, args.seed)

    report["secondary_arena_final_population"] = contrast(
        scores, viable, 1, "elite_mean_physical_distance_m",
        args.n_resamples, args.seed)

    report["secondaries"] = {
        f"arena{arena}_{field}": contrast(scores, viable, arena, field,
                                          args.n_resamples, args.seed)
        for arena in (0, 1)
        for field in ("top1_physical_distance_m",
                      "elite_mean_of_physical_distances_m",
                      "rank_correlation_model_vs_physical")
    }

    # Context for diagnosing a floor rather than reporting one as a null.
    report["context"] = {
        f"arena{arena}": {
            "population_spread_m_median": float(np.median([
                scores[s]["original"][arena]["population_physical_max_m"]
                - scores[s]["original"][arena]["population_physical_min_m"]
                for s in viable])),
            "out_of_bounds_fraction_mean": {
                arm: float(np.mean([scores[s][arm][arena][
                    "elite_mean_out_of_bounds_fraction"] for s in viable]))
                for arm in ("original", *SEEDS)},
            "executed_steps_median": {
                arm: float(np.median([scores[s][arm][arena][
                    "elite_mean_executed_steps"] for s in viable]))
                for arm in ("original", *SEEDS)},
        } for arena in (0, 1)
    }

    p = report["primary"]
    report["verdict"] = (
        "CONFIRMED_CONTINUATION_IMPROVES_CEM_REFIT" if p["pass"]
        else "NOT_CONFIRMED")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
