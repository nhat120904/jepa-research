#!/usr/bin/env python3
"""Paired Cube closed-loop analysis, including immediate-success prevalence."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.evaluation.read_text())
    arms = list(data)
    roots = [[(x["episode"], x["start_step"]) for x in data[a]["results"]] for a in arms]
    if not all(r == roots[0] and len(r) == 50 for r in roots):
        raise ValueError("Arms must have 50 identical episode/start pairs")
    rng = np.random.default_rng(20260928)
    draws = rng.integers(0, 50, size=(20000, 50))
    success = {a: np.array([x["success"] for x in data[a]["results"]], bool) for a in arms}
    distance = {a: np.array([x["final_cube_distance_m"] for x in data[a]["results"]], float) for a in arms}
    steps = {a: np.array([x["steps"] for x in data[a]["results"]], int) for a in arms}
    comparisons = {}
    requested = (("rank_scorer", "prediction_scorer"),
                 ("rank_scorer", "prediction_native"),
                 ("prediction_scorer", "prediction_native"),
                 ("rank_native", "prediction_native"),
                 ("response_native", "prediction_native"),
                 ("response_mixed", "prediction_mixed"))
    for left, right in (pair for pair in requested if all(a in arms for a in pair)):
        delta = success[left].astype(int) - success[right].astype(int)
        reduction = distance[right] - distance[left]
        comparisons[f"{left}_vs_{right}"] = {
            "paired_gain_percentage_points": float(delta.mean() * 100),
            "gain_bootstrap_95_percent_interval_points":
                (np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]) * 100).tolist(),
            "left_only_successes": int(np.sum(success[left] & ~success[right])),
            "right_only_successes": int(np.sum(~success[left] & success[right])),
            "mean_distance_reduction_m": float(reduction.mean()),
            "distance_reduction_bootstrap_95_percent_interval_m":
                np.quantile(reduction[draws].mean(axis=1), [0.025, 0.975]).tolist(),
        }
    report = {
        "evaluation": str(args.evaluation),
        "arms": {a: {"successes": int(success[a].sum()),
                     "mean_final_distance_m": float(distance[a].mean()),
                     "successes_in_at_most_3_steps": int(np.sum(success[a] & (steps[a] <= 3)))}
                 for a in arms},
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
