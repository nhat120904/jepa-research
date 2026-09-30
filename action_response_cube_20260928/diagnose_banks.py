#!/usr/bin/env python3
"""Measure whether collected CEM neighborhoods contain decision headroom."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.cache) as data:
        distance = data["physical_distance_m"].astype(float)
        success = data["success"].astype(bool)
        episodes = data["episode"]
    if len(np.unique(episodes)) != len(episodes):
        raise ValueError("branch banks repeat an episode")
    best = distance.min(axis=1)
    report = {
        "banks": len(distance), "candidates_per_bank": distance.shape[1],
        "distinct_episodes": len(np.unique(episodes)),
        "any_success_banks": int(success.any(axis=1).sum()),
        "baseline_rank0_success_banks": int(success[:, 0].sum()),
        "median_best_to_worst_distance_spread_m": float(np.median(distance.max(axis=1) - best)),
        "median_rank0_physical_regret_m": float(np.median(distance[:, 0] - best)),
        "mean_rank0_physical_regret_m": float(np.mean(distance[:, 0] - best)),
        "fraction_rank0_is_physical_best": float(np.mean(distance[:, 0] <= best + 1e-9)),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
