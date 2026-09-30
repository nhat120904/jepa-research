#!/usr/bin/env python3
"""Summarize independent fine-tuning seeds on common PushT tasks."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs=3, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = [json.loads((p / "evaluation.json").read_text()) for p in args.runs]
    rows = [[arm["row"] for arm in data["prediction"]["results"]]
            for data in records]
    if not all(len(row) == 50 and row == rows[0] for row in rows):
        raise ValueError("Three seeds must share exactly the same 50 roots")
    for data in records:
        if [x["row"] for x in data["response"]["results"]] != rows[0]:
            raise ValueError("Unmatched control and response roots")
    p = np.array([[x["success"] for x in data["prediction"]["results"]]
                  for data in records], dtype=np.int8)
    r = np.array([[x["success"] for x in data["response"]["results"]]
                  for data in records], dtype=np.int8)
    d = r - p
    rng = np.random.default_rng(20260928)
    boot = np.empty(20000, dtype=np.float64)
    for i in range(len(boot)):
        seeds = rng.integers(0, 3, size=3)
        roots = rng.integers(0, 50, size=50)
        boot[i] = d[np.ix_(seeds, roots)].mean() * 100
    report = {
        "runs": [str(p) for p in args.runs],
        "n_training_seeds": 3,
        "n_common_roots_per_seed": 50,
        "prediction_successes_per_seed": p.sum(axis=1).tolist(),
        "response_successes_per_seed": r.sum(axis=1).tolist(),
        "prediction_mean_success_percent": float(p.mean() * 100),
        "response_mean_success_percent": float(r.mean() * 100),
        "mean_paired_gain_percentage_points": float(d.mean() * 100),
        "two_level_paired_bootstrap_95_percent_interval_points":
            np.quantile(boot, [0.025, 0.975]).tolist(),
        "note": "Bootstrap resamples training seeds and shared task roots; three seeds limit precision. Published 96% is an external, unpaired reference."
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
