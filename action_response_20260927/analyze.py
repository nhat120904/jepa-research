#!/usr/bin/env python3
"""Paired closed-loop summary for one completed action-response run."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    data = json.loads((run / "evaluation.json").read_text())
    pred = data["prediction"]["results"]
    resp = data["response"]["results"]
    if len(pred) != 50 or len(resp) != 50:
        raise ValueError("Expected 50 completed roots in both arms")
    if [r["row"] for r in pred] != [r["row"] for r in resp]:
        raise ValueError("Prediction and response used different roots")
    p = np.array([r["success"] for r in pred], dtype=np.int8)
    r = np.array([x["success"] for x in resp], dtype=np.int8)
    d = r - p
    won = int(np.sum(d == 1))
    lost = int(np.sum(d == -1))
    discordant = won + lost
    # Conditional exact two-sided McNemar test, with no SciPy dependency.
    mc_p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(won, lost) + 1)) / 2**discordant)
    rng = np.random.default_rng(20260928)
    draws = rng.integers(0, len(d), size=(20000, len(d)))
    gains = d[draws].mean(axis=1)
    position_gain = None
    position_interval = None
    if all("position_error" in x for x in pred + resp):
        pos_d = np.array([x["position_error"] - y["position_error"]
                          for x, y in zip(pred, resp)])
        position_gain = float(pos_d.mean())
        position_interval = np.quantile(pos_d[draws].mean(axis=1),
                                        [0.025, 0.975]).tolist()
    report = {
        "n_paired_roots": len(d),
        "prediction_successes": int(p.sum()),
        "response_successes": int(r.sum()),
        "response_only_successes": won,
        "prediction_only_successes": lost,
        "both_successes": int(np.sum((p == 1) & (r == 1))),
        "neither_successes": int(np.sum((p == 0) & (r == 0))),
        "paired_gain_percentage_points": float(100 * d.mean()),
        "paired_bootstrap_95_percent_interval_percentage_points": (100 * np.quantile(gains, [0.025, 0.975])).tolist(),
        "exact_mcnemar_two_sided_p": mc_p,
        "mean_position_error_reduction": position_gain,
        "paired_bootstrap_95_percent_interval_position_error_reduction": position_interval,
        "run_dir": str(run),
        "interpretation": "Single fine-tuning seed; paired bootstrap only measures evaluation-root uncertainty."
    }
    (run / "paired_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
