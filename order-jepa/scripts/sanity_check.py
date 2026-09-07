#!/usr/bin/env python3
"""Analytic ORDER checks; intentionally submitted to a CPU compute node."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from order_jepa.core import order_vector_metrics  # noqa: E402


def true_unicycle(order: str, h: float) -> np.ndarray:
    if order == "RT":
        return np.array([h * np.cos(h), h * np.sin(h), h], dtype=np.float64)
    if order == "TR":
        return np.array([h, 0.0, h], dtype=np.float64)
    raise ValueError(order)


def commuting_predictor(order: str, h: float) -> np.ndarray:
    del order
    return np.array([h, 0.0, h], dtype=np.float64)


def run() -> dict:
    rows = []
    for h in (0.2, 0.1, 0.05):
        true_rt, true_tr = true_unicycle("RT", h), true_unicycle("TR", h)
        pred_rt, pred_tr = commuting_predictor("RT", h), commuting_predictor("TR", h)
        endpoint_mse_sum = float(
            np.mean(np.square(pred_rt - true_rt)) + np.mean(np.square(pred_tr - true_tr))
        )
        metrics = order_vector_metrics(true_rt, true_tr, pred_rt, pred_tr)
        rows.append(
            {
                "h": h,
                "endpoint_mse_sum": endpoint_mse_sum,
                "order_error_over_h2": metrics.error_norm / (h**2),
            }
        )

    rng = np.random.default_rng(20260907)
    max_cost_identity_error = 0.0
    max_decomposition_error = 0.0
    for _ in range(10_000):
        z1, z2, goal = rng.normal(size=(3, 8))
        mean, difference = (z1 + z2) / 2.0, z1 - z2
        lhs = np.sum(np.square(z1 - goal)) - np.sum(np.square(z2 - goal))
        rhs = 2.0 * np.dot(mean - goal, difference)
        max_cost_identity_error = max(max_cost_identity_error, abs(lhs - rhs))

        p1, p2 = rng.normal(size=(2, 8))
        e1, e2 = p1 - z1, p2 - z2
        branch = np.sum(e1**2) + np.sum(e2**2)
        decomposed = 2.0 * np.sum(((e1 + e2) / 2.0) ** 2) + 0.5 * np.sum((e1 - e2) ** 2)
        max_decomposition_error = max(max_decomposition_error, abs(branch - decomposed))

    return {
        "rows": rows,
        "max_cost_identity_error": max_cost_identity_error,
        "max_decomposition_error": max_decomposition_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs/sanity/sanity_results.json")
    parser.add_argument("--allow-login", action="store_true", help="Only for tiny CI environments")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ and not args.allow_login:
        raise RuntimeError("Submit this script with scripts/slurm_sanity.sh")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
