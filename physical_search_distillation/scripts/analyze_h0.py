#!/usr/bin/env python3
"""Aggregate held-out results and apply the locked H0 decision rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def paired_summary(reference: np.ndarray, method: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    improvement = reference - method
    indices = rng.integers(0, len(improvement), size=(n_boot, len(improvement)))
    bootstrap = improvement[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean_improvement_m": float(improvement.mean()),
        "ci95_low_m": float(low), "ci95_high_m": float(high),
        "wins": int(np.sum(improvement > 0)), "ties": int(np.sum(improvement == 0)),
        "n": int(len(improvement)),
    }


def main() -> None:
    args = parse_args()
    rows = [json.loads(path.read_text()) for path in sorted(args.eval_dir.glob("snapshot_*/summary.json"))]
    if len(rows) != 32:
        raise RuntimeError(f"locked evaluation requires 32 test states, got {len(rows)}")
    orders = [row["snapshot"] for row in rows]
    if len(set(orders)) != 32 or any(order % 8 not in (0, 4) for order in orders):
        raise RuntimeError("held-out order gate failed")
    names = list(rows[0]["results"])
    if set(names) != {"native", "pointwise", "listwise", "elite", "operator", "operator_metric"}:
        raise RuntimeError(f"missing locked arm: {names}")
    distance = {name: np.asarray([row["results"][name]["physical_distance_m"] for row in rows]) for name in names}
    success = {name: np.asarray([row["results"][name]["success"] for row in rows]) for name in names}
    latency = {name: np.asarray([row["results"][name]["solve_seconds"] for row in rows]) for name in names}
    rng = np.random.default_rng(args.seed)
    aggregate = {
        name: {
            "distance_mean_m": float(distance[name].mean()),
            "distance_median_m": float(np.median(distance[name])),
            "success_rate": float(success[name].mean()),
            "solve_seconds_mean": float(latency[name].mean()),
        }
        for name in names
    }
    comparisons = {
        "operator_vs_native": paired_summary(distance["native"], distance["operator"], rng, args.bootstrap),
        "operator_vs_listwise": paired_summary(distance["listwise"], distance["operator"], rng, args.bootstrap),
        "metric_vs_operator": paired_summary(distance["operator"], distance["operator_metric"], rng, args.bootstrap),
    }
    comparisons["operator_vs_native"]["success_delta"] = float(success["operator"].mean() - success["native"].mean())
    comparisons["operator_vs_listwise"]["success_delta"] = float(success["operator"].mean() - success["listwise"].mean())
    comparisons["metric_vs_operator"]["success_delta"] = float(success["operator_metric"].mean() - success["operator"].mean())

    op_native = comparisons["operator_vs_native"]
    op_rank = comparisons["operator_vs_listwise"]
    metric = comparisons["metric_vs_operator"]
    if op_rank["mean_improvement_m"] <= 0 or op_rank["success_delta"] < 0:
        decision = "STOP_OPERATOR_NOVELTY"
    elif (
        op_native["ci95_low_m"] > 0 and op_rank["ci95_low_m"] > 0
        and op_native["success_delta"] >= 0 and op_rank["success_delta"] >= 0
    ):
        decision = "GO_REPRESENTATION" if metric["ci95_low_m"] > 0 and metric["success_delta"] >= 0 else "GO_OPERATOR"
    else:
        decision = "HOLD_SCALE"
    report = {
        "decision": decision, "n_test_states": len(rows), "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "interpretation": {
            "GO_OPERATOR": "operator target clears native and matched listwise; authorize one DAgger relabel round",
            "GO_REPRESENTATION": "operator clears gate and residual latent metric adds a further reliable gain",
            "STOP_OPERATOR_NOVELTY": "operator does not improve the key matched ranking baseline; do not develop PERD claim",
            "HOLD_SCALE": "point estimates are favorable but H0 uncertainty remains; scale held-out states once without loss changes",
        }[decision],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
