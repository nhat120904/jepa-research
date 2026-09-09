#!/usr/bin/env python3
"""Paired A1 gate analysis; must run on a compute node by repository policy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from skill_handoff_wm.sim import require_slurm  # noqa: E402


def paired_interval(values: np.ndarray, rng: np.random.Generator, samples: int = 20_000) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--gate-points", type=float, default=10.0)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.episodes).read_text().splitlines() if line.strip()]
    grouped: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for record in records:
        grouped[record["arm"]][(int(record["task_id"]), int(record["seed"]))] = record
    keys = sorted(grouped["oracle"])
    if not keys or any(set(rows) != set(keys) for rows in grouped.values()):
        raise ValueError("all arms must contain the same paired task/seed cases")
    nonoracle = sorted(arm for arm in grouped if arm != "oracle")
    rates = {
        arm: float(np.mean([grouped[arm][key]["success"] for key in keys]))
        for arm in sorted(grouped)
    }
    strongest = max(nonoracle, key=lambda arm: (rates[arm], -np.mean(
        [grouped[arm][key]["primitive_steps"] for key in keys]
    )))
    oracle_success = np.asarray([grouped["oracle"][key]["success"] for key in keys], dtype=np.float64)
    baseline_success = np.asarray([grouped[strongest][key]["success"] for key in keys], dtype=np.float64)
    paired_points = 100.0 * (oracle_success - baseline_success)
    rng = np.random.default_rng(args.seed)
    payload = {
        "protocol": "direction_a_a1_gate_v1",
        "n_paired_cases": len(keys),
        "success_rates": rates,
        "strongest_nonoracle": strongest,
        "oracle_minus_strongest_points": float(paired_points.mean()),
        "paired_bootstrap_95_ci_points": paired_interval(paired_points, rng),
        "gate_threshold_points": args.gate_points,
        "a1_pass": bool(paired_points.mean() >= args.gate_points),
        "mean_primitive_steps": {
            arm: float(np.mean([grouped[arm][key]["primitive_steps"] for key in keys]))
            for arm in sorted(grouped)
        },
        "oracle_simulator_query_steps": int(
            sum(grouped["oracle"][key]["simulator_query_steps"] for key in keys)
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
