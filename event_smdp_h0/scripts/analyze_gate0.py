#!/usr/bin/env python3
"""Aggregate the paired Event-SMDP H0 oracle-interface gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("terminal_only", "event_state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--first-index", type=int, default=1)
    parser.add_argument("--expected-shards", type=int, default=64)
    parser.add_argument("--primary-budget", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--min-effect", type=float, default=0.10)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    boot = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def exact_mcnemar_one_sided(event_only: int, terminal_only: int) -> float:
    discordant = event_only + terminal_only
    if discordant == 0:
        return 1.0
    return float(
        sum(math.comb(discordant, k) for k in range(event_only, discordant + 1))
        / (2**discordant)
    )


def main() -> None:
    args = parse_args()
    indices = list(range(args.first_index, args.first_index + args.expected_shards))
    records: list[dict[str, Any]] = []
    reference_config: dict[str, Any] | None = None
    for index in indices:
        path = args.shards / str(index) / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text())
        if int(record["snapshot"]) != index or not record["repeat_gate"]["pass"]:
            raise RuntimeError(f"invalid or non-repeatable shard {path}")
        config = record["config"]
        locked = {
            key: config[key]
            for key in (
                "goal_offset", "action_block", "depth", "branching", "budgets",
                "exploration", "noise_scale", "noise_rho", "settle_steps",
                "stable_dwell", "success_tolerance_m", "near_tolerance_m", "seed",
            )
        }
        if reference_config is None:
            reference_config = locked
        elif locked != reference_config:
            raise RuntimeError(f"configuration drift in shard {path}")
        records.append(record)

    assert reference_config is not None
    budgets = [int(x) for x in reference_config["budgets"]]
    if args.primary_budget not in budgets:
        raise ValueError("primary budget absent from locked shards")
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    by_budget: dict[str, Any] = {}
    for budget in budgets:
        successes: dict[str, np.ndarray] = {}
        distances: dict[str, np.ndarray] = {}
        stages: dict[str, np.ndarray] = {}
        for arm in ARMS:
            finals = [record["results"][str(budget)][arm]["final"] for record in records]
            successes[arm] = np.asarray([int(final["stable_success"]) for final in finals])
            distances[arm] = np.asarray([float(final["final_distance_m"]) for final in finals])
            stages[arm] = np.asarray([int(final["final_stage"]) for final in finals])

        paired_success = successes["event_state"] - successes["terminal_only"]
        event_only = int(np.sum(paired_success == 1))
        terminal_only = int(np.sum(paired_success == -1))
        by_budget[str(budget)] = {
            "terminal_success": mean_ci(successes["terminal_only"], rng, args.bootstrap),
            "event_success": mean_ci(successes["event_state"], rng, args.bootstrap),
            "event_minus_terminal_success": mean_ci(paired_success, rng, args.bootstrap),
            "event_minus_terminal_final_distance_m": mean_ci(
                distances["event_state"] - distances["terminal_only"], rng, args.bootstrap
            ),
            "event_minus_terminal_final_stage": mean_ci(
                stages["event_state"] - stages["terminal_only"], rng, args.bootstrap
            ),
            "discordant": {
                "event_only_success": event_only,
                "terminal_only_success": terminal_only,
                "exact_mcnemar_one_sided_p": exact_mcnemar_one_sided(
                    event_only, terminal_only
                ),
            },
        }
        for record, terminal, event, dt, de, st, se in zip(
            records,
            successes["terminal_only"],
            successes["event_state"],
            distances["terminal_only"],
            distances["event_state"],
            stages["terminal_only"],
            stages["event_state"],
        ):
            rows.append(
                {
                    "snapshot": int(record["snapshot"]),
                    "budget": budget,
                    "start_distance_m": float(record["start_distance_m"]),
                    "nominal_support_success": int(record["nominal_support"]["stable_success"]),
                    "terminal_success": int(terminal),
                    "event_success": int(event),
                    "terminal_final_distance_m": float(dt),
                    "event_final_distance_m": float(de),
                    "terminal_final_stage": int(st),
                    "event_final_stage": int(se),
                }
            )

    nominal = np.asarray([int(record["nominal_support"]["stable_success"]) for record in records])
    primary = by_budget[str(args.primary_budget)]
    delta = primary["event_minus_terminal_success"]
    p_value = primary["discordant"]["exact_mcnemar_one_sided_p"]
    effect_pass = delta["mean"] >= args.min_effect
    significance_pass = delta["ci_low"] > 0.0 and p_value < 0.05
    if effect_pass and significance_pass:
        verdict = "GO_CAUSAL_ROOM"
    elif delta["ci_high"] < args.min_effect:
        verdict = "STOP_NO_MATERIAL_CAUSAL_ROOM"
    else:
        verdict = "INCONCLUSIVE_EXTEND_OR_REDESIGN"

    event_curve = [by_budget[str(b)]["event_success"]["mean"] for b in budgets]
    search_degradation = any(
        later < earlier - 0.05 for earlier, later in zip(event_curve, event_curve[1:])
    )
    output = {
        "scope": (
            "Paired oracle-interface H0. A GO licenses a learned event-head/action-prior "
            "experiment; it is not evidence that Hazard-JEPA itself improves success."
        ),
        "n_snapshots": len(records),
        "indices": [indices[0], indices[-1]],
        "locked_config": reference_config,
        "bootstrap": {"draws": args.bootstrap, "seed": args.seed},
        "nominal_support_success": mean_ci(nominal, rng, args.bootstrap),
        "by_budget": by_budget,
        "primary": {
            "budget": args.primary_budget,
            "minimum_material_effect": args.min_effect,
            "effect_pass": effect_pass,
            "significance_pass": significance_pass,
            "search_intensity_degradation_gt_5pp": search_degradation,
        },
        "verdict": verdict,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    with (args.out_dir / "paired_snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decision = [
        "# Event-SMDP H0 decision",
        "",
        f"Verdict: **{verdict}**.",
        "",
        f"Primary budget: {args.primary_budget} UCT simulations per replan; n={len(records)} paired snapshots.",
        f"Event minus terminal stable-success: {delta['mean']:.3f} "
        f"[{delta['ci_low']:.3f}, {delta['ci_high']:.3f}].",
        f"One-sided exact McNemar p={p_value:.6g}.",
        "",
        "This gate uses oracle event labels and a privileged support-matched proposal lattice. "
        "It only tests whether the event-state interface creates finite-search causal room.",
    ]
    (args.out_dir / "DECISION.md").write_text("\n".join(decision) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
