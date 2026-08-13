#!/usr/bin/env python3
"""Aggregate corrected OGBench true-endpoint audit shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def interval(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_finite": 0,
        }
    bootstrap = rng.choice(finite, size=(draws, len(finite)), replace=True).mean(axis=1)
    return {
        "mean": float(finite.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "n_finite": int(len(finite)),
    }


def paired_contrast(
    rows: dict[str, dict[int, dict[str, Any]]],
    baseline: str,
    method: str,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, dict[str, float | int]]:
    ids = sorted(rows[baseline])
    if ids != sorted(rows[method]):
        raise RuntimeError(f"snapshot mismatch for {baseline} versus {method}")
    regret_gain = np.asarray(
        [
            rows[baseline][idx]["selection_regret_m"]
            - rows[method][idx]["selection_regret_m"]
            for idx in ids
        ]
    )
    success_gain = np.asarray(
        [
            rows[method][idx]["selected_success"]
            - rows[baseline][idx]["selected_success"]
            for idx in ids
        ]
    )
    return {
        "regret_gain_m": interval(regret_gain, rng, draws),
        "success_gain": interval(success_gain, rng, draws),
    }


def main() -> None:
    args = parse_args()
    shard_dirs = sorted(
        (path for path in args.shards.iterdir() if path.is_dir()),
        key=lambda path: int(path.name),
    )
    if len(shard_dirs) != args.expected:
        raise RuntimeError(f"expected {args.expected} shards, found {len(shard_dirs)}")

    rows: dict[str, dict[int, dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for shard in shard_dirs:
        summary = json.loads((shard / "summary.json").read_text())
        if not summary["gate"]["pass"]:
            raise RuntimeError(f"shard {shard.name} failed its locked gate")
        summaries.append(summary)
        with (shard / "snapshot_metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                parsed = {
                    key: (
                        value
                        if key == "selector"
                        else int(value)
                        if key in {"snapshot", "selected_candidate"}
                        else float(value)
                    )
                    for key, value in row.items()
                }
                selector = parsed["selector"]
                snapshot = parsed["snapshot"]
                if snapshot in rows.setdefault(selector, {}):
                    raise RuntimeError(f"duplicate {selector} snapshot {snapshot}")
                rows[selector][snapshot] = parsed
                all_rows.append(parsed)
                seen.add(snapshot)
        with gzip.open(shard / "candidate_costs.csv.gz", "rt", newline="") as handle:
            candidate_rows.extend(csv.DictReader(handle))

    expected_ids = set(range(args.expected))
    if seen != expected_ids:
        raise RuntimeError(f"snapshot coverage mismatch: {sorted(seen)}")
    required = {
        "learned_predicted_goal",
        "true_dataset_goal",
        "true_rendered_goal",
    }
    if set(rows) != required:
        raise RuntimeError(f"selector mismatch: {sorted(rows)}")
    if any(set(selector_rows) != expected_ids for selector_rows in rows.values()):
        raise RuntimeError("one or more selectors lack complete snapshot coverage")

    reference_config = {
        key: value
        for key, value in summaries[0]["config"].items()
        if key != "snapshot_index"
    }
    for summary in summaries[1:]:
        config = {
            key: value
            for key, value in summary["config"].items()
            if key != "snapshot_index"
        }
        if config != reference_config:
            raise RuntimeError("shard configuration mismatch")

    rng = np.random.default_rng(args.seed + 99)
    metric_names = [
        "selected_physical_distance_m",
        "selected_success",
        "selection_regret_m",
        "success_gap",
        "spearman_physical",
        "top10pct_recall_physical",
        "false_elite_rate_physical",
    ]
    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for selector, selector_rows in sorted(rows.items()):
        ordered = [selector_rows[index] for index in range(args.expected)]
        metrics[selector] = {
            name: interval(
                np.asarray([row[name] for row in ordered]), rng, args.bootstrap
            )
            for name in metric_names
        }

    oracle_success = interval(
        np.asarray([summary["oracle_candidate_success"] for summary in summaries]),
        rng,
        args.bootstrap,
    )
    oracle_distance = interval(
        np.asarray([summary["oracle_physical_distance_m"] for summary in summaries]),
        rng,
        args.bootstrap,
    )
    domain = {
        name: interval(
            np.asarray([summary["domain"][name] for summary in summaries]),
            rng,
            args.bootstrap,
        )
        for name in (
            "same_state_init_cost",
            "same_state_goal_cost",
            "dataset_task_cost",
            "rendered_task_cost",
            "domain_ratio",
        )
    }
    contrasts = {
        "true_rendered_vs_learned": paired_contrast(
            rows,
            "learned_predicted_goal",
            "true_rendered_goal",
            rng,
            args.bootstrap,
        ),
        "true_dataset_vs_learned": paired_contrast(
            rows,
            "learned_predicted_goal",
            "true_dataset_goal",
            rng,
            args.bootstrap,
        ),
        "true_rendered_vs_true_dataset": paired_contrast(
            rows,
            "true_dataset_goal",
            "true_rendered_goal",
            rng,
            args.bootstrap,
        ),
    }

    primary = metrics["true_rendered_goal"]
    regret_positive = primary["selection_regret_m"]["ci_low"] > 0
    success_gap_positive = primary["success_gap"]["ci_low"] > 0
    coverage_positive = oracle_success["ci_low"] > 0
    strong = regret_positive and success_gap_positive and coverage_positive
    distance_only = regret_positive and coverage_positive and not success_gap_positive
    representation_gate = {
        "status": (
            "strong_support"
            if strong
            else "distance_only_support"
            if distance_only
            else "no_support"
        ),
        "primary_selector": "true_rendered_goal",
        "criteria": {
            "representation_regret_ci_low_gt_zero": regret_positive,
            "representation_success_gap_ci_low_gt_zero": success_gap_positive,
            "oracle_candidate_success_ci_low_gt_zero": coverage_positive,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(
            sorted(all_rows, key=lambda row: (row["snapshot"], row["selector"]))
        )
    with gzip.open(args.out / "candidate_costs.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    combined = {
        "config": {**reference_config, "processed_snapshots": args.expected},
        "metrics": metrics,
        "paired_contrasts": contrasts,
        "physical_oracle": {
            "candidate_success": oracle_success,
            "distance_m": oracle_distance,
        },
        "domain": domain,
        "representation_gate": representation_gate,
        "gate_worst": {
            "domain_ratio": max(summary["domain"]["domain_ratio"] for summary in summaries),
            "repeat_physical_distance_m": max(
                summary["repeat"]["physical_distance_m_max_abs"]
                for summary in summaries
            ),
            "repeat_endpoint_pixel_max_abs": max(
                summary["repeat"]["endpoint_pixels"]["max_abs"]
                for summary in summaries
            ),
            "repeat_dataset_goal_cost_max_abs": max(
                summary["repeat"]["true_dataset_goal_cost_max_abs"]
                for summary in summaries
            ),
            "repeat_rendered_goal_cost_max_abs": max(
                summary["repeat"]["true_rendered_goal_cost_max_abs"]
                for summary in summaries
            ),
            "reference_physical_distance_m": max(
                summary["reference_corrected_physical"]["physical_distance_m_max_abs"]
                for summary in summaries
            ),
        },
        "scope": (
            "corrected offline true-endpoint diagnostic on locked candidates; "
            "same-renderer goal is primary and simulator information is evaluation-only"
        ),
        "aggregation": "10,000 bootstrap draws over 32 distinct snapshots",
    }
    (args.out / "summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    print(json.dumps(representation_gate, indent=2, sort_keys=True))
    print("OGB_TRUE_ENDPOINT_AGGREGATE_DONE")


if __name__ == "__main__":
    main()
