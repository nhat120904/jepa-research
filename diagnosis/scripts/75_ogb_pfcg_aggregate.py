#!/usr/bin/env python3
"""Aggregate locked PFCG same-candidate pilot shards."""

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
    boot = rng.choice(finite, size=(draws, len(finite)), replace=True).mean(axis=1)
    return {
        "mean": float(finite.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_finite": int(len(finite)),
    }


def paired_contrast(
    rows: dict[str, dict[int, dict[str, Any]]],
    baseline: str,
    method: str,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, dict[str, float]]:
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
    all_rows = []
    candidate_rows = []
    configs = []
    geometry_rows = []
    artifact_reproduction = []
    repeat_replay = []
    seen = set()
    for shard in shard_dirs:
        summary = json.loads((shard / "summary.json").read_text())
        configs.append(summary["config"])
        geometry_rows.append(summary["geometry"])
        artifact_reproduction.append(summary["artifact_reproduction"])
        repeat_replay.append(summary["repeat_replay"])
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
    if any(set(selector_rows) != expected_ids for selector_rows in rows.values()):
        raise RuntimeError("one or more selectors lack complete snapshot coverage")
    reference = {
        key: value for key, value in configs[0].items() if key != "snapshot_index"
    }
    for config in configs[1:]:
        if {key: value for key, value in config.items() if key != "snapshot_index"} != reference:
            raise RuntimeError("shard configuration mismatch")

    rng = np.random.default_rng(args.seed + 99)
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    metric_names = [
        "selected_physical_distance_m",
        "selected_success",
        "selection_regret_m",
        "success_gap",
        "spearman_physical",
        "top10pct_recall_physical",
        "false_elite_rate_physical",
    ]
    for selector, selector_rows in sorted(rows.items()):
        ordered = [selector_rows[idx] for idx in range(args.expected)]
        metrics[selector] = {
            name: interval(np.asarray([row[name] for row in ordered]), rng, args.bootstrap)
            for name in metric_names
        }

    contrasts = {
        "pred_pfcg_vs_pred_l2": paired_contrast(
            rows, "pred_l2", "pred_pfcg", rng, args.bootstrap
        ),
        "pred_pfcg_vs_pred_projected": paired_contrast(
            rows, "pred_projected", "pred_pfcg", rng, args.bootstrap
        ),
        "pred_pfcg_vs_pred_diag": paired_contrast(
            rows, "pred_diag", "pred_pfcg", rng, args.bootstrap
        ),
        "pred_pfcg_vs_pred_random": paired_contrast(
            rows, "pred_random", "pred_pfcg", rng, args.bootstrap
        ),
        "pred_projected_vs_pred_l2": paired_contrast(
            rows, "pred_l2", "pred_projected", rng, args.bootstrap
        ),
        "pred_random_vs_pred_l2": paired_contrast(
            rows, "pred_l2", "pred_random", rng, args.bootstrap
        ),
    }
    pred_pair = contrasts["pred_pfcg_vs_pred_l2"]
    pred_regret_positive = pred_pair["regret_gain_m"]["ci_low"] > 0
    pred_success_positive = pred_pair["success_gain"]["ci_low"] > 0
    pred_success_mean = pred_pair["success_gain"]["mean"]
    beats_projected_mean = (
        metrics["pred_pfcg"]["selection_regret_m"]["mean"]
        < metrics["pred_projected"]["selection_regret_m"]["mean"]
    )
    beats_random_mean = (
        metrics["pred_pfcg"]["selection_regret_m"]["mean"]
        < metrics["pred_random"]["selection_regret_m"]["mean"]
    )
    strong = (
        pred_regret_positive
        and pred_success_positive
        and beats_projected_mean
        and beats_random_mean
    )
    conditional = (
        pred_regret_positive
        and pred_success_mean > 0
        and metrics["pred_pfcg"]["selection_regret_m"]["mean"]
        <= metrics["pred_projected"]["selection_regret_m"]["mean"]
    )
    verdict = {
        "status": "strong_go" if strong else "conditional_expand" if conditional else "no_go",
        "criteria": {
            "predicted_regret_gain_ci_low_gt_zero": pred_regret_positive,
            "predicted_success_gain_ci_low_gt_zero": pred_success_positive,
            "predicted_success_gain_mean_gt_zero": pred_success_mean > 0,
            "predicted_pfcg_beats_projected_mean_regret": beats_projected_mean,
            "predicted_pfcg_beats_random_mean_regret": beats_random_mean,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (row["snapshot"], row["selector"])))
    with gzip.open(args.out / "candidate_costs.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    combined = {
        "config": {**reference, "processed_snapshots": args.expected},
        "metrics": metrics,
        "paired_contrasts": contrasts,
        "verdict": verdict,
        "geometry": {
            "rank_min": min(row["rank"] for row in geometry_rows),
            "rank_max": max(row["rank"] for row in geometry_rows),
            "rank_mean": float(np.mean([row["rank"] for row in geometry_rows])),
            "response_energy_retained_min": min(
                row["response_energy_retained"] for row in geometry_rows
            ),
        },
        "artifact_reproduction_worst": {
            key: max(row[key] for row in artifact_reproduction)
            for key in artifact_reproduction[0]
        },
        "repeat_replay_worst": {
            key: max(row[key] for row in repeat_replay)
            for key in repeat_replay[0]
        },
        "scope": (
            "same persisted L2-generated candidates; deployable PFCG choices fixed "
            "before simulator evaluation; no simulator rendering/encoding arm"
        ),
        "aggregation": "10,000 bootstrap draws over 32 distinct snapshots",
    }
    (args.out / "summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print("OGB_PFCG_AGGREGATE_DONE")


if __name__ == "__main__":
    main()
