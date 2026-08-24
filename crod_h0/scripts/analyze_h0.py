#!/usr/bin/env python3
"""Aggregate the preregistered fresh-cohort CROD H0 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY = "crod_directional"
PRIMARY_CONTROL = "action_diverse"
CONTROLS = (
    "action_diverse",
    "rejected_action_diverse",
    "dino_best_rejected",
    "native_uncertainty_diverse",
    "random_rejected",
)
METRICS = (
    "inversion_hit",
    "best_corrective_advantage_m",
    "best_any_advantage_m",
    "best_improvement_per_query_m",
    "any_success_gain",
    "selected_success_any",
    "best_selected_distance_m",
    "native_rejected_fraction",
    "auxiliary_prefers_fraction",
    "directional_positive_fraction",
    "mean_crod_score",
    "mean_native_rank_fraction",
    "mean_auxiliary_rank_fraction",
    "mean_action_distance_from_anchor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap input must be finite and nonempty")
    boot = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n": int(len(values)),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.shards.glob("*/summary.json"), key=lambda p: int(p.parent.name))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    configs = []
    diagnostics = []
    for path in paths:
        record = json.loads(path.read_text())
        if not record["repeat_gate"]["pass"]:
            raise RuntimeError(f"repeatability gate failed in {path}")
        configs.append(record["config"])
        diagnostics.append(record["diagnostics"])
        for arm, values in record["arms"].items():
            rows.append(
                {
                    "snapshot": int(record["snapshot"]),
                    "arm": arm,
                    "anchor_physical_distance_m": float(
                        record["anchor"]["physical_distance_m"]
                    ),
                    "anchor_success": int(record["anchor"]["success"]),
                    **{name: float(values[name]) for name in METRICS},
                }
            )
    if any(config != configs[0] for config in configs[1:]):
        raise RuntimeError("shard configuration mismatch")
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[int(row["snapshot"])] = row
    if set(by_arm) != {PRIMARY, *CONTROLS}:
        raise RuntimeError(f"unexpected arms: {sorted(by_arm)}")
    ids = list(range(args.expected_shards))
    if any(sorted(records) != ids for records in by_arm.values()):
        raise RuntimeError("incomplete arm/snapshot pairing")

    rng = np.random.default_rng(args.seed + 91)
    arm_summary = {
        arm: {
            metric: mean_ci(
                np.asarray([by_arm[arm][idx][metric] for idx in ids]), rng, args.bootstrap
            )
            for metric in ("anchor_physical_distance_m", "anchor_success", *METRICS)
        }
        for arm in sorted(by_arm)
    }
    contrasts: dict[str, dict[str, dict[str, float]]] = {}
    for control in CONTROLS:
        contrasts[f"{PRIMARY}_minus_{control}"] = {
            metric: mean_ci(
                np.asarray(
                    [
                        by_arm[PRIMARY][idx][metric] - by_arm[control][idx][metric]
                        for idx in ids
                    ]
                ),
                rng,
                args.bootstrap,
            )
            for metric in (
                "inversion_hit",
                "best_corrective_advantage_m",
                "best_improvement_per_query_m",
                "any_success_gain",
            )
        }
    primary_contrast = contrasts[f"{PRIMARY}_minus_{PRIMARY_CONTROL}"]
    hit = primary_contrast["inversion_hit"]
    advantage = primary_contrast["best_corrective_advantage_m"]
    if hit["ci_low"] > 0 and advantage["mean"] > 0:
        verdict = "GO_CROD_H1_SIMPLE_BC"
    elif hit["mean"] > 0:
        verdict = "HOLD_CROD_DIRECTIONAL_GAIN_NOT_CI_CLEAN"
    else:
        verdict = "STOP_CROD_NO_GAIN_OVER_ACTION_DIVERSITY"

    report = {
        "scope": (
            "Fresh 128-state cohort, episode-disjoint from Phase 0/1a/1a-v2. "
            "All selectors are outcome-blind and charged nine branches per state."
        ),
        "primary_arm": PRIMARY,
        "primary_control": PRIMARY_CONTROL,
        "primary_metric": "corrective inversion-hit rate at a 2 cm physical margin",
        "gate": (
            "GO only if the paired 95% bootstrap CI lower bound for CROD minus "
            "action diversity is above zero on corrective hit rate and the mean "
            "best-corrective-advantage contrast is positive."
        ),
        "config": {
            "n_snapshots": args.expected_shards,
            "bootstrap_draws": args.bootstrap,
            "bootstrap_seed": args.seed,
            **configs[0],
        },
        "population_diagnostics": {
            "mean_native_rejected_candidates": float(
                np.mean([row["native_rejected_candidates"] for row in diagnostics])
            ),
            "mean_directional_positive_candidates": float(
                np.mean([row["directional_positive_candidates"] for row in diagnostics])
            ),
        },
        "arm_summary": arm_summary,
        "paired_contrasts": contrasts,
        "verdict": verdict,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "arm_snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["snapshot"], r["arm"])))
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# CROD H0 — fresh matched-budget result",
        "",
        f"- **Verdict:** {verdict}",
        (
            "- CROD − action diversity corrective-hit gain: "
            f"{hit['mean'] * 100:+.1f} pp "
            f"[{hit['ci_low'] * 100:+.1f}, {hit['ci_high'] * 100:+.1f}]"
        ),
        (
            "- CROD − action diversity best corrective advantage: "
            f"{advantage['mean'] * 100:+.2f} cm "
            f"[{advantage['ci_low'] * 100:+.2f}, {advantage['ci_high'] * 100:+.2f}]"
        ),
        "",
        "No simulator outcome is used by any selector.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
