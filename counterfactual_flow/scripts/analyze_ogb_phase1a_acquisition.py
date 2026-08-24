#!/usr/bin/env python3
"""Aggregate the preregistered Phase-1a matched-query acquisition test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY = "proxy_rejected_stratified_diverse"
CONTROLS = ("random_final_population", "proxy_hard", "action_diverse")
METRICS = (
    "inversion_hit",
    "best_corrective_advantage_m",
    "best_any_advantage_m",
    "any_success_gain",
    "selected_success_any",
    "best_selected_distance_m",
    "proxy_rejected_fraction",
    "mean_proxy_rank_fraction",
    "mean_action_distance_from_anchor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite nonempty vector")
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n": int(len(values)),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.shards.glob("*/summary.json"), key=lambda path: int(path.parent.name))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} Phase-1a shards, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for path in paths:
        record = json.loads(path.read_text())
        if not record["repeat_gate"]["pass"]:
            raise RuntimeError(f"anchor repeatability gate failed in {path}")
        configs.append(record["config"])
        for arm, values in record["arms"].items():
            rows.append({
                "snapshot": int(record["snapshot"]),
                "arm": arm,
                "anchor_physical_distance_m": float(record["anchor"]["physical_distance_m"]),
                "anchor_success": int(record["anchor"]["success"]),
                **{metric: float(values[metric]) for metric in METRICS},
            })
    ref = configs[0]
    for config in configs[1:]:
        if config != ref:
            raise RuntimeError("Phase-1a shard configuration mismatch")
    expected_ids = list(range(args.expected_shards))
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[row["snapshot"]] = row
    expected_arms = {PRIMARY, *CONTROLS}
    if set(by_arm) != expected_arms:
        raise RuntimeError(f"unexpected arms: {sorted(by_arm)}")
    if any(sorted(records) != expected_ids for records in by_arm.values()):
        raise RuntimeError("incomplete arm/snapshot pairing")

    rng = np.random.default_rng(args.seed + 99)
    arm_summary = {
        arm: {
            metric: mean_ci(np.asarray([by_arm[arm][idx][metric] for idx in expected_ids]), rng, args.bootstrap)
            for metric in ("anchor_physical_distance_m", "anchor_success", *METRICS)
        }
        for arm in sorted(by_arm)
    }
    contrasts: dict[str, dict[str, dict[str, float]]] = {}
    for control in CONTROLS:
        contrasts[f"{PRIMARY}_minus_{control}"] = {
            "inversion_hit_gain": mean_ci(np.asarray([
                by_arm[PRIMARY][idx]["inversion_hit"] - by_arm[control][idx]["inversion_hit"]
                for idx in expected_ids
            ]), rng, args.bootstrap),
            "best_corrective_advantage_gain_m": mean_ci(np.asarray([
                by_arm[PRIMARY][idx]["best_corrective_advantage_m"]
                - by_arm[control][idx]["best_corrective_advantage_m"]
                for idx in expected_ids
            ]), rng, args.bootstrap),
            "success_gain_gain": mean_ci(np.asarray([
                by_arm[PRIMARY][idx]["any_success_gain"] - by_arm[control][idx]["any_success_gain"]
                for idx in expected_ids
            ]), rng, args.bootstrap),
        }
    versus_random = contrasts[f"{PRIMARY}_minus_random_final_population"]
    verdict = (
        "GO_MATCHED_BUDGET_POLICY_PILOT"
        if (
            versus_random["inversion_hit_gain"]["ci_low"] > 0.0
            and versus_random["best_corrective_advantage_gain_m"]["ci_low"] > 0.0
        )
        else "HOLD_NO_CI_CLEAN_ACQUISITION_GAIN_VS_RANDOM"
    )
    report = {
        "scope": (
            "Fresh episode-disjoint OGBench-Cube snapshots. Each arm chooses K alternatives "
            "before physics and is charged K+1 physical branches including the CEM-returned anchor."
        ),
        "selection_rule": (
            "primary arm partitions proxy-rejected final-CEM candidates into K proxy-rank strata "
            "and selects the most action-novel candidate in each stratum; no physical labels enter selection"
        ),
        "config": {"n_snapshots": args.expected_shards, "bootstrap_draws": args.bootstrap, "bootstrap_seed": args.seed, **ref},
        "primary_arm": PRIMARY,
        "controls": list(CONTROLS),
        "arm_summary": arm_summary,
        "paired_contrasts": contrasts,
        "verdict": verdict,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "arm_snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["snapshot"], row["arm"])))
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    hit = versus_random["inversion_hit_gain"]
    advantage = versus_random["best_corrective_advantage_gain_m"]
    lines = [
        "# Phase 1a — matched-budget ordinal-inversion acquisition",
        "",
        f"- **Verdict:** {verdict}",
        f"- Physical budget per arm/state: {ref['physical_budget_per_arm']} branches.",
        (
            "- Primary − random inversion-hit gain: "
            f"{hit['mean'] * 100:+.1f} pp [{hit['ci_low'] * 100:+.1f}, {hit['ci_high'] * 100:+.1f}]"
        ),
        (
            "- Primary − random best-corrective-advantage gain: "
            f"{advantage['mean'] * 100:+.2f} cm "
            f"[{advantage['ci_low'] * 100:+.2f}, {advantage['ci_high'] * 100:+.2f}]"
        ),
        "",
        "Selections were frozen before all MuJoCo rollouts; this is acquisition evidence, not policy evidence.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
