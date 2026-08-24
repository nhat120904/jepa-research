#!/usr/bin/env python3
"""Aggregate the locked fresh-cohort Phase-1a-v2 acquisition replication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY = "proxy_instability_diverse"
CONTROLS = ("action_diverse", "random_final_population", "cem_disagreement_diverse")
METRICS = (
    "inversion_hit", "best_corrective_advantage_m", "best_any_advantage_m",
    "any_success_gain", "selected_success_any", "best_selected_distance_m",
    "proxy_rejected_fraction", "mean_proxy_rank_fraction",
    "mean_action_distance_from_anchor", "mean_proxy_instability",
    "mean_cem_disagreement",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite nonempty vector")
    boot = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()), "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)), "n": int(len(values)),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.shards.glob("*/summary.json"), key=lambda path: int(path.parent.name))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for path in paths:
        record = json.loads(path.read_text())
        if not record["repeat_gate"]["pass"]:
            raise RuntimeError(f"repeat gate failed in {path}")
        configs.append(record["config"])
        for arm, values in record["arms"].items():
            rows.append({
                "snapshot": int(record["snapshot"]), "arm": arm,
                "anchor_physical_distance_m": float(record["anchor"]["physical_distance_m"]),
                "anchor_success": int(record["anchor"]["success"]),
                **{name: float(values[name]) for name in METRICS},
            })
    if any(config != configs[0] for config in configs[1:]):
        raise RuntimeError("shard configuration mismatch")
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[row["snapshot"]] = row
    if set(by_arm) != {PRIMARY, *CONTROLS}:
        raise RuntimeError(f"unexpected arms: {sorted(by_arm)}")
    expected_ids = list(range(args.expected_shards))
    if any(sorted(records) != expected_ids for records in by_arm.values()):
        raise RuntimeError("incomplete arm/snapshot pairing")

    rng = np.random.default_rng(args.seed + 99)
    arm_summary = {
        arm: {
            name: mean_ci(np.asarray([by_arm[arm][idx][name] for idx in expected_ids]), rng, args.bootstrap)
            for name in ("anchor_physical_distance_m", "anchor_success", *METRICS)
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
    versus_diverse = contrasts[f"{PRIMARY}_minus_action_diverse"]
    required = (
        versus_random["inversion_hit_gain"]["ci_low"] > 0.0,
        versus_random["best_corrective_advantage_gain_m"]["ci_low"] > 0.0,
        versus_diverse["inversion_hit_gain"]["ci_low"] > 0.0,
        versus_diverse["best_corrective_advantage_gain_m"]["ci_low"] > 0.0,
    )
    verdict = (
        "GO_MATCHED_BUDGET_POLICY_PILOT"
        if all(required)
        else "STOP_NO_ROBUST_MODEL_SIGNAL_BEYOND_DIVERSITY"
    )
    report = {
        "scope": (
            "Locked test on a fresh cohort episode-disjoint from both Phase 0d and Phase 1a. "
            "All selectors are frozen-model-only and use nine accounted physical branches/state."
        ),
        "primary_arm": PRIMARY,
        "controls": list(CONTROLS),
        "gate": (
            "primary inversion-hit and corrective-advantage gains must both have CI_low>0 "
            "against random and pure action diversity"
        ),
        "config": {"n_snapshots": args.expected_shards, "bootstrap_draws": args.bootstrap, "bootstrap_seed": args.seed, **configs[0]},
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
    r_hit = versus_random["inversion_hit_gain"]
    d_hit = versus_diverse["inversion_hit_gain"]
    r_adv = versus_random["best_corrective_advantage_gain_m"]
    d_adv = versus_diverse["best_corrective_advantage_gain_m"]
    lines = [
        "# Phase 1a-v2 — proxy-instability acquisition replication",
        "", f"- **Verdict:** {verdict}",
        (
            "- Primary − random inversion-hit gain: "
            f"{r_hit['mean'] * 100:+.1f} pp [{r_hit['ci_low'] * 100:+.1f}, {r_hit['ci_high'] * 100:+.1f}]"
        ),
        (
            "- Primary − action-diverse inversion-hit gain: "
            f"{d_hit['mean'] * 100:+.1f} pp [{d_hit['ci_low'] * 100:+.1f}, {d_hit['ci_high'] * 100:+.1f}]"
        ),
        (
            "- Primary − random corrective-advantage gain: "
            f"{r_adv['mean'] * 100:+.2f} cm [{r_adv['ci_low'] * 100:+.2f}, {r_adv['ci_high'] * 100:+.2f}]"
        ),
        (
            "- Primary − action-diverse corrective-advantage gain: "
            f"{d_adv['mean'] * 100:+.2f} cm [{d_adv['ci_low'] * 100:+.2f}, {d_adv['ci_high'] * 100:+.2f}]"
        ),
        "", "No physical label or outcome is used by any selector.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
