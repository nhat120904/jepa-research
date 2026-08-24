#!/usr/bin/env python3
"""Aggregate deployed-CEM-action selection regret with snapshot bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
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
        raise RuntimeError(f"expected {args.expected_shards} Phase-0d shards, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        record = json.loads(path.read_text())
        if not record["repeat_gate"]["pass"]:
            raise RuntimeError(f"repeatability gate failed in {path}")
        corrective = record["verified_proxy_rejected_corrective"]
        final = record["final_population"]
        rows.append({
            "snapshot": int(record["snapshot"]),
            "returned_proxy_cost": float(record["returned"]["proxy_cost"]),
            "returned_physical_distance_m": float(record["returned"]["physical_distance_m"]),
            "returned_success": int(record["returned"]["success"]),
            "returned_selection_regret_m": float(final["returned_selection_regret_m"]),
            "returned_success_gap": int(final["returned_success_gap"]),
            "physical_oracle_candidate_distance_m": float(final["physical_oracle_distance_m"]),
            "physical_oracle_candidate_success": int(final["physical_oracle_success"]),
            "proxy_better_than_returned_fraction": float(final["proxy_better_than_returned_fraction"]),
            "verified_proxy_rejected_corrective": int(corrective["exists"]),
            "corrective_physical_advantage_m": (
                float(corrective["physical_advantage_m"]) if corrective["exists"] else 0.0
            ),
        })
    snapshots = [row["snapshot"] for row in rows]
    if snapshots != list(range(args.expected_shards)):
        raise RuntimeError(f"incomplete or unordered snapshot IDs: {snapshots}")
    rng = np.random.default_rng(args.seed)
    names = [name for name in rows[0] if name != "snapshot"]
    summary = {name: mean_ci(np.asarray([row[name] for row in rows]), rng, args.bootstrap) for name in names}
    regret = summary["returned_selection_regret_m"]
    correction = summary["verified_proxy_rejected_corrective"]
    verdict = (
        "GO_MINIMAL_POLICY_PILOT"
        if regret["ci_low"] > 0.0 and correction["ci_low"] > 0.0
        else "HOLD_NO_CI_CLEAN_DEPLOYED_ACTION_SELECTION_REGRET"
    )
    report = {
        "scope": (
            "exact final proposal mean returned by CEM, evaluated post-hoc against final-population "
            "candidates from the same restored physical state. This is a mechanism gate, not policy evidence."
        ),
        "config": {"n_snapshots": args.expected_shards, "bootstrap_draws": args.bootstrap, "bootstrap_seed": args.seed},
        "definitions": {
            "returned_selection_regret_m": "physical distance of returned CEM mean minus physical-best final-population candidate",
            "verified_proxy_rejected_corrective": "a final candidate has strictly worse proxy cost than returned mean but is at least 2 cm physically better",
        },
        "summary": summary,
        "verdict": verdict,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Phase 0d — deployed CEM-action physical selection regret",
        "",
        f"- **Verdict:** {verdict}",
        (
            "- Returned-plan selection regret: "
            f"{regret['mean'] * 100:.2f} cm [{regret['ci_low'] * 100:.2f}, {regret['ci_high'] * 100:.2f}]"
        ),
        (
            "- Verified proxy-rejected corrective rate: "
            f"{correction['mean'] * 100:.1f}% [{correction['ci_low'] * 100:.1f}, {correction['ci_high'] * 100:.1f}]"
        ),
        "",
        "All physical evaluations are snapshot/restore post-hoc replays. No simulator outcome was available to CEM.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
