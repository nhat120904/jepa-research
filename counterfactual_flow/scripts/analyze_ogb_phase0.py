#!/usr/bin/env python3
"""Aggregate the Phase-0 counterfactual-mining shards without loading models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--min-deceptive-snapshots", type=int, default=8)
    parser.add_argument("--min-regret-m", type=float, default=0.02)
    parser.add_argument("--max-control-gap-m", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = sorted(args.shards.glob("*/summary.json"))
    if len(summaries) != args.expected_shards:
        raise RuntimeError(
            f"expected {args.expected_shards} successful shards, found {len(summaries)}"
        )
    rows: list[dict[str, Any]] = []
    for path in summaries:
        data = json.loads(path.read_text())
        if not data.get("repeat_gate", {}).get("pass", False):
            raise RuntimeError(f"repeat gate failed in {path}")
        row: dict[str, Any] = {
            "snapshot": data["snapshot"],
            "pool_physical_best_m": data["pool_physical_best_m"],
            "pool_success_available": data["pool_success_available"],
            "has_deceptive": int(data["group"]["has_deceptive"]),
        }
        if row["has_deceptive"]:
            row.update({
                "deceptive_regret_m": data["deceptive"]["physical_regret_m"],
                "deceptive_proxy_rank_fraction": data["deceptive"]["proxy_rank_fraction"],
                "control_regret_gap_m": data["matched_control"]["absolute_regret_gap_m"],
            })
        rows.append(row)

    deceptive = [row for row in rows if row["has_deceptive"]]
    valid_controls = [
        row for row in deceptive if row["control_regret_gap_m"] <= args.max_control_gap_m
    ]
    decision = bool(
        len(deceptive) >= args.min_deceptive_snapshots
        and sum(row["deceptive_regret_m"] >= args.min_regret_m for row in deceptive)
        >= args.min_deceptive_snapshots
        and len(valid_controls) >= args.min_deceptive_snapshots
    )
    report = {
        "expected_shards": args.expected_shards,
        "n_shards": len(rows),
        "n_deceptive": len(deceptive),
        "deceptive_coverage": len(deceptive) / len(rows),
        "n_matched_controls": len(valid_controls),
        "thresholds": {
            "min_deceptive_snapshots": args.min_deceptive_snapshots,
            "min_regret_m": args.min_regret_m,
            "max_control_gap_m": args.max_control_gap_m,
        },
        "phase0_decision": "GO" if decision else "NO_GO",
        "scope": "exploratory dataset-construction gate; not held-out policy evidence",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.out_dir / "snapshot_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Phase-0 OGBench-Cube counterfactual mining",
        "",
        f"- **Decision:** {report['phase0_decision']}",
        f"- Successful shards: {len(rows)}/{args.expected_shards}",
        f"- Snapshots with a proxy-deceptive candidate: {len(deceptive)}/{len(rows)}",
        f"- Hardness-matched controls within {args.max_control_gap_m * 100:.1f} cm: {len(valid_controls)}",
        "",
        "This is an exploratory data gate. It establishes neither policy improvement nor generalization.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
