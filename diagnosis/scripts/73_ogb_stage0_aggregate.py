#!/usr/bin/env python3
"""Aggregate snapshot-sharded OGBench Stage-0 candidate audits."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def load_audit_module():
    source = Path(__file__).with_name("72_ogb_stage0_candidate_audit.py")
    spec = importlib.util.spec_from_file_location("ogb_stage0_audit", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    audit = load_audit_module()
    shard_dirs = sorted(
        (path for path in args.shards.iterdir() if path.is_dir()),
        key=lambda path: int(path.name),
    )
    if len(shard_dirs) != args.expected:
        raise RuntimeError(f"expected {args.expected} shards, found {len(shard_dirs)}")

    snapshot_rows = []
    candidate_rows = []
    restorations = []
    configs = []
    normalizer_errors = []
    seen = set()
    for shard in shard_dirs:
        summary = json.loads((shard / "summary.json").read_text())
        configs.append(summary["config"])
        normalizer_errors.append(summary["action_normalizer_roundtrip_max_abs"])
        restorations.extend(summary["restoration"])
        with (shard / "snapshot_metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                parsed = {
                    key: value if key == "population" else int(value) if key == "snapshot" else float(value)
                    for key, value in row.items()
                }
                snapshot_rows.append(parsed)
                seen.add(parsed["snapshot"])
        with gzip.open(shard / "candidate_costs.csv.gz", "rt", newline="") as handle:
            candidate_rows.extend(csv.DictReader(handle))

    if seen != set(range(args.expected)):
        raise RuntimeError(f"snapshot coverage mismatch: {sorted(seen)}")
    reference = {
        key: value
        for key, value in configs[0].items()
        if key not in {"snapshot_index", "processed_snapshots"}
    }
    for config in configs[1:]:
        comparable = {
            key: value
            for key, value in config.items()
            if key not in {"snapshot_index", "processed_snapshots"}
        }
        if comparable != reference:
            raise RuntimeError("shard configuration mismatch")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(snapshot_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(snapshot_rows, key=lambda row: (row["snapshot"], row["population"])))
    with gzip.open(args.out / "candidate_costs.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    (args.out / "manifest.json").write_text((shard_dirs[0] / "manifest.json").read_text())

    metrics = audit.bootstrap_summary(snapshot_rows, args.bootstrap, args.seed + 99)
    combined = {
        "config": {**reference, "processed_snapshots": args.expected, "snapshot_index": None},
        "action_normalizer_roundtrip_max_abs": max(normalizer_errors),
        "restoration": sorted(restorations, key=lambda row: row["snapshot"]),
        "metrics": metrics,
        "representation_gate": audit.representation_gate(metrics),
        "scope": "offline matched-candidate audit; simulator was not exposed to the planner",
        "aggregation": "one independent Slurm shard per precommitted snapshot",
    }
    (args.out / "summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    print(json.dumps(combined["representation_gate"], indent=2, sort_keys=True))
    print("OGB_STAGE0_AGGREGATE_DONE")


if __name__ == "__main__":
    main()
