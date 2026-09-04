#!/usr/bin/env python3
"""Aggregate true-q posterior rank on replayed failure trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROTOCOL = "scene_event_posterior_support_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(records: list[dict], label: str) -> dict:
    probability = np.asarray(
        [float(record["true_joint_probability"]) for record in records], dtype=float
    )
    rank = np.asarray([int(record["true_joint_rank"]) for record in records], dtype=int)
    return {
        "subset": label,
        "n": len(records),
        "mean_true_probability": float(probability.mean()),
        "median_true_probability": float(np.median(probability)),
        "top1_coverage": float(np.mean(rank <= 1)),
        "top2_coverage": float(np.mean(rank <= 2)),
        "top3_coverage": float(np.mean(rank <= 3)),
        "median_rank": float(np.median(rank)),
        "rank_p90": float(np.quantile(rank, 0.9)),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob("*/result.json"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")
    expected_seeds = set(range(84500, 84564))
    seen_seeds: set[int] = set()
    records: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError(f"unexpected protocol in {path}")
        reset_seed = int(payload["reset_seed"])
        if reset_seed in seen_seeds:
            raise RuntimeError(f"duplicate reset {reset_seed}")
        seen_seeds.add(reset_seed)
        if payload["source_success"] != payload["replay_success"]:
            raise RuntimeError(f"source/replay success mismatch in {path}")
        for record in payload["records"]:
            records.append({"reset_seed": reset_seed, **record})
    if seen_seeds != expected_seeds:
        raise RuntimeError("replay reset set does not match the locked replication range")

    hard_errors = [record for record in records if not record["hard_correct"]]
    catastrophic = [
        record
        for record in records
        if int(record["true_state"]["cube_stage"]) == 1
        and int(record["true_state"]["window_stage"]) == 2
    ]
    catastrophic_errors = [record for record in catastrophic if not record["hard_correct"]]
    summaries = [
        summarize(records, "all_visits"),
        summarize(hard_errors, "hard_map_errors"),
        summarize(catastrophic, "true_q_1_2"),
        summarize(catastrophic_errors, "true_q_1_2_hard_errors"),
    ]
    primary = next(row for row in summaries if row["subset"] == "hard_map_errors")
    if (
        primary["top3_coverage"] >= 0.75
        and primary["median_true_probability"] >= 0.05
    ):
        verdict = "SOFT_POSTERIOR_SUPPORT_PASS"
    elif (
        primary["top3_coverage"] < 0.50
        or primary["median_true_probability"] < 0.01
    ):
        verdict = "HISTORY_REQUIRED"
    else:
        verdict = "HYBRID_SOFT_HISTORY_REQUIRED"
    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "primary": primary,
        "summaries": summaries,
        "num_resets": len(seen_seeds),
        "num_visits": len(records),
        "scope": (
            "diagnostic replay of completed seed-1 task-5 trajectories; no new "
            "success-rate claim"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.out_dir / "DECISION.md").write_text(
        "# Scene event-posterior support gate\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"On {primary['n']} hard-MAP errors, true-q top-3 coverage is "
        f"{100.0 * primary['top3_coverage']:.2f}% and median probability is "
        f"{primary['median_true_probability']:.6f}.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
