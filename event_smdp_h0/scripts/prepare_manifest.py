#!/usr/bin/env python3
"""Prepare a fresh non-trivial, episode-disjoint OGBench-Cube gate manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--num-snapshots", type=int, default=65)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--min-task-distance-m", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_snapshots < 2 or args.goal_offset <= 0 or args.min_task_distance_m <= 0:
        raise ValueError("invalid manifest configuration")

    import stable_worldmodel as swm

    excluded: set[int] = set()
    for path in args.exclude_manifest:
        if path.exists():
            excluded.update(int(row["episode"]) for row in json.loads(path.read_text()))

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=[])
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    offsets = np.asarray(dataset.offsets, dtype=np.int64)
    position_key = next(
        (
            key
            for key in ("privileged_block_0_pos", "privileged/block_0_pos")
            if key in dataset.column_names
        ),
        None,
    )
    if position_key is None:
        raise KeyError(
            "cube-position column absent; available columns="
            + ",".join(sorted(dataset.column_names))
        )
    positions = np.asarray(dataset.get_col_data(position_key), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise RuntimeError(f"unexpected cube-position column shape: {positions.shape}")

    rng = np.random.default_rng(args.seed)
    candidates: list[dict[str, int | float]] = []
    for episode, (offset, length) in enumerate(zip(offsets, lengths)):
        if episode in excluded or length <= args.goal_offset:
            continue
        starts = np.arange(int(offset), int(offset + length - args.goal_offset), dtype=np.int64)
        goal_rows = starts + args.goal_offset
        distances = np.linalg.norm(positions[goal_rows] - positions[starts], axis=1)
        valid = np.flatnonzero(distances >= args.min_task_distance_m)
        if not len(valid):
            continue
        local = int(rng.choice(valid))
        storage_row = int(starts[local])
        candidates.append(
            {
                "episode": int(episode),
                "start_step": int(storage_row - offset),
                "storage_row": storage_row,
                "task_distance_m": float(distances[local]),
            }
        )

    if len(candidates) < args.num_snapshots:
        raise RuntimeError(
            f"only {len(candidates)} eligible episode-disjoint windows for "
            f"{args.num_snapshots} requested snapshots"
        )
    chosen = np.sort(rng.choice(len(candidates), size=args.num_snapshots, replace=False))
    rows = []
    for order, index in enumerate(chosen):
        row = dict(candidates[int(index)])
        row.update(order=order, reset_seed=args.seed + 100_000 + order)
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "dataset": args.dataset,
        "goal_offset": args.goal_offset,
        "min_task_distance_m": args.min_task_distance_m,
        "n_snapshots": len(rows),
        "seed": args.seed,
        "excluded_episodes": len(excluded),
        "position_column": position_key,
        "exclude_manifests": [str(path) for path in args.exclude_manifest],
        "smoke_index": 0,
        "locked_indices": [1, args.num_snapshots - 1],
        "task_distance_m": {
            "min": float(min(row["task_distance_m"] for row in rows)),
            "median": float(np.median([row["task_distance_m"] for row in rows])),
            "max": float(max(row["task_distance_m"] for row in rows)),
        },
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
