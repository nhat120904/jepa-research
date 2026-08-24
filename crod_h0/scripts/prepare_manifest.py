#!/usr/bin/env python3
"""Create the fresh, episode-disjoint 128-state CROD H0 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--num-snapshots", type=int, default=128)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_snapshots <= 0 or args.goal_offset <= 0:
        raise ValueError("snapshot count and goal offset must be positive")
    import stable_worldmodel as swm

    excluded: set[int] = set()
    for path in args.exclude_manifest:
        excluded.update(int(row["episode"]) for row in json.loads(path.read_text()))
    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=[])
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    offsets = np.asarray(dataset.offsets, dtype=np.int64)
    eligible = np.asarray(
        [i for i in np.flatnonzero(lengths > args.goal_offset) if int(i) not in excluded],
        dtype=np.int64,
    )
    if len(eligible) < args.num_snapshots:
        raise RuntimeError(
            f"only {len(eligible)} episode-disjoint choices for {args.num_snapshots} states"
        )
    rng = np.random.default_rng(args.seed)
    episodes = np.sort(rng.choice(eligible, size=args.num_snapshots, replace=False))
    rows = []
    for order, episode in enumerate(episodes):
        max_start = int(lengths[episode] - args.goal_offset)
        start = int(rng.integers(max_start))
        rows.append(
            {
                "order": order,
                "episode": int(episode),
                "start_step": start,
                "storage_row": int(offsets[episode] + start),
                "reset_seed": int(args.seed + 100_000 + order),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_snapshots": len(rows),
                "excluded_episodes": len(excluded),
                "excluded_manifests": [str(p) for p in args.exclude_manifest],
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
