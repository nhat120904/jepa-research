#!/usr/bin/env python3
"""Create a fresh episode-disjoint manifest for the locked Phase-1a test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest", type=Path, action="append", default=[],
        help="additional manifests whose entire episodes must be excluded",
    )
    parser.add_argument("--num-snapshots", type=int, default=128)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_snapshots <= 0 or args.goal_offset <= 0:
        raise ValueError("snapshot count and goal offset must be positive")
    import stable_worldmodel as swm

    excluded_episodes: set[int] = set()
    for path in (args.audit_manifest, *args.exclude_manifest):
        excluded_episodes.update(int(row["episode"]) for row in json.loads(path.read_text()))
    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=[])
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    offsets = np.asarray(dataset.offsets, dtype=np.int64)
    eligible = np.flatnonzero(lengths > args.goal_offset)
    eligible = np.asarray([idx for idx in eligible if int(idx) not in excluded_episodes])
    if len(eligible) < args.num_snapshots:
        raise RuntimeError(
            f"only {len(eligible)} episode-disjoint states available for {args.num_snapshots} snapshots"
        )
    rng = np.random.default_rng(args.seed)
    episodes = np.sort(rng.choice(eligible, size=args.num_snapshots, replace=False))
    rows = []
    for order, episode in enumerate(episodes):
        max_start = int(lengths[episode] - args.goal_offset)
        start_step = int(rng.integers(max_start))
        rows.append({
            "order": order,
            "episode": int(episode),
            "start_step": start_step,
            "storage_row": int(offsets[episode] + start_step),
            "reset_seed": int(args.seed + 100_000 + order),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "n_snapshots": len(rows),
        "episode_disjoint_from_phase0d": True,
        "excluded_episodes": len(excluded_episodes),
        "excluded_manifests": [str(args.audit_manifest), *map(str, args.exclude_manifest)],
        "seed": args.seed,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
