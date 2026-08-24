#!/usr/bin/env python3
"""Create an expert-sequence cache manifest matched to off-policy sequence count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--offpolicy-manifest", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=15_360)
    parser.add_argument("--primitive-horizon", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(args.dataset, keys_to_load=["action"])
    excluded = {int(row["episode"]) for row in json.loads(args.offpolicy_manifest.read_text())}
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    offsets = np.asarray(dataset.offsets, dtype=np.int64)
    counts = np.maximum(lengths - args.primitive_horizon, 0)
    for episode in excluded:
        counts[episode] = 0
    total = int(counts.sum())
    if args.num_sequences > total:
        raise ValueError(f"requested {args.num_sequences}, only {total} valid starts")

    rng = np.random.default_rng(args.seed)
    ordinals = np.sort(rng.choice(total, size=args.num_sequences, replace=False))
    eligible = np.flatnonzero(counts)
    cumulative = np.cumsum(counts[eligible])
    rows = []
    for order, ordinal in enumerate(ordinals):
        local_episode = int(np.searchsorted(cumulative, ordinal, side="right"))
        episode = int(eligible[local_episode])
        previous = int(cumulative[local_episode - 1]) if local_episode else 0
        start = int(ordinal - previous)
        rows.append(
            {
                "order": order,
                "episode": episode,
                "start_step": start,
                "storage_row": int(offsets[episode] + start),
            }
        )
    if len({(row["episode"], row["start_step"]) for row in rows}) != len(rows):
        raise RuntimeError("expert manifest contains duplicate sequence starts")
    if any(row["episode"] in excluded for row in rows):
        raise RuntimeError("episode exclusion failure")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    summary = {
        "num_sequences": len(rows),
        "num_episodes": len({row["episode"] for row in rows}),
        "excluded_offpolicy_episodes": len(excluded),
        "seed": args.seed,
        "primitive_horizon": args.primitive_horizon,
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

