#!/usr/bin/env python3
"""Create a deterministic Stage-A PushT qualification manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np


ORIGINAL_DINO_WM_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_block_refs(lengths: list[int], block_steps: int) -> list[tuple[int, int]]:
    refs: list[tuple[int, int]] = []
    for episode, length in enumerate(lengths):
        refs.extend((episode, start) for start in range(0, int(length) - block_steps + 1))
    return refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="PushT train or val split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchors", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--order-pairs", type=int, default=2)
    parser.add_argument("--block-steps", type=int, default=5)
    parser.add_argument(
        "--num-hist",
        type=int,
        default=3,
        help="frames retained for diagnostics/training; official planning scorer uses the last frame",
    )
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Submit manifest creation through Slurm")
    if args.candidates < 2 * args.order_pairs + 1:
        raise ValueError(
            "candidates must include both orders for every designated pair plus one witness"
        )

    lengths_path = args.dataset / "seq_lengths.pkl"
    with lengths_path.open("rb") as handle:
        lengths = [int(x) for x in pickle.load(handle)]
    minimum_anchor = (args.num_hist - 1) * args.block_steps
    eligible = [
        (episode, step)
        for episode, length in enumerate(lengths)
        for step in range(minimum_anchor, length - 2 * args.block_steps + 1)
    ]
    if args.anchors > len(eligible):
        raise ValueError(f"requested {args.anchors} anchors but only {len(eligible)} are eligible")
    block_refs = valid_block_refs(lengths, args.block_steps)
    if not block_refs:
        raise ValueError("dataset has no complete action blocks")

    rng = np.random.default_rng(args.seed)
    anchor_indices = rng.choice(len(eligible), size=args.anchors, replace=False)
    records = []
    for anchor_id, eligible_index in enumerate(anchor_indices.tolist()):
        episode, step = eligible[eligible_index]
        candidates = []
        order_pairs = []
        for pair_id in range(args.order_pairs):
            first, second = rng.choice(len(block_refs), size=2, replace=False)
            ref_i, ref_j = block_refs[int(first)], block_refs[int(second)]
            ij_index = len(candidates)
            candidates.append({"kind": "order_ij", "pair_id": pair_id, "blocks": [ref_i, ref_j]})
            ji_index = len(candidates)
            candidates.append({"kind": "order_ji", "pair_id": pair_id, "blocks": [ref_j, ref_i]})
            order_pairs.append({"pair_id": pair_id, "ij": ij_index, "ji": ji_index})
        # The suffix used to construct the feasible goal is included as a
        # labeled witness.  This makes coverage/ranking failures separable;
        # witness panels are reported separately from ordinary candidate sets.
        candidates.append(
            {
                "kind": "witness",
                "pair_id": None,
                "blocks": [(episode, step), (episode, step + args.block_steps)],
            }
        )
        while len(candidates) < args.candidates:
            first, second = rng.choice(len(block_refs), size=2, replace=False)
            candidates.append(
                {
                    "kind": "ordinary",
                    "pair_id": None,
                    "blocks": [block_refs[int(first)], block_refs[int(second)]],
                }
            )
        records.append(
            {
                "anchor_id": anchor_id,
                "episode": episode,
                "step": step,
                "sim_seed": args.seed + anchor_id,
                "order_pairs": order_pairs,
                "candidates": candidates,
            }
        )

    payload = {
        "schema": "order-jepa-stage-a-manifest-v2-original-dino-wm",
        "implementation": "gaoyuezhou/dino_wm",
        "source_commit": ORIGINAL_DINO_WM_COMMIT,
        "checkpoint_family": "authors-official-osf-pusht",
        "dataset": str(args.dataset.resolve()),
        "dataset_files": {
            name: sha256(args.dataset / name)
            for name in ("seq_lengths.pkl", "states.pth", "velocities.pth", "rel_actions.pth")
        },
        "seed": args.seed,
        "block_steps": args.block_steps,
        "num_hist": args.num_hist,
        "anchors": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "anchors": len(records), "sha256": sha256(args.out)}))


if __name__ == "__main__":
    main()
