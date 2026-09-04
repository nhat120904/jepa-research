#!/usr/bin/env python3
"""Is the deployed observer failing on states its training set never contained?

Compares the event states reachable as canonical milestone roots (the original
observer's entire training support) against the states the deployed planner
actually visits, taken from the completed 128-reset replication artifacts.
Reports observer exact-q accuracy conditioned on whether the visited state is
inside the canonical support, which separates a coverage failure from genuine
single-frame aliasing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.scene_history_dataset import build_sequences  # noqa: E402
from event_smdp_h0.scripts.train_scene_h1 import load_split  # noqa: E402


PROTOCOL = "scene_state_coverage_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def support(dataset: dict[str, np.ndarray], endpoints: bool) -> dict[int, set[tuple]]:
    keep = np.ones(len(dataset["cube"]), dtype=bool)
    if not endpoints:
        keep &= dataset["is_endpoint"] == 0
    out: dict[int, set[tuple]] = defaultdict(set)
    for task, cube, window, stable in zip(
        dataset["task_id"][keep],
        dataset["cube"][keep],
        dataset["window"][keep],
        dataset["stable"][keep],
    ):
        out[int(task)].add((int(cube), int(window), int(stable >= 0.5)))
    return out


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("coverage diagnosis must run inside a Slurm compute job")

    train_np, train_paths = load_split(args.data_root, "train")
    dataset = build_sequences(train_np, "latent")
    canonical = support(dataset, endpoints=False)
    full = support(dataset, endpoints=True)

    shards = sorted(args.replication_root.glob("*/result.json"))
    if not shards:
        raise FileNotFoundError(f"no replication shards under {args.replication_root}")

    visits: list[dict[str, object]] = []
    for shard in shards:
        payload = json.loads(shard.read_text())
        task_id = int(payload["task_id"])
        for row in payload["results"]:
            if row.get("observer_seed") is None:
                continue
            for replan in row["replans"]:
                true_state = replan["true_state"]
                key = (
                    int(true_state["cube_stage"]),
                    int(true_state["window_stage"]),
                    int(int(true_state["stable_count"]) >= 3),
                )
                visits.append(
                    {
                        "task_id": task_id,
                        "observer_seed": int(row["observer_seed"]),
                        "state": key,
                        "in_canonical": key in canonical.get(task_id, set()),
                        "in_full": key in full.get(task_id, set()),
                        "correct": bool(replan["exact_q_correct"]),
                    }
                )

    def rate(rows: list[dict[str, object]]) -> dict[str, float]:
        if not rows:
            return {"n": 0, "exact_q_accuracy": float("nan")}
        return {
            "n": len(rows),
            "exact_q_accuracy": sum(int(bool(r["correct"])) for r in rows) / len(rows),
        }

    inside = [row for row in visits if row["in_canonical"]]
    outside = [row for row in visits if not row["in_canonical"]]
    outside_but_full = [row for row in outside if row["in_full"]]
    outside_of_full = [row for row in visits if not row["in_full"]]

    missed = Counter(
        (row["task_id"], row["state"]) for row in outside
    )
    top_missing = [
        {
            "task_id": task_id,
            "state": list(state),
            "visits": count,
            "in_full_support": state in full.get(task_id, set()),
            "exact_q_accuracy": rate(
                [
                    row
                    for row in visits
                    if row["task_id"] == task_id and row["state"] == state
                ]
            )["exact_q_accuracy"],
        }
        for (task_id, state), count in missed.most_common(10)
    ]

    summary = {
        "protocol": PROTOCOL,
        "num_replication_shards": len(shards),
        "num_visits": len(visits),
        "canonical_support": {
            str(task): sorted(list(state) for state in states)
            for task, states in canonical.items()
        },
        "support_sizes": {
            str(task): {
                "canonical_states": len(canonical.get(task, set())),
                "full_states": len(full.get(task, set())),
            }
            for task in sorted(full)
        },
        "training_samples": {
            "canonical_roots": int((dataset["is_endpoint"] == 0).sum()),
            "with_endpoints": int(len(dataset["is_endpoint"])),
        },
        "conditioned_accuracy": {
            "visited_state_in_canonical_support": rate(inside),
            "visited_state_outside_canonical_support": rate(outside),
            "outside_canonical_but_inside_full_support": rate(outside_but_full),
            "outside_full_support": rate(outside_of_full),
        },
        "top_uncovered_visited_states": top_missing,
        "source_train_shards": train_paths,
        "scope": (
            "descriptive audit of already completed artifacts; no new success claim"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "canonical_support"}, sort_keys=True))


if __name__ == "__main__":
    main()
