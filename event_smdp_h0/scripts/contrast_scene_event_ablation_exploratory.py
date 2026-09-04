#!/usr/bin/env python3
"""Exploratory, not preregistered: contrasts the locked ablation gate omitted.

These are reported as exploratory throughout.  The locked verdict in
`analyze_scene_event_ablation.py` does not depend on them.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np


BOOTSTRAP = 10000
SEEDS = (0, 1, 2)
LEARNED = ("frame_full", "obs_history_full", "action_only_full", "history_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("must run inside a Slurm compute job")

    table: dict[tuple[int, int], dict[str, dict[object, bool]]] = {}
    for shard in sorted(args.eval_root.glob("*/result.json"), key=lambda p: int(p.parent.name)):
        payload = json.loads(shard.read_text())
        entry: dict[str, dict[object, bool]] = defaultdict(dict)
        for row in payload["results"]:
            seed = row["observer_seed"]
            entry[row["arm"]]["single" if seed is None else int(seed)] = bool(row["success"])
        table[(int(payload["task_id"]), int(payload["reset_seed"]))] = entry

    keys = sorted(table)

    def vector(arm: str, task: int | None = None) -> np.ndarray:
        rows = []
        for key in keys:
            if task is not None and key[0] != task:
                continue
            cell = table[key][arm]
            rows.append(
                float(np.mean([float(cell[s]) for s in SEEDS]))
                if arm in LEARNED
                else float(cell["single"])
            )
        return np.asarray(rows, dtype=np.float64)

    def ci(values: np.ndarray) -> dict[str, float]:
        rng = np.random.default_rng(args.bootstrap_seed)
        draws = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
        means = values[draws].mean(axis=1)
        return {
            "mean_points": 100.0 * float(values.mean()),
            "ci_low_points": 100.0 * float(np.percentile(means, 2.5)),
            "ci_high_points": 100.0 * float(np.percentile(means, 97.5)),
            "n": int(len(values)),
        }

    pairs = [
        ("obs_history_full", "oracle_event"),
        ("history_full", "oracle_event"),
        ("obs_history_full", "action_only_full"),
        ("obs_history_full", "openloop_transition"),
    ]
    out: dict[str, object] = {
        "protocol": "scene_event_ablation_exploratory_v1",
        "scope": (
            "exploratory contrasts, not part of the locked ablation gate; reported "
            "as exploratory and not used to license any decision"
        ),
        "pooled": {
            f"{left}-{right}": ci(vector(left) - vector(right)) for left, right in pairs
        },
        "per_task": {
            str(task): {
                f"{left}-{right}": ci(vector(left, task) - vector(right, task))
                for left, right in pairs
            }
            for task in sorted({key[0] for key in keys})
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "exploratory_contrasts.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
