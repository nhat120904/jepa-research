#!/usr/bin/env python3
"""A0: verify that the frozen skill reaches held-out short goals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from skill_handoff_wm.data import GoalConditionedDataset  # noqa: E402
from skill_handoff_wm.policy import PolicyRunner  # noqa: E402
from skill_handoff_wm.sim import make_env, set_from_observation  # noqa: E402


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="antmaze-medium-navigate-v0")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=91_001)
    parser.add_argument("--horizons", type=parse_ints, default=parse_ints("10,20,40,80"))
    parser.add_argument("--cases-per-horizon", type=int, default=50)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    dataset = GoalConditionedDataset(args.val_data)
    cases = dataset.short_goal_cases(
        np.random.default_rng(args.seed), args.horizons, args.cases_per_horizon
    )
    env = make_env(args.dataset_name, args.dataset_dir)
    raw = env.unwrapped
    raw.reset(seed=args.seed, options={"task_id": 1})
    policy = PolicyRunner.load(args.checkpoint, args.device)

    records = []
    for case_id, case in enumerate(cases):
        start_state = np.asarray(case.pop("start_state"))
        goal_xy = np.asarray(case["goal_xy"])
        set_from_observation(raw, start_state)
        raw.set_goal(goal_xy=goal_xy)
        initial_distance = float(np.linalg.norm(raw.get_xy() - goal_xy))
        min_distance = initial_distance
        reached_at = None
        for step in range(1, int(case["horizon"]) + 1):
            action = policy.act(raw.get_ob(), goal_xy)
            raw.step(action)
            distance = float(np.linalg.norm(raw.get_xy() - goal_xy))
            min_distance = min(min_distance, distance)
            if reached_at is None and distance <= args.radius:
                reached_at = step
                break
        record = {
            "case_id": case_id,
            **{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in case.items()},
            "initial_distance": initial_distance,
            "final_distance": distance,
            "min_distance": min_distance,
            "success": reached_at is not None,
            "reached_at": reached_at,
            "primitive_steps": step,
        }
        records.append(record)

    with (out_dir / "episodes.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    by_horizon = defaultdict(list)
    for record in records:
        by_horizon[record["horizon"]].append(record)
    summary = {
        "protocol": "direction_a_a0_v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "dataset_name": args.dataset_name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "val_sha256": sha256(args.val_data),
        "radius": args.radius,
        "num_cases": len(records),
        "overall_success": float(np.mean([row["success"] for row in records])),
        "by_horizon": {
            str(horizon): {
                "n": len(rows),
                "success": float(np.mean([row["success"] for row in rows])),
                "mean_min_distance": float(np.mean([row["min_distance"] for row in rows])),
            }
            for horizon, rows in sorted(by_horizon.items())
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
