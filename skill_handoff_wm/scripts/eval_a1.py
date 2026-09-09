#!/usr/bin/env python3
"""A1/A2 screen: fixed, geometric, heuristic, and physics-oracle handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from skill_handoff_wm.maze import path_waypoints  # noqa: E402
from skill_handoff_wm.policy import PolicyRunner  # noqa: E402
from skill_handoff_wm.sim import make_env, paired_reset  # noqa: E402
from skill_handoff_wm.switching import oracle_duration, rollout_skill  # noqa: E402
from skill_handoff_wm.termination import TerminationRunner, rollout_learned_termination  # noqa: E402


def ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_arm(
    env, policy, termination, task_id, seed, arm, durations, probe_steps, radius, stride, budget
):
    _, info = paired_reset(env, task_id, seed)
    raw = env.unwrapped
    final_goal = np.asarray(info["goal"][:2], dtype=np.float32)
    deployed_steps = 0
    simulator_query_steps = 0
    proposed_waypoints = 0
    replans = 0
    handoffs = []
    while deployed_steps < budget and not raw.compute_success():
        waypoints = path_waypoints(raw, final_goal, stride=stride)
        proposed_waypoints += len(waypoints)
        cycle_start_steps = deployed_steps
        for index, target in enumerate(waypoints):
            if deployed_steps >= budget or raw.compute_success():
                break
            next_target = waypoints[min(index + 1, len(waypoints) - 1)]
            is_final = index == len(waypoints) - 1
            start_observation = raw.get_ob().copy()
            branches = None
            termination_decisions = None
            result = None
            if arm.startswith("fixed_"):
                duration = int(arm.split("_", 1)[1])
                stop_on_target = False
            elif arm == "heuristic":
                duration = max(durations)
                stop_on_target = True
            elif arm == "learned":
                if termination is None:
                    raise ValueError("learned arm requires --termination-checkpoint")
                result, duration, termination_decisions = rollout_learned_termination(
                    raw,
                    policy,
                    termination,
                    target,
                    next_target,
                    durations,
                    budget - deployed_steps,
                    radius,
                    is_final,
                )
                stop_on_target = False
            elif arm == "oracle":
                remaining = budget - deployed_steps
                candidates = sorted(set(min(duration, remaining) for duration in durations if remaining > 0))
                duration, branches = oracle_duration(
                    raw,
                    policy,
                    target,
                    next_target,
                    candidates,
                    0 if is_final else probe_steps,
                    radius,
                )
                simulator_query_steps += sum(int(row["simulator_steps"]) for row in branches)
                stop_on_target = False
            else:
                raise ValueError(f"unknown arm {arm}")
            if result is None:
                allowed = min(duration, budget - deployed_steps)
                result = rollout_skill(raw, policy, target, allowed, radius, stop_on_target)
            deployed_steps += result.primitive_steps
            handoffs.append(
                {
                    "replan": replans,
                    "skill_index": index,
                    "start_observation": start_observation.tolist(),
                    "target_xy": target.tolist(),
                    "next_target_xy": next_target.tolist(),
                    "is_final_waypoint": is_final,
                    "chosen_duration": duration,
                    "deployed_steps": result.primitive_steps,
                    "reached_target": result.reached_target,
                    "min_target_distance": result.min_target_distance,
                    "exit_xy": raw.get_xy().tolist(),
                    "exit_velocity_xy": raw.data.qvel[:2].tolist(),
                    "oracle_branches": branches,
                    "termination_decisions": termination_decisions,
                }
            )
        replans += 1
        if deployed_steps == cycle_start_steps:
            break
    return {
        "task_id": task_id,
        "seed": seed,
        "arm": arm,
        "success": bool(raw.compute_success()),
        "final_distance": float(np.linalg.norm(raw.get_xy() - final_goal)),
        "primitive_steps": deployed_steps,
        "simulator_query_steps": simulator_query_steps,
        "num_waypoints": proposed_waypoints,
        "num_replans": replans,
        "num_handoffs": len(handoffs),
        "handoffs": handoffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="antmaze-medium-navigate-v0")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-ids", type=ints, default=ints("1,2,3,4,5"))
    parser.add_argument("--seeds", type=ints, default=ints("92001,92002,92003,92004,92005"))
    parser.add_argument("--durations", type=ints, default=ints("10,20,40,80"))
    parser.add_argument("--arms", type=strings, default=None)
    parser.add_argument("--termination-checkpoint")
    parser.add_argument("--probe-steps", type=int, default=40)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--waypoint-stride", type=int, default=1)
    parser.add_argument("--primitive-budget", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    policy = PolicyRunner.load(args.checkpoint, args.device)
    termination = (
        TerminationRunner.load(args.termination_checkpoint, args.device)
        if args.termination_checkpoint
        else None
    )
    env = make_env(args.dataset_name, args.dataset_dir)
    arms = args.arms or [f"fixed_{duration}" for duration in args.durations] + ["heuristic", "oracle"]
    records = []
    for task_id in args.task_ids:
        for seed in args.seeds:
            for arm in arms:
                record = evaluate_arm(
                    env,
                    policy,
                    termination,
                    task_id,
                    seed,
                    arm,
                    args.durations,
                    args.probe_steps,
                    args.radius,
                    args.waypoint_stride,
                    args.primitive_budget,
                )
                records.append(record)
                print(json.dumps({key: value for key, value in record.items() if key != "handoffs"}), flush=True)
    with (out_dir / "episodes.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = {}
    for arm in arms:
        rows = [row for row in records if row["arm"] == arm]
        summary[arm] = {
            "n": len(rows),
            "success": float(np.mean([row["success"] for row in rows])),
            "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
            "deployed_steps": int(sum(row["primitive_steps"] for row in rows)),
            "simulator_query_steps": int(sum(row["simulator_query_steps"] for row in rows)),
        }
    payload = {
        "protocol": "direction_a_a1_v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "dataset_name": args.dataset_name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "termination_checkpoint_sha256": (
            sha256(args.termination_checkpoint) if args.termination_checkpoint else None
        ),
        "config": vars(args),
        "arms": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
