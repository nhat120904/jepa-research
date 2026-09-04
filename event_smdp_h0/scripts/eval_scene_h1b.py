#!/usr/bin/env python3
"""Evaluate the abstract-state-closed SMDP in the paired Scene loop."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.scene_abstract_smdp import AbstractSMDPEvaluator  # noqa: E402
from event_smdp_h0.scene_core import (  # noqa: E402
    SKILLS,
    feedback_reward,
    initial_milestones,
    uct_plan_search,
)
from event_smdp_h0.scripts.run_scene_gate0 import (  # noqa: E402
    SceneSnapshotManager,
    SkillLibrary,
    make_world,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=(4, 5), required=True)
    parser.add_argument("--reset-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budgets", default="14,28")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--exploration", type=float, default=0.55)
    parser.add_argument("--duration-cost", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("H1b evaluation must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("H1b evaluation requires a GPU allocation")
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",") if value}))
    if not budgets or min(budgets) < len(SKILLS):
        raise ValueError("each budget must expand all seven root skills")
    max_decisions = args.max_decisions or (6 if args.task_id == 4 else 10)
    evaluator = AbstractSMDPEvaluator(args.checkpoint, device="cuda")

    world, raw = make_world(args.task_id, args.reset_seed)
    try:
        snapshots = SceneSnapshotManager(raw)
        library = SkillLibrary(raw, stable_dwell=3)
        root = snapshots.capture()
        root_signature = snapshots.signature()
        results: list[dict] = []
        for budget in budgets:
            snapshots.restore(root)
            true_state = initial_milestones(args.task_id)
            deployed: list[int] = []
            replans: list[dict] = []
            deploy_env_steps = 0
            model_queries = 0
            for decision in range(max_decisions):
                state_before = true_state

                def evaluate(sequence: tuple[int, ...]) -> float:
                    nonlocal model_queries
                    model_queries += 1
                    return evaluator.score_sequence(
                        args.task_id,
                        true_state,
                        sequence,
                        duration_cost=args.duration_cost,
                    )

                search_seed = (
                    args.seed
                    + 1_000_003 * args.reset_seed
                    + 10_007 * args.task_id
                    + decision
                )
                search = uct_plan_search(
                    horizon=args.horizon,
                    simulations=budget,
                    search_seed=search_seed,
                    exploration=args.exploration,
                    evaluate=evaluate,
                    record_evaluations=True,
                )
                true_state, record = library.execute(search.selected_action, true_state)
                deploy_env_steps += int(record["env_steps"])
                deployed.append(search.selected_action)
                replans.append(
                    {
                        "decision": decision,
                        "true_state_before": asdict(state_before),
                        "selected_skill": SKILLS[search.selected_action],
                        "search": asdict(search),
                        "deployed": record,
                    }
                )
                if true_state.stable_success:
                    break
            true_state, hold_steps = library.hold(true_state, 3)
            deploy_env_steps += hold_steps
            results.append(
                {
                    "task_id": args.task_id,
                    "reset_seed": args.reset_seed,
                    "model_seed": args.model_seed,
                    "feature_view": "abstract_state",
                    "head": "abstract_smdp",
                    "budget": budget,
                    "success": bool(true_state.stable_success),
                    "final_event_reward": feedback_reward(true_state, "event_state"),
                    "final_state": asdict(true_state),
                    "deployed_skills": [SKILLS[index] for index in deployed],
                    "num_replans": len(replans),
                    "model_queries": model_queries,
                    "deploy_env_steps": deploy_env_steps,
                    "replans": replans,
                }
            )
        output = {
            "protocol": "scene_h1b_abstract_closure_v1",
            "interpretation": (
                "learned q_{k+1},duration transition closed recursively in q; "
                "current q is simulator-monitored at each physical replan"
            ),
            "task_id": args.task_id,
            "reset_seed": args.reset_seed,
            "model_seed": args.model_seed,
            "budgets": list(budgets),
            "horizon": args.horizon,
            "max_decisions": max_decisions,
            "exploration": args.exploration,
            "duration_cost": args.duration_cost,
            "root_signature": root_signature,
            "checkpoint": {"path": str(args.checkpoint), "sha256": file_sha256(args.checkpoint)},
            "results": results,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.out_dir / "result.json"
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "successes": sum(int(row["success"]) for row in results),
                    "rows": len(results),
                },
                sort_keys=True,
            )
        )
    finally:
        world.close()


if __name__ == "__main__":
    main()

