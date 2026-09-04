#!/usr/bin/env python3
"""Run terminal-only planning through the same H1b abstract transition."""

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

from event_smdp_h0.core import ARM_TERMINAL  # noqa: E402
from event_smdp_h0.scene_abstract_smdp import AbstractSMDPEvaluator  # noqa: E402
from event_smdp_h0.scene_core import SKILLS, feedback_reward, initial_milestones, uct_plan_search  # noqa: E402
from event_smdp_h0.scripts.run_scene_gate0 import SceneSnapshotManager, SkillLibrary, make_world  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=(4, 5), required=True)
    parser.add_argument("--reset-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budgets", default="7,14,28,56,112")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--exploration", type=float, default=0.55)
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
        raise RuntimeError("factorial evaluation must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("factorial evaluation requires a GPU allocation")
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",") if value}))
    evaluator = AbstractSMDPEvaluator(args.checkpoint, device="cuda")
    max_decisions = 6 if args.task_id == 4 else 10
    world, raw = make_world(args.task_id, args.reset_seed)
    try:
        snapshots = SceneSnapshotManager(raw)
        library = SkillLibrary(raw, stable_dwell=3)
        root = snapshots.capture()
        root_signature = snapshots.signature()
        results: list[dict] = []
        for budget in budgets:
            snapshots.restore(root)
            state = initial_milestones(args.task_id)
            deployed: list[str] = []
            for decision in range(max_decisions):
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
                    evaluate=lambda sequence: evaluator.score_sequence(
                        args.task_id, state, sequence, arm=ARM_TERMINAL
                    ),
                )
                state, _ = library.execute(search.selected_action, state)
                deployed.append(SKILLS[search.selected_action])
                if state.stable_success:
                    break
            state, _ = library.hold(state, 3)
            results.append(
                {
                    "task_id": args.task_id,
                    "reset_seed": args.reset_seed,
                    "model_seed": args.model_seed,
                    "budget": budget,
                    "arm": "abstract_terminal_probability",
                    "success": bool(state.stable_success),
                    "final_event_reward": feedback_reward(state, "event_state"),
                    "final_state": asdict(state),
                    "deployed_skills": deployed,
                }
            )
        output = {
            "protocol": "scene_abstract_factorial_v1",
            "task_id": args.task_id,
            "reset_seed": args.reset_seed,
            "model_seed": args.model_seed,
            "budgets": list(budgets),
            "horizon": args.horizon,
            "exploration": args.exploration,
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

