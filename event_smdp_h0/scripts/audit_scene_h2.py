#!/usr/bin/env python3
"""Exact selected-candidate audit as Event-SMDP search width increases."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.scene_abstract_smdp import AbstractSMDPEvaluator  # noqa: E402
from event_smdp_h0.scene_core import (  # noqa: E402
    SKILLS,
    MilestoneState,
    feedback_reward,
    initial_milestones,
    uct_plan_search,
)
from event_smdp_h0.scripts.run_scene_gate0 import (  # noqa: E402
    SceneSnapshotManager,
    SkillLibrary,
    known_solution,
    make_world,
)


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


def bernoulli_nll(probability: float, target: bool) -> float:
    p = float(np.clip(probability, 1e-7, 1.0 - 1e-7))
    return -math.log(p if target else 1.0 - p)


def exact_candidate(
    *,
    evaluator: AbstractSMDPEvaluator,
    snapshots: SceneSnapshotManager,
    library: SkillLibrary,
    root_snapshot: Any,
    root_state: MilestoneState,
    task_id: int,
    sequence: tuple[int, ...],
) -> dict[str, Any]:
    snapshots.restore(root_snapshot)
    predicted = evaluator.rollout_details(task_id, root_state, sequence)
    state = root_state
    teacher_nll = 0.0
    exact_mode_steps = 0
    actual_steps: list[dict[str, Any]] = []
    for skill_index in sequence:
        before = state
        distribution = evaluator.transition_distribution(task_id, before, skill_index)
        state, record = library.execute(skill_index, state)
        cube_probability = np.asarray(distribution["cube_probability"])
        window_probability = np.asarray(distribution["window_probability"])
        stable_target = bool(record["after"]["native_success"])
        teacher_nll += -math.log(max(float(cube_probability[state.cube_stage]), 1e-7))
        teacher_nll += -math.log(max(float(window_probability[state.window_stage]), 1e-7))
        teacher_nll += bernoulli_nll(
            float(distribution["stable_probability"]), stable_target
        )
        mode_matches = (
            int(cube_probability.argmax()) == state.cube_stage
            and int(window_probability.argmax()) == state.window_stage
            and (float(distribution["stable_probability"]) >= 0.5) == stable_target
        )
        exact_mode_steps += int(mode_matches)
        actual_steps.append(
            {
                "skill": SKILLS[skill_index],
                "state_before": asdict(before),
                "state_after": asdict(state),
                "teacher_forced_nll": teacher_nll,
                "mode_matches": mode_matches,
            }
        )
    true_reward = feedback_reward(state, "event_state")
    success_probability = float(predicted["stable_probability"])
    true_success = bool(state.stable_success)
    return {
        "sequence": list(sequence),
        "skill_names": [SKILLS[index] for index in sequence],
        "predicted_reward": float(predicted["score"]),
        "predicted_success_probability": success_probability,
        "predicted_final_state": asdict(predicted["state"]),
        "true_reward": float(true_reward),
        "true_success": true_success,
        "true_final_state": asdict(state),
        "reward_signed_error": float(predicted["score"]) - float(true_reward),
        "reward_absolute_error": abs(float(predicted["score"]) - float(true_reward)),
        "success_brier": (success_probability - float(true_success)) ** 2,
        "success_log_loss": bernoulli_nll(success_probability, true_success),
        "teacher_forced_transition_nll": teacher_nll / max(len(sequence), 1),
        "mode_step_accuracy": exact_mode_steps / max(len(sequence), 1),
        "actual_steps": actual_steps,
    }


def choose_root_sequence(search: Any) -> tuple[int, ...]:
    eligible = [
        item
        for item in search.evaluations
        if int(item["sequence"][0]) == search.selected_action
    ]
    if not eligible:
        raise RuntimeError("robust root action has no evaluated sequence")
    best = max(
        enumerate(eligible),
        key=lambda pair: (float(pair[1]["predicted_reward"]), -pair[0]),
    )[1]
    return tuple(int(value) for value in best["sequence"])


def run_closed_loop(
    *,
    evaluator: AbstractSMDPEvaluator,
    snapshots: SceneSnapshotManager,
    library: SkillLibrary,
    initial_snapshot: Any,
    task_id: int,
    reset_seed: int,
    budget: int,
    horizon: int,
    exploration: float,
    seed: int,
) -> dict[str, Any]:
    snapshots.restore(initial_snapshot)
    state = initial_milestones(task_id)
    max_decisions = 6 if task_id == 4 else 10
    deployed: list[str] = []
    for decision in range(max_decisions):
        search_seed = seed + 1_000_003 * reset_seed + 10_007 * task_id + decision
        search = uct_plan_search(
            horizon=horizon,
            simulations=budget,
            search_seed=search_seed,
            exploration=exploration,
            evaluate=lambda sequence: evaluator.score_sequence(task_id, state, sequence),
        )
        state, _ = library.execute(search.selected_action, state)
        deployed.append(SKILLS[search.selected_action])
        if state.stable_success:
            break
    state, _ = library.hold(state, 3)
    return {
        "budget": budget,
        "success": bool(state.stable_success),
        "final_reward": feedback_reward(state, "event_state"),
        "final_state": asdict(state),
        "deployed_skills": deployed,
    }


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("H2 audit must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("H2 audit requires a GPU allocation")
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",") if value}))
    if not budgets or min(budgets) < len(SKILLS):
        raise ValueError("budgets must be nonempty and expand every root skill")
    evaluator = AbstractSMDPEvaluator(args.checkpoint, device="cuda")
    world, raw = make_world(args.task_id, args.reset_seed)
    try:
        snapshots = SceneSnapshotManager(raw)
        library = SkillLibrary(raw, stable_dwell=3)
        initial_snapshot = snapshots.capture()
        initial_signature = snapshots.signature()
        state = initial_milestones(args.task_id)
        roots: list[dict[str, Any]] = []
        path = known_solution(args.task_id)
        for root_index in range(len(path) + 1):
            root_snapshot = snapshots.capture()
            root_signature = snapshots.signature()
            searches: dict[int, Any] = {}
            for budget in budgets:
                search_seed = (
                    args.seed
                    + 1_000_003 * args.reset_seed
                    + 10_007 * args.task_id
                    + root_index
                )
                searches[budget] = uct_plan_search(
                    horizon=args.horizon,
                    simulations=budget,
                    search_seed=search_seed,
                    exploration=args.exploration,
                    evaluate=lambda sequence: evaluator.score_sequence(
                        args.task_id, state, sequence
                    ),
                    record_evaluations=True,
                )
            max_trace = searches[max(budgets)].evaluations
            for budget, search in searches.items():
                if search.evaluations != max_trace[:budget]:
                    raise RuntimeError(f"search trace is not nested at budget {budget}")
            sequences = {
                tuple(int(value) for value in item["sequence"])
                for item in max_trace
            }
            candidate_map = {
                sequence: exact_candidate(
                    evaluator=evaluator,
                    snapshots=snapshots,
                    library=library,
                    root_snapshot=root_snapshot,
                    root_state=state,
                    task_id=args.task_id,
                    sequence=sequence,
                )
                for sequence in sorted(sequences)
            }
            search_rows: list[dict[str, Any]] = []
            for budget in budgets:
                search = searches[budget]
                prefix_sequences = [
                    tuple(int(value) for value in item["sequence"])
                    for item in search.evaluations
                ]
                model_best = tuple(search.best_sequence)
                root_selected = choose_root_sequence(search)
                search_rows.append(
                    {
                        "budget": budget,
                        "selected_action": int(search.selected_action),
                        "selected_skill": SKILLS[search.selected_action],
                        "model_best_sequence": list(model_best),
                        "root_selected_sequence": list(root_selected),
                        "max_predicted_reward": float(search.max_reward),
                        "model_best_truth": candidate_map[model_best],
                        "root_selected_truth": candidate_map[root_selected],
                        "candidate_sequences": [list(sequence) for sequence in prefix_sequences],
                    }
                )
            roots.append(
                {
                    "root_index": root_index,
                    "root_signature": root_signature,
                    "true_state": asdict(state),
                    "searches": search_rows,
                    "candidates": list(candidate_map.values()),
                }
            )
            snapshots.restore(root_snapshot)
            if root_index < len(path):
                state, _ = library.execute(path[root_index], state)

        closed_loop = [
            run_closed_loop(
                evaluator=evaluator,
                snapshots=snapshots,
                library=library,
                initial_snapshot=initial_snapshot,
                task_id=args.task_id,
                reset_seed=args.reset_seed,
                budget=budget,
                horizon=args.horizon,
                exploration=args.exploration,
                seed=args.seed,
            )
            for budget in budgets
        ]
        output = {
            "protocol": "scene_h2_search_width_audit_v1",
            "task_id": args.task_id,
            "reset_seed": args.reset_seed,
            "model_seed": args.model_seed,
            "budgets": list(budgets),
            "horizon": args.horizon,
            "exploration": args.exploration,
            "initial_signature": initial_signature,
            "checkpoint": {"path": str(args.checkpoint), "sha256": file_sha256(args.checkpoint)},
            "canonical_path": [SKILLS[index] for index in path],
            "roots": roots,
            "closed_loop": closed_loop,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.out_dir / "result.json"
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "roots": len(roots),
                    "candidates": sum(len(root["candidates"]) for root in roots),
                    "closed_loop_successes": sum(int(row["success"]) for row in closed_loop),
                },
                sort_keys=True,
            )
        )
    finally:
        world.close()


if __name__ == "__main__":
    main()

