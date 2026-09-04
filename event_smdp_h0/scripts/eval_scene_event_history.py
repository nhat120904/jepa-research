#!/usr/bin/env python3
"""Paired Skill-UCT evaluation of the coverage/history observer factorial.

One fresh reset per shard.  Every arm replans from the same restored snapshot
with the same frozen abstract transition model, the same tree budget and the
same search seeds; arms differ only in where the planning event state comes
from.  Learned arms never read simulator q.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.core import ARM_EVENT, ARM_TERMINAL  # noqa: E402
from event_smdp_h0.scene_abstract_smdp import AbstractSMDPEvaluator  # noqa: E402
from event_smdp_h0.scene_core import (  # noqa: E402
    SKILLS,
    feedback_reward,
    initial_milestones,
    uct_plan_search,
)
from event_smdp_h0.scene_event_history import (  # noqa: E402
    ARMS,
    NO_SKILL,
    HistoryObserverEvaluator,
)
from event_smdp_h0.scene_learning import goal_feature  # noqa: E402
from event_smdp_h0.scripts.collect_scene_h1 import encode_images, resize_render  # noqa: E402
from event_smdp_h0.scripts.run_scene_gate0 import (  # noqa: E402
    SceneSnapshotManager,
    SkillLibrary,
    make_world,
)


PROTOCOL = "scene_event_history_eval_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=(4, 5), required=True)
    parser.add_argument("--reset-seed", type=int, required=True)
    parser.add_argument("--observer-seeds", default="0,1,2")
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--transition-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--budget", type=int, default=112)
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
        raise RuntimeError("history eval must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("history eval requires a GPU allocation")
    observer_seeds = tuple(
        sorted({int(value) for value in args.observer_seeds.split(",") if value})
    )
    if observer_seeds != (0, 1, 2):
        raise ValueError(f"locked observer seeds are (0, 1, 2), got {observer_seeds}")

    transition = AbstractSMDPEvaluator(args.transition_checkpoint, device="cuda")
    observers: dict[tuple[str, int], HistoryObserverEvaluator] = {}
    observer_meta: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        for model_seed in observer_seeds:
            path = args.observer_root / arm / f"seed{model_seed}" / "observer.pt"
            observer = HistoryObserverEvaluator(path, device="cuda")
            if observer.arm != arm:
                raise ValueError(f"checkpoint {path} declares arm {observer.arm}")
            observers[(arm, model_seed)] = observer
            observer_meta[f"{arm}/seed{model_seed}"] = {
                "path": str(path),
                "sha256": file_sha256(path),
                "history_length": observer.history_length,
            }

    import stable_worldmodel as swm

    visual_model: Any = swm.wm.utils.load_pretrained(args.visual_checkpoint).cuda().eval()
    visual_model.requires_grad_(False)
    if hasattr(visual_model, "interpolate_pos_encoding"):
        visual_model.interpolate_pos_encoding = True

    max_decisions = 6 if args.task_id == 4 else 10
    world, raw = make_world(args.task_id, args.reset_seed)
    try:
        snapshots = SceneSnapshotManager(raw)
        library = SkillLibrary(raw, stable_dwell=3)
        root = snapshots.capture()
        root_signature = snapshots.signature()
        goal = goal_feature(raw)
        results: list[dict[str, Any]] = []

        arm_specs: list[tuple[str, int | None, str]] = [
            ("oracle_event", None, ARM_EVENT),
            ("abstract_terminal", None, ARM_TERMINAL),
        ]
        arm_specs.extend(
            (arm, model_seed, ARM_EVENT)
            for arm in ARMS
            for model_seed in observer_seeds
        )

        for arm_name, observer_seed, feedback_arm in arm_specs:
            snapshots.restore(root)
            true_state = initial_milestones(args.task_id)
            deployed: list[str] = []
            replans: list[dict[str, Any]] = []
            features: list[Any] = []
            prev_skills: list[int] = []
            for decision in range(max_decisions):
                if observer_seed is None:
                    planning_state = true_state
                    observer_details: dict[str, object] | None = None
                else:
                    features.append(
                        encode_images(visual_model, resize_render(raw)[None], batch_size=1)[0]
                    )
                    prev_skills.append(
                        NO_SKILL if decision == 0 else int(SKILLS.index(deployed[-1]))
                    )
                    planning_state, observer_details = observers[
                        (arm_name, observer_seed)
                    ].predict(features, prev_skills, goal, args.task_id)
                cube_correct = planning_state.cube_stage == true_state.cube_stage
                window_correct = planning_state.window_stage == true_state.window_stage
                stable_correct = planning_state.stable_success == true_state.stable_success
                search_seed = (
                    args.seed
                    + 1_000_003 * args.reset_seed
                    + 10_007 * args.task_id
                    + decision
                )
                search = uct_plan_search(
                    horizon=args.horizon,
                    simulations=args.budget,
                    search_seed=search_seed,
                    exploration=args.exploration,
                    evaluate=lambda sequence: transition.score_sequence(
                        args.task_id, planning_state, sequence, arm=feedback_arm
                    ),
                )
                state_before = true_state
                true_state, record = library.execute(search.selected_action, true_state)
                deployed.append(SKILLS[search.selected_action])
                replans.append(
                    {
                        "decision": decision,
                        "true_state": asdict(state_before),
                        "planning_state": asdict(planning_state),
                        "cube_correct": cube_correct if observer_seed is not None else None,
                        "window_correct": window_correct if observer_seed is not None else None,
                        "stable_correct": stable_correct if observer_seed is not None else None,
                        "exact_q_correct": (
                            cube_correct and window_correct and stable_correct
                            if observer_seed is not None
                            else None
                        ),
                        "history_steps": (
                            observer_details["history_steps"]
                            if observer_details is not None
                            else None
                        ),
                        "beyond_trained_history": (
                            observer_details["beyond_trained_history"]
                            if observer_details is not None
                            else None
                        ),
                        "selected_skill": SKILLS[search.selected_action],
                        "deployed": record,
                    }
                )
                if true_state.stable_success:
                    break
            true_state, _ = library.hold(true_state, 3)
            results.append(
                {
                    "task_id": args.task_id,
                    "reset_seed": args.reset_seed,
                    "arm": arm_name,
                    "observer_seed": observer_seed,
                    "budget": args.budget,
                    "success": bool(true_state.stable_success),
                    "final_event_reward": feedback_reward(true_state, ARM_EVENT),
                    "final_state": asdict(true_state),
                    "deployed_skills": deployed,
                    "num_replans": len(replans),
                    "replans": replans,
                }
            )

        output = {
            "protocol": PROTOCOL,
            "task_id": args.task_id,
            "reset_seed": args.reset_seed,
            "observer_seeds": list(observer_seeds),
            "arms": list(ARMS),
            "budget": args.budget,
            "horizon": args.horizon,
            "exploration": args.exploration,
            "root_signature": root_signature,
            "observers": observer_meta,
            "transition": {
                "path": str(args.transition_checkpoint),
                "sha256": file_sha256(args.transition_checkpoint),
            },
            "results": results,
            "scope": (
                "fresh held-out reset; learned arms infer current q from rendered "
                "observations and their own deployed skill history"
            ),
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.out_dir / "result.json"
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "rows": len(results),
                    "successes": sum(int(row["success"]) for row in results),
                },
                sort_keys=True,
            )
        )
    finally:
        world.close()


if __name__ == "__main__":
    main()
