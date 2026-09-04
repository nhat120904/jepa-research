#!/usr/bin/env python3
"""Evaluate event-progress planning without simulator-provided current q."""

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

from event_smdp_h0.scene_abstract_smdp import AbstractSMDPEvaluator  # noqa: E402
from event_smdp_h0.scene_core import SKILLS, feedback_reward, initial_milestones, uct_plan_search  # noqa: E402
from event_smdp_h0.scene_event_perception import EventObserverEvaluator  # noqa: E402
from event_smdp_h0.scene_learning import goal_feature, raw_state_feature  # noqa: E402
from event_smdp_h0.scripts.collect_scene_h1 import encode_images, resize_render  # noqa: E402
from event_smdp_h0.scripts.run_scene_gate0 import SceneSnapshotManager, SkillLibrary, make_world  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=(4, 5), required=True)
    parser.add_argument("--reset-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--transition-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--views", default="latent,privileged")
    parser.add_argument("--budgets", default="14,28,56,112")
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
        raise RuntimeError("event-perception eval must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("event-perception eval requires a GPU allocation")
    views = tuple(value for value in args.views.split(",") if value)
    if not set(views) <= {"latent", "privileged"}:
        raise ValueError(f"unsupported views: {views}")
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",") if value}))
    transition = AbstractSMDPEvaluator(args.transition_checkpoint, device="cuda")
    observers: dict[str, EventObserverEvaluator] = {}
    observer_meta: dict[str, dict[str, str]] = {}
    for view in views:
        path = args.observer_root / view / f"seed{args.model_seed}" / "observer.pt"
        observers[view] = EventObserverEvaluator(path, device="cuda")
        observer_meta[view] = {"path": str(path), "sha256": file_sha256(path)}

    visual_model: Any | None = None
    if "latent" in views:
        import stable_worldmodel as swm

        visual_model = swm.wm.utils.load_pretrained(args.visual_checkpoint).cuda().eval()
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
        for view in views:
            observer = observers[view]
            for budget in budgets:
                snapshots.restore(root)
                true_state = initial_milestones(args.task_id)
                replans: list[dict[str, Any]] = []
                deployed: list[str] = []
                for decision in range(max_decisions):
                    if view == "latent":
                        feature = encode_images(
                            visual_model, resize_render(raw)[None], batch_size=1
                        )[0]
                    else:
                        feature = raw_state_feature(raw)
                    observed_state, observer_details = observer.predict(
                        feature, goal, args.task_id
                    )
                    cube_correct = observed_state.cube_stage == true_state.cube_stage
                    window_correct = observed_state.window_stage == true_state.window_stage
                    stable_correct = observed_state.stable_success == true_state.stable_success
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
                        evaluate=lambda sequence: transition.score_sequence(
                            args.task_id, observed_state, sequence
                        ),
                    )
                    state_before = true_state
                    true_state, record = library.execute(
                        search.selected_action, true_state
                    )
                    deployed.append(SKILLS[search.selected_action])
                    replans.append(
                        {
                            "decision": decision,
                            "true_state": asdict(state_before),
                            "observed_state": asdict(observed_state),
                            "cube_correct": cube_correct,
                            "window_correct": window_correct,
                            "stable_correct": stable_correct,
                            "exact_q_correct": cube_correct and window_correct and stable_correct,
                            "stable_probability": observer_details["stable_probability"],
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
                        "model_seed": args.model_seed,
                        "feature_view": view,
                        "budget": budget,
                        "success": bool(true_state.stable_success),
                        "final_event_reward": feedback_reward(true_state, "event_state"),
                        "final_state": asdict(true_state),
                        "deployed_skills": deployed,
                        "num_replans": len(replans),
                        "replans": replans,
                    }
                )
        output = {
            "protocol": "scene_event_perception_v1",
            "task_id": args.task_id,
            "reset_seed": args.reset_seed,
            "model_seed": args.model_seed,
            "budgets": list(budgets),
            "horizon": args.horizon,
            "exploration": args.exploration,
            "root_signature": root_signature,
            "observers": observer_meta,
            "transition": {
                "path": str(args.transition_checkpoint),
                "sha256": file_sha256(args.transition_checkpoint),
            },
            "interpretation": (
                "planner current q is inferred from the current observation; true q is used "
                "only for physical bookkeeping and evaluation"
            ),
            "results": results,
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

