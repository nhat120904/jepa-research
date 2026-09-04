#!/usr/bin/env python3
"""3x2 grid over event-state source and planner-facing scalar feedback.

See docs/SCENE_STATE_VS_FEEDBACK_PROTOCOL.md.  Arms are named
`<state_source>__<feedback>`; the three `event_progress` cells reproduce
configurations already measured, which doubles as a determinism check.
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

from event_smdp_h0.scene_abstract_smdp import AbstractSMDPEvaluator  # noqa: E402
from event_smdp_h0.scene_core import (  # noqa: E402
    SKILLS,
    initial_milestones,
    uct_plan_search,
)
from event_smdp_h0.scene_event_history import NO_SKILL, HistoryObserverEvaluator  # noqa: E402
from event_smdp_h0.scene_feedback import FEEDBACKS, scalar  # noqa: E402
from event_smdp_h0.scene_learning import goal_feature  # noqa: E402
from event_smdp_h0.scripts.collect_scene_h1 import encode_images, resize_render  # noqa: E402
from event_smdp_h0.scripts.run_scene_gate0 import (  # noqa: E402
    SceneSnapshotManager,
    SkillLibrary,
    make_world,
)


PROTOCOL = "scene_state_vs_feedback_v1"
ORACLE = "oracle"
LEARNED_SOURCES = ("frame_full", "obs_history_full")
STATE_SOURCES = (*LEARNED_SOURCES, ORACLE)


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
        raise RuntimeError("grid eval must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("grid eval requires a GPU allocation")
    observer_seeds = tuple(
        sorted({int(value) for value in args.observer_seeds.split(",") if value})
    )
    if observer_seeds != (0, 1, 2):
        raise ValueError(f"locked observer seeds are (0, 1, 2), got {observer_seeds}")

    transition = AbstractSMDPEvaluator(args.transition_checkpoint, device="cuda")
    observers: dict[tuple[str, int], HistoryObserverEvaluator] = {}
    observer_meta: dict[str, dict[str, Any]] = {}
    for source in LEARNED_SOURCES:
        for model_seed in observer_seeds:
            path = args.observer_root / source / f"seed{model_seed}" / "observer.pt"
            observer = HistoryObserverEvaluator(path, device="cuda")
            if observer.arm != source:
                raise ValueError(f"checkpoint {path} declares arm {observer.arm}")
            observers[(source, model_seed)] = observer
            observer_meta[f"{source}/seed{model_seed}"] = {
                "path": str(path),
                "sha256": file_sha256(path),
                "ablation": observer.ablation,
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

        arm_specs: list[tuple[str, str, int | None]] = []
        for feedback in FEEDBACKS:
            arm_specs.append((ORACLE, feedback, None))
            arm_specs.extend(
                (source, feedback, model_seed)
                for source in LEARNED_SOURCES
                for model_seed in observer_seeds
            )

        for source, feedback, observer_seed in arm_specs:
            snapshots.restore(root)
            true_state = initial_milestones(args.task_id)
            deployed: list[str] = []
            replans: list[dict[str, Any]] = []
            features: list[Any] = []
            prev_skills: list[int] = []
            for decision in range(max_decisions):
                if observer_seed is None:
                    planning_state = true_state
                else:
                    features.append(
                        encode_images(visual_model, resize_render(raw)[None], batch_size=1)[0]
                    )
                    prev_skills.append(
                        NO_SKILL if decision == 0 else int(SKILLS.index(deployed[-1]))
                    )
                    planning_state, _ = observers[(source, observer_seed)].predict(
                        features, prev_skills, goal, args.task_id
                    )
                search_seed = (
                    args.seed
                    + 1_000_003 * args.reset_seed
                    + 10_007 * args.task_id
                    + decision
                )

                def evaluate(sequence: Any, state=planning_state, kind=feedback) -> float:
                    details = transition.rollout_details(args.task_id, state, sequence)
                    return scalar(details["state"], kind)

                search = uct_plan_search(
                    horizon=args.horizon,
                    simulations=args.budget,
                    search_seed=search_seed,
                    exploration=args.exploration,
                    evaluate=evaluate,
                )
                state_before = true_state
                true_state, record = library.execute(search.selected_action, true_state)
                deployed.append(SKILLS[search.selected_action])
                replans.append(
                    {
                        "decision": decision,
                        "true_state": asdict(state_before),
                        "planning_state": asdict(planning_state),
                        "exact_q_correct": (
                            None
                            if observer_seed is None
                            else (
                                planning_state.cube_stage == state_before.cube_stage
                                and planning_state.window_stage == state_before.window_stage
                                and planning_state.stable_success == state_before.stable_success
                            )
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
                    "arm": f"{source}__{feedback}",
                    "state_source": source,
                    "feedback": feedback,
                    "observer_seed": observer_seed,
                    "budget": args.budget,
                    "success": bool(true_state.stable_success),
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
            "state_sources": list(STATE_SOURCES),
            "feedbacks": list(FEEDBACKS),
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
                "3x2 grid over event-state source and planner scalar feedback on the "
                "ablation reset band; event_progress cells are reproductions"
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
