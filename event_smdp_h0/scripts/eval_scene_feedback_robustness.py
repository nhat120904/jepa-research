#!/usr/bin/env python3
"""Feedback-robustness sweep over the branch-weight family plus two designs.

See docs/SCENE_FEEDBACK_ROBUSTNESS_PROTOCOL.md.  Arms are
`<state_source>__<feedback>`; `branch_w050` and `branch_w062` reproduce the
task-5 half of the 3x2 grid and act as the determinism check.
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
    MilestoneState,
    initial_milestones,
    uct_plan_search,
)
from event_smdp_h0.scene_event_history import NO_SKILL, HistoryObserverEvaluator  # noqa: E402
from event_smdp_h0.scene_feedback import (  # noqa: E402
    ANTI_LIVELOCK_PENALTY,
    SHAPED_GAMMA,
    SWEEP_FEEDBACKS,
    branch_weighted,
    is_branch_family,
    sweep_weight,
)
from event_smdp_h0.scene_learning import goal_feature  # noqa: E402
from event_smdp_h0.scripts.collect_scene_h1 import encode_images, resize_render  # noqa: E402
from event_smdp_h0.scripts.run_scene_gate0 import (  # noqa: E402
    SceneSnapshotManager,
    SkillLibrary,
    make_world,
)


PROTOCOL = "scene_feedback_robustness_v1"
ORACLE = "oracle"
LEARNED_SOURCES = ("frame_full", "obs_history_full")
STATE_SOURCES = (*LEARNED_SOURCES, ORACLE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=(5,), required=True)
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


def state_key(state: MilestoneState) -> tuple[int, int, bool]:
    return (int(state.cube_stage), int(state.window_stage), bool(state.stable_success))


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("sweep eval must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("sweep eval requires a GPU allocation")
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
            }

    import stable_worldmodel as swm

    visual_model: Any = swm.wm.utils.load_pretrained(args.visual_checkpoint).cuda().eval()
    visual_model.requires_grad_(False)
    if hasattr(visual_model, "interpolate_pos_encoding"):
        visual_model.interpolate_pos_encoding = True

    max_decisions = 10
    world, raw = make_world(args.task_id, args.reset_seed)
    try:
        snapshots = SceneSnapshotManager(raw)
        library = SkillLibrary(raw, stable_dwell=3)
        root = snapshots.capture()
        root_signature = snapshots.signature()
        goal = goal_feature(raw)
        results: list[dict[str, Any]] = []

        arm_specs: list[tuple[str, str, int | None]] = []
        for feedback in SWEEP_FEEDBACKS:
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
            blocked: set[tuple[tuple[int, int, bool], int]] = set()
            pending: tuple[tuple[int, int, bool], int] | None = None
            repeated_blocked = 0
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
                believed = state_key(planning_state)
                if pending is not None and pending[0] == believed:
                    # The believed event state did not move, so that pair is a
                    # known no-progress action from this believed state.
                    blocked.add(pending)
                pending = None

                def evaluate(sequence: Any, state=planning_state, kind=feedback,
                             here=believed) -> float:
                    details = transition.rollout_details(args.task_id, state, sequence)
                    if is_branch_family(kind):
                        return branch_weighted(details["state"], sweep_weight(kind))
                    if kind == "anti_livelock":
                        value = branch_weighted(details["state"], 0.5)
                        if sequence and (here, int(sequence[0])) in blocked:
                            value -= ANTI_LIVELOCK_PENALTY
                        return value
                    if kind == "shaped_gamma09":
                        total = 0.0
                        discount = 1.0
                        previous = branch_weighted(state, 0.5)
                        for step in details["steps"]:
                            current = branch_weighted(step["state_after"], 0.5)
                            total += discount * (current - previous)
                            discount *= SHAPED_GAMMA
                            previous = current
                        return min(max(total, 0.0), 1.0)
                    raise ValueError(f"unknown feedback: {kind}")

                search = uct_plan_search(
                    horizon=args.horizon,
                    simulations=args.budget,
                    search_seed=(
                        args.seed
                        + 1_000_003 * args.reset_seed
                        + 10_007 * args.task_id
                        + decision
                    ),
                    exploration=args.exploration,
                    evaluate=evaluate,
                )
                chosen = int(search.selected_action)
                if (believed, chosen) in blocked:
                    repeated_blocked += 1
                state_before = true_state
                true_state, record = library.execute(chosen, true_state)
                deployed.append(SKILLS[chosen])
                pending = (believed, chosen)
                replans.append(
                    {
                        "decision": decision,
                        "true_state": asdict(state_before),
                        "planning_state": asdict(planning_state),
                        "exact_q_correct": (
                            None
                            if observer_seed is None
                            else state_key(planning_state) == state_key(state_before)
                        ),
                        "selected_skill": SKILLS[chosen],
                        "deployed": record,
                    }
                )
                if true_state.stable_success:
                    break
            true_state, _ = library.hold(true_state, 3)
            unique_skills = len(set(deployed))
            results.append(
                {
                    "task_id": args.task_id,
                    "reset_seed": args.reset_seed,
                    "arm": f"{source}__{feedback}",
                    "state_source": source,
                    "feedback": feedback,
                    "observer_seed": observer_seed,
                    "success": bool(true_state.stable_success),
                    "final_state": asdict(true_state),
                    "deployed_skills": deployed,
                    "num_replans": len(replans),
                    "exhausted_budget": len(replans) >= max_decisions
                    and not true_state.stable_success,
                    "repeated_skill_rate": 1.0 - unique_skills / max(len(deployed), 1),
                    "repeated_blocked_choices": repeated_blocked,
                    "replans": replans,
                }
            )

        output = {
            "protocol": PROTOCOL,
            "task_id": args.task_id,
            "reset_seed": args.reset_seed,
            "observer_seeds": list(observer_seeds),
            "state_sources": list(STATE_SOURCES),
            "feedbacks": list(SWEEP_FEEDBACKS),
            "anti_livelock_penalty": ANTI_LIVELOCK_PENALTY,
            "shaped_gamma": SHAPED_GAMMA,
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
            "scope": "task-5 feedback-robustness sweep on the shared 88500-88563 band",
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "result.json").write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n"
        )
        print(
            json.dumps(
                {
                    "output": str(args.out_dir / "result.json"),
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
