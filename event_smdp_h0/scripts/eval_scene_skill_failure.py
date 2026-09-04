#!/usr/bin/env python3
"""Skill-failure dose-response over event-state sources.

See docs/SCENE_SKILL_FAILURE_PROTOCOL.md.  A failed skill is executed and then
rolled back to its pre-skill snapshot, so the attempt costs physical time and
changes nothing.  Only arms that infer progress from the attempt rather than
from the scene should be hurt by it.
"""

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

import torch


os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.core import ARM_EVENT  # noqa: E402
from event_smdp_h0.scene_abstract_smdp import AbstractSMDPEvaluator  # noqa: E402
from event_smdp_h0.scene_core import (  # noqa: E402
    SKILLS,
    MilestoneState,
    initial_milestones,
    uct_plan_search,
)
from event_smdp_h0.scene_event_history import NO_SKILL, HistoryObserverEvaluator  # noqa: E402
from event_smdp_h0.scene_feedback import branch_weighted  # noqa: E402
from event_smdp_h0.scene_learning import goal_feature  # noqa: E402
from event_smdp_h0.scripts.collect_scene_h1 import encode_images, resize_render  # noqa: E402
from event_smdp_h0.scripts.run_scene_gate0 import (  # noqa: E402
    SceneSnapshotManager,
    SkillLibrary,
    make_world,
)


PROTOCOL = "scene_skill_failure_v1"
ORACLE = "oracle_event"
OPENLOOP = "openloop_transition"
LEARNED_SOURCES = ("frame_full", "action_only_full", "obs_history_full", "history_full")
FAILURE_RATES = (0.00, 0.10, 0.20, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def attempt_fails(reset_seed: int, decision: int, skill: int, rate: float) -> bool:
    """Deterministic in the attempt, so arms share the same failure draws."""

    if rate <= 0.0:
        return False
    key = f"{reset_seed}:{decision}:{skill}:{rate:.4f}".encode()
    draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(1 << 64)
    return draw < rate


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("skill-failure eval must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("skill-failure eval requires a GPU allocation")
    observer_seeds = tuple(
        sorted({int(value) for value in args.observer_seeds.split(",") if value})
    )
    if observer_seeds != (0, 1, 2):
        raise ValueError(f"locked observer seeds are (0, 1, 2), got {observer_seeds}")
    task_id = 5

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

    # A failed attempt still consumes a decision, so a fixed budget would make
    # every arm, oracle included, simply run out of turns; the smoke check hit
    # exactly that at p >= 0.20.  Scale the budget with the failure rate so the
    # expected number of *effective* decisions is constant, and the sweep
    # measures mis-inference after a failure rather than budget exhaustion.
    def decision_budget(rate: float) -> int:
        return int(math.ceil(10.0 / (1.0 - rate)))

    world, raw = make_world(task_id, args.reset_seed)
    try:
        snapshots = SceneSnapshotManager(raw)
        library = SkillLibrary(raw, stable_dwell=3)
        root = snapshots.capture()
        root_signature = snapshots.signature()
        goal = goal_feature(raw)
        results: list[dict[str, Any]] = []

        arm_specs: list[tuple[str, int | None]] = [(ORACLE, None), (OPENLOOP, None)]
        arm_specs.extend(
            (source, model_seed)
            for source in LEARNED_SOURCES
            for model_seed in observer_seeds
        )

        for rate in FAILURE_RATES:
            max_decisions = decision_budget(rate)
            for source, observer_seed in arm_specs:
                snapshots.restore(root)
                true_state = initial_milestones(task_id)
                belief = initial_milestones(task_id)
                deployed: list[str] = []
                replans: list[dict[str, Any]] = []
                features: list[Any] = []
                prev_skills: list[int] = []
                injected = 0
                for decision in range(max_decisions):
                    if source == OPENLOOP:
                        planning_state: MilestoneState = belief
                    elif observer_seed is None:
                        planning_state = true_state
                    else:
                        features.append(
                            encode_images(
                                visual_model, resize_render(raw)[None], batch_size=1
                            )[0]
                        )
                        prev_skills.append(
                            NO_SKILL if decision == 0 else int(SKILLS.index(deployed[-1]))
                        )
                        planning_state, _ = observers[(source, observer_seed)].predict(
                            features, prev_skills, goal, task_id
                        )
                    inferred = observer_seed is not None or source == OPENLOOP

                    def evaluate(sequence: Any, state=planning_state) -> float:
                        details = transition.rollout_details(task_id, state, sequence)
                        return branch_weighted(details["state"], 0.5)

                    search = uct_plan_search(
                        horizon=args.horizon,
                        simulations=args.budget,
                        search_seed=(
                            args.seed
                            + 1_000_003 * args.reset_seed
                            + 10_007 * task_id
                            + decision
                        ),
                        exploration=args.exploration,
                        evaluate=evaluate,
                    )
                    chosen = int(search.selected_action)
                    fails = attempt_fails(args.reset_seed, decision, chosen, rate)
                    before_snapshot = snapshots.capture()
                    state_before = true_state
                    candidate_state, record = library.execute(chosen, true_state)
                    if fails:
                        # The attempt cost physical time and achieved nothing.
                        snapshots.restore(before_snapshot)
                        injected += 1
                    else:
                        true_state = candidate_state
                    deployed.append(SKILLS[chosen])
                    if source == OPENLOOP:
                        belief = transition.rollout_details(task_id, belief, [chosen])["state"]
                    replans.append(
                        {
                            "decision": decision,
                            "true_state": asdict(state_before),
                            "planning_state": asdict(planning_state),
                            "exact_q_correct": (
                                None
                                if not inferred
                                else (
                                    planning_state.cube_stage == state_before.cube_stage
                                    and planning_state.window_stage == state_before.window_stage
                                    and planning_state.stable_success == state_before.stable_success
                                )
                            ),
                            "selected_skill": SKILLS[chosen],
                            "injected_failure": bool(fails),
                        "beyond_trained_history": (
                            decision >= 10 and observer_seed is not None
                        ),
                            "deployed": record,
                        }
                    )
                    if true_state.stable_success:
                        break
                true_state, _ = library.hold(true_state, 3)
                results.append(
                    {
                        "task_id": task_id,
                        "reset_seed": args.reset_seed,
                        "failure_rate": rate,
                        "arm": source,
                        "state_source": source,
                        "observer_seed": observer_seed,
                        "success": bool(true_state.stable_success),
                        "final_state": asdict(true_state),
                        "deployed_skills": deployed,
                        "num_replans": len(replans),
                        "injected_failures": injected,
                        "decision_budget": max_decisions,
                        "exhausted_budget": len(replans) >= max_decisions
                        and not true_state.stable_success,
                        "replans": replans,
                    }
                )

        output = {
            "protocol": PROTOCOL,
            "task_id": task_id,
            "reset_seed": args.reset_seed,
            "failure_rates": list(FAILURE_RATES),
            "decision_budgets": {str(r): decision_budget(r) for r in FAILURE_RATES},
            "state_sources": [ORACLE, OPENLOOP, *LEARNED_SOURCES],
            "observer_seeds": list(observer_seeds),
            "feedback": "branch_w050",
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
            "scope": "task-5 skill-failure dose response on the shared 88500-88563 band",
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
