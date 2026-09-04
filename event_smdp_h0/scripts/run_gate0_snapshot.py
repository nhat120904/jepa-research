#!/usr/bin/env python3
"""Oracle finite-budget event-feedback versus terminal-only Skill-UCT.

This is an H0 causal-room experiment.  Both arms receive the exact same fixed
nominal-plus-noise skill lattice and simulator-call budget.  The only changed
variable is the scalar backed up through UCT: stable terminal success, or an
ordinal multi-state event summary.  MuJoCo is the oracle world model here; no
learned hazard/event head is evaluated by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "diagnosis"
sys.path.insert(0, str(REPO))

from event_smdp_h0.core import (  # noqa: E402
    ARM_EVENT,
    ARM_TERMINAL,
    EventSummary,
    make_skill_lattice,
    self_check,
    summarize_events,
    uct_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--branching", type=int, default=5)
    parser.add_argument("--budgets", default="16,32,64")
    parser.add_argument("--exploration", type=float, default=0.65)
    parser.add_argument("--noise-scale", default="0.24,0.24,0.18,0.16,0.10")
    parser.add_argument("--noise-rho", type=float, default=0.60)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--stable-dwell", type=int, default=3)
    parser.add_argument("--success-tolerance-m", type=float, default=0.04)
    parser.add_argument("--near-tolerance-m", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(value.tobytes()).hexdigest()


def build_robot_contact_classifier(raw_env: Any):
    """Resolve cube contacts through MuJoCo body IDs, without name guessing."""

    import mujoco

    model = raw_env._model
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint_0")
    if joint_id < 0:
        raise RuntimeError("object_joint_0 was not found")
    cube_body = int(np.asarray(model.jnt_bodyid).reshape(-1)[joint_id])
    geom_body = np.asarray(model.geom_bodyid).reshape(-1)
    cube_geoms = frozenset(int(x) for x in np.flatnonzero(geom_body == cube_body))
    if not cube_geoms:
        raise RuntimeError("cube body owns no geoms")

    def classify() -> bool:
        for index in range(int(raw_env._data.ncon)):
            contact = raw_env._data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            in1, in2 = g1 in cube_geoms, g2 in cube_geoms
            if in1 == in2:
                continue
            other = g2 if in1 else g1
            # OGBench's table/floor is attached to the world body.  Contacts
            # with every other body are robot--cube contacts in Cube-single.
            if int(geom_body[other]) != 0:
                return True
        return False

    return classify, {
        "joint_id": joint_id,
        "cube_body_id": cube_body,
        "cube_geom_ids": sorted(cube_geoms),
        "n_geoms": int(len(geom_body)),
    }


def event_to_json(summary: EventSummary) -> dict[str, Any]:
    result = asdict(summary)
    result["stage_trace"] = list(summary.stage_trace)
    result["final_cube_position"] = list(summary.final_cube_position)
    result["terminal_reward"] = summary.reward(ARM_TERMINAL)
    result["event_reward"] = summary.reward(ARM_EVENT)
    return result


def main() -> None:
    args = parse_args()
    checks = self_check()
    budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    noise_scale = np.asarray([float(x) for x in args.noise_scale.split(",")])
    if not budgets or min(budgets) < args.branching:
        raise ValueError("every budget must cover every root proposal")
    if args.goal_offset % args.action_block:
        raise ValueError("goal offset must be divisible by action block")
    if args.settle_steps < args.stable_dwell:
        raise ValueError("settle steps must be at least stable dwell")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "event_gate_audit")
    corrected = load_module(
        DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "event_gate_corrected"
    )
    import stable_worldmodel as swm
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest):
        raise ValueError("snapshot index outside manifest")
    manifest_row = dict(manifest[args.snapshot_index])
    task_distance = float(manifest_row.pop("task_distance_m"))
    snapshot = audit.Snapshot(**manifest_row)
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("manifest order/index mismatch")

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]
    action_data = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    nominal = action_data[snapshot.storage_row : snapshot.storage_row + args.goal_offset]
    if nominal.shape != (args.goal_offset, 5) or not np.isfinite(nominal).all():
        raise RuntimeError(f"invalid nominal action window: {nominal.shape}")
    lattice_seed = args.seed + 1_000_003 * snapshot.order
    skills = make_skill_lattice(
        nominal,
        action_block=args.action_block,
        branching=args.branching,
        noise_scale=noise_scale,
        noise_rho=args.noise_rho,
        seed=lattice_seed,
    )
    total_depth = skills.shape[0]

    world, raw_env, visual_signature, visual_shapes = corrected.make_world(swm, snapshot)
    raw_env._terminate_at_goal = False
    classify_contact, contact_info = build_robot_contact_classifier(raw_env)
    target = np.asarray(audit.goal_field(goal_row, "block_0_pos"), dtype=np.float64)
    hold_action = np.zeros((args.settle_steps, nominal.shape[1]), dtype=np.float32)

    def restore_start() -> tuple[np.ndarray, float]:
        corrected.restore_complete(
            raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit
        )
        cube = np.asarray(raw_env._data.joint("object_joint_0").qpos[:3], dtype=np.float64).copy()
        return cube, float(np.linalg.norm(cube - target))

    start_cube, start_distance = restore_start()

    def rollout(actions: np.ndarray) -> tuple[EventSummary, np.ndarray, np.ndarray]:
        restored_cube, restored_distance = restore_start()
        if not np.array_equal(restored_cube, start_cube) or restored_distance != start_distance:
            raise RuntimeError("start-state restoration drifted")
        cubes: list[np.ndarray] = []
        distances: list[float] = []
        contacts: list[bool] = []
        sequence = np.concatenate([np.asarray(actions, dtype=np.float32), hold_action], axis=0)
        for action in sequence:
            _, _, _, truncated, _ = raw_env.step(action)
            if truncated:
                raise RuntimeError("oracle rollout hit an unexpected time limit")
            cube = np.asarray(
                raw_env._data.joint("object_joint_0").qpos[:3], dtype=np.float64
            ).copy()
            cubes.append(cube)
            distances.append(float(np.linalg.norm(cube - target)))
            contacts.append(classify_contact())
        summary = summarize_events(
            np.asarray(distances),
            np.asarray(cubes),
            np.asarray(contacts),
            start_cube_position=start_cube,
            start_distance_m=start_distance,
            success_tolerance_m=args.success_tolerance_m,
            near_tolerance_m=args.near_tolerance_m,
            stable_dwell=args.stable_dwell,
        )
        return summary, raw_env._data.qpos.copy(), raw_env._data.qvel.copy()

    nominal_summary_1, nominal_qpos_1, nominal_qvel_1 = rollout(nominal)
    nominal_summary_2, nominal_qpos_2, nominal_qvel_2 = rollout(nominal)
    repeat_gate = {
        "qpos_max_abs": float(np.max(np.abs(nominal_qpos_1 - nominal_qpos_2))),
        "qvel_max_abs": float(np.max(np.abs(nominal_qvel_1 - nominal_qvel_2))),
        "event_equal": nominal_summary_1 == nominal_summary_2,
    }
    repeat_gate["pass"] = bool(
        repeat_gate["qpos_max_abs"] == 0.0
        and repeat_gate["qvel_max_abs"] == 0.0
        and repeat_gate["event_equal"]
    )
    if not repeat_gate["pass"]:
        raise RuntimeError(f"repeatability gate failed: {repeat_gate}")

    def execute_arm(arm: str, budget: int) -> dict[str, Any]:
        deployed: list[int] = []
        replans: list[dict[str, Any]] = []
        search_rollouts = 0
        for current_depth in range(total_depth):
            def evaluate(suffix: tuple[int, ...]) -> float:
                nonlocal search_rollouts
                indices = tuple(deployed) + suffix
                if len(indices) != total_depth:
                    raise RuntimeError("UCT returned an incomplete skill sequence")
                actions = np.concatenate(
                    [skills[depth, variant] for depth, variant in enumerate(indices)], axis=0
                )
                event, _, _ = rollout(actions)
                search_rollouts += 1
                return event.reward(arm)

            search_seed = args.seed + 10_007 * snapshot.order + 97 * current_depth
            result = uct_search(
                start_depth=current_depth,
                total_depth=total_depth,
                branching=args.branching,
                simulations=budget,
                seed=search_seed,
                exploration=args.exploration,
                evaluate=evaluate,
            )
            deployed.append(result.selected_action)
            replans.append(asdict(result))

        deployed_actions = np.concatenate(
            [skills[depth, variant] for depth, variant in enumerate(deployed)], axis=0
        )
        final_event, final_qpos, final_qvel = rollout(deployed_actions)
        calls_per_rollout = args.goal_offset + args.settle_steps
        expected_rollouts = budget * total_depth
        if search_rollouts != expected_rollouts:
            raise RuntimeError("search rollout accounting mismatch")
        return {
            "arm": arm,
            "budget_per_replan": budget,
            "selected_skill_indices": deployed,
            "selected_actions_sha256": array_hash(deployed_actions),
            "search_rollouts": search_rollouts,
            "oracle_transition_calls": search_rollouts * calls_per_rollout,
            "deployment_transition_calls": calls_per_rollout,
            "total_transition_calls": (search_rollouts + 1) * calls_per_rollout,
            "final": event_to_json(final_event),
            "final_qpos_sha256": array_hash(final_qpos),
            "final_qvel_sha256": array_hash(final_qvel),
            "replans": replans,
        }

    results: dict[str, Any] = {}
    try:
        for budget in budgets:
            # Order is fixed, but every simulator query independently restores
            # the full root state, so the second arm cannot inherit state.
            terminal = execute_arm(ARM_TERMINAL, budget)
            event = execute_arm(ARM_EVENT, budget)
            if terminal["total_transition_calls"] != event["total_transition_calls"]:
                raise RuntimeError("matched-budget invariant failed")
            results[str(budget)] = {ARM_TERMINAL: terminal, ARM_EVENT: event}
    finally:
        world.close()

    output = {
        "scope": (
            "H0 oracle-interface gate only: privileged future demonstration chunks provide "
            "the fixed proposal support to both arms; no learned event model is evaluated."
        ),
        "snapshot": snapshot.order,
        "episode": snapshot.episode,
        "start_step": snapshot.start_step,
        "task_distance_m_from_manifest": task_distance,
        "start_distance_m": start_distance,
        "target_position": target.tolist(),
        "start_cube_position": start_cube.tolist(),
        "config": {
            "dataset": args.dataset,
            "goal_offset": args.goal_offset,
            "action_block": args.action_block,
            "depth": total_depth,
            "branching": args.branching,
            "budgets": budgets,
            "exploration": args.exploration,
            "noise_scale": noise_scale.tolist(),
            "noise_rho": args.noise_rho,
            "settle_steps": args.settle_steps,
            "stable_dwell": args.stable_dwell,
            "success_tolerance_m": args.success_tolerance_m,
            "near_tolerance_m": args.near_tolerance_m,
            "seed": args.seed,
        },
        "self_check": checks,
        "repeat_gate": repeat_gate,
        "nominal_actions_sha256": array_hash(nominal),
        "skill_lattice_sha256": array_hash(skills),
        "nominal_support": event_to_json(nominal_summary_1),
        "contact_resolution": contact_info,
        "visual_signature": visual_signature,
        "visual_signature_shapes": visual_shapes,
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "summary.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
