#!/usr/bin/env python3
"""Oracle Skill-UCT causal-room gate on OGBench-Scene tasks 4 and 5.

This is deliberately not a learned-world-model result.  MuJoCo supplies exact
skill transitions and the official OGBench Markov controllers supply a fixed,
task-agnostic skill support.  The paired arms share every proposal, search
hyperparameter, and skill-query budget; only the feedback backed up through
UCT changes from stable terminal success to a task-conditioned event state.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.core import ARM_EVENT, ARM_TERMINAL  # noqa: E402
from event_smdp_h0.scene_core import (  # noqa: E402
    SKILLS,
    MilestoneState,
    ScenePredicates,
    advance_milestones,
    feedback_reward,
    initial_milestones,
    uct_plan_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=(4, 5), required=True)
    parser.add_argument("--reset-seed", type=int, required=True)
    parser.add_argument("--budgets", default="14,28")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--exploration", type=float, default=0.55)
    parser.add_argument("--stable-dwell", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--skip-support-check", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


@dataclass
class EnvSnapshot:
    integration_state: np.ndarray
    button_states: np.ndarray
    prev_button_states: np.ndarray
    prev_qpos: np.ndarray
    prev_qvel: np.ndarray
    prev_ob_info: dict[str, Any]
    drawer_target_site_pos: np.ndarray
    window_target_site_pos: np.ndarray
    success: bool


class SceneSnapshotManager:
    def __init__(self, raw_env: Any):
        import mujoco

        self.raw = raw_env
        self.mujoco = mujoco
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_size = mujoco.mj_stateSize(raw_env._model, self.spec)

    def capture(self) -> EnvSnapshot:
        state = np.empty(self.state_size, dtype=np.float64)
        self.mujoco.mj_getState(
            self.raw._model, self.raw._data, state, self.spec
        )
        return EnvSnapshot(
            integration_state=state,
            button_states=np.asarray(self.raw._cur_button_states).copy(),
            prev_button_states=np.asarray(self.raw._prev_button_states).copy(),
            prev_qpos=np.asarray(self.raw._prev_qpos).copy(),
            prev_qvel=np.asarray(self.raw._prev_qvel).copy(),
            prev_ob_info=copy.deepcopy(self.raw._prev_ob_info),
            drawer_target_site_pos=self.raw._model.site(
                "drawer_handle_center_target"
            ).pos.copy(),
            window_target_site_pos=self.raw._model.site(
                "window_handle_center_target"
            ).pos.copy(),
            success=bool(self.raw._success),
        )

    def restore(self, snapshot: EnvSnapshot) -> None:
        self.mujoco.mj_setState(
            self.raw._model,
            self.raw._data,
            snapshot.integration_state,
            self.spec,
        )
        self.raw._cur_button_states = snapshot.button_states.copy()
        self.raw._prev_button_states = snapshot.prev_button_states.copy()
        self.raw._apply_button_states()
        self.raw._model.site("drawer_handle_center_target").pos[:] = (
            snapshot.drawer_target_site_pos
        )
        self.raw._model.site("window_handle_center_target").pos[:] = (
            snapshot.window_target_site_pos
        )
        self.mujoco.mj_forward(self.raw._model, self.raw._data)
        self.raw._prev_qpos = snapshot.prev_qpos.copy()
        self.raw._prev_qvel = snapshot.prev_qvel.copy()
        self.raw._prev_ob_info = copy.deepcopy(snapshot.prev_ob_info)
        self.raw._success = bool(snapshot.success)
        self.raw._reset_next_step = False

    def signature(self) -> str:
        snapshot = self.capture()
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(snapshot.integration_state).tobytes())
        digest.update(np.ascontiguousarray(snapshot.button_states).tobytes())
        digest.update(snapshot.drawer_target_site_pos.tobytes())
        digest.update(snapshot.window_target_site_pos.tobytes())
        digest.update(bytes([snapshot.success]))
        return digest.hexdigest()


class SkillLibrary:
    def __init__(self, raw_env: Any, stable_dwell: int):
        from ogbench.manipspace.oracles.markov.button_markov import (
            ButtonMarkovOracle,
        )
        from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
        from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
        from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle

        self.raw = raw_env
        self.stable_dwell = stable_dwell
        self.oracle_classes = {
            "button": ButtonMarkovOracle,
            "cube": CubeMarkovOracle,
            "drawer": DrawerMarkovOracle,
            "window": WindowMarkovOracle,
        }
        self.max_steps = {"button": 70, "drawer": 85, "window": 85, "cube": 200}

    def predicates(self) -> ScenePredicates:
        cube = np.asarray(
            self.raw._data.joint("object_joint_0").qpos[:3], dtype=np.float64
        )
        goal = np.asarray(
            self.raw._data.mocap_pos[self.raw._cube_target_mocap_ids[0]],
            dtype=np.float64,
        )
        drawer = float(self.raw._data.joint("drawer_slide").qpos[0])
        window = float(self.raw._data.joint("window_slide").qpos[0])
        return ScenePredicates(
            button_0=int(self.raw._cur_button_states[0]),
            button_1=int(self.raw._cur_button_states[1]),
            drawer_open=drawer <= -0.12,
            drawer_closed=drawer >= -0.04,
            window_open=window >= 0.16,
            window_closed=window <= 0.04,
            cube_in_drawer=bool(self.raw._is_in_drawer(cube)),
            cube_at_goal=bool(np.linalg.norm(cube - goal) <= 0.04),
            native_success=bool(self.raw._success),
        )

    def _configure(self, skill: str) -> tuple[str, dict[str, Any]]:
        import mujoco

        info = self.raw.compute_ob_info()
        if skill.startswith("toggle_button_"):
            button = int(skill.rsplit("_", 1)[1])
            target_state = (int(self.raw._cur_button_states[button]) + 1) % 2
            info.update(
                {
                    "privileged/target_button": button,
                    "privileged/target_button_state": target_state,
                    "privileged/target_button_top_pos": self.raw._data.site_xpos[
                        self.raw._button_site_ids[button]
                    ].copy(),
                }
            )
            return "button", info

        if skill in ("drawer_open", "drawer_close"):
            target = -0.16 if skill.endswith("open") else 0.0
            self.raw._model.site("drawer_handle_center_target").pos[1] = target
            mujoco.mj_kinematics(self.raw._model, self.raw._data)
            info = self.raw.compute_ob_info()
            info["privileged/target_drawer_handle_pos"] = self.raw._data.site_xpos[
                self.raw._drawer_target_site_id
            ].copy()
            return "drawer", info

        if skill in ("window_open", "window_close"):
            target = 0.2 if skill.endswith("open") else 0.0
            self.raw._model.site("window_handle_center_target").pos[0] = target
            mujoco.mj_kinematics(self.raw._model, self.raw._data)
            info = self.raw.compute_ob_info()
            info["privileged/target_window_handle_pos"] = self.raw._data.site_xpos[
                self.raw._window_target_site_id
            ].copy()
            return "window", info

        if skill == "place_cube_in_drawer":
            info.update(
                {
                    "privileged/target_block": 0,
                    # This is the official Scene data-collection target used
                    # for putting a cube into an open drawer.  The final task
                    # goal is reached after the drawer subsequently closes.
                    "privileged/target_block_pos": self.raw._drawer_center.copy(),
                    "privileged/target_block_yaw": np.array([0.0]),
                }
            )
            return "cube", info
        raise ValueError(f"unknown skill: {skill}")

    def _refresh_info(self, target_fields: dict[str, Any]) -> dict[str, Any]:
        # Rebuild moving proprio/object fields while keeping the target fixed
        # for the entire closed-loop skill.  Recomputing a button target after
        # it toggles would incorrectly request a second toggle.
        info = self.raw.compute_ob_info()
        info.update(target_fields)
        return info

    def execute(
        self, skill_index: int, milestones: MilestoneState
    ) -> tuple[MilestoneState, dict[str, Any]]:
        skill = SKILLS[skill_index]
        before = self.predicates()
        kind, info = self._configure(skill)
        target_fields = {
            key: copy.deepcopy(value)
            for key, value in info.items()
            if key.startswith("privileged/target_")
        }
        oracle = self.oracle_classes[kind](env=self.raw, min_norm=0.4)
        oracle.reset(None, info)
        oracle._max_step = self.max_steps[kind]
        # Remove the oracle's random post-skill pose from the experiment.
        oracle._final_pos = np.array([0.53, 0.15, 0.31], dtype=np.float64)
        oracle._final_yaw = 0.0

        state = milestones
        steps = 0
        for _ in range(self.max_steps[kind]):
            action = np.asarray(oracle.select_action(None, info), dtype=np.float32)
            _, _, _, truncated, _ = self.raw.step(action)
            if truncated:
                raise RuntimeError("raw Scene rollout unexpectedly truncated")
            steps += 1
            state = advance_milestones(state, self.predicates())
            if oracle.done:
                break
            info = self._refresh_info(target_fields)

        after = self.predicates()
        return state, {
            "skill": skill,
            "env_steps": steps,
            "before": asdict(before),
            "after": asdict(after),
            "new_events": list(state.transitions[len(milestones.transitions) :]),
        }

    def hold(
        self, milestones: MilestoneState, steps: int
    ) -> tuple[MilestoneState, int]:
        state = milestones
        for _ in range(steps):
            _, _, _, truncated, _ = self.raw.step(np.zeros(5, dtype=np.float32))
            if truncated:
                raise RuntimeError("raw Scene hold unexpectedly truncated")
            state = advance_milestones(state, self.predicates())
        return state, steps


def make_world(task_id: int, reset_seed: int) -> tuple[Any, Any]:
    import mujoco
    import stable_worldmodel as swm

    world = swm.World(
        "swm/OGBScene-v0",
        num_envs=1,
        max_episode_steps=2000,
        add_pixels=False,
        ob_type="states",
        multiview=False,
        visualize_info=False,
        terminate_at_goal=False,
        mode="task",
        reward_task_id=task_id,
    )
    raw = world.envs.envs[0].unwrapped
    raw.reset(seed=reset_seed, options={"variation": []})
    raw._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    return world, raw


def run_sequence(
    library: SkillLibrary,
    sequence: tuple[int, ...],
    milestones: MilestoneState,
) -> tuple[MilestoneState, list[dict[str, Any]], int]:
    state = milestones
    records: list[dict[str, Any]] = []
    env_steps = 0
    for skill_index in sequence:
        state, record = library.execute(skill_index, state)
        records.append(record)
        env_steps += int(record["env_steps"])
    return state, records, env_steps


def known_solution(task_id: int) -> tuple[int, ...]:
    by_name = {name: index for index, name in enumerate(SKILLS)}
    drawer_branch = (
        by_name["toggle_button_0"],
        by_name["drawer_open"],
        by_name["place_cube_in_drawer"],
        by_name["drawer_close"],
    )
    if task_id == 4:
        return drawer_branch
    return drawer_branch + (
        by_name["toggle_button_0"],
        by_name["toggle_button_1"],
        by_name["window_open"],
        by_name["toggle_button_1"],
    )


def main() -> None:
    args = parse_args()
    budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    if not budgets or min(budgets) < len(SKILLS):
        raise ValueError("every budget must cover all seven root skills")
    if args.stable_dwell != 3:
        raise ValueError("the locked automaton currently requires stable-dwell=3")
    max_decisions = args.max_decisions or (6 if args.task_id == 4 else 10)

    world, raw = make_world(args.task_id, args.reset_seed)
    try:
        snapshots = SceneSnapshotManager(raw)
        library = SkillLibrary(raw, args.stable_dwell)
        root = snapshots.capture()
        root_signature = snapshots.signature()
        root_predicates = asdict(library.predicates())

        support: dict[str, Any] = {"skipped": bool(args.skip_support_check)}
        if not args.skip_support_check:
            snapshots.restore(root)
            support_state, support_trace, support_steps = run_sequence(
                library,
                known_solution(args.task_id),
                initial_milestones(args.task_id),
            )
            support_state, hold_steps = library.hold(support_state, args.stable_dwell)
            support = {
                "skipped": False,
                "sequence": [SKILLS[index] for index in known_solution(args.task_id)],
                "success": support_state.stable_success,
                "milestones": asdict(support_state),
                "env_steps": support_steps + hold_steps,
                "trace": support_trace,
            }
            if not support_state.stable_success:
                raise RuntimeError("known skill composition did not solve the task")

        results: list[dict[str, Any]] = []
        for budget in budgets:
            for arm in (ARM_TERMINAL, ARM_EVENT):
                snapshots.restore(root)
                state = initial_milestones(args.task_id)
                deployed: list[int] = []
                replans: list[dict[str, Any]] = []
                eval_skill_calls = 0
                eval_env_steps = 0
                deploy_env_steps = 0

                for decision in range(max_decisions):
                    decision_snapshot = snapshots.capture()
                    decision_signature = snapshots.signature()
                    base_state = state

                    def evaluate(sequence: tuple[int, ...]) -> float:
                        nonlocal eval_skill_calls, eval_env_steps
                        snapshots.restore(decision_snapshot)
                        simulated, _, steps = run_sequence(
                            library, sequence, base_state
                        )
                        eval_skill_calls += len(sequence)
                        eval_env_steps += steps
                        return feedback_reward(simulated, arm)

                    search_seed = (
                        args.seed
                        + 1_000_003 * args.reset_seed
                        + 10_007 * args.task_id
                        + 503 * budget
                        + decision
                    )
                    search = uct_plan_search(
                        horizon=args.horizon,
                        simulations=budget,
                        search_seed=search_seed,
                        exploration=args.exploration,
                        evaluate=evaluate,
                    )
                    snapshots.restore(decision_snapshot)
                    if snapshots.signature() != decision_signature:
                        raise RuntimeError("decision-state restoration drifted")
                    state, deployed_record = library.execute(
                        search.selected_action, state
                    )
                    deploy_env_steps += int(deployed_record["env_steps"])
                    deployed.append(search.selected_action)
                    replans.append(
                        {
                            "decision": decision,
                            "state_before": asdict(base_state),
                            "selected_skill": SKILLS[search.selected_action],
                            "search": asdict(search),
                            "deployed": deployed_record,
                            "state_after": asdict(state),
                        }
                    )
                    if state.stable_success:
                        break

                state, hold_steps = library.hold(state, args.stable_dwell)
                deploy_env_steps += hold_steps
                final_predicates = library.predicates()
                results.append(
                    {
                        "task_id": args.task_id,
                        "reset_seed": args.reset_seed,
                        "budget_per_replan": budget,
                        "arm": arm,
                        "success": state.stable_success,
                        "final_reward": feedback_reward(state, arm),
                        "final_milestones": asdict(state),
                        "final_predicates": asdict(final_predicates),
                        "deployed_skills": [SKILLS[index] for index in deployed],
                        "num_replans": len(replans),
                        "eval_skill_calls": eval_skill_calls,
                        "eval_env_steps": eval_env_steps,
                        "deploy_env_steps": deploy_env_steps,
                        "replans": replans,
                    }
                )

        output = {
            "protocol": "scene_skill_uct_h0_v1",
            "interpretation": (
                "oracle causal-room gate only; not a learned world-model result"
            ),
            "task_id": args.task_id,
            "reset_seed": args.reset_seed,
            "skills": list(SKILLS),
            "budgets": budgets,
            "horizon": args.horizon,
            "max_decisions": max_decisions,
            "exploration": args.exploration,
            "stable_dwell": args.stable_dwell,
            "root_signature": root_signature,
            "root_predicates": root_predicates,
            "support_check": support,
            "results": results,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / "result.json"
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(out_path), "rows": results}, sort_keys=True))
    finally:
        world.close()


if __name__ == "__main__":
    main()
