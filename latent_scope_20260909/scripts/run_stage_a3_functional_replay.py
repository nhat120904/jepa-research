#!/usr/bin/env python3
"""Camera-free 10 Hz functional replay for the Stage-A execution decision."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def require_slurm() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("RoboCasa replay must run inside an sbatch job")
    return job_id


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def clone_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def history_state(env: Any, fields: list[str]) -> dict[str, Any]:
    return {field: clone_value(getattr(env, field)) for field in fields}


def value_changed(first: Any, second: Any) -> bool:
    return json.dumps(jsonable(first), sort_keys=True) != json.dumps(
        jsonable(second), sort_keys=True
    )


def make_env(dataset_path: Path, control_freq: int) -> Any:
    import robocasa.utils.lerobot_utils as lerobot_utils
    import robosuite

    env_meta = lerobot_utils.get_env_metadata(dataset_path)
    kwargs = copy.deepcopy(env_meta["env_kwargs"])
    kwargs["env_name"] = env_meta["env_name"]
    kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "ignore_done": True,
            "control_freq": control_freq,
        }
    )
    return robosuite.make(**kwargs)


def state_error(actual: np.ndarray, expected: np.ndarray, env: Any) -> dict[str, float]:
    delta = np.asarray(actual) - np.asarray(expected)
    nq = int(env.sim.model.nq)
    nv = int(env.sim.model.nv)
    pieces = {
        "time": delta[:1],
        "qpos": delta[1 : 1 + nq],
        "qvel": delta[1 + nq : 1 + nq + nv],
    }
    return {
        name: float(np.max(np.abs(values), initial=0.0))
        for name, values in pieces.items()
    }


def replay_episode(
    env: Any,
    dataset_path: Path,
    episode_index: int,
    history_fields: list[str],
    tolerance: float,
) -> dict[str, Any]:
    import robocasa.utils.lerobot_utils as lerobot_utils
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    states = lerobot_utils.get_episode_states(dataset_path, episode_index)
    actions = lerobot_utils.get_episode_actions(
        dataset_path, episode_index, abs_actions=False
    )
    initial = {
        "states": states[0],
        "model": lerobot_utils.get_episode_model_xml(dataset_path, episode_index),
        "ep_meta": json.dumps(
            lerobot_utils.get_episode_meta(dataset_path, episode_index)
        ),
    }
    reset_to(env, initial)

    initial_history = history_state(env, history_fields)
    previous_history = initial_history
    changed_fields: set[str] = set()
    history_change_steps: list[int] = []
    first_history_values: dict[str, Any] = {}
    success_step = None
    first_divergence_step = None
    state_errors: list[float] = []
    component_max = {"time": 0.0, "qpos": 0.0, "qvel": 0.0}

    for step, action in enumerate(actions):
        _, reward, _, _ = env.step(action)
        if reward > 0 and success_step is None:
            success_step = step

        current_history = history_state(env, history_fields)
        step_changed = False
        for field in history_fields:
            if value_changed(previous_history[field], current_history[field]):
                changed_fields.add(field)
                first_history_values.setdefault(field, jsonable(current_history[field]))
                step_changed = True
        if step_changed:
            history_change_steps.append(step)
        previous_history = current_history

        if step + 1 < len(states):
            actual = np.asarray(env.sim.get_state().flatten())
            components = state_error(actual, states[step + 1], env)
            maximum = max(components.values())
            state_errors.append(maximum)
            for name, value in components.items():
                component_max[name] = max(component_max[name], value)
            if first_divergence_step is None and maximum > tolerance:
                first_divergence_step = step

    error_array = np.asarray(state_errors, dtype=np.float64)
    return {
        "episode_index": episode_index,
        "num_states": int(len(states)),
        "num_actions": int(len(actions)),
        "native_success": success_step is not None,
        "native_success_step": success_step,
        "initial_history": initial_history,
        "final_history": previous_history,
        "changed_history_fields": sorted(changed_fields),
        "history_change_steps": history_change_steps,
        "first_history_values": first_history_values,
        "first_divergence_step": first_divergence_step,
        "released_replay_component_max_abs": component_max,
        "released_replay_error_quantiles": {
            "q0": float(np.quantile(error_array, 0.0)),
            "q50": float(np.quantile(error_array, 0.5)),
            "q90": float(np.quantile(error_array, 0.9)),
            "q99": float(np.quantile(error_array, 0.99)),
            "q100": float(np.quantile(error_array, 1.0)),
        },
    }


def decide_verdict(task_rows: list[dict[str, Any]]) -> str:
    episodes = [episode for task in task_rows for episode in task["episodes"]]
    all_success = all(episode["native_success"] for episode in episodes)
    all_tasks_have_history = all(task["has_history_change"] for task in task_rows)
    if all_success and all_tasks_have_history:
        return "A3_FUNCTIONAL_PASS"
    if all_success:
        return "A3_HISTORY_SIGNAL_FAIL"
    return "A3_FUNCTIONAL_FAIL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    job_id = require_slurm()
    config = json.loads(args.config.read_text())
    runtime_root = Path(config["runtime_root"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "robocasa": runtime_root / "src" / "robocasa365",
        "robosuite": runtime_root / "src" / "robosuite",
        "diffusion_policy": runtime_root / "src" / "diffusion_policy",
    }
    resolved = {name: git_head(path) for name, path in source_paths.items()}
    if resolved != config["source_revisions"]:
        raise RuntimeError(f"Source revisions differ from locked config: {resolved}")

    import mujoco
    import robocasa
    import robosuite

    task_rows = []
    for task in config["tasks"]:
        dataset_path = runtime_root / "datasets" / task["dataset_relative_path"]
        env = make_env(dataset_path, int(config["control_freq"]))
        try:
            episodes = [
                replay_episode(
                    env,
                    dataset_path,
                    episode_index,
                    task["history_fields"],
                    float(config["released_state_max_abs_tolerance"]),
                )
                for episode_index in range(int(config["episodes_per_task"]))
            ]
        finally:
            env.close()
        task_rows.append(
            {
                "task": task["name"],
                "dataset_path": str(dataset_path),
                "history_fields": task["history_fields"],
                "native_successes": sum(
                    episode["native_success"] for episode in episodes
                ),
                "has_history_change": any(
                    episode["history_change_steps"] for episode in episodes
                ),
                "episodes": episodes,
            }
        )

    result = {
        "verdict": decide_verdict(task_rows),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm": {"job_id": job_id, "node": platform.node()},
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "robosuite": getattr(robosuite, "__version__", "unknown"),
            "robocasa": getattr(robocasa, "__version__", "unknown"),
        },
        "source_revisions": resolved,
        "config": config,
        "tasks": task_rows,
    }
    output_path = args.output_dir / "stage_a3_result.json"
    output_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(result["verdict"], flush=True)
    for row in task_rows:
        print(
            f"{row['task']}: success={row['native_successes']}/{len(row['episodes'])}; "
            f"history_changed={row['has_history_change']}",
            flush=True,
        )
    print(f"Result: {output_path}", flush=True)


if __name__ == "__main__":
    main()
