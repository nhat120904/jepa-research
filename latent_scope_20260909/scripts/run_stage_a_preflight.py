#!/usr/bin/env python3
"""Replay native RoboCasa demonstrations and audit history-bearing restoration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
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


def task_state(env: Any, fields: list[str]) -> dict[str, Any]:
    return {field: clone_value(getattr(env, field)) for field in fields}


def simple_fingerprint(value: Any) -> str | None:
    if isinstance(value, np.ndarray):
        if value.size > 10_000:
            return None
        array = np.ascontiguousarray(value)
        payload = str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()
    elif value is None or isinstance(value, (bool, int, float, str, np.generic)):
        payload = repr(jsonable(value)).encode()
    elif isinstance(value, (list, tuple)):
        if len(value) > 1_000:
            return None
        children = [simple_fingerprint(item) for item in value]
        if any(child is None for child in children):
            return None
        payload = repr((type(value).__name__, children)).encode()
    else:
        return None
    return hashlib.sha256(payload).hexdigest()


def simple_attribute_fingerprints(env: Any) -> dict[str, str]:
    rows = {}
    for name, value in vars(env).items():
        fingerprint = simple_fingerprint(value)
        if fingerprint is not None:
            rows[name] = fingerprint
    return rows


def object_simple_state(obj: Any) -> dict[str, Any]:
    state = {}
    for name, value in vars(obj).items():
        if isinstance(value, np.ndarray):
            state[name] = value.copy()
        elif value is None or isinstance(value, (bool, int, float, str, np.generic)):
            state[name] = clone_value(value)
    return state


def controller_state(env: Any) -> list[dict[str, Any]]:
    rows = []
    keep = {
        "goal_pos",
        "goal_ori",
        "goal_qpos",
        "goal_qvel",
        "goal_vel",
        "goal_torque",
    }
    for robot_index, robot in enumerate(env.robots):
        composite = getattr(robot, "composite_controller", None)
        parts = getattr(composite, "part_controllers", {}) if composite else {}
        for part_name, controller in parts.items():
            attrs = {
                name: clone_value(getattr(controller, name))
                for name in keep
                if hasattr(controller, name)
            }
            interpolators = {
                name: object_simple_state(value)
                for name, value in vars(controller).items()
                if name.startswith("interpolator") and value is not None
            }
            rows.append(
                {
                    "robot_index": robot_index,
                    "part_name": part_name,
                    "attrs": attrs,
                    "interpolators": interpolators,
                }
            )
    return rows


def restore_controller_state(env: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        robot = env.robots[row["robot_index"]]
        controller = robot.composite_controller.part_controllers[row["part_name"]]
        for name, value in row["attrs"].items():
            setattr(controller, name, clone_value(value))
        for name, state in row["interpolators"].items():
            interpolator = getattr(controller, name)
            for attr_name, value in state.items():
                setattr(interpolator, attr_name, clone_value(value))


def buffer_state(env: Any) -> list[dict[str, Any]]:
    rows = []
    for robot_index, robot in enumerate(env.robots):
        for name, value in vars(robot).items():
            candidates = value.items() if isinstance(value, dict) else [(None, value)]
            for key, candidate in candidates:
                if candidate is None or not hasattr(candidate, "push"):
                    continue
                state = object_simple_state(candidate)
                if state:
                    rows.append(
                        {
                            "robot_index": robot_index,
                            "name": name,
                            "key": key,
                            "state": state,
                        }
                    )
    return rows


def restore_buffer_state(env: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        owner = getattr(env.robots[row["robot_index"]], row["name"])
        buffer = owner[row["key"]] if row["key"] is not None else owner
        for name, value in row["state"].items():
            setattr(buffer, name, clone_value(value))


def sim_dynamics_state(env: Any) -> dict[str, Any]:
    data = env.sim.data
    names = (
        "act",
        "ctrl",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "eq_active",
        "mocap_pos",
        "mocap_quat",
        "userdata",
    )
    return {
        name: np.asarray(getattr(data, name)).copy()
        for name in names
        if hasattr(data, name)
    }


def restore_sim_dynamics_state(env: Any, state: dict[str, Any]) -> None:
    for name, value in state.items():
        getattr(env.sim.data, name)[...] = value


def capture_snapshot(env: Any, history_fields: list[str]) -> dict[str, Any]:
    rng_state = None
    if hasattr(env, "rng") and hasattr(env.rng, "bit_generator"):
        rng_state = copy.deepcopy(env.rng.bit_generator.state)
    return {
        "sim_state": np.asarray(env.sim.get_state().flatten()).copy(),
        "sim_dynamics_state": sim_dynamics_state(env),
        "task_state": task_state(env, history_fields),
        "controller_state": controller_state(env),
        "buffer_state": buffer_state(env),
        "timestep": int(env.timestep),
        "cur_time": float(env.cur_time),
        "done": bool(env.done),
        "env_rng": rng_state,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def restore_snapshot(env: Any, snapshot: dict[str, Any]) -> None:
    env.sim.set_state_from_flattened(snapshot["sim_state"])
    restore_sim_dynamics_state(env, snapshot["sim_dynamics_state"])
    env.sim.forward()
    for field, value in snapshot["task_state"].items():
        setattr(env, field, clone_value(value))
    restore_controller_state(env, snapshot["controller_state"])
    restore_buffer_state(env, snapshot["buffer_state"])
    env.timestep = snapshot["timestep"]
    env.cur_time = snapshot["cur_time"]
    env.done = snapshot["done"]
    if snapshot["env_rng"] is not None:
        env.rng.bit_generator.state = copy.deepcopy(snapshot["env_rng"])
    np.random.set_state(snapshot["numpy_rng"])
    random.setstate(snapshot["python_rng"])


def image_observations(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value).copy()
        for key, value in obs.items()
        if key.endswith("_image") and isinstance(value, np.ndarray)
    }


def monitor_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return json.dumps(jsonable(before), sort_keys=True) != json.dumps(
        jsonable(after), sort_keys=True
    )


def run_suffix(
    env: Any,
    actions: np.ndarray,
    history_fields: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for action in actions:
        obs, reward, done, _ = env.step(action)
        rows.append(
            {
                "sim_state": np.asarray(env.sim.get_state().flatten()).copy(),
                "sim_dynamics_state": sim_dynamics_state(env),
                "reward": float(reward),
                "done": bool(done),
                "task_state": task_state(env, history_fields),
                "images": image_observations(obs),
            }
        )
    return rows


def compare_suffix(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    state_max = 0.0
    dynamics_max = 0.0
    image_max = 0
    reward_equal = True
    done_equal = True
    task_equal = True
    image_keys_equal = True
    dynamics_keys_equal = True
    for row_a, row_b in zip(first, second):
        state_max = max(
            state_max,
            float(np.max(np.abs(row_a["sim_state"] - row_b["sim_state"]))),
        )
        dynamics_names = set(row_a["sim_dynamics_state"]) & set(
            row_b["sim_dynamics_state"]
        )
        dynamics_keys_equal = dynamics_keys_equal and set(
            row_a["sim_dynamics_state"]
        ) == set(row_b["sim_dynamics_state"])
        for name in dynamics_names:
            first_state = row_a["sim_dynamics_state"][name]
            second_state = row_b["sim_dynamics_state"][name]
            if first_state.size:
                dynamics_max = max(
                    dynamics_max,
                    float(np.max(np.abs(first_state - second_state))),
                )
        reward_equal = reward_equal and row_a["reward"] == row_b["reward"]
        done_equal = done_equal and row_a["done"] == row_b["done"]
        task_equal = task_equal and not monitor_changed(
            row_a["task_state"], row_b["task_state"]
        )
        keys_a = set(row_a["images"])
        keys_b = set(row_b["images"])
        image_keys_equal = image_keys_equal and keys_a == keys_b
        for key in keys_a & keys_b:
            delta = np.abs(
                row_a["images"][key].astype(np.int16)
                - row_b["images"][key].astype(np.int16)
            )
            image_max = max(image_max, int(delta.max(initial=0)))
    return {
        "length_equal": len(first) == len(second),
        "state_max_abs": state_max,
        "dynamics_max_abs": dynamics_max,
        "image_max_abs": image_max,
        "reward_equal": reward_equal,
        "done_equal": done_equal,
        "task_state_equal": task_equal,
        "image_keys_equal": image_keys_equal,
        "dynamics_keys_equal": dynamics_keys_equal,
    }


def save_event_images(
    images: dict[str, np.ndarray], output_dir: Path, task: str, episode: int, step: int
) -> list[str]:
    from PIL import Image

    saved = []
    event_dir = output_dir / "event_frames" / task
    event_dir.mkdir(parents=True, exist_ok=True)
    for key, array in images.items():
        path = event_dir / f"episode_{episode:06d}_step_{step:05d}_{key}.png"
        Image.fromarray(array).save(path)
        saved.append(str(path))
    return saved


def replay_episode(
    env: Any,
    dataset_path: Path,
    episode_index: int,
    history_fields: list[str],
    suffix_steps: int,
    released_state_tolerance: float,
    output_dir: Path,
) -> dict[str, Any]:
    import robocasa.utils.lerobot_utils as lerobot_utils
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    states = lerobot_utils.get_episode_states(dataset_path, episode_index)
    actions = lerobot_utils.get_episode_actions(dataset_path, episode_index, abs_actions=False)
    initial = {
        "states": states[0],
        "model": lerobot_utils.get_episode_model_xml(dataset_path, episode_index),
        "ep_meta": json.dumps(lerobot_utils.get_episode_meta(dataset_path, episode_index)),
    }
    reset_to(env, initial)

    max_abs_error = 0.0
    first_divergence_step = None
    success_step = None
    history_change_steps = []
    event_images = []
    camera_stats: dict[str, dict[str, Any]] = {}
    previous_monitor = task_state(env, history_fields)
    previous_attributes = simple_attribute_fingerprints(env)
    unlisted_changed_attributes: set[str] = set()

    for step, action in enumerate(actions):
        obs, reward, _, _ = env.step(action)
        if step + 1 < len(states):
            replayed = np.asarray(env.sim.get_state().flatten())
            expected = np.asarray(states[step + 1])
            error = float(np.max(np.abs(replayed - expected)))
            max_abs_error = max(max_abs_error, error)
            if first_divergence_step is None and error > released_state_tolerance:
                first_divergence_step = step
        if reward > 0 and success_step is None:
            success_step = step
        monitor = task_state(env, history_fields)
        attributes = simple_attribute_fingerprints(env)
        ignored = set(history_fields) | {"timestep", "cur_time", "done"}
        unlisted_changed_attributes.update(
            name
            for name in previous_attributes.keys() & attributes.keys()
            if name not in ignored and previous_attributes[name] != attributes[name]
        )
        previous_attributes = attributes
        images = image_observations(obs)
        for key, array in images.items():
            stats = camera_stats.setdefault(
                key,
                {"shape": list(array.shape), "min": 255, "max": 0, "changed": False},
            )
            stats["min"] = min(stats["min"], int(array.min(initial=255)))
            stats["max"] = max(stats["max"], int(array.max(initial=0)))
            stats["changed"] = stats["changed"] or stats.get("first_sum") not in {
                None,
                int(array.astype(np.uint64).sum()),
            }
            stats.setdefault("first_sum", int(array.astype(np.uint64).sum()))
        if monitor_changed(previous_monitor, monitor):
            history_change_steps.append(step)
            if len(event_images) < 6:
                event_images.extend(
                    save_event_images(images, output_dir, env.__class__.__name__, episode_index, step)
                )
        previous_monitor = monitor

    replay_success = success_step is not None

    reset_to(env, initial)
    latest_prefix = max(1, len(actions) - suffix_steps)
    eligible_event_prefixes = [
        step + 1 for step in history_change_steps if step + 1 <= latest_prefix
    ]
    if eligible_event_prefixes:
        prefix = eligible_event_prefixes[len(eligible_event_prefixes) // 2]
    else:
        prefix = max(1, min(latest_prefix, len(actions) // 2))
    for action in actions[:prefix]:
        env.step(action)
    snapshot = capture_snapshot(env, history_fields)
    suffix_actions = actions[prefix : prefix + suffix_steps]
    suffix_first = run_suffix(env, suffix_actions, history_fields)
    restore_snapshot(env, snapshot)
    suffix_second = run_suffix(env, suffix_actions, history_fields)
    suffix_comparison = compare_suffix(suffix_first, suffix_second)

    return {
        "episode_index": episode_index,
        "num_states": int(len(states)),
        "num_actions": int(len(actions)),
        "released_replay_max_abs_error": max_abs_error,
        "first_divergence_step": first_divergence_step,
        "native_success": replay_success,
        "native_success_step": success_step,
        "history_change_steps": history_change_steps,
        "unlisted_changed_attributes": sorted(unlisted_changed_attributes),
        "event_images": event_images,
        "camera_stats": camera_stats,
        "snapshot_prefix_steps": prefix,
        "snapshot_suffix_steps": int(len(suffix_actions)),
        "suffix_comparison": suffix_comparison,
    }


def make_env(dataset_path: Path, camera_names: list[str], height: int, width: int) -> Any:
    import robocasa.utils.lerobot_utils as lerobot_utils
    import robosuite

    env_meta = lerobot_utils.get_env_metadata(dataset_path)
    kwargs = copy.deepcopy(env_meta["env_kwargs"])
    kwargs["env_name"] = env_meta["env_name"]
    kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": camera_names,
            "camera_heights": height,
            "camera_widths": width,
            "ignore_done": True,
        }
    )
    return robosuite.make(**kwargs)


def decide_verdict(config: dict[str, Any], task_rows: list[dict[str, Any]]) -> str:
    episodes = [episode for task in task_rows for episode in task["episodes"]]
    if not all(episode["native_success"] for episode in episodes):
        return "REPLAY_DIVERGED"
    if any(
        episode["released_replay_max_abs_error"]
        > config["released_state_max_abs_tolerance"]
        for episode in episodes
    ):
        return "REPLAY_DIVERGED"
    for episode in episodes:
        comparison = episode["suffix_comparison"]
        if (
            not comparison["length_equal"]
            or not comparison["reward_equal"]
            or not comparison["done_equal"]
            or not comparison["task_state_equal"]
            or not comparison["image_keys_equal"]
            or not comparison["dynamics_keys_equal"]
            or comparison["state_max_abs"]
            > config["restored_state_max_abs_tolerance"]
            or comparison["dynamics_max_abs"]
            > config["restored_state_max_abs_tolerance"]
            or comparison["image_max_abs"]
            > config["restored_image_max_abs_tolerance"]
        ):
            return "HISTORY_RESTORE_BROKEN"
    if any(not task["has_history_change"] for task in task_rows):
        return "VISUAL_SIGNAL_UNRESOLVED"
    required = {f"{camera}_image" for camera in config["camera_names"]}
    for episode in episodes:
        if not required.issubset(episode["camera_stats"]):
            return "VISUAL_SIGNAL_UNRESOLVED"
        if not all(
            episode["camera_stats"][key]["changed"]
            for key in required
        ):
            return "VISUAL_SIGNAL_UNRESOLVED"
    return "STAGE_A_PASS"


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
        raise RuntimeError(
            f"Source revisions differ from locked config: {resolved}"
        )

    import mujoco
    import robocasa
    import robosuite

    task_rows = []
    for task in config["tasks"]:
        dataset_path = runtime_root / "datasets" / task["dataset_relative_path"]
        env = make_env(
            dataset_path,
            config["camera_names"],
            config["camera_height"],
            config["camera_width"],
        )
        try:
            episodes = [
                replay_episode(
                    env,
                    dataset_path,
                    episode_index,
                    task["history_fields"],
                    config["snapshot_suffix_steps"],
                    config["released_state_max_abs_tolerance"],
                    args.output_dir,
                )
                for episode_index in range(config["episodes_per_task"])
            ]
        finally:
            env.close()
        task_rows.append(
            {
                "task": task["name"],
                "dataset_path": str(dataset_path),
                "history_fields": task["history_fields"],
                "has_history_change": any(
                    episode["history_change_steps"] for episode in episodes
                ),
                "episodes": episodes,
            }
        )

    verdict = decide_verdict(config, task_rows)
    result = {
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm": {
            "job_id": job_id,
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": platform.node(),
        },
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
    output_path = args.output_dir / "stage_a_result.json"
    output_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(verdict, flush=True)
    print(f"Result: {output_path}", flush=True)
    if verdict != "STAGE_A_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
