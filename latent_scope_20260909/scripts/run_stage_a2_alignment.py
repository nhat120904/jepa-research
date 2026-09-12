#!/usr/bin/env python3
"""Bounded one-step diagnosis of released RoboCasa action/state alignment."""

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
import pandas as pd


def require_slurm() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("RoboCasa diagnostics must run inside an sbatch job")
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


def state_error(actual: np.ndarray, expected: np.ndarray, env: Any) -> dict[str, Any]:
    delta = np.asarray(actual) - np.asarray(expected)
    nq = int(env.sim.model.nq)
    nv = int(env.sim.model.nv)
    segments = {
        "time": delta[:1],
        "qpos": delta[1 : 1 + nq],
        "qvel": delta[1 + nq : 1 + nq + nv],
    }
    index = int(np.argmax(np.abs(delta))) if delta.size else None
    return {
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "l2": float(np.linalg.norm(delta)),
        "max_flat_index": index,
        "signed_error_at_max": float(delta[index]) if index is not None else 0.0,
        "segment_max_abs": {
            name: float(np.max(np.abs(values), initial=0.0))
            for name, values in segments.items()
        },
    }


def action_metadata(env: Any) -> dict[str, Any]:
    low, high = env.action_spec
    robots = []
    for robot_index, robot in enumerate(env.robots):
        composite = robot.composite_controller
        parts = {}
        for name, controller in composite.part_controllers.items():
            parts[name] = {
                "class": controller.__class__.__name__,
                "input_type": getattr(controller, "input_type", None),
                "input_ref_frame": getattr(controller, "input_ref_frame", None),
                "control_dim": getattr(controller, "control_dim", None),
            }
        robots.append(
            {
                "robot_index": robot_index,
                "robot_class": robot.__class__.__name__,
                "action_split_indexes": getattr(composite, "_action_split_indexes", {}),
                "part_controllers": parts,
            }
        )
    return {
        "shape": list(np.asarray(low).shape),
        "low": np.asarray(low),
        "high": np.asarray(high),
        "control_freq": getattr(env, "control_freq", None),
        "model_timestep": getattr(env, "model_timestep", None),
        "control_timestep": getattr(env, "control_timestep", None),
        "robots": robots,
    }


def make_env(dataset_path: Path) -> Any:
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
        }
    )
    return robosuite.make(**kwargs)


def initial_state(dataset_path: Path, episode_index: int, states: np.ndarray) -> dict[str, Any]:
    import robocasa.utils.lerobot_utils as lerobot_utils

    return {
        "states": states[0],
        "model": lerobot_utils.get_episode_model_xml(dataset_path, episode_index),
        "ep_meta": json.dumps(
            lerobot_utils.get_episode_meta(dataset_path, episode_index)
        ),
    }


def raw_lerobot_actions(dataset_path: Path, episode_index: int) -> np.ndarray:
    files = list(dataset_path.glob(f"data/*/episode_{episode_index:06d}.parquet"))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected one parquet file for episode {episode_index}, found {len(files)}"
        )
    frame = pd.read_parquet(files[0], columns=["action"])
    return np.stack(frame["action"].to_list())


def run_action_trial(
    env: Any,
    reset_payload: dict[str, Any],
    action: np.ndarray,
    states: np.ndarray,
    target_indices: list[int],
) -> dict[str, Any]:
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    reset_to(env, reset_payload)
    before = np.asarray(env.sim.get_state().flatten()).copy()
    env.step(np.asarray(action))
    after = np.asarray(env.sim.get_state().flatten()).copy()
    errors = {
        str(index): state_error(after, states[index], env)
        for index in target_indices
        if index < len(states)
    }
    best_index = min(errors, key=lambda key: errors[key]["max_abs"])
    return {
        "action": np.asarray(action),
        "action_min": float(np.min(action)),
        "action_max": float(np.max(action)),
        "action_l2": float(np.linalg.norm(action)),
        "reset_error_to_state_0": state_error(before, states[0], env),
        "errors_by_target_state_index": errors,
        "best_target_state_index": int(best_index),
        "best_target_error": errors[best_index],
    }


def task_probe(task: dict[str, Any], runtime_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    import robocasa
    import robocasa.utils.lerobot_utils as lerobot_utils
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    dataset_path = runtime_root / "datasets" / task["dataset_relative_path"]
    episode_index = int(task["episode_index"])
    states = lerobot_utils.get_episode_states(dataset_path, episode_index)
    canonical = lerobot_utils.get_episode_actions(
        dataset_path, episode_index, abs_actions=False
    )
    raw = raw_lerobot_actions(dataset_path, episode_index)
    dataset_meta = json.loads(
        (dataset_path / "extras" / "dataset_meta.json").read_text()
    )
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    reset_payload = initial_state(dataset_path, episode_index, states)

    env = make_env(dataset_path)
    try:
        reset_to(env, reset_payload)
        state_zero_after_reset = np.asarray(env.sim.get_state().flatten()).copy()
        reset_to(env, {"states": states[1]})
        direct_state_one = np.asarray(env.sim.get_state().flatten()).copy()
        direct_state_error = state_error(direct_state_one, states[1], env)

        trials = {
            "canonical_action_0": run_action_trial(
                env, reset_payload, canonical[0], states, config["target_state_indices"]
            ),
            "raw_lerobot_action_0": run_action_trial(
                env, reset_payload, raw[0], states, config["target_state_indices"]
            ),
            "canonical_action_1": run_action_trial(
                env, reset_payload, canonical[1], states, config["target_state_indices"]
            ),
            "zero_action": run_action_trial(
                env,
                reset_payload,
                np.zeros_like(canonical[0]),
                states,
                config["target_state_indices"],
            ),
        }
        canonical_error = trials["canonical_action_0"][
            "errors_by_target_state_index"
        ]["1"]["max_abs"]
        raw_error = trials["raw_lerobot_action_0"][
            "errors_by_target_state_index"
        ]["1"]["max_abs"]
        shifted_action_error = trials["canonical_action_1"][
            "errors_by_target_state_index"
        ]["1"]["max_abs"]
        canonical_best_target = trials["canonical_action_0"]["best_target_state_index"]

        tolerance = float(config["released_state_max_abs_tolerance"])
        direct_tolerance = float(config["direct_state_max_abs_tolerance"])
        if direct_state_error["max_abs"] > direct_tolerance:
            diagnosis = "DIRECT_STATE_PLAYBACK_BROKEN"
        elif canonical_error <= tolerance:
            diagnosis = "CANONICAL_ONE_STEP_MATCH"
        elif raw_error < canonical_error / 10:
            diagnosis = "ACTION_ORDERING_MISMATCH"
        elif shifted_action_error < canonical_error / 10:
            diagnosis = "ACTION_INDEX_MISMATCH"
        elif canonical_best_target not in (0, 1):
            diagnosis = "STATE_INDEX_MISMATCH"
        else:
            diagnosis = "OPEN_LOOP_DYNAMICS_INCOMPATIBLE"

        return {
            "task": task["name"],
            "dataset_path": str(dataset_path),
            "episode_index": episode_index,
            "dataset_versions": {
                key: dataset_meta["env_args"].get(key)
                for key in ("env_version", "robosuite_version", "mujoco_version")
            },
            "installed_robocasa_version": getattr(robocasa, "__version__", "unknown"),
            "dataset_fps": info.get("fps"),
            "num_states": int(len(states)),
            "num_actions": int(len(canonical)),
            "state_dimension": int(states.shape[1]),
            "action_dimension": int(canonical.shape[1]),
            "canonical_differs_from_raw": bool(not np.array_equal(canonical, raw)),
            "initial_reset_error": state_error(state_zero_after_reset, states[0], env),
            "direct_state_1_error": direct_state_error,
            "recorded_transition_0_to_1": state_error(states[1], states[0], env),
            "action_metadata": action_metadata(env),
            "trials": trials,
            "diagnosis": diagnosis,
        }
    finally:
        env.close()


def aggregate_verdict(rows: list[dict[str, Any]]) -> str:
    diagnoses = {row["diagnosis"] for row in rows}
    if diagnoses == {"CANONICAL_ONE_STEP_MATCH"}:
        return "A2_CANONICAL_MATCH"
    if "DIRECT_STATE_PLAYBACK_BROKEN" in diagnoses:
        return "A2_STATE_PLAYBACK_BROKEN"
    if diagnoses & {"ACTION_ORDERING_MISMATCH", "ACTION_INDEX_MISMATCH", "STATE_INDEX_MISMATCH"}:
        return "A2_ALIGNMENT_BUG_LOCALIZED"
    return "A2_OPEN_LOOP_INCOMPATIBLE"


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

    rows = [task_probe(task, runtime_root, config) for task in config["tasks"]]
    result = {
        "verdict": aggregate_verdict(rows),
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
        "tasks": rows,
    }
    output_path = args.output_dir / "stage_a2_result.json"
    output_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(result["verdict"], flush=True)
    for row in rows:
        canonical_error = row["trials"]["canonical_action_0"][
            "errors_by_target_state_index"
        ]["1"]["max_abs"]
        print(
            f"{row['task']}: {row['diagnosis']}; canonical step-0 error={canonical_error}",
            flush=True,
        )
    print(f"Result: {output_path}", flush=True)


if __name__ == "__main__":
    main()
