#!/usr/bin/env python3
"""Test action repeat and control-frequency explanations for Stage-A divergence."""

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
    pieces = {
        "time": delta[:1],
        "qpos": delta[1 : 1 + nq],
        "qvel": delta[1 + nq : 1 + nq + nv],
    }
    segment_max = {
        name: float(np.max(np.abs(values), initial=0.0))
        for name, values in pieces.items()
    }
    return {
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "physical_max_abs": max(segment_max["qpos"], segment_max["qvel"]),
        "l2": float(np.linalg.norm(delta)),
        "segment_max_abs": segment_max,
    }


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


def reset_payload(dataset_path: Path, episode_index: int, state: np.ndarray) -> dict[str, Any]:
    import robocasa.utils.lerobot_utils as lerobot_utils

    return {
        "states": state,
        "model": lerobot_utils.get_episode_model_xml(dataset_path, episode_index),
        "ep_meta": json.dumps(
            lerobot_utils.get_episode_meta(dataset_path, episode_index)
        ),
    }


def trial(
    env: Any,
    initial: dict[str, Any],
    actions: list[np.ndarray],
    target: np.ndarray,
) -> dict[str, Any]:
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    reset_to(env, initial)
    trace = []
    for action in actions:
        env.step(action)
        trace.append(float(env.sim.data.time))
    final = np.asarray(env.sim.get_state().flatten()).copy()
    return {
        "num_env_steps": len(actions),
        "sim_time_trace": trace,
        "error_to_released_state_1": state_error(final, target, env),
    }


def task_probe(task: dict[str, Any], runtime_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    import robocasa.utils.lerobot_utils as lerobot_utils

    dataset_path = runtime_root / "datasets" / task["dataset_relative_path"]
    episode_index = int(task["episode_index"])
    states = lerobot_utils.get_episode_states(dataset_path, episode_index)
    actions = lerobot_utils.get_episode_actions(
        dataset_path, episode_index, abs_actions=False
    )
    initial = reset_payload(dataset_path, episode_index, states[0])
    candidates: dict[str, dict[str, Any]] = {}

    for control_freq in config["control_frequency_candidates"]:
        env = make_env(dataset_path, int(control_freq))
        try:
            if control_freq == 20:
                for repeat in config["repeat_factors"]:
                    candidates[f"freq20_hold_action0_repeat{repeat}"] = trial(
                        env,
                        initial,
                        [actions[0]] * int(repeat),
                        states[1],
                    )
                for repeat in config["repeat_factors"]:
                    candidates[f"freq20_sequential_repeat{repeat}"] = trial(
                        env,
                        initial,
                        [actions[index] for index in range(int(repeat))],
                        states[1],
                    )
            else:
                candidates[f"freq{control_freq}_action0_once"] = trial(
                    env, initial, [actions[0]], states[1]
                )
        finally:
            env.close()

    best_name = min(
        candidates,
        key=lambda name: candidates[name]["error_to_released_state_1"][
            "physical_max_abs"
        ],
    )
    baseline = candidates["freq20_hold_action0_repeat1"][
        "error_to_released_state_1"
    ]["physical_max_abs"]
    best_error = candidates[best_name]["error_to_released_state_1"]
    tolerance = float(config["released_state_max_abs_tolerance"])

    if best_error["max_abs"] <= tolerance:
        diagnosis = "REPLAY_TIMING_MATCH_LOCALIZED"
    elif best_error["physical_max_abs"] < baseline / 10:
        diagnosis = "REPLAY_TIMING_STRONGLY_SUPPORTED"
    elif best_error["physical_max_abs"] < baseline / 2:
        diagnosis = "REPLAY_TIMING_PARTIALLY_SUPPORTED"
    else:
        diagnosis = "REPLAY_TIMING_REFUTED"

    return {
        "task": task["name"],
        "episode_index": episode_index,
        "released_state_0_time": float(states[0][0]),
        "released_state_1_time": float(states[1][0]),
        "released_delta_time": float(states[1][0] - states[0][0]),
        "candidates": candidates,
        "best_candidate": best_name,
        "best_error": best_error,
        "baseline_physical_max_abs": baseline,
        "diagnosis": diagnosis,
    }


def aggregate_verdict(rows: list[dict[str, Any]]) -> str:
    if all(row["diagnosis"] == "REPLAY_TIMING_MATCH_LOCALIZED" for row in rows):
        return "A2B_TIMING_MATCH_LOCALIZED"
    if all(
        row["diagnosis"]
        in {"REPLAY_TIMING_MATCH_LOCALIZED", "REPLAY_TIMING_STRONGLY_SUPPORTED"}
        for row in rows
    ):
        return "A2B_TIMING_STRONGLY_SUPPORTED"
    if any(row["diagnosis"] == "REPLAY_TIMING_REFUTED" for row in rows):
        return "A2B_TIMING_NOT_GENERAL"
    return "A2B_TIMING_PARTIALLY_SUPPORTED"


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
    output_path = args.output_dir / "stage_a2b_result.json"
    output_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(result["verdict"], flush=True)
    for row in rows:
        print(
            f"{row['task']}: {row['diagnosis']}; best={row['best_candidate']}; "
            f"physical_max={row['best_error']['physical_max_abs']}",
            flush=True,
        )
    print(f"Result: {output_path}", flush=True)


if __name__ == "__main__":
    main()
