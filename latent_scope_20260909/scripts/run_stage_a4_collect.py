#!/usr/bin/env python3
"""Collect successful current-runtime GR00T trajectories for Stage A4."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np

import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.dataset_registry_utils import get_task_horizon

from run_baseline_sim_client import InferenceClient, load_multistep_wrapper, process_alive


def jsonable(value):
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def wait_for_server(client: InferenceClient, pid: int, state_path: Path) -> None:
    for _ in range(200):
        if not process_alive(pid):
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            raise RuntimeError(f"Policy server exited before readiness: {state}")
        try:
            if client.ping():
                client.timeout_ms = 120_000
                client._open_socket()
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Policy server did not become ready within 10 minutes")


def task_history(task_env, fields: list[str]) -> dict:
    return {field: jsonable(getattr(task_env, field)) for field in fields}


def batch_observation(observation: dict) -> dict:
    """Match the one-environment SyncVectorEnv layout expected by GR00T."""
    return {
        key: np.expand_dims(value, axis=0) if isinstance(value, np.ndarray) else (value,)
        for key, value in observation.items()
    }


def save_successful_episode(
    directory: Path,
    metadata: dict,
    model_xml: str,
    ep_meta: dict,
    initial_state: np.ndarray,
    final_state: np.ndarray,
    actions: list[dict[str, np.ndarray]],
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    with gzip.open(directory / "model.xml.gz", "wt", encoding="utf-8") as handle:
        handle.write(model_xml)
    write_json(directory / "ep_meta.json", ep_meta)
    write_json(directory / "metadata.json", metadata)
    arrays = {
        "initial_state": initial_state,
        "final_state": final_state,
    }
    for key in actions[0]:
        arrays[f"action::{key}"] = np.stack([action[key] for action in actions])
    np.savez_compressed(directory / "trajectory.npz", **arrays)


def run_attempt(client, wrapper_class, task_cfg: dict, split: str, n_action_steps: int, seed: int):
    task = task_cfg["name"]
    horizon = get_task_horizon(task)

    def create_env():
        base = gym.make(f"robocasa/{task}", split=split, enable_render=True)
        return wrapper_class(
            base,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=n_action_steps,
            max_episode_steps=horizon,
        )

    wrapped = create_env()
    gym_env = wrapped.unwrapped
    task_env = gym_env.env
    actions: list[dict[str, np.ndarray]] = []
    success = False
    chunks = 0
    started = time.monotonic()
    try:
        obs, _ = wrapped.reset(seed=seed)
        initial_state = np.asarray(task_env.sim.get_state().flatten()).copy()
        model_xml = task_env.sim.model.get_xml()
        ep_meta = task_env.get_ep_meta()
        initial_history = task_history(task_env, task_cfg["history_fields"])
        while True:
            predicted = client.get_action(batch_observation(obs))
            batched_chunk = predicted.get("actions", predicted)
            action_chunk = {
                key: np.asarray(value)[0].copy() for key, value in batched_chunk.items()
            }
            obs, reward, terminated, truncated, info = wrapped.step(action_chunk)
            executed = len(info["rewards"])
            if executed <= 0:
                raise RuntimeError(f"No native actions executed for {task}")
            for step in range(executed):
                actions.append(
                    {
                        key: np.asarray(value)[step].copy()
                        for key, value in action_chunk.items()
                    }
                )
            chunks += 1
            success = success or bool(reward) or bool(np.asarray(info["success"]).reshape(-1)[-1])
            if bool(terminated) or bool(truncated):
                break
        final_state = np.asarray(task_env.sim.get_state().flatten()).copy()
        final_history = task_history(task_env, task_cfg["history_fields"])
        return {
            "success": success,
            "native_steps": len(actions),
            "chunks": chunks,
            "seed": seed,
            "horizon": horizon,
            "elapsed_seconds": time.monotonic() - started,
            "initial_history": initial_history,
            "final_history": final_history,
        }, model_xml, ep_meta, initial_state, final_state, actions
    finally:
        wrapped.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--server-state", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("A4 collection must run inside an sbatch job")
    config = json.loads(args.config.read_text())
    result_path = args.output_dir / "collection_result.json"
    result = {
        "verdict": "A4_COLLECTION_RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": job_id,
        "node": platform.node(),
        "config": config,
        "tasks": {},
    }
    write_json(result_path, result)

    client = InferenceClient("localhost", args.port, timeout_ms=1_000)
    try:
        wait_for_server(client, args.server_pid, args.server_state)
        wrapper_class = load_multistep_wrapper(
            Path(config["runtime_root"]) / "src" / "Isaac-GR00T"
        )
        required = config["collection"]["successful_episodes_per_task"]
        seed_base = config["collection"]["seed_base"]
        for task_index, task_cfg in enumerate(config["collection"]["tasks"]):
            task = task_cfg["name"]
            task_root = args.output_dir / "episodes" / task
            task_row = {"attempts": [], "successful_episodes": []}
            result["tasks"][task] = task_row
            for attempt in range(task_cfg["max_attempts"]):
                seed = seed_base + task_index * 10_000 + attempt
                payload = run_attempt(
                    client,
                    wrapper_class,
                    task_cfg,
                    config["collection"]["split"],
                    config["policy"]["n_action_steps"],
                    seed,
                )
                metadata, model_xml, ep_meta, initial_state, final_state, actions = payload
                metadata["attempt"] = attempt
                task_row["attempts"].append(metadata)
                if metadata["success"]:
                    success_index = len(task_row["successful_episodes"])
                    episode_dir = task_root / f"success_{success_index:02d}"
                    metadata["success_index"] = success_index
                    save_successful_episode(
                        episode_dir,
                        metadata,
                        model_xml,
                        ep_meta,
                        initial_state,
                        final_state,
                        actions,
                    )
                    task_row["successful_episodes"].append(str(episode_dir))
                task_row["num_attempts"] = attempt + 1
                task_row["num_successes"] = len(task_row["successful_episodes"])
                write_json(result_path, result)
                print(
                    f"{task} attempt {attempt + 1}: success={metadata['success']} "
                    f"retained={task_row['num_successes']}/{required}",
                    flush=True,
                )
                if task_row["num_successes"] >= required:
                    break
            if task_row.get("num_successes", 0) < required:
                result["verdict"] = "A4_COLLECTION_INSUFFICIENT_SUCCESS"
                result["completed_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(result_path, result)
                raise RuntimeError(f"Could not collect {required} successes for {task}")

        result["verdict"] = "A4_COLLECTION_COMPLETE"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["policy_server_state"] = json.loads(args.server_state.read_text())
        write_json(result_path, result)
        print("A4_COLLECTION_COMPLETE", flush=True)
    except Exception as error:
        if result["verdict"] == "A4_COLLECTION_RUNNING":
            result["verdict"] = "A4_COLLECTION_ERROR"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        write_json(result_path, result)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
