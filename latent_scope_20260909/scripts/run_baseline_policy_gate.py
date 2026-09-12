#!/usr/bin/env python3
"""Evaluate the official target-trained GR00T policy on two selected tasks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def require_slurm() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Baseline evaluation must run inside an sbatch job")
    return job_id


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    job_id = require_slurm()
    config = json.loads(args.config.read_text())
    runtime_root = Path(config["runtime_root"])
    shared_root = Path(config["shared_robocasa_runtime_root"])
    checkpoint_path = (
        runtime_root
        / "checkpoints"
        / "robocasa365_checkpoints"
        / config["checkpoint"]["relative_path"]
    )
    setup_result = runtime_root / "outputs" / "setup_result.json"
    if not setup_result.is_file():
        raise RuntimeError(f"Setup result missing: {setup_result}")
    if json.loads(setup_result.read_text()).get("verdict") != "BASELINE_SETUP_COMPLETE":
        raise RuntimeError("Baseline setup did not complete cleanly")

    import torch
    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.eval.simulation import (
        MultiStepConfig,
        SimulationConfig,
        SimulationInferenceClient,
        VideoConfig,
    )
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the GPU evaluation job")

    expected_revisions = config["source_revisions"]
    source_paths = {
        "isaac_groot": runtime_root / "src" / "Isaac-GR00T",
        "robocasa": shared_root / "src" / "robocasa365",
        "robosuite": shared_root / "src" / "robosuite",
    }
    source_revisions = {name: git_head(path) for name, path in source_paths.items()}
    if source_revisions != expected_revisions:
        raise RuntimeError(
            f"Source revision mismatch: {source_revisions} != {expected_revisions}"
        )

    evaluation = config["evaluation"]
    registered = set(TASK_SET_REGISTRY[evaluation["expected_task_set"]])
    if not set(evaluation["tasks"]).issubset(registered):
        raise RuntimeError("Selected tasks are not all in the locked official task set")

    started = datetime.now(timezone.utc)
    result_path = args.output_dir / "baseline_policy_result.json"
    payload = {
        "verdict": "BASELINE_EVAL_RUNNING",
        "created_utc": started.isoformat(),
        "slurm_job_id": job_id,
        "node": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "source_revisions": source_revisions,
        "checkpoint": {
            "repo_id": config["checkpoint"]["repo_id"],
            "revision": config["checkpoint"]["revision"],
            "path": str(checkpoint_path),
        },
        "policy": config["policy"],
        "evaluation": evaluation,
        "gate": config["gate"],
        "task_results": {},
    }
    write_result(result_path, payload)

    try:
        data_config = DATA_CONFIG_MAP[config["policy"]["data_config"]]
        policy = Gr00tPolicy(
            model_path=str(checkpoint_path),
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=config["policy"]["embodiment_tag"],
            denoising_steps=config["policy"]["denoising_steps"],
            device="cuda",
        )
        server = RobotInferenceServer(policy, port=args.port)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        simulation_client = SimulationInferenceClient(host="localhost", port=args.port)
        connection_error = None
        for _ in range(60):
            try:
                modality_config = simulation_client.get_modality_config()
                connection_error = None
                break
            except Exception as error:  # server startup is asynchronous
                connection_error = error
                time.sleep(2)
        if connection_error is not None:
            raise RuntimeError("Inference server did not become ready") from connection_error
        payload["modality_keys"] = sorted(modality_config.keys())

        for task in evaluation["tasks"]:
            task_started = time.monotonic()
            sim_config = SimulationConfig(
                env_name=f"robocasa/{task}",
                split=evaluation["split"],
                n_episodes=evaluation["episodes_per_task"],
                n_envs=evaluation["n_envs"],
                video=VideoConfig(video_dir=None),
                multistep=MultiStepConfig(
                    n_action_steps=config["policy"]["n_action_steps"],
                    max_episode_steps=get_task_horizon(task),
                ),
            )
            _, successes = simulation_client.run_simulation(sim_config)
            successes = [bool(value) for value in successes]
            if len(successes) != evaluation["episodes_per_task"]:
                raise RuntimeError(
                    f"Expected exactly {evaluation['episodes_per_task']} results for "
                    f"{task}, got {len(successes)}"
                )
            payload["task_results"][task] = {
                "episode_successes": successes,
                "successes": sum(successes),
                "episodes": len(successes),
                "success_rate": float(np.mean(successes)),
                "horizon": get_task_horizon(task),
                "elapsed_seconds": time.monotonic() - task_started,
            }
            write_result(result_path, payload)

        minimum = config["gate"]["minimum_successes_per_task"]
        passed = all(
            row["successes"] >= minimum
            for row in payload["task_results"].values()
        )
        payload["verdict"] = (
            "BASELINE_USABLE_FOR_A4" if passed else "BASELINE_NOT_USABLE_STOP_ARENA"
        )
        payload["gate_passed"] = passed
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        payload["elapsed_seconds"] = (
            datetime.now(timezone.utc) - started
        ).total_seconds()
        payload["packages"] = {
            name: importlib.metadata.version(name)
            for name in (
                "flash-attn",
                "gr00t",
                "mujoco",
                "robocasa",
                "robosuite",
                "torch",
                "transformers",
            )
        }
        write_result(result_path, payload)
        print(payload["verdict"], flush=True)
    except Exception as error:
        payload["verdict"] = "BASELINE_EVAL_ERROR"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        payload["error"] = repr(error)
        payload["traceback"] = traceback.format_exc()
        write_result(result_path, payload)
        raise


if __name__ == "__main__":
    main()
