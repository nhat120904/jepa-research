#!/usr/bin/env python3
"""Run RoboCasa evaluation without importing GR00T into the simulator environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import zmq

import robocasa  # noqa: F401 - registers the native Gym environments
import robosuite  # noqa: F401
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class InferenceClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 120_000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = None
        self._open_socket()

    def _open_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    @staticmethod
    def _encode(data) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def _decode(data: bytes):
        return torch.load(BytesIO(data), weights_only=False)

    def call(self, endpoint: str, data=None, requires_input: bool = True):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self.socket.send(self._encode(request))
            response = self._decode(self.socket.recv())
        except zmq.error.Again:
            self._open_socket()
            raise
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Policy server error: {response['error']}")
        return response

    def ping(self) -> bool:
        return self.call("ping", requires_input=False).get("status") == "ok"

    def get_action(self, observations: dict) -> dict:
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations = observations.copy()
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        return self.call("get_action", observations)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.context.term()


def load_multistep_wrapper(groot_source: Path):
    source = groot_source / "gr00t" / "eval" / "wrappers" / "multistep_wrapper.py"
    spec = importlib.util.spec_from_file_location("locked_groot_multistep_wrapper", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official MultiStepWrapper from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiStepWrapper


def run_task(client: InferenceClient, wrapper_class, task: str, split: str, episodes: int, n_action_steps: int):
    horizon = get_task_horizon(task)

    def create_env():
        env = gym.make(f"robocasa/{task}", split=split, enable_render=True)
        return wrapper_class(
            env,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=n_action_steps,
            max_episode_steps=horizon,
        )

    env = gym.vector.SyncVectorEnv([create_env])
    successes = []
    current_success = False
    obs, _ = env.reset()
    started = time.monotonic()
    try:
        while len(successes) < episodes:
            action_dict = client.get_action(obs)
            actions = action_dict.get("actions", action_dict)
            obs, _, terminations, truncations, infos = env.step(actions)
            current_success |= bool(infos["success"][0][0])
            if bool(terminations[0]) or bool(truncations[0]):
                successes.append(current_success)
                print(
                    f"{task} episode {len(successes)}/{episodes}: success={current_success}",
                    flush=True,
                )
                current_success = False
    finally:
        env.close()
    return {
        "episode_successes": successes,
        "successes": sum(successes),
        "episodes": len(successes),
        "success_rate": float(np.mean(successes)),
        "horizon": horizon,
        "elapsed_seconds": time.monotonic() - started,
    }


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
        raise RuntimeError("Simulator client must run inside an sbatch job")
    config = json.loads(args.config.read_text())
    runtime_root = Path(config["runtime_root"])
    shared_root = Path(config["shared_robocasa_runtime_root"])
    result_path = args.output_dir / "baseline_policy_result.json"
    started = datetime.now(timezone.utc)
    payload = {
        "verdict": "BASELINE_EVAL_RUNNING",
        "created_utc": started.isoformat(),
        "slurm_job_id": job_id,
        "node": platform.node(),
        "runtime_layout": "split_policy_server_and_simulator_client",
        "policy": config["policy"],
        "evaluation": config["evaluation"],
        "gate": config["gate"],
        "task_results": {},
    }
    write_result(result_path, payload)

    client = InferenceClient("localhost", args.port, timeout_ms=1_000)
    try:
        ready = False
        for _ in range(200):
            if not process_alive(args.server_pid):
                state = json.loads(args.server_state.read_text()) if args.server_state.is_file() else {}
                raise RuntimeError(f"Policy server exited before readiness: {state}")
            try:
                if client.ping():
                    ready = True
                    break
            except (zmq.error.ZMQError, RuntimeError):
                pass
            time.sleep(2)
        if not ready:
            raise RuntimeError("Policy server did not become ready within 10 minutes")
        client.timeout_ms = 120_000
        client._open_socket()

        groot_source = runtime_root / "src" / "Isaac-GR00T"
        expected_groot = config["source_revisions"]["isaac_groot"]
        actual_groot = subprocess.check_output(
            ["git", "-C", str(groot_source), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_groot != expected_groot:
            raise RuntimeError(f"GR00T revision mismatch: {actual_groot}")
        wrapper_class = load_multistep_wrapper(groot_source)

        evaluation = config["evaluation"]
        if not set(evaluation["tasks"]).issubset(
            set(TASK_SET_REGISTRY[evaluation["expected_task_set"]])
        ):
            raise RuntimeError("Selected tasks are absent from the locked task set")
        for task in evaluation["tasks"]:
            payload["task_results"][task] = run_task(
                client,
                wrapper_class,
                task,
                evaluation["split"],
                evaluation["episodes_per_task"],
                config["policy"]["n_action_steps"],
            )
            write_result(result_path, payload)

        minimum = config["gate"]["minimum_successes_per_task"]
        passed = all(
            row["successes"] >= minimum for row in payload["task_results"].values()
        )
        payload["verdict"] = "BASELINE_USABLE_FOR_A4" if passed else "BASELINE_NOT_USABLE_STOP_ARENA"
        payload["gate_passed"] = passed
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        payload["elapsed_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
        payload["sim_packages"] = {
            name: importlib.metadata.version(name)
            for name in ("gymnasium", "mujoco", "pyzmq", "robocasa", "robosuite", "torch", "torchvision")
        }
        payload["policy_server_state"] = json.loads(args.server_state.read_text())
        write_result(result_path, payload)
        print(payload["verdict"], flush=True)
    except Exception as error:
        payload["verdict"] = "BASELINE_EVAL_ERROR"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        payload["error"] = repr(error)
        payload["traceback"] = traceback.format_exc()
        if args.server_state.is_file():
            payload["policy_server_state"] = json.loads(args.server_state.read_text())
        write_result(result_path, payload)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
