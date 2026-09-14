#!/usr/bin/env python3
"""Bounded seeded vector/direct differential audit; compute node only.

Uses the released MultiStepWrapper in both paths. This is NOT an independent
reproduction of the full upstream evaluator or an estimate of task success.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
import traceback
from pathlib import Path

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Submit with sbatch; no simulator work on login nodes")

import gymnasium as gym
import numpy as np
import run_baseline_sim_client as base


def digest(obs):
    h = hashlib.sha256()
    for key, value in sorted(obs.items()):
        a = np.asarray(value)
        h.update(key.encode())
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes() if a.dtype.kind not in "OUS" else str(a.tolist()).encode())
    return h.hexdigest()


class Recorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episodes = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        native = self.env.unwrapped.env
        self.record = {
            "initial_observation_sha256": digest(obs),
            "scene": {k: str(getattr(native, k, None)) for k in
                      ("layout_id", "style_id", "control_freq")},
            "steps": [],
        }
        self.episodes.append(self.record)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.record["steps"].append({
            "step": len(self.record["steps"]) + 1,
            "success": bool(info["success"]),
            "action_sha256": digest(action),
            "observation_sha256": digest(obs),
        })
        return obs, reward, done, truncated, info


def rollout(client, wrapper, task, vector, seed):
    random.seed(seed)
    np.random.seed(seed)
    base.torch.manual_seed(seed)
    holder = []

    def factory():
        recorder = Recorder(gym.make(f"robocasa/{task}", split="target",
                                     enable_render=True, seed=seed))
        holder.append(recorder)
        return wrapper(recorder, video_delta_indices=np.array([0]),
                       state_delta_indices=np.array([0]), n_action_steps=16,
                       max_episode_steps=base.get_task_horizon(task))

    env = gym.vector.SyncVectorEnv([factory]) if vector else factory()
    try:
        obs, _ = env.reset(seed=seed)
        record = holder[0].record  # retain pre-autoreset episode
        client.call("set_seed", {"seed": seed + 100000})
        sampled_success = False
        while True:
            batched = obs if vector else {
                k: np.expand_dims(v, 0) if not isinstance(v, str) else (v,)
                for k, v in obs.items()
            }
            result = client.get_action(batched)
            actions = result.get("actions", result)
            if not vector:
                actions = {k: np.asarray(v)[0] for k, v in actions.items()}
            obs, _, done, truncated, info = env.step(actions)
            sampled_success |= bool(np.asarray(info["success"]).reshape(-1)[0])
            if bool(np.asarray(done).any()) or bool(np.asarray(truncated).any()):
                break
        successes = [s["step"] for s in record["steps"] if s["success"]]
        record.update(mode="vector" if vector else "direct", env_seed=seed,
                      policy_seed=seed + 100000, sampled_success=sampled_success,
                      native_any_success=bool(successes),
                      first_success_step=successes[0] if successes else None)
        return record
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--server-pid", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    out = args.output_dir / "execution_audit_result.json"
    report = {"job_id": os.environ["SLURM_JOB_ID"], "verdict": "RUNNING",
              "scope": "one seeded episode per task, vector vs direct; cadence 16",
              "config": config, "tasks": {},
              "packages": {n: importlib.metadata.version(n) for n in
                           ("gymnasium", "mujoco", "robocasa", "robosuite", "torch")}}
    base.write_result(out, report)
    client = base.InferenceClient("localhost", args.port, timeout_ms=1000)
    try:
        for _ in range(200):
            if not base.process_alive(args.server_pid):
                raise RuntimeError("Policy server died during startup")
            try:
                if client.ping():
                    break
            except base.zmq.error.ZMQError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("Server readiness timeout")
        client.timeout_ms = 120000
        client._open_socket()
        wrapper = base.load_multistep_wrapper(Path(config["runtime_root"]) / "src/Isaac-GR00T")
        for index, task in enumerate(config["evaluation"]["tasks"]):
            records = []
            report["tasks"][task] = {"records": records}
            for vector in (True, False):
                records.append(rollout(client, wrapper, task, vector, 204000 + index))
                base.write_result(out, report)
            a, b = records
            report["tasks"][task]["pass"] = (
                a["initial_observation_sha256"] == b["initial_observation_sha256"]
                and a["steps"] == b["steps"]
                and all(r["sampled_success"] == r["native_any_success"] for r in records))
            base.write_result(out, report)
        report["verdict"] = "PASS" if all(r["pass"] for r in report["tasks"].values()) else "MISMATCH"
        base.write_result(out, report)
        if report["verdict"] != "PASS":
            raise RuntimeError("Differential audit mismatch; inspect saved traces before scaling")
    except Exception:
        report["error"] = traceback.format_exc()
        if report["verdict"] == "RUNNING":
            report["verdict"] = "ERROR"
        base.write_result(out, report)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
