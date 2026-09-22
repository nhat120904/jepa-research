#!/usr/bin/env python3
"""Record GR00T ScrubCuttingBoard rollouts as video with the latched success predicate overlaid.

The point of this script is diagnostic, not statistical: it renders what the released
policy actually does and annotates, frame by frame, the three cumulative conditions the
native RoboCasa evaluator requires (>=5 distinct board contacts, >=0.10 m sweep extent,
sponge released >0.15 m from the gripper). A failing episode is then attributable to a
named milestone rather than to "the policy did not succeed".
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

import robocasa  # noqa: F401 - registers the native Gym environments
import robosuite  # noqa: F401
import robocasa.utils.object_utils as OU
from robocasa.utils.dataset_registry_utils import get_task_horizon

from run_baseline_sim_client import InferenceClient, load_multistep_wrapper, process_alive
from run_stage_a4_collect import batch_observation, jsonable, wait_for_server, write_json

CONTACT_MIN = 5
SWEEP_MIN = 0.1
RELEASE_TH = 0.15


def sweep_range(positions) -> float:
    if not len(positions):
        return 0.0
    array = np.asarray(positions)
    return float(np.linalg.norm(array.max(axis=0) - array.min(axis=0)))


class FrameTap(gym.Wrapper):
    """Capture one frame and one ground-truth progress probe per native control step."""

    def __init__(self, env, cameras, stride):
        super().__init__(env)
        self.cameras = cameras
        self.stride = stride
        self.frames: list[np.ndarray] = []
        self.probes: list[dict] = []
        self._native_step = 0

    @property
    def task_env(self):
        return self.env.unwrapped.env

    def _probe(self) -> dict:
        task_env = self.task_env
        positions = list(getattr(task_env, "board_contact_positions", []))
        contacts = int(getattr(task_env, "board_contact_timer", 0))
        extent = sweep_range(positions)
        return {
            "native_step": self._native_step,
            "contacts": contacts,
            "sweep": extent,
            "grasped": bool(OU.check_obj_grasped(task_env, "sponge")),
            "gripper_far": bool(OU.gripper_obj_far(task_env, "sponge", th=RELEASE_TH)),
            "contacts_ok": contacts >= CONTACT_MIN,
            "sweep_ok": extent >= SWEEP_MIN,
            "success": bool(task_env._check_success()),
        }

    def _capture(self, observation, probe) -> None:
        tiles = [np.asarray(observation[key], dtype=np.uint8) for key in self.cameras]
        frame = np.concatenate(tiles, axis=1)
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        self.frames.append(annotate(frame, probe))

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.frames = []
        self.probes = []
        self._native_step = 0
        probe = self._probe()
        self.probes.append(probe)
        self._capture(observation, probe)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self._native_step += 1
        probe = self._probe()
        self.probes.append(probe)
        if self._native_step % self.stride == 0:
            self._capture(observation, probe)
        return observation, reward, terminated, truncated, info


def annotate(frame: np.ndarray, probe: dict) -> np.ndarray:
    height, width = frame.shape[:2]
    bar = np.zeros((132, width, 3), dtype=np.uint8)
    canvas = np.concatenate([frame, bar], axis=0)
    ok = (90, 220, 90)
    bad = (90, 90, 235)
    grey = (190, 190, 190)

    def line(text, row, color):
        cv2.putText(
            canvas,
            text,
            (12, height + row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    line(f"native step {probe['native_step']:4d}   grasped={probe['grasped']}", 24, grey)
    line(
        f"[1] distinct board contacts {probe['contacts']:2d} / {CONTACT_MIN}",
        50,
        ok if probe["contacts_ok"] else bad,
    )
    line(
        f"[2] sweep extent {probe['sweep']:.3f} / {SWEEP_MIN:.2f} m",
        74,
        ok if probe["sweep_ok"] else bad,
    )
    line(
        f"[3] sponge released >{RELEASE_TH:.2f} m  {probe['gripper_far']}",
        98,
        ok if probe["gripper_far"] else bad,
    )
    line(
        "SUCCESS" if probe["success"] else "not yet successful",
        122,
        ok if probe["success"] else bad,
    )
    return canvas


def classify(probes: list[dict]) -> tuple[str, str]:
    final = probes[-1]
    if any(p["success"] for p in probes):
        return "SUCCESS", "All three cumulative conditions held simultaneously."
    peak_contacts = max(p["contacts"] for p in probes)
    peak_sweep = max(p["sweep"] for p in probes)
    ever_grasped = any(p["grasped"] for p in probes)
    if not ever_grasped:
        return "NEVER_GRASPED", "The sponge was never grasped, so no contact can be latched."
    if peak_contacts < CONTACT_MIN and peak_sweep < SWEEP_MIN:
        return (
            "NO_SCRUB_COVERAGE",
            f"Only {peak_contacts} distinct contacts over {peak_sweep:.3f} m; the scrub never happened.",
        )
    if peak_contacts < CONTACT_MIN:
        return (
            "CONTACT_COUNT_SHORT",
            f"Swept {peak_sweep:.3f} m but latched only {peak_contacts}/{CONTACT_MIN} distinct contacts.",
        )
    if peak_sweep < SWEEP_MIN:
        return (
            "SWEEP_EXTENT_SHORT",
            f"Latched {peak_contacts} contacts but confined to {peak_sweep:.3f} m; scrubbed one spot.",
        )
    return (
        "NEVER_RELEASED",
        "Both scrub conditions were latched; the policy never put the sponge down far enough.",
    )


def run_episode(client, wrapper_class, task, split, n_action_steps, seed, cameras, stride):
    horizon = get_task_horizon(task)
    base = gym.make(f"robocasa/{task}", split=split, enable_render=True)
    tap = FrameTap(base, cameras, stride)
    wrapped = wrapper_class(
        tap,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=n_action_steps,
        max_episode_steps=horizon,
    )
    started = time.monotonic()
    try:
        observation, _ = wrapped.reset(seed=seed)
        chunks = 0
        while True:
            predicted = client.get_action(batch_observation(observation))
            batched = predicted.get("actions", predicted)
            chunk = {key: np.asarray(value)[0].copy() for key, value in batched.items()}
            observation, _, terminated, truncated, _ = wrapped.step(chunk)
            chunks += 1
            if bool(terminated) or bool(truncated):
                break
        verdict, explanation = classify(tap.probes)
        final = tap.probes[-1]
        return {
            "seed": seed,
            "task": task,
            "horizon": horizon,
            "chunks": chunks,
            "native_steps": final["native_step"],
            "success": verdict == "SUCCESS",
            "failure_mode": verdict,
            "explanation": explanation,
            "peak_contacts": max(p["contacts"] for p in tap.probes),
            "peak_sweep": max(p["sweep"] for p in tap.probes),
            "ever_grasped": any(p["grasped"] for p in tap.probes),
            "ever_released": any(p["gripper_far"] for p in tap.probes),
            "steps_with_scrub_conditions_met": sum(
                1 for p in tap.probes if p["contacts_ok"] and p["sweep_ok"]
            ),
            "steps_scrub_met_but_holding": sum(
                1
                for p in tap.probes
                if p["contacts_ok"] and p["sweep_ok"] and not p["gripper_far"]
            ),
            "final": final,
            "elapsed_seconds": time.monotonic() - started,
        }, tap.frames, tap.probes
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
        raise RuntimeError("Failure demo must run inside an sbatch job")
    config = json.loads(args.config.read_text())
    demo = config["demo"]
    result_path = args.output_dir / "failure_demo_result.json"
    video_dir = args.output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "verdict": "FAILURE_DEMO_RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": job_id,
        "node": platform.node(),
        "config": config,
        "episodes": [],
    }
    write_json(result_path, result)

    client = InferenceClient("localhost", args.port, timeout_ms=1_000)
    try:
        wait_for_server(client, args.server_pid, args.server_state)
        wrapper_class = load_multistep_wrapper(
            Path(config["runtime_root"]) / "src" / "Isaac-GR00T"
        )
        for index in range(demo["episodes"]):
            seed = demo["seed_base"] + index
            row, frames, probes = run_episode(
                client,
                wrapper_class,
                demo["task"],
                demo["split"],
                config["policy"]["n_action_steps"],
                seed,
                demo["cameras"],
                demo["frame_stride"],
            )
            tag = "success" if row["success"] else row["failure_mode"].lower()
            name = f"ep{index:02d}_seed{seed}_{tag}"
            video_path = video_dir / f"{name}.mp4"
            imageio.mimwrite(
                video_path,
                frames,
                fps=demo["fps"],
                codec="libx264",
                quality=7,
                macro_block_size=1,
            )
            np.savez_compressed(
                video_dir / f"{name}_probes.npz",
                **{
                    key: np.asarray([p[key] for p in probes])
                    for key in ("native_step", "contacts", "sweep", "grasped", "gripper_far", "success")
                },
            )
            row["video"] = str(video_path)
            row["frames"] = len(frames)
            result["episodes"].append(row)
            print(f"{name}: {row['explanation']}", flush=True)
            write_json(result_path, result)

        episodes = result["episodes"]
        modes: dict[str, int] = {}
        for row in episodes:
            modes[row["failure_mode"]] = modes.get(row["failure_mode"], 0) + 1
        result["summary"] = {
            "episodes": len(episodes),
            "successes": sum(1 for row in episodes if row["success"]),
            "failure_modes": modes,
        }
        result["verdict"] = "FAILURE_DEMO_COMPLETE"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["policy_server_state"] = json.loads(args.server_state.read_text())
        write_json(result_path, result)
        print(json.dumps(jsonable(result["summary"]), indent=2), flush=True)
    except Exception as error:
        result["verdict"] = "FAILURE_DEMO_ERROR"
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        write_json(result_path, result)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
