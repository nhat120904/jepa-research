#!/usr/bin/env python3
"""Profile Stage-B one-intervention oracle headroom with the frozen GR00T policy."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import robocasa  # noqa: F401 - registers gym environments
import robosuite  # noqa: F401
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.dataset_registry_utils import get_task_horizon

from run_baseline_sim_client import InferenceClient, load_multistep_wrapper
from run_stage_a4_collect import batch_observation, jsonable, task_history, wait_for_server
from run_stage_a4_calibrate import progress_label, state_max_abs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def image_max_abs(first: np.ndarray, second: np.ndarray) -> int:
    if first.shape != second.shape:
        return 2**31 - 1
    return int(np.abs(first.astype(np.int16) - second.astype(np.int16)).max(initial=0))


def create_env(wrapper_class, task: str, split: str, chunk_steps: int):
    horizon = get_task_horizon(task)
    base = gym.make(f"robocasa/{task}", split=split, enable_render=True)
    wrapped = wrapper_class(
        base,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=chunk_steps,
        max_episode_steps=horizon,
    )
    return wrapped, wrapped.unwrapped, wrapped.unwrapped.env, horizon


def policy_action_chunk(client: InferenceClient, obs: dict, chunk_steps: int) -> dict[str, np.ndarray]:
    predicted = client.get_action(batch_observation(obs))
    batched = predicted.get("actions", predicted)
    chunk = {key: np.asarray(value)[0, :chunk_steps].copy() for key, value in batched.items()}
    if any(value.shape[0] != chunk_steps for value in chunk.values()):
        raise RuntimeError("Policy returnedLess than the locked candidate chunk length")
    return chunk


def progress_score(task: str, label: dict) -> float:
    """Map native latched milestones to [0, 1], excluding terminal success."""
    if task == "ScrubCuttingBoard":
        contact = min(float(label["contact_milestone"]), 5.0) / 5.0
        sweep = float(label["sweep_milestone"])
        return 0.5 * contact + 0.5 * sweep
    if task == "RinseSinkBasin":
        return float(label["washed_count"]) / 3.0
    raise ValueError(f"Unsupported Stage-B task: {task}")


def flatten_chunk(chunk: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    steps = next(iter(chunk.values())).shape[0]
    return [
        {key: value[step].copy() for key, value in chunk.items()}
        for step in range(steps)
    ]


def stack_actions(actions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([action[key] for action in actions])
        for key in actions[0]
    }


def sync_wrapper_after_reset(wrapped, gym_env, task_env) -> dict:
    obs = gym_env.get_observation(task_env._get_observations(force_update=True))
    wrapped.obs = deque([obs] * (wrapped.max_steps_needed + 1), maxlen=wrapped.max_steps_needed + 1)
    wrapped.reward = []
    wrapped.done = []
    wrapped.info = defaultdict(lambda: deque(maxlen=wrapped.max_steps_needed + 1))
    return wrapped._get_obs(wrapped.video_delta_indices, wrapped.state_delta_indices)


def reset_carrier(wrapped, gym_env, task_env, source: dict) -> dict:
    wrapped.reset(seed=source["seed"])
    task_env.rng = np.random.default_rng(source["seed"])
    reset_to(
        task_env,
        {
            "model": source["model_xml"],
            "ep_meta": json.dumps(source["ep_meta"]),
            "states": source["initial_state"],
        },
    )
    return sync_wrapper_after_reset(wrapped, gym_env, task_env)


def replay_prefix(wrapped, obs: dict, prefix_actions: list[dict[str, np.ndarray]], chunk_steps: int):
    if len(prefix_actions) % chunk_steps:
        raise RuntimeError("Prefix length is not divisible by the locked chunk length")
    success = False
    for start in range(0, len(prefix_actions), chunk_steps):
        obs, reward, terminated, truncated, info = wrapped.step(
            stack_actions(prefix_actions[start : start + chunk_steps])
        )
        success = success or bool(reward)
        if terminated or truncated:
            raise RuntimeError("Carrier terminated before the intervention prefix")
    return obs, success


def capture_images(obs: dict, camera_keys: list[str]) -> dict[str, np.ndarray]:
    return {key: np.asarray(obs[key][-1]).copy() for key in camera_keys if key in obs}


def generate_source_prefix(
    client,
    wrapper_class,
    task_cfg,
    profile,
    task_index: int,
    attempt: int,
):
    task = task_cfg["name"]
    seed = profile["environment_seed_base"] + task_index * 10_000 + attempt
    wrapped, gym_env, task_env, horizon = create_env(
        wrapper_class, task, profile["split"], profile["candidate_chunk_steps"]
    )
    try:
        obs, _ = wrapped.reset(seed=seed)
        source = {
            "seed": seed,
            "model_xml": task_env.sim.model.get_xml(),
            "ep_meta": task_env.get_ep_meta(),
            "initial_state": np.asarray(task_env.sim.get_state().flatten()).copy(),
        }
        chunk_steps = profile["candidate_chunk_steps"]
        prefix_actions: list[dict[str, np.ndarray]] = []
        prefix_success = False
        set_policy_seed(client, profile["proposal_seed_base"] - 10_000 + seed)
        if profile.get("anchor_mode", "fixed_fraction") == "event_aligned":
            previous_label = progress_label(task, task_env, False)
            previous_score = progress_score(task, previous_label)
            events = []
            rollout_success = False
            terminated = truncated = False
            while not (terminated or truncated):
                chunk = policy_action_chunk(client, obs, chunk_steps)
                obs, reward, terminated, truncated, info = wrapped.step(chunk)
                succeeded = bool(info["success"][-1])
                next_label = progress_label(task, task_env, succeeded)
                next_score = progress_score(task, next_label)
                if next_score > previous_score + 1e-12:
                    fraction = len(prefix_actions) / float(horizon)
                    if (
                        fraction >= profile["event_anchor_min_fraction"]
                        and fraction <= profile["event_anchor_max_fraction"]
                    ):
                        events.append(
                            {
                                "prefix_actions": list(prefix_actions),
                                "native_step": len(prefix_actions),
                                "fraction": fraction,
                                "before_label": previous_label,
                                "after_label": next_label,
                                "before_score": previous_score,
                                "after_score": next_score,
                            }
                        )
                prefix_actions.extend(flatten_chunk(chunk))
                rollout_success = rollout_success or bool(reward)
                previous_label = next_label
                previous_score = next_score
            source["source_rollout_success"] = rollout_success
            source["source_events"] = [
                {key: value for key, value in event.items() if key != "prefix_actions"}
                for event in events
            ]
            if not events or (
                profile.get("event_anchor_require_source_success", True)
                and not rollout_success
            ):
                source["prefix_actions"] = []
                source["source_prefix_success"] = False
                source["anchor_found"] = False
                source["horizon"] = horizon
                return source
            selected = events[-1]
            prefix_actions = selected["prefix_actions"]
            source["anchor_found"] = True
            source["anchor_event"] = {
                key: value for key, value in selected.items() if key != "prefix_actions"
            }
        else:
            target_steps = int(horizon * profile["prefix_fraction"])
            target_steps -= target_steps % chunk_steps
            while len(prefix_actions) < target_steps:
                chunk = policy_action_chunk(client, obs, chunk_steps)
                obs, reward, terminated, truncated, _ = wrapped.step(chunk)
                prefix_actions.extend(flatten_chunk(chunk))
                prefix_success = prefix_success or bool(reward)
                if terminated or truncated:
                    break
            source["anchor_found"] = True
        source["prefix_actions"] = prefix_actions
        source["source_prefix_success"] = prefix_success
        source["horizon"] = horizon
        return source
    finally:
        wrapped.close()


def reconstruct_canonical(wrapper_class, task_cfg, profile, source):
    wrapped, gym_env, task_env, _ = create_env(
        wrapper_class,
        task_cfg["name"],
        profile["split"],
        profile["candidate_chunk_steps"],
    )
    obs = reset_carrier(wrapped, gym_env, task_env, source)
    obs, success = replay_prefix(
        wrapped, obs, source["prefix_actions"], profile["candidate_chunk_steps"]
    )
    return wrapped, gym_env, task_env, obs, success


def set_policy_seed(client: InferenceClient, seed: int) -> None:
    response = client.call("set_seed", {"seed": int(seed)})
    if int(response["seed"]) != int(seed):
        raise RuntimeError("Policy server did not accept the requested seed")


def run_candidate(
    client,
    wrapper_class,
    task_cfg,
    profile,
    source,
    canonical,
    candidate_chunk,
    continuation_seed: int,
):
    task = task_cfg["name"]
    camera_keys = profile["camera_keys"]
    chunk_steps = profile["candidate_chunk_steps"]
    wrapped, gym_env, task_env, obs, prefix_success = reconstruct_canonical(
        wrapper_class, task_cfg, profile, source
    )
    started = time.monotonic()
    try:
        before_state = np.asarray(task_env.sim.get_state().flatten()).copy()
        before_history = task_history(task_env, task_cfg["history_fields"])
        before_images = capture_images(obs, camera_keys)
        prefix_comparison = {
            "state_max_abs": state_max_abs(before_state, canonical["state"]),
            "history_equal": before_history == canonical["history"],
            "image_max_abs": {
                key: image_max_abs(before_images[key], canonical["images"][key])
                for key in set(before_images) & set(canonical["images"])
            },
            "camera_keys_equal": set(before_images) == set(canonical["images"]) == set(camera_keys),
        }
        prefix_comparison["exact"] = (
            prefix_comparison["state_max_abs"] <= profile["state_max_abs_tolerance"]
            and prefix_comparison["history_equal"]
            and prefix_comparison["camera_keys_equal"]
            and max(prefix_comparison["image_max_abs"].values(), default=2**31 - 1) == 0
        )
        if prefix_success:
            raise RuntimeError(f"{task} canonical prefix already succeeded")

        obs, reward, terminated, truncated, info = wrapped.step(candidate_chunk)
        success = bool(reward)
        post_state = np.asarray(task_env.sim.get_state().flatten()).copy()
        post_history = task_history(task_env, task_cfg["history_fields"])
        post_images = capture_images(obs, camera_keys)
        progress = [
            {
                "native_step": len(wrapped.reward),
                "history": post_history,
                "label": progress_label(task, task_env, bool(info["success"][-1])),
            }
        ]
        set_policy_seed(client, continuation_seed)
        while not (terminated or truncated):
            continuation = policy_action_chunk(client, obs, chunk_steps)
            obs, reward, terminated, truncated, info = wrapped.step(continuation)
            success = success or bool(reward)
            progress.append(
                {
                    "native_step": len(wrapped.reward),
                    "history": task_history(task_env, task_cfg["history_fields"]),
                    "label": progress_label(task, task_env, bool(info["success"][-1])),
                }
            )
        final_state = np.asarray(task_env.sim.get_state().flatten()).copy()
        final_images = capture_images(obs, camera_keys)
        return {
            "success": success,
            "prefix_comparison": prefix_comparison,
            "post_state": post_state,
            "post_history": post_history,
            "final_state": final_state,
            "final_history": task_history(task_env, task_cfg["history_fields"]),
            "native_steps": len(wrapped.reward),
            "progress": progress,
            "elapsed_seconds": time.monotonic() - started,
            "post_images": post_images,
            "final_images": final_images,
        }
    finally:
        wrapped.close()


def run_prefix(
    client,
    wrapper_class,
    task_cfg,
    profile,
    task_index: int,
    prefix_index: int,
    source: dict,
    output_dir: Path,
):
    task = task_cfg["name"]
    wrapped, gym_env, task_env, obs, canonical_success = reconstruct_canonical(
        wrapper_class, task_cfg, profile, source
    )
    try:
        if canonical_success:
            return None
        canonical = {
            "state": np.asarray(task_env.sim.get_state().flatten()).copy(),
            "history": task_history(task_env, task_cfg["history_fields"]),
            "images": capture_images(obs, profile["camera_keys"]),
        }
        if set(canonical["images"]) != set(profile["camera_keys"]):
            raise RuntimeError(f"Missing canonical cameras for {task}")
        candidate_chunks = []
        for candidate_index in range(profile["candidate_chunks_per_prefix"]):
            proposal_seed = (
                profile["proposal_seed_base"]
                + task_index * 100_000
                + prefix_index * 100
                + candidate_index
            )
            set_policy_seed(client, proposal_seed)
            candidate_chunks.append(
                policy_action_chunk(client, obs, profile["candidate_chunk_steps"])
            )
    finally:
        wrapped.close()

    prefix_dir = output_dir / "prefixes" / task / f"prefix_{prefix_index:02d}"
    prefix_dir.mkdir(parents=True, exist_ok=False)
    with gzip.open(prefix_dir / "model.xml.gz", "wt", encoding="utf-8") as handle:
        handle.write(source["model_xml"])
    write_json(prefix_dir / "ep_meta.json", source["ep_meta"])
    arrays = {
        "initial_state": source["initial_state"],
        "canonical_prefix_state": canonical["state"],
    }
    for camera_key, value in canonical["images"].items():
        arrays[f"canonical_obs::{camera_key}"] = value
    for action_key in source["prefix_actions"][0]:
        arrays[f"prefix_action::{action_key}"] = np.stack(
            [action[action_key] for action in source["prefix_actions"]]
        )

    rows = []
    continuation_seed = (
        profile["continuation_seed_base"] + task_index * 10_000 + prefix_index
    )
    progress_path = prefix_dir / "progress.json"
    for candidate_index, candidate_chunk in enumerate(candidate_chunks):
        row = run_candidate(
            client,
            wrapper_class,
            task_cfg,
            profile,
            source,
            canonical,
            candidate_chunk,
            continuation_seed,
        )
        row["candidate_index"] = candidate_index
        rows.append(row)
        for key, value in candidate_chunk.items():
            arrays[f"candidate_{candidate_index}::action::{key}"] = value
        arrays[f"candidate_{candidate_index}::post_state"] = row.pop("post_state")
        arrays[f"candidate_{candidate_index}::final_state"] = row.pop("final_state")
        for camera_key, value in row.pop("post_images").items():
            arrays[f"candidate_{candidate_index}::post_obs::{camera_key}"] = value
        for camera_key, value in row.pop("final_images").items():
            arrays[f"candidate_{candidate_index}::final_obs::{camera_key}"] = value
        write_json(progress_path, {"candidates": rows})
        print(
            f"{task} prefix {prefix_index + 1} candidate {candidate_index + 1}/"
            f"{len(candidate_chunks)} success={row['success']}",
            flush=True,
        )

    post_states = [arrays[f"candidate_{index}::post_state"] for index in range(len(rows))]
    diversity = max(
        (
            state_max_abs(post_states[first], post_states[second])
            for first in range(len(rows))
            for second in range(first + 1, len(rows))
        ),
        default=0.0,
    )
    np.savez_compressed(prefix_dir / "branches.npz", **arrays)
    prefix_row = {
        "task": task,
        "prefix_index": prefix_index,
        "environment_seed": source["seed"],
        "prefix_native_steps": len(source["prefix_actions"]),
        "canonical_history": canonical["history"],
        "candidate_successes": [bool(row["success"]) for row in rows],
        "baseline_candidate_success": bool(rows[0]["success"]),
        "oracle_candidate_success": any(row["success"] for row in rows),
        "outcome_varies": len({bool(row["success"]) for row in rows}) > 1,
        "candidate_max_progress_scores": [
            max(progress_score(task, point["label"]) for point in row["progress"])
            for row in rows
        ],
        "candidate_state_diversity": diversity,
        "candidate_state_diverse": diversity >= profile["minimum_candidate_state_diversity"],
        "all_prefix_carriers_exact": all(row["prefix_comparison"]["exact"] for row in rows),
        "candidate_elapsed_seconds": [row["elapsed_seconds"] for row in rows],
        "artifact": str(prefix_dir / "branches.npz"),
        "progress_artifact": str(progress_path),
    }
    scores = prefix_row["candidate_max_progress_scores"]
    prefix_row["baseline_candidate_max_progress_score"] = scores[0]
    prefix_row["oracle_candidate_max_progress_score"] = max(scores)
    prefix_row["progress_oracle_gain"] = max(scores) - scores[0]
    prefix_row["progress_outcome_varies"] = max(scores) - min(scores) > 1e-12
    if "anchor_event" in source:
        prefix_row["anchor_event"] = source["anchor_event"]
    write_json(prefix_dir / "metadata.json", prefix_row)
    return prefix_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--server-state", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--prefixes-per-task", type=int)
    args = parser.parse_args()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Stage-B profiling must run inside an sbatch job")

    config = json.loads(args.config.read_text())
    profile = config["profile"]
    if args.candidate_count is not None:
        if args.candidate_count not in (8, 32):
            raise ValueError("The locked candidate-count options are 8 and 32")
        profile["candidate_chunks_per_prefix"] = args.candidate_count
    if args.prefixes_per_task is not None:
        if args.prefixes_per_task < 1:
            raise ValueError("prefixes-per-task must be positive")
        profile["prefixes_per_task"] = args.prefixes_per_task
    result_path = args.output_dir / "stage_b_profile_result.json"
    result = {
        "verdict": "B0_PROFILE_RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": job_id,
        "node": platform.node(),
        "config": config,
        "tasks": {},
    }
    write_json(result_path, result)
    started = time.monotonic()
    client = InferenceClient("localhost", args.port, timeout_ms=1_000)
    try:
        wait_for_server(client, args.server_pid, args.server_state)
        wrapper_class = load_multistep_wrapper(
            Path(config["runtime_root"]) / "src" / "Isaac-GR00T"
        )
        for task_index, task_cfg in enumerate(profile["tasks"]):
            task = task_cfg["name"]
            task_row = {"prefixes": [], "attempts": []}
            result["tasks"][task] = task_row
            for attempt in range(profile["maximum_prefix_attempts_per_task"]):
                source = generate_source_prefix(
                    client, wrapper_class, task_cfg, profile, task_index, attempt
                )
                task_row["attempts"].append(
                    {
                        "attempt": attempt,
                        "seed": source["seed"],
                        "source_prefix_success": source["source_prefix_success"],
                        "source_rollout_success": source.get("source_rollout_success"),
                        "anchor_found": source.get("anchor_found", True),
                        "source_events": source.get("source_events"),
                    }
                )
                if not source.get("anchor_found", True):
                    write_json(result_path, result)
                    continue
                row = run_prefix(
                    client,
                    wrapper_class,
                    task_cfg,
                    profile,
                    task_index,
                    len(task_row["prefixes"]),
                    source,
                    args.output_dir,
                )
                if row is not None:
                    task_row["prefixes"].append(row)
                    write_json(result_path, result)
                if len(task_row["prefixes"]) == profile["prefixes_per_task"]:
                    break
            if len(task_row["prefixes"]) != profile["prefixes_per_task"]:
                raise RuntimeError(f"Could not produce enough unsolved prefixes for {task}")

        prefixes = [row for task in result["tasks"].values() for row in task["prefixes"]]
        baseline_rate = float(np.mean([row["baseline_candidate_success"] for row in prefixes]))
        oracle_rate = float(np.mean([row["oracle_candidate_success"] for row in prefixes]))
        baseline_progress = float(
            np.mean([row["baseline_candidate_max_progress_score"] for row in prefixes])
        )
        oracle_progress = float(
            np.mean([row["oracle_candidate_max_progress_score"] for row in prefixes])
        )
        elapsed = time.monotonic() - started
        projected = (
            elapsed
            * config["expansion"]["validation_prefixes_per_task"]
            / profile["prefixes_per_task"]
            / 3600.0
        )
        result["summary"] = {
            "prefixes": len(prefixes),
            "candidate_branches": len(prefixes) * profile["candidate_chunks_per_prefix"],
            "baseline_candidate_success_rate": baseline_rate,
            "oracle_candidate_success_rate": oracle_rate,
            "profile_oracle_gain_percentage_points": 100.0 * (oracle_rate - baseline_rate),
            "prefixes_with_outcome_variation": sum(row["outcome_varies"] for row in prefixes),
            "baseline_candidate_max_progress_score": baseline_progress,
            "oracle_candidate_max_progress_score": oracle_progress,
            "profile_progress_oracle_gain_percentage_points": 100.0
            * (oracle_progress - baseline_progress),
            "prefixes_with_progress_variation": sum(
                row["progress_outcome_varies"] for row in prefixes
            ),
            "prefixes_with_positive_progress_headroom": sum(
                row["progress_oracle_gain"] > 1e-12 for row in prefixes
            ),
            "all_prefix_carriers_exact": all(row["all_prefix_carriers_exact"] for row in prefixes),
            "all_candidate_states_diverse": all(row["candidate_state_diverse"] for row in prefixes),
            "elapsed_seconds": elapsed,
            "projected_serial_gpu_hours_for_40_prefixes_per_task": projected,
            "within_additional_gpu_hour_cap": projected <= config["expansion"]["additional_gpu_hour_cap"],
        }
        if not result["summary"]["all_prefix_carriers_exact"]:
            result["verdict"] = "B0_PREFIX_RECONSTRUCTION_FAIL"
        elif not result["summary"]["all_candidate_states_diverse"]:
            result["verdict"] = "B0_NO_ACTION_DIVERSITY"
        elif not result["summary"]["within_additional_gpu_hour_cap"]:
            result["verdict"] = "B0_EXPANSION_OVER_BUDGET"
        else:
            result["verdict"] = "B0_PROFILE_PASS_EXPANSION_READY"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["policy_server_state"] = json.loads(args.server_state.read_text())
        write_json(result_path, result)
        print(result["verdict"], flush=True)
    except Exception as error:
        result["verdict"] = "B0_PROFILE_ERROR"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        write_json(result_path, result)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
