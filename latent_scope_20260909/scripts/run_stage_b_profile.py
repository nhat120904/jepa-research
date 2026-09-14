#!/usr/bin/env python3
"""Profile Stage-B one-intervention oracle headroom with the frozen GR00T policy."""

from __future__ import annotations

import argparse
import copy
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


def image_alignment_metrics(first: np.ndarray, second: np.ndarray) -> dict:
    if first.shape != second.shape:
        return {
            "shape_equal": False,
            "max_abs": 2**31 - 1,
            "normalized_mae": float("inf"),
            "p99_abs": float("inf"),
        }
    delta = np.abs(first.astype(np.float32) - second.astype(np.float32))
    return {
        "shape_equal": True,
        "max_abs": int(delta.max(initial=0)),
        "normalized_mae": float(delta.mean() / 255.0),
        "p99_abs": float(np.quantile(delta, 0.99)),
    }


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


def continuation_cadence(profile: dict) -> int:
    return int(profile.get("continuation_cadence_steps", profile["candidate_chunk_steps"]))


def intervention_horizons(profile: dict) -> list[int]:
    values = profile.get("candidate_intervention_steps", [profile["candidate_chunk_steps"]])
    horizons = [int(value) for value in values]
    cadence = continuation_cadence(profile)
    if not horizons or any(value <= 0 or value % cadence for value in horizons):
        raise ValueError("Every candidate intervention horizon must be a positive cadence multiple")
    if horizons != sorted(set(horizons)):
        raise ValueError("Candidate intervention horizons must be sorted and unique")
    return horizons


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


def restore_task_history(task_env, saved_history: dict) -> None:
    for field, value in saved_history.items():
        if field == "board_contact_positions":
            value = [np.asarray(position, dtype=np.float64).copy() for position in value]
        else:
            value = copy.deepcopy(value)
        setattr(task_env, field, value)


def reset_to_event(wrapped, gym_env, task_env, source: dict) -> dict:
    wrapped.reset(seed=source["seed"])
    task_env.rng = np.random.default_rng(source["seed"])
    reset_to(
        task_env,
        {
            "model": source["model_xml"],
            "ep_meta": json.dumps(source["ep_meta"]),
            "states": source["anchor_state"],
        },
    )
    restore_task_history(task_env, source["anchor_history"])
    obs = sync_wrapper_after_reset(wrapped, gym_env, task_env)
    remaining_steps = int(source["horizon"]) - int(source["anchor_event"]["native_step"])
    if remaining_steps <= 0:
        raise RuntimeError("Event snapshot has no remaining native horizon")
    wrapped.max_episode_steps = remaining_steps
    return obs


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


def capture_fresh_images(gym_env, task_env, camera_keys: list[str]) -> dict[str, np.ndarray]:
    """Render cameras from the current simulator state, bypassing wrapper observation caches."""
    obs = gym_env.get_observation(task_env._get_observations(force_update=True))
    return {key: np.asarray(obs[key]).copy() for key in camera_keys if key in obs}


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
    cadence = continuation_cadence(profile)
    wrapped, gym_env, task_env, horizon = create_env(
        wrapper_class, task, profile["split"], cadence
    )
    try:
        obs, _ = wrapped.reset(seed=seed)
        source = {
            "seed": seed,
            "model_xml": task_env.sim.model.get_xml(),
            "ep_meta": task_env.get_ep_meta(),
            "initial_state": np.asarray(task_env.sim.get_state().flatten()).copy(),
        }
        chunk_steps = cadence
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
                before_images = capture_fresh_images(
                    gym_env, task_env, profile["camera_keys"]
                )
                before_state = np.asarray(task_env.sim.get_state().flatten()).copy()
                before_history = task_history(task_env, task_cfg["history_fields"])
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
                                "anchor_state": before_state,
                                "anchor_history": before_history,
                                "anchor_images": before_images,
                            }
                        )
                prefix_actions.extend(flatten_chunk(chunk))
                rollout_success = rollout_success or bool(reward)
                previous_label = next_label
                previous_score = next_score
            source["source_rollout_success"] = rollout_success
            source["source_events"] = [
                {
                    key: value
                    for key, value in event.items()
                    if key
                    not in ("prefix_actions", "anchor_state", "anchor_history", "anchor_images")
                }
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
                key: value
                for key, value in selected.items()
                if key not in ("prefix_actions", "anchor_state", "anchor_history", "anchor_images")
            }
            source["anchor_state"] = selected["anchor_state"]
            source["anchor_history"] = selected["anchor_history"]
            source["anchor_images"] = selected["anchor_images"]
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
    cadence = continuation_cadence(profile)
    wrapped, gym_env, task_env, _ = create_env(
        wrapper_class,
        task_cfg["name"],
        profile["split"],
        cadence,
    )
    if profile.get("anchor_mode", "fixed_fraction") == "event_aligned":
        obs = reset_to_event(wrapped, gym_env, task_env, source)
        success = bool(source["anchor_event"]["before_label"].get("success", 0)) or bool(
            source["anchor_event"]["before_label"].get("released_success", 0)
        )
    else:
        obs = reset_carrier(wrapped, gym_env, task_env, source)
        obs, success = replay_prefix(
            wrapped, obs, source["prefix_actions"], cadence
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
    candidate_bank,
    intervention_steps: int,
    continuation_seed: int,
):
    task = task_cfg["name"]
    camera_keys = profile["camera_keys"]
    chunk_steps = continuation_cadence(profile)
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

        success = False
        terminated = truncated = False
        progress = []
        anchor_native_step = int(source.get("anchor_event", {}).get("native_step", 0))
        for start in range(0, intervention_steps, chunk_steps):
            candidate_chunk = {
                key: value[start : start + chunk_steps]
                for key, value in candidate_bank.items()
            }
            obs, reward, terminated, truncated, info = wrapped.step(candidate_chunk)
            success = success or bool(reward)
            progress.append(
                {
                    "phase": "intervention",
                    "native_step": anchor_native_step + len(wrapped.reward),
                    "history": task_history(task_env, task_cfg["history_fields"]),
                    "label": progress_label(task, task_env, bool(info["success"][-1])),
                }
            )
            if terminated or truncated:
                break
        post_state = np.asarray(task_env.sim.get_state().flatten()).copy()
        post_history = task_history(task_env, task_cfg["history_fields"])
        post_images = capture_images(obs, camera_keys)
        set_policy_seed(client, continuation_seed)
        while not (terminated or truncated):
            continuation = policy_action_chunk(client, obs, chunk_steps)
            obs, reward, terminated, truncated, info = wrapped.step(continuation)
            success = success or bool(reward)
            progress.append(
                {
                    "phase": "continuation",
                    "native_step": anchor_native_step + len(wrapped.reward),
                    "history": task_history(task_env, task_cfg["history_fields"]),
                    "label": progress_label(task, task_env, bool(info["success"][-1])),
                }
            )
        final_state = np.asarray(task_env.sim.get_state().flatten()).copy()
        final_images = capture_images(obs, camera_keys)
        return {
            "success": success,
            "intervention_steps": intervention_steps,
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
    horizons = intervention_horizons(profile)
    bank_steps = max(horizons)
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
        if profile.get("anchor_mode", "fixed_fraction") == "event_aligned":
            image_metrics = {
                key: image_alignment_metrics(
                    canonical["images"][key], source["anchor_images"][key]
                )
                for key in profile["camera_keys"]
            }
            max_normalized_mae = float(
                profile.get("source_image_max_normalized_mae", 0.0)
            )
            max_p99_abs = float(profile.get("source_image_max_p99_abs", 0.0))
            image_gate_mode = profile.get("source_image_gate_mode", "threshold")
            if image_gate_mode not in ("threshold", "diagnostic_only"):
                raise ValueError(f"Unknown source_image_gate_mode: {image_gate_mode}")
            image_within_tolerance = all(
                row["shape_equal"]
                and row["normalized_mae"] <= max_normalized_mae
                and row["p99_abs"] <= max_p99_abs
                for row in image_metrics.values()
            )
            source_alignment = {
                "state_max_abs": state_max_abs(canonical["state"], source["anchor_state"]),
                "history_equal": canonical["history"] == source["anchor_history"],
                "image_max_abs": {
                    key: row["max_abs"] for key, row in image_metrics.items()
                },
                "image_metrics": image_metrics,
                "image_gate": {
                    "mode": image_gate_mode,
                    "max_normalized_mae": max_normalized_mae,
                    "max_p99_abs": max_p99_abs,
                    "within_tolerance": image_within_tolerance,
                },
            }
            source_alignment["exact"] = (
                source_alignment["state_max_abs"] <= profile["state_max_abs_tolerance"]
                and source_alignment["history_equal"]
                and all(row["shape_equal"] for row in image_metrics.values())
                and (image_gate_mode == "diagnostic_only" or image_within_tolerance)
            )
            if not source_alignment["exact"]:
                raise RuntimeError(
                    f"{task} restored carrier does not match the intended source event: "
                    f"{source_alignment}"
                )
        else:
            source_alignment = {"exact": True, "mode": "replayed_fixed_fraction"}

        candidate_banks = []
        for candidate_index in range(profile["candidate_chunks_per_prefix"]):
            proposal_seed = (
                profile["proposal_seed_base"]
                + task_index * 100_000
                + prefix_index * 100
                + candidate_index
            )
            set_policy_seed(client, proposal_seed)
            candidate_banks.append(policy_action_chunk(client, obs, bank_steps))
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
    if "anchor_state" in source:
        arrays["intended_source_event_state"] = source["anchor_state"]
        for camera_key, value in source["anchor_images"].items():
            arrays[f"intended_source_event_obs::{camera_key}"] = value
    for camera_key, value in canonical["images"].items():
        arrays[f"canonical_obs::{camera_key}"] = value
    for action_key in source["prefix_actions"][0]:
        arrays[f"prefix_action::{action_key}"] = np.stack(
            [action[action_key] for action in source["prefix_actions"]]
        )
    for candidate_index, candidate_bank in enumerate(candidate_banks):
        for key, value in candidate_bank.items():
            arrays[f"candidate_{candidate_index}::action_bank::{key}"] = value

    rows_by_horizon = {}
    horizon_results = {}
    continuation_seed = (
        profile["continuation_seed_base"] + task_index * 10_000 + prefix_index
    )
    progress_path = prefix_dir / "progress.json"
    for intervention_steps in horizons:
        rows = []
        rows_by_horizon[str(intervention_steps)] = rows
        for candidate_index, candidate_bank in enumerate(candidate_banks):
            row = run_candidate(
                client,
                wrapper_class,
                task_cfg,
                profile,
                source,
                canonical,
                candidate_bank,
                intervention_steps,
                continuation_seed,
            )
            row["candidate_index"] = candidate_index
            rows.append(row)
            prefix_key = f"h{intervention_steps}::candidate_{candidate_index}"
            arrays[f"{prefix_key}::post_state"] = row.pop("post_state")
            arrays[f"{prefix_key}::final_state"] = row.pop("final_state")
            for camera_key, value in row.pop("post_images").items():
                arrays[f"{prefix_key}::post_obs::{camera_key}"] = value
            for camera_key, value in row.pop("final_images").items():
                arrays[f"{prefix_key}::final_obs::{camera_key}"] = value
            write_json(progress_path, {"candidates_by_intervention_steps": rows_by_horizon})
            print(
                f"{task} prefix {prefix_index + 1} horizon={intervention_steps} "
                f"candidate {candidate_index + 1}/{len(candidate_banks)} "
                f"success={row['success']}",
                flush=True,
            )

        post_states = [
            arrays[f"h{intervention_steps}::candidate_{index}::post_state"]
            for index in range(len(rows))
        ]
        diversity = max(
            (
                state_max_abs(post_states[first], post_states[second])
                for first in range(len(rows))
                for second in range(first + 1, len(rows))
            ),
            default=0.0,
        )
        scores = [
            max(progress_score(task, point["label"]) for point in row["progress"])
            for row in rows
        ]
        horizon_results[str(intervention_steps)] = {
            "candidate_successes": [bool(row["success"]) for row in rows],
            "baseline_candidate_success": bool(rows[0]["success"]),
            "oracle_candidate_success": any(row["success"] for row in rows),
            "outcome_varies": len({bool(row["success"]) for row in rows}) > 1,
            "candidate_max_progress_scores": scores,
            "baseline_candidate_max_progress_score": scores[0],
            "oracle_candidate_max_progress_score": max(scores),
            "progress_oracle_gain": max(scores) - scores[0],
            "progress_outcome_varies": max(scores) - min(scores) > 1e-12,
            "candidate_state_diversity": diversity,
            "candidate_state_diverse": diversity >= profile["minimum_candidate_state_diversity"],
            "all_prefix_carriers_exact": all(
                row["prefix_comparison"]["exact"] for row in rows
            ),
            "candidate_elapsed_seconds": [row["elapsed_seconds"] for row in rows],
        }

    np.savez_compressed(prefix_dir / "branches.npz", **arrays)
    primary = horizon_results[str(horizons[0])]
    prefix_row = {
        "task": task,
        "prefix_index": prefix_index,
        "environment_seed": source["seed"],
        "prefix_native_steps": len(source["prefix_actions"]),
        "canonical_history": canonical["history"],
        "source_event_alignment": source_alignment,
        "candidate_bank_steps": bank_steps,
        "candidate_intervention_steps": horizons,
        "nested_candidate_bank": len(horizons) > 1,
        "interventions": horizon_results,
        "artifact": str(prefix_dir / "branches.npz"),
        "progress_artifact": str(progress_path),
    }
    prefix_row.update(primary)
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
        horizons = intervention_horizons(profile)
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
        intervention_summaries = {}
        for intervention_steps in horizons:
            key = str(intervention_steps)
            rows = [prefix["interventions"][key] for prefix in prefixes]
            baseline_success = float(np.mean([row["baseline_candidate_success"] for row in rows]))
            oracle_success = float(np.mean([row["oracle_candidate_success"] for row in rows]))
            baseline_score = float(
                np.mean([row["baseline_candidate_max_progress_score"] for row in rows])
            )
            oracle_score = float(
                np.mean([row["oracle_candidate_max_progress_score"] for row in rows])
            )
            intervention_summaries[key] = {
                "baseline_candidate_success_rate": baseline_success,
                "oracle_candidate_success_rate": oracle_success,
                "oracle_gain_percentage_points": 100.0 * (oracle_success - baseline_success),
                "prefixes_with_outcome_variation": sum(row["outcome_varies"] for row in rows),
                "baseline_candidate_max_progress_score": baseline_score,
                "oracle_candidate_max_progress_score": oracle_score,
                "progress_oracle_gain_percentage_points": 100.0
                * (oracle_score - baseline_score),
                "prefixes_with_progress_variation": sum(
                    row["progress_outcome_varies"] for row in rows
                ),
                "all_candidate_states_diverse": all(
                    row["candidate_state_diverse"] for row in rows
                ),
                "all_prefix_carriers_exact": all(
                    row["all_prefix_carriers_exact"] for row in rows
                ),
            }
        result["summary"] = {
            "prefixes": len(prefixes),
            "candidate_branches": len(prefixes)
            * profile["candidate_chunks_per_prefix"]
            * len(horizons),
            "candidate_bank_steps": max(horizons),
            "candidate_intervention_steps": horizons,
            "interventions": intervention_summaries,
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
            "all_source_event_restores_exact": all(
                row["source_event_alignment"]["exact"] for row in prefixes
            ),
            "all_prefix_carriers_exact": all(
                summary["all_prefix_carriers_exact"]
                for summary in intervention_summaries.values()
            ),
            "all_candidate_states_diverse": all(
                summary["all_candidate_states_diverse"]
                for summary in intervention_summaries.values()
            ),
            "elapsed_seconds": elapsed,
            "projected_serial_gpu_hours_for_40_prefixes_per_task": projected,
            "within_additional_gpu_hour_cap": projected <= config["expansion"]["additional_gpu_hour_cap"],
        }
        if not result["summary"]["all_source_event_restores_exact"]:
            result["verdict"] = "B0_SOURCE_EVENT_ALIGNMENT_FAIL"
        elif not result["summary"]["all_prefix_carriers_exact"]:
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
