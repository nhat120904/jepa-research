#!/usr/bin/env python3
"""Replay A4 rollouts and calibrate deterministic one-step transition branches."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import robocasa  # noqa: F401 - registers gym environments
import robosuite  # noqa: F401
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def history(task_env: Any, fields: list[str]) -> dict:
    return {field: jsonable(getattr(task_env, field)) for field in fields}


def state_max_abs(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("inf")
    if first.size == 0:
        return 0.0
    return float(np.max(np.abs(first - second)))


def image_max_abs(first: np.ndarray, second: np.ndarray) -> int:
    if first.shape != second.shape:
        return 2**31 - 1
    delta = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return int(delta.max(initial=0))


def progress_label(task: str, task_env: Any, succeeded: bool) -> dict:
    if task == "ScrubCuttingBoard":
        positions = np.asarray(task_env.board_contact_positions)
        sweep_range = 0.0
        if positions.size:
            sweep_range = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        contact_count = int(task_env.board_contact_timer)
        return {
            "contact_milestone": min(contact_count, 5),
            "sweep_milestone": int(sweep_range >= 0.1),
            "released_success": int(succeeded),
            "contact_count": contact_count,
            "sweep_range": sweep_range,
        }
    if task == "RinseSinkBasin":
        washed = tuple(bool(value) for value in task_env.washed_loc)
        return {
            "washed_mask": [int(value) for value in washed],
            "washed_count": int(sum(washed)),
            "success": int(succeeded),
        }
    raise ValueError(f"Unsupported A4 task: {task}")


def label_signature(label: dict) -> tuple:
    if "contact_milestone" in label:
        return (
            label["contact_milestone"],
            label["sweep_milestone"],
            label["released_success"],
        )
    return (*label["washed_mask"], label["success"])


def load_episode(directory: Path) -> dict:
    with gzip.open(directory / "model.xml.gz", "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    metadata = json.loads((directory / "metadata.json").read_text())
    ep_meta = json.loads((directory / "ep_meta.json").read_text())
    with np.load(directory / "trajectory.npz") as archive:
        initial_state = archive["initial_state"].copy()
        final_state = archive["final_state"].copy()
        action_keys = sorted(key.removeprefix("action::") for key in archive.files if key.startswith("action::"))
        actions = [
            {key: archive[f"action::{key}"][step].copy() for key in action_keys}
            for step in range(metadata["native_steps"])
        ]
    return {
        "directory": directory,
        "metadata": metadata,
        "ep_meta": ep_meta,
        "model_xml": model_xml,
        "initial_state": initial_state,
        "final_state": final_state,
        "actions": actions,
    }


def create_env(task: str, split: str):
    managed_env = gym.make(f"robocasa/{task}", split=split, enable_render=True)
    gym_env = managed_env.unwrapped
    return managed_env, gym_env, gym_env.env


def reset_episode(task_env: Any, episode: dict) -> None:
    task_env.rng = np.random.default_rng(int(episode["metadata"]["seed"]))
    reset_to(
        task_env,
        {
            "model": episode["model_xml"],
            "ep_meta": json.dumps(episode["ep_meta"]),
            "states": episode["initial_state"],
        },
    )


def replay_episode(task: str, task_cfg: dict, split: str, episode: dict, tolerance: float) -> dict:
    managed_env, gym_env, task_env = create_env(task, split)
    labels = []
    success_steps = []
    try:
        reset_episode(task_env, episode)
        labels.append(progress_label(task, task_env, False))
        for step, action in enumerate(episode["actions"]):
            _, reward, _, _, info = gym_env.step({key: value.copy() for key, value in action.items()})
            succeeded = bool(reward) or bool(info.get("success", False))
            labels.append(progress_label(task, task_env, succeeded))
            if succeeded:
                success_steps.append(step)
        replayed_final = np.asarray(task_env.sim.get_state().flatten()).copy()
        final_error = state_max_abs(replayed_final, episode["final_state"])
        history_equal = history(task_env, task_cfg["history_fields"]) == episode["metadata"]["final_history"]
        return {
            "episode_dir": str(episode["directory"]),
            "native_steps": len(episode["actions"]),
            "final_state_max_abs": final_error,
            "final_state_exact": final_error <= tolerance,
            "final_history_equal": history_equal,
            "success_reproduced": bool(success_steps),
            "first_success_step": success_steps[0] if success_steps else None,
            "labels": labels,
            "label_signatures": [list(label_signature(label)) for label in labels],
        }
    finally:
        managed_env.close()


def candidate_steps(replay: dict) -> list[int]:
    length = replay["native_steps"]
    candidates = {0, max(0, length // 4), max(0, length // 2), max(0, (3 * length) // 4), length - 1}
    signatures = replay["label_signatures"]
    for step in range(length):
        if signatures[step] != signatures[step + 1]:
            candidates.add(step)
            candidates.add(step + 1)
    if replay["first_success_step"] is not None:
        candidates.add(replay["first_success_step"])
    return sorted(step for step in candidates if 0 <= step < length)


def select_anchors(replays: list[dict], count: int) -> list[tuple[int, int]]:
    pools = [candidate_steps(replay) for replay in replays]
    selected: list[tuple[int, int]] = []
    for episode_index, pool in enumerate(pools):
        fraction = episode_index / max(1, len(pools) - 1)
        chosen = pool[round(fraction * (len(pool) - 1))]
        selected.append((episode_index, chosen))
    seen_labels = {
        tuple(replays[episode_index]["label_signatures"][step])
        for episode_index, step in selected
    }
    remaining = [
        (episode_index, step)
        for episode_index, pool in enumerate(pools)
        for step in pool
        if (episode_index, step) not in selected
    ]
    while remaining and len(selected) < count:
        unseen = [
            pair
            for pair in remaining
            if tuple(replays[pair[0]]["label_signatures"][pair[1]]) not in seen_labels
        ]
        pair = min(unseen or remaining, key=lambda value: (value[1], value[0]))
        remaining.remove(pair)
        selected.append(pair)
        seen_labels.add(tuple(replays[pair[0]]["label_signatures"][pair[1]]))
    if len(selected) != count:
        raise RuntimeError(f"Could only select {len(selected)}/{count} unique anchors")
    return selected


def branch_actions(recorded: dict[str, np.ndarray], epsilon: float) -> dict[str, dict[str, np.ndarray]]:
    position_key = "action.end_effector_position"
    if position_key not in recorded or recorded[position_key].size < 2:
        raise RuntimeError(f"Expected {position_key} with at least two dimensions")
    candidates = {"recorded": {key: value.copy() for key, value in recorded.items()}}
    for dimension, axis in enumerate(("x", "y")):
        for direction, sign in (("plus", 1.0), ("minus", -1.0)):
            action = {key: value.copy() for key, value in recorded.items()}
            action[position_key][dimension] = np.clip(
                action[position_key][dimension] + sign * epsilon, -1.0, 1.0
            )
            candidates[f"pos_{axis}_{direction}"] = action
    return candidates


def mapped_observation(gym_env: Any, task_env: Any) -> dict[str, Any]:
    return gym_env.get_observation(task_env._get_observations(force_update=True))


def calibrate_anchor(
    task: str,
    task_cfg: dict,
    split: str,
    episode: dict,
    episode_index: int,
    step: int,
    calibration: dict,
    anchor_dir: Path,
) -> dict:
    camera_keys = calibration["camera_keys"]
    repeats = calibration["branch_repeats"]
    candidates = branch_actions(episode["actions"][step], calibration["position_perturbation"])
    branch_rows: dict[str, list[dict]] = {}
    arrays: dict[str, np.ndarray] = {}
    missing_cameras: set[str] = set()
    for candidate_name, candidate in candidates.items():
        rows = []
        for repeat in range(repeats):
            # A fresh simulator carrier avoids inheriting hidden controller or renderer
            # state from another branch. Every carrier is rebuilt from the recorded XML
            # and initial state, then receives exactly the same prefix.
            managed_env, gym_env, task_env = create_env(task, split)
            try:
                reset_episode(task_env, episode)
                for prefix_action in episode["actions"][:step]:
                    gym_env.step(
                        {key: value.copy() for key, value in prefix_action.items()}
                    )
                before_obs = mapped_observation(gym_env, task_env)
                missing_cameras.update(set(camera_keys) - set(before_obs))
                before_state = np.asarray(task_env.sim.get_state().flatten()).copy()
                before_history = history(task_env, task_cfg["history_fields"])
                label = progress_label(task, task_env, all(task_env.washed_loc) if task == "RinseSinkBasin" else False)
                post_obs, reward, done, truncated, info = gym_env.step(
                    {key: value.copy() for key, value in candidate.items()}
                )
                row = {
                    "before_sim_state": before_state,
                    "before_history": before_history,
                    "progress_label": label,
                    "sim_state": np.asarray(task_env.sim.get_state().flatten()).copy(),
                    "reward": float(reward),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "success": bool(info.get("success", False)),
                    "history": history(task_env, task_cfg["history_fields"]),
                    "before_images": {key: np.asarray(before_obs[key]).copy() for key in camera_keys if key in before_obs},
                    "post_images": {key: np.asarray(post_obs[key]).copy() for key in camera_keys if key in post_obs},
                }
                rows.append(row)
                arrays[f"{candidate_name}::repeat_{repeat}::before_sim_state"] = before_state
                for action_key, value in candidate.items():
                    arrays[f"{candidate_name}::repeat_{repeat}::action::{action_key}"] = value
                arrays[f"{candidate_name}::repeat_{repeat}::post_sim_state"] = row["sim_state"]
                for camera_key, value in row["before_images"].items():
                    arrays[f"{candidate_name}::repeat_{repeat}::before_obs::{camera_key}"] = value
                for camera_key, value in row["post_images"].items():
                    arrays[f"{candidate_name}::repeat_{repeat}::post_obs::{camera_key}"] = value
            finally:
                managed_env.close()
        branch_rows[candidate_name] = rows

    comparisons = {}
    deterministic = not missing_cameras
    for candidate_name, rows in branch_rows.items():
        reference = rows[0]
        candidate_comparisons = []
        for row in rows[1:]:
            comparison = {
                    "before_state_max_abs": state_max_abs(reference["before_sim_state"], row["before_sim_state"]),
                    "before_history_equal": reference["before_history"] == row["before_history"],
                    "progress_label_equal": reference["progress_label"] == row["progress_label"],
                    "state_max_abs": state_max_abs(reference["sim_state"], row["sim_state"]),
                    "reward_equal": reference["reward"] == row["reward"],
                    "done_equal": reference["done"] == row["done"],
                    "truncated_equal": reference["truncated"] == row["truncated"],
                    "success_equal": reference["success"] == row["success"],
                    "history_equal": reference["history"] == row["history"],
                    "before_image_max_abs": {
                        key: image_max_abs(reference["before_images"][key], row["before_images"][key])
                        for key in set(reference["before_images"]) & set(row["before_images"])
                    },
                    "post_image_max_abs": {
                        key: image_max_abs(reference["post_images"][key], row["post_images"][key])
                        for key in set(reference["post_images"]) & set(row["post_images"])
                    },
                    "image_keys_equal": set(reference["post_images"]) == set(row["post_images"]) == set(camera_keys),
                }
            comparison["exact"] = (
                    comparison["before_state_max_abs"] <= calibration["state_max_abs_tolerance"]
                    and comparison["before_history_equal"]
                    and comparison["progress_label_equal"]
                    and comparison["state_max_abs"] <= calibration["state_max_abs_tolerance"]
                    and comparison["reward_equal"]
                    and comparison["done_equal"]
                    and comparison["truncated_equal"]
                    and comparison["success_equal"]
                    and comparison["history_equal"]
                    and comparison["image_keys_equal"]
                    and max(comparison["before_image_max_abs"].values(), default=2**31 - 1)
                    <= calibration["image_max_abs_tolerance"]
                    and max(comparison["post_image_max_abs"].values(), default=2**31 - 1)
                    <= calibration["image_max_abs_tolerance"]
                )
            deterministic = deterministic and comparison["exact"]
            candidate_comparisons.append(comparison)
        comparisons[candidate_name] = candidate_comparisons

    outcomes = {name: rows[0]["sim_state"] for name, rows in branch_rows.items()}
    pairwise = {
            f"{first}__{second}": state_max_abs(outcomes[first], outcomes[second])
            for first, second in itertools.combinations(outcomes, 2)
    }
    diversity = max(pairwise.values(), default=0.0)
    np.savez_compressed(anchor_dir / "branches.npz", **arrays)
    reference_label = branch_rows["recorded"][0]["progress_label"]
    row = {
            "task": task,
            "episode_index": episode_index,
            "episode_dir": str(episode["directory"]),
            "step": step,
            "progress_label": reference_label,
            "progress_signature": list(label_signature(reference_label)),
            "missing_camera_keys": sorted(missing_cameras),
            "candidate_names": list(candidates),
            "comparisons": comparisons,
            "deterministic": deterministic,
            "pairwise_outcome_state_max_abs": pairwise,
            "outcome_state_diversity": diversity,
            "diverse": diversity >= calibration["minimum_outcome_state_diversity"],
            "artifact": str(anchor_dir / "branches.npz"),
    }
    write_json(anchor_dir / "metadata.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("A4 calibration must run inside an sbatch job")

    config = json.loads(args.config.read_text())
    collection_result = json.loads((args.collection_dir / "collection_result.json").read_text())
    result_path = args.output_dir / "stage_a4_result.json"
    result = {
        "verdict": "A4_CALIBRATION_RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": job_id,
        "node": platform.node(),
        "collection_dir": str(args.collection_dir),
        "collection_job_id": collection_result.get("slurm_job_id"),
        "config": config,
        "tasks": {},
    }
    write_json(result_path, result)
    try:
        if collection_result.get("verdict") != "A4_COLLECTION_COMPLETE":
            raise RuntimeError(f"Collection gate is {collection_result.get('verdict')}")
        calibration = config["calibration"]
        all_replays_exact = True
        all_branches_deterministic = True
        all_anchors_diverse = True
        all_progress_covered = True
        all_camera_keys_present = True
        for task_cfg in config["collection"]["tasks"]:
            task = task_cfg["name"]
            directories = [Path(path) for path in collection_result["tasks"][task]["successful_episodes"]]
            episodes = [load_episode(directory) for directory in directories]
            replays = [
                replay_episode(
                    task,
                    task_cfg,
                    config["collection"]["split"],
                    episode,
                    calibration["state_max_abs_tolerance"],
                )
                for episode in episodes
            ]
            replay_exact = all(
                row["final_state_exact"] and row["final_history_equal"] and row["success_reproduced"]
                for row in replays
            )
            all_replays_exact = all_replays_exact and replay_exact
            selected = select_anchors(replays, calibration["anchors_per_task"])
            anchors = []
            for anchor_index, (episode_index, step) in enumerate(selected):
                anchor_dir = args.output_dir / "anchors" / task / f"anchor_{anchor_index:02d}"
                anchor_dir.mkdir(parents=True, exist_ok=False)
                row = calibrate_anchor(
                    task,
                    task_cfg,
                    config["collection"]["split"],
                    episodes[episode_index],
                    episode_index,
                    step,
                    calibration,
                    anchor_dir,
                )
                anchors.append(row)
                write_json(result_path, result)
            distinct_labels = sorted({tuple(row["progress_signature"]) for row in anchors})
            progress_covered = len(distinct_labels) >= calibration["minimum_distinct_progress_labels"]
            branches_deterministic = all(row["deterministic"] for row in anchors)
            anchors_diverse = all(row["diverse"] for row in anchors)
            cameras_present = all(not row["missing_camera_keys"] for row in anchors)
            result["tasks"][task] = {
                "replays": replays,
                "replay_exact": replay_exact,
                "anchors": anchors,
                "num_anchors": len(anchors),
                "distinct_anchor_progress_labels": [list(value) for value in distinct_labels],
                "progress_covered": progress_covered,
                "branches_deterministic": branches_deterministic,
                "anchors_diverse": anchors_diverse,
                "camera_keys_present": cameras_present,
            }
            all_progress_covered = all_progress_covered and progress_covered
            all_branches_deterministic = all_branches_deterministic and branches_deterministic
            all_anchors_diverse = all_anchors_diverse and anchors_diverse
            all_camera_keys_present = all_camera_keys_present and cameras_present
            write_json(result_path, result)
            print(
                f"{task}: replay={replay_exact} deterministic={branches_deterministic} "
                f"diverse={anchors_diverse} labels={len(distinct_labels)} cameras={cameras_present}",
                flush=True,
            )

        result["gate"] = {
            "all_replays_exact": all_replays_exact,
            "all_branches_deterministic": all_branches_deterministic,
            "all_anchors_diverse": all_anchors_diverse,
            "all_progress_covered": all_progress_covered,
            "all_camera_keys_present": all_camera_keys_present,
        }
        if all(result["gate"].values()):
            result["verdict"] = "A4_PASS"
        elif not all_replays_exact:
            result["verdict"] = "A4_REPLAY_FAIL"
        elif not all_camera_keys_present or not all_branches_deterministic:
            result["verdict"] = "A4_BRANCH_NONDETERMINISTIC"
        elif not all_anchors_diverse:
            result["verdict"] = "A4_NO_ACTION_DIVERSITY"
        else:
            result["verdict"] = "A4_PROGRESS_LABEL_FAIL"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(result_path, result)
        print(result["verdict"], flush=True)
    except Exception as error:
        result["verdict"] = "A4_CALIBRATION_ERROR"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
