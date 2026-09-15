#!/usr/bin/env python3
"""Step-1 data readiness check for the Scrub in-segment mechanism test.

Questions, in order:
1. Is the released ScrubCuttingBoard LeRobot data (states, actions, videos) available,
   and how large is the full set?
2. Does recorded-state playback in the current runtime reproduce the native
   distinct-contact monitor well enough to serve as labels? Reference: the recorded
   per-frame success reward.
3. Are accepted contact events dense enough, at 8-128 step segment lengths, to test
   prediction and composition of accumulated in-segment effects, and how non-additive
   is the native filter across segment boundaries?

No policy, no rendering, no training. Labels come from calling the native
`update_state` after loading each recorded simulator state.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CONTACT_RADIUS = 0.02  # native ScrubCuttingBoard.update_state
REQUIRED_CONTACTS = 5  # native _check_success
REQUIRED_SWEEP = 0.1
WINDOW_LENGTHS = (8, 16, 32, 64, 128)
WINDOW_STRIDE = 4


def require_slurm() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Scrub data check must run inside an sbatch job")
    return job_id


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Availability


def list_remote(config: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    source = config["dataset_source"]
    api = HfApi()
    totals: dict[str, dict[str, int]] = {}
    episodes: set[str] = set()
    for entry in api.list_repo_tree(
        repo_id=source["repo_id"],
        path_in_repo=source["path_in_repo"],
        repo_type="dataset",
        revision=source["revision"],
        recursive=True,
    ):
        size = getattr(entry, "size", None)
        if size is None:
            continue
        relative = entry.path[len(source["path_in_repo"]) + 1 :]
        top = relative.split("/", 1)[0]
        if top == "extras":
            group = "extras/" + relative.rsplit("/", 1)[-1]
            parts = relative.split("/")
            if len(parts) > 2 and parts[1].startswith("episode_"):
                episodes.add(parts[1])
        elif top == "videos":
            group = "videos/" + relative.split("/")[2] if relative.count("/") >= 3 else "videos"
        else:
            group = top
        row = totals.setdefault(group, {"files": 0, "bytes": 0})
        row["files"] += 1
        row["bytes"] += int(size)
    total_bytes = sum(row["bytes"] for row in totals.values())
    return {
        "groups": totals,
        "episodes_with_extras": len(episodes),
        "total_gb": total_bytes / 1e9,
        "non_video_gb": sum(v["bytes"] for k, v in totals.items() if not k.startswith("videos")) / 1e9,
    }


def download_files(config: dict[str, Any], dataset_dir: Path, episodes: list[int], video_episodes: list[int]) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    source = config["dataset_source"]
    cache_dir = Path(config["hf_cache"])
    relative = [
        "meta/info.json",
        "meta/modality.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/stats.json",
        "meta/embodiment.json",
        "extras/dataset_meta.json",
    ]
    for episode in episodes:
        name = f"episode_{episode:06d}"
        relative.append(f"data/chunk-000/{name}.parquet")
        relative.extend(f"extras/{name}/{item}" for item in ("states.npz", "model.xml.gz", "ep_meta.json"))
    for episode in video_episodes:
        for camera in config["camera_names"]:
            relative.append(f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4")
    downloaded = 0
    failures = []
    start = time.time()
    for item in relative:
        destination = dataset_dir / item
        if destination.is_file():
            continue
        try:
            cached = hf_hub_download(
                repo_id=source["repo_id"],
                filename=f"{source['path_in_repo']}/{item}",
                repo_type="dataset",
                revision=source["revision"],
                cache_dir=cache_dir,
            )
        except Exception as error:  # record and continue; availability is a measured outcome
            failures.append({"file": item, "error": repr(error)[:300]})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        downloaded += 1
    return {"requested": len(relative), "downloaded": downloaded, "failures": failures, "seconds": time.time() - start}


def check_videos(config: dict[str, Any], dataset_dir: Path, episode: int) -> dict[str, Any]:
    import av
    import pandas as pd

    rows = len(pd.read_parquet(dataset_dir / f"data/chunk-000/episode_{episode:06d}.parquet"))
    result: dict[str, Any] = {"episode": episode, "parquet_rows": rows, "cameras": {}}
    for camera in config["camera_names"]:
        path = dataset_dir / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
        if not path.is_file():
            result["cameras"][camera] = {"present": False}
            continue
        with av.open(str(path)) as container:
            frames = 0
            shape = None
            for frame in container.decode(video=0):
                if shape is None:
                    shape = list(frame.to_ndarray(format="rgb24").shape)
                frames += 1
        result["cameras"][camera] = {
            "present": True,
            "bytes": path.stat().st_size,
            "frames": frames,
            "shape": shape,
            "frames_match_parquet": frames == rows,
        }
    return result


# ---------------------------------------------------------------------------
# Native-monitor playback


def make_env(dataset_dir: Path) -> Any:
    import copy

    import robocasa.utils.lerobot_utils as lerobot_utils
    import robosuite

    env_meta = lerobot_utils.get_env_metadata(dataset_dir)
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


def body_geoms(env: Any, body: int) -> list[str]:
    model = env.sim.model
    return [model.geom_id2name(gid) for gid in range(model.ngeom) if model.geom_bodyid[gid] == body]


def playback_episode(env: Any, dataset_dir: Path, episode: int) -> dict[str, Any]:
    import pandas as pd
    import robocasa.utils.lerobot_utils as lerobot_utils
    import robocasa.utils.object_utils as OU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    states = lerobot_utils.get_episode_states(dataset_dir, episode)
    frame = pd.read_parquet(dataset_dir / f"data/chunk-000/episode_{episode:06d}.parquet")
    recorded_reward = frame["next.reward"].to_numpy()
    start = time.time()
    reset_to(
        env,
        {
            "states": states[0],
            "model": lerobot_utils.get_episode_model_xml(dataset_dir, episode),
            "ep_meta": json.dumps(lerobot_utils.get_episode_meta(dataset_dir, episode)),
        },
    )
    load_seconds = time.time() - start
    initial_error = float(np.max(np.abs(np.asarray(env.sim.get_state().flatten()) - states[0])))

    env.board_contact_positions = []
    env.board_contact_timer = 0
    sponge_body = env.obj_body_id["sponge"]
    sponge_geoms = body_geoms(env, sponge_body)
    board_geoms = body_geoms(env, env.obj_body_id["cutting_board"])

    n = len(states)
    count = np.zeros(n, dtype=np.int32)
    sweep = np.zeros(n, dtype=np.float64)
    contact = np.zeros(n, dtype=bool)
    grasped = np.zeros(n, dtype=bool)
    far = np.zeros(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    sponge_xy = np.zeros((n, 2), dtype=np.float64)
    load_error = np.zeros(n, dtype=np.float64)
    start = time.time()
    for t in range(n):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        load_error[t] = float(np.max(np.abs(np.asarray(env.sim.get_state().flatten()) - states[t])))
        grasped[t] = bool(OU.check_obj_grasped(env, "sponge"))
        contact[t] = bool(env.check_contact(sponge_geoms, board_geoms))
        sponge_xy[t] = np.asarray(env.sim.data.body_xpos[sponge_body])[:2]
        far[t] = bool(OU.gripper_obj_far(env, "sponge", th=0.15))
        env.update_state()  # native monitor update, same call as Kitchen._post_action
        count[t] = int(env.board_contact_timer)
        if env.board_contact_positions:
            positions = np.asarray(env.board_contact_positions)
            sweep[t] = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        success[t] = bool(env._check_success())
    playback_seconds = time.time() - start

    # The native monitor recomputes contact/grasp inside update_state; verify that the
    # offline replication of its filter from the stored raw signals gives the same counts.
    offline_count = native_filter_counts(sponge_xy, contact & grasped)

    recorded_first = int(np.argmax(recorded_reward > 0)) if (recorded_reward > 0).any() else None
    playback_first = int(np.argmax(success)) if success.any() else None
    return {
        "episode": episode,
        "frames": int(n),
        "state_dim": int(states.shape[1]),
        "initial_state_error": initial_error,
        "max_state_load_error": float(load_error.max()),
        "load_seconds": load_seconds,
        "playback_seconds": playback_seconds,
        "recorded_first_success": recorded_first,
        "playback_first_success": playback_first,
        "playback_success_at_end": bool(success[-1]),
        "success_frame_offset": None if recorded_first is None or playback_first is None else playback_first - recorded_first,
        "offline_filter_matches_native": bool(np.array_equal(offline_count, count)),
        "final_contact_count": int(count[-1]),
        "final_sweep": float(sweep[-1]),
        "contact_and_grasp_frames": int((contact & grasped).sum()),
        "arrays": {
            "count": count,
            "sweep": sweep,
            "contact": contact,
            "grasped": grasped,
            "far": far,
            "success": success,
            "sponge_xy": sponge_xy,
            "recorded_reward": recorded_reward,
        },
    }


def native_filter_counts(xy: np.ndarray, active: np.ndarray, history: list[np.ndarray] | None = None) -> np.ndarray:
    accepted = [] if history is None else list(history)
    base = len(accepted)
    counts = np.zeros(len(xy), dtype=np.int32)
    for t in range(len(xy)):
        if active[t] and all(np.linalg.norm(xy[t] - p) >= CONTACT_RADIUS for p in accepted):
            accepted.append(xy[t])
        counts[t] = len(accepted) - base
    return counts


def accepted_positions(xy: np.ndarray, active: np.ndarray, end: int) -> list[np.ndarray]:
    accepted: list[np.ndarray] = []
    for t in range(end):
        if active[t] and all(np.linalg.norm(xy[t] - p) >= CONTACT_RADIUS for p in accepted):
            accepted.append(xy[t])
    return accepted


# ---------------------------------------------------------------------------
# Segment statistics


def segment_statistics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    phase_rows = []
    for row in episodes:
        arrays = row["arrays"]
        count = arrays["count"]
        events = np.flatnonzero(np.diff(np.concatenate([[0], count])) > 0)
        sweep_ok = np.flatnonzero(arrays["sweep"] >= REQUIRED_SWEEP)
        phase_rows.append(
            {
                "episode": row["episode"],
                "frames": row["frames"],
                "first_event": int(events[0]) if len(events) else None,
                "fifth_event": int(events[REQUIRED_CONTACTS - 1]) if len(events) >= REQUIRED_CONTACTS else None,
                "last_event": int(events[-1]) if len(events) else None,
                "events": int(len(events)),
                "sweep_reached": int(sweep_ok[0]) if len(sweep_ok) else None,
                "event_gaps_median": float(np.median(np.diff(events))) if len(events) > 1 else None,
            }
        )
    result["phases"] = phase_rows

    for length in WINDOW_LENGTHS:
        whole = {"windows": 0, "ge1": 0, "ge2": 0, "ge3": 0}
        active = {"windows": 0, "ge1": 0, "ge2": 0, "ge3": 0, "increments": []}
        nonadditive = {"windows_with_events": 0, "history_changes_increment": 0, "overcount_total": 0}
        split = {"windows": 0, "split_sum_exceeds_whole": 0, "excess_total": 0}
        for row in episodes:
            arrays = row["arrays"]
            count = arrays["count"]
            xy = arrays["sponge_xy"]
            live = arrays["contact"] & arrays["grasped"]
            events = np.flatnonzero(np.diff(np.concatenate([[0], count])) > 0)
            n = len(count)
            for t in range(0, n - length, WINDOW_STRIDE):
                u = t + length
                before = count[t - 1] if t > 0 else 0
                increment = int(count[u - 1] - before)  # frames t..u-1, half-open
                whole["windows"] += 1
                whole["ge1"] += increment >= 1
                whole["ge2"] += increment >= 2
                whole["ge3"] += increment >= 3
                if len(events) and events[0] - length <= t <= events[-1]:
                    active["windows"] += 1
                    active["ge1"] += increment >= 1
                    active["ge2"] += increment >= 2
                    active["ge3"] += increment >= 3
                    active["increments"].append(increment)
                if not live[t:u].any():
                    continue
                scratch = int(native_filter_counts(xy[t:u], live[t:u])[-1])
                if scratch or increment:
                    nonadditive["windows_with_events"] += 1
                    if scratch != increment:
                        nonadditive["history_changes_increment"] += 1
                        nonadditive["overcount_total"] += scratch - increment
                if length >= 16:
                    mid = t + length // 2
                    left = int(native_filter_counts(xy[t:mid], live[t:mid])[-1])
                    right = int(native_filter_counts(xy[mid:u], live[mid:u])[-1])
                    if scratch:
                        split["windows"] += 1
                        if left + right > scratch:
                            split["split_sum_exceeds_whole"] += 1
                            split["excess_total"] += left + right - scratch
        increments = np.asarray(active.pop("increments"), dtype=np.int32)
        result[f"L{length}"] = {
            "all_windows": {k: int(v) for k, v in whole.items()},
            "all_fraction_ge2": whole["ge2"] / max(whole["windows"], 1),
            "scrub_phase_windows": {k: int(v) for k, v in active.items()},
            "scrub_phase_fraction_ge1": active["ge1"] / max(active["windows"], 1),
            "scrub_phase_fraction_ge2": active["ge2"] / max(active["windows"], 1),
            "scrub_phase_increment_histogram": np.bincount(increments, minlength=6).tolist() if len(increments) else [],
            "history_nonadditivity": {k: int(v) for k, v in nonadditive.items()},
            "history_nonadditive_fraction": nonadditive["history_changes_increment"] / max(nonadditive["windows_with_events"], 1),
            "midpoint_split": {k: int(v) for k, v in split.items()},
            "midpoint_split_double_count_fraction": split["split_sum_exceeds_whole"] / max(split["windows"], 1),
        }

    # Label robustness: re-run the native filter on every second frame of the raw signal.
    subsample = []
    for row in episodes:
        arrays = row["arrays"]
        live = arrays["contact"] & arrays["grasped"]
        full = int(native_filter_counts(arrays["sponge_xy"], live)[-1])
        half = int(native_filter_counts(arrays["sponge_xy"][::2], live[::2])[-1])
        subsample.append({"episode": row["episode"], "full_rate_final": full, "half_rate_final": half})
    result["frame_subsampling"] = subsample
    return result


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    job_id = require_slurm()
    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(config["dataset_dir"])
    deadline = time.time() + float(config["soft_limit_seconds"])

    report: dict[str, Any] = {
        "job_id": job_id,
        "node": platform.node(),
        "python": sys.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "scope": "recorded-state playback with native Scrub monitor; no policy, rendering or training",
    }
    import importlib.metadata as metadata

    report["packages"] = {name: metadata.version(name) for name in ("robocasa", "robosuite", "mujoco", "huggingface_hub", "av")}

    try:
        report["remote"] = list_remote(config)
    except Exception as error:
        report["remote"] = {"error": repr(error)[:500]}

    rng = np.random.default_rng(config["episode_seed"])
    total = int(config["total_episodes"])
    fixed = list(range(config["always_include_first"]))
    sampled = sorted(rng.choice(np.arange(len(fixed), total), size=config["episodes"] - len(fixed), replace=False).tolist())
    episodes = fixed + sampled
    report["episode_selection"] = episodes
    local_seed = Path(config["local_seed_dir"])
    if local_seed.is_dir():  # reuse the five Stage-A files instead of re-downloading them
        for path in local_seed.rglob("*"):
            destination = dataset_dir / path.relative_to(local_seed)
            if path.is_file() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    report["download"] = download_files(config, dataset_dir, episodes, config["video_episodes"])
    report["videos"] = []
    for episode in config["video_episodes"]:
        try:
            report["videos"].append(check_videos(config, dataset_dir, episode))
        except Exception as error:
            report["videos"].append({"episode": episode, "error": repr(error)[:500]})

    env = make_env(dataset_dir)
    rows = []
    errors = []
    for episode in episodes:
        if time.time() > deadline:
            errors.append({"episode": episode, "error": "soft time limit reached before episode"})
            continue
        if not (dataset_dir / f"extras/episode_{episode:06d}/states.npz").is_file():
            errors.append({"episode": episode, "error": "missing after download"})
            continue
        try:
            rows.append(playback_episode(env, dataset_dir, episode))
        except Exception as error:
            errors.append({"episode": episode, "error": repr(error)[:500]})
        print(f"episode {episode} done ({len(rows)} ok, {len(errors)} errors)", flush=True)

    np.savez_compressed(
        args.output_dir / "scrub_monitor_arrays.npz",
        **{f"ep{row['episode']:06d}__{key}": value for row in rows for key, value in row["arrays"].items()},
    )
    report["episodes"] = [{k: v for k, v in row.items() if k != "arrays"} for row in rows]
    report["episode_errors"] = errors

    offsets = [row["success_frame_offset"] for row in rows if row["success_frame_offset"] is not None]
    report["label_fidelity"] = {
        "episodes": len(rows),
        "max_initial_state_error": max((row["initial_state_error"] for row in rows), default=None),
        "max_state_load_error": max((row["max_state_load_error"] for row in rows), default=None),
        "recorded_success_episodes": sum(row["recorded_first_success"] is not None for row in rows),
        "playback_success_episodes": sum(row["playback_first_success"] is not None for row in rows),
        "success_frame_offset_exact": sum(offset == 0 for offset in offsets),
        "success_frame_offset_within_2": sum(abs(offset) <= 2 for offset in offsets),
        "success_frame_offsets": offsets,
        "offline_filter_matches_native": sum(row["offline_filter_matches_native"] for row in rows),
    }
    if rows:
        report["segments"] = segment_statistics(rows)

    out = args.output_dir / "scrub_data_check_result.json"
    out.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n")
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
