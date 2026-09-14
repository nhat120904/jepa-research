#!/usr/bin/env python3
"""Calibrate visual variability when restoring exact Stage-B event snapshots."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_baseline_sim_client import InferenceClient, load_multistep_wrapper
from run_stage_a4_collect import jsonable, task_history, wait_for_server
from run_stage_a4_calibrate import state_max_abs
from run_stage_b_profile import (
    capture_fresh_images,
    capture_images,
    generate_source_prefix,
    reconstruct_canonical,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def image_metrics(first: np.ndarray, second: np.ndarray) -> dict:
    if first.shape != second.shape:
        return {
            "shape_equal": False,
            "first_shape": list(first.shape),
            "second_shape": list(second.shape),
        }
    delta = np.abs(first.astype(np.float32) - second.astype(np.float32))
    mae = float(delta.mean())
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    return {
        "shape_equal": True,
        "max_abs": int(delta.max(initial=0)),
        "mae": mae,
        "normalized_mae": mae / 255.0,
        "rmse": rmse,
        "psnr_db": float("inf") if rmse == 0 else 20.0 * math.log10(255.0 / rmse),
        "p99_abs": float(np.quantile(delta, 0.99)),
        "changed_fraction": float(np.mean(delta > 0)),
        "changed_gt_8_fraction": float(np.mean(delta > 8)),
    }


def max_metric(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row.get("shape_equal") and field in row]
    return max(values) if values else None


def calibrate_task(client, wrapper_class, task_cfg, profile, task_index: int, output_dir: Path):
    attempts = []
    source = None
    for attempt in range(profile["maximum_prefix_attempts_per_task"]):
        candidate = generate_source_prefix(
            client, wrapper_class, task_cfg, profile, task_index, attempt
        )
        attempts.append(
            {
                "attempt": attempt,
                "seed": candidate["seed"],
                "anchor_found": candidate.get("anchor_found", True),
                "source_rollout_success": candidate.get("source_rollout_success"),
                "source_events": candidate.get("source_events"),
            }
        )
        if candidate.get("anchor_found", True):
            source = candidate
            break
    if source is None:
        raise RuntimeError(f"Could not find an event-aligned source for {task_cfg['name']}")

    task = task_cfg["name"]
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=False)
    with gzip.open(task_dir / "model.xml.gz", "wt", encoding="utf-8") as handle:
        handle.write(source["model_xml"])
    write_json(task_dir / "ep_meta.json", source["ep_meta"])

    arrays = {"source_event_state": source["anchor_state"]}
    for key, value in source["anchor_images"].items():
        arrays[f"source::{key}"] = value

    repeats = []
    repeat_images = []
    for repeat in range(profile["restore_calibration_repeats"]):
        wrapped, gym_env, task_env, obs, success = reconstruct_canonical(
            wrapper_class, task_cfg, profile, source
        )
        try:
            state = np.asarray(task_env.sim.get_state().flatten()).copy()
            history = task_history(task_env, task_cfg["history_fields"])
            wrapper_images = capture_images(obs, profile["camera_keys"])
            fresh_first = capture_fresh_images(
                gym_env, task_env, profile["camera_keys"]
            )
            fresh_second = capture_fresh_images(
                gym_env, task_env, profile["camera_keys"]
            )
            repeat_images.append(fresh_first)
            row = {
                "repeat": repeat,
                "already_succeeded": success,
                "state_max_abs": state_max_abs(state, source["anchor_state"]),
                "history_equal": history == source["anchor_history"],
                "source_to_restore": {
                    key: image_metrics(source["anchor_images"][key], fresh_first[key])
                    for key in profile["camera_keys"]
                },
                "wrapper_to_fresh": {
                    key: image_metrics(wrapper_images[key], fresh_first[key])
                    for key in profile["camera_keys"]
                },
                "same_carrier_rerender": {
                    key: image_metrics(fresh_first[key], fresh_second[key])
                    for key in profile["camera_keys"]
                },
            }
            repeats.append(row)
            arrays[f"repeat_{repeat}::state"] = state
            for key, value in fresh_first.items():
                arrays[f"repeat_{repeat}::{key}"] = value
        finally:
            wrapped.close()

    cross_restore = {key: [] for key in profile["camera_keys"]}
    for first in range(len(repeat_images)):
        for second in range(first + 1, len(repeat_images)):
            for key in profile["camera_keys"]:
                cross_restore[key].append(
                    image_metrics(repeat_images[first][key], repeat_images[second][key])
                )
    np.savez_compressed(task_dir / "restore_calibration.npz", **arrays)

    per_camera = {}
    for key in profile["camera_keys"]:
        source_rows = [row["source_to_restore"][key] for row in repeats]
        rerender_rows = [row["same_carrier_rerender"][key] for row in repeats]
        per_camera[key] = {
            "max_source_to_restore_normalized_mae": max_metric(
                source_rows, "normalized_mae"
            ),
            "max_source_to_restore_p99_abs": max_metric(source_rows, "p99_abs"),
            "max_cross_restore_normalized_mae": max_metric(
                cross_restore[key], "normalized_mae"
            ),
            "max_cross_restore_p99_abs": max_metric(cross_restore[key], "p99_abs"),
            "max_same_carrier_rerender_normalized_mae": max_metric(
                rerender_rows, "normalized_mae"
            ),
            "max_same_carrier_rerender_p99_abs": max_metric(
                rerender_rows, "p99_abs"
            ),
        }

    return {
        "attempts": attempts,
        "anchor_event": source["anchor_event"],
        "source_anchor_history": source["anchor_history"],
        "all_states_exact": all(
            row["state_max_abs"] <= profile["state_max_abs_tolerance"] for row in repeats
        ),
        "all_histories_exact": all(row["history_equal"] for row in repeats),
        "repeats": repeats,
        "cross_restore": cross_restore,
        "per_camera_summary": per_camera,
        "artifact": str(task_dir / "restore_calibration.npz"),
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
        raise RuntimeError("Event-restore calibration must run inside an sbatch job")

    config = json.loads(args.config.read_text())
    result_path = args.output_dir / "event_restore_calibration_result.json"
    result = {
        "verdict": "EVENT_RESTORE_CALIBRATION_RUNNING",
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
        for task_index, task_cfg in enumerate(config["profile"]["tasks"]):
            task = task_cfg["name"]
            result["tasks"][task] = calibrate_task(
                client,
                wrapper_class,
                task_cfg,
                config["profile"],
                task_index,
                args.output_dir,
            )
            write_json(result_path, result)
        physical_exact = all(
            row["all_states_exact"] and row["all_histories_exact"]
            for row in result["tasks"].values()
        )
        result["verdict"] = (
            "EVENT_RESTORE_CALIBRATION_COMPLETE"
            if physical_exact
            else "EVENT_RESTORE_PHYSICAL_ALIGNMENT_FAIL"
        )
        result["summary"] = {
            "physical_alignment_exact": physical_exact,
            "restore_repeats_per_task": config["profile"]["restore_calibration_repeats"],
            "elapsed_seconds": time.monotonic() - started,
        }
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(result_path, result)
        print(result["verdict"], flush=True)
        if not physical_exact:
            raise RuntimeError("Exact physical event restoration failed")
    except Exception as error:
        if result["verdict"] == "EVENT_RESTORE_CALIBRATION_RUNNING":
            result["verdict"] = "EVENT_RESTORE_CALIBRATION_ERROR"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        write_json(result_path, result)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
