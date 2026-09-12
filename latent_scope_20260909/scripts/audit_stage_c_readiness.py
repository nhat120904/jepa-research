#!/usr/bin/env python3
"""Compute-node readiness audit for Stage-C data and its Stage-B prerequisite."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def task_files(task_cfg: dict, data_cfg: dict) -> dict:
    root = Path(task_cfg["root"])
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return {"complete": False, "error": f"Missing {info_path}"}
    info = json.loads(info_path.read_text())
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    expected = task_cfg["expected_episodes"]
    parquet_present = 0
    video_present = {key: 0 for key in data_cfg["camera_keys"]}
    for episode in range(expected):
        chunk = episode // info["chunks_size"]
        parquet_path = root / info["data_path"].format(
            episode_chunk=chunk, episode_index=episode
        )
        parquet_present += int(parquet_path.is_file())
        for key in data_cfg["camera_keys"]:
            video_path = root / info["video_path"].format(
                episode_chunk=chunk, video_key=key, episode_index=episode
            )
            video_present[key] += int(video_path.is_file())
    complete = (
        info.get("total_episodes") == expected
        and len(episodes) == expected
        and parquet_present == expected
        and all(count == expected for count in video_present.values())
    )
    return {
        "complete": complete,
        "metadata_episodes": len(episodes),
        "declared_episodes": info.get("total_episodes"),
        "expected_episodes": expected,
        "parquet_present": parquet_present,
        "videos_present": video_present,
    }


def manifest_status(path: Path, forbidden_source: str) -> dict:
    if not path.is_file():
        return {"complete": False, "error": f"Missing {path}"}
    manifest = json.loads(path.read_text())
    entries = manifest.get("episodes", [])
    train = [entry for entry in entries if entry["split"] == "train"]
    candidate = [entry for entry in entries if entry["split"] == "candidate_eval"]
    leaked = [
        entry["episode_uid"]
        for entry in train
        if not entry.get("allow_training", False) or entry.get("source") == forbidden_source
    ]
    positive = sum(int(entry.get("positive_frames", 0)) for entry in train)
    negative = sum(int(entry.get("negative_frames", 0)) for entry in train)
    progress_entries = sum(bool(entry.get("has_progress", False)) for entry in train)
    candidate_groups = len(
        {entry.get("candidate_group_id") for entry in candidate if entry.get("candidate_group_id") is not None}
    )
    feature_files_present = sum((path.parent / entry["path"]).is_file() for entry in entries)
    complete = (
        bool(train)
        and positive > 0
        and negative > 0
        and progress_entries > 0
        and bool(candidate)
        and candidate_groups > 0
        and not leaked
        and feature_files_present == len(entries)
    )
    return {
        "complete": complete,
        "entries": len(entries),
        "train_entries": len(train),
        "candidate_entries": len(candidate),
        "candidate_groups": candidate_groups,
        "train_positive_frames": positive,
        "train_negative_frames": negative,
        "train_entries_with_progress": progress_entries,
        "forbidden_training_entries": leaked[:20],
        "feature_files_present": feature_files_present,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-b1-result", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Stage-C readiness audit must run inside an sbatch job")
    config = json.loads(args.config.read_text())
    data_cfg = config["data"]
    tasks = {
        task_cfg["name"]: task_files(task_cfg, data_cfg)
        for task_cfg in data_cfg["tasks"]
    }
    if args.stage_b1_result and args.stage_b1_result.is_file():
        stage_b = json.loads(args.stage_b1_result.read_text())
        stage_b_status = {
            "path": str(args.stage_b1_result),
            "verdict": stage_b.get("verdict"),
            "complete": stage_b.get("verdict") == config["gate"]["required_stage_b_verdict"],
        }
    else:
        stage_b_status = {"complete": False, "error": "Stage-B1 result is not available"}
    manifest = manifest_status(
        Path(data_cfg["manifest"]), data_cfg["forbidden_training_source"]
    )
    ready = all(task["complete"] for task in tasks.values()) and stage_b_status["complete"] and manifest["complete"]
    result = {
        "verdict": "STAGE_C_READY" if ready else "STAGE_C_NOT_READY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "tasks": tasks,
        "stage_b1": stage_b_status,
        "feature_manifest": manifest,
    }
    write_json(args.output, result)
    print(result["verdict"])


if __name__ == "__main__":
    main()
