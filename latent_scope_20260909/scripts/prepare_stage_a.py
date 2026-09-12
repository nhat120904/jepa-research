#!/usr/bin/env python3
"""Install-time data preparation for the pinned Stage-A RoboCasa runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


def require_slurm() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Stage-A setup must run inside an sbatch job")
    return job_id


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def dataset_complete(path: Path, minimum_episodes: int) -> bool:
    if not (
        (path / "meta" / "info.json").is_file()
        and (path / "extras" / "dataset_meta.json").is_file()
    ):
        return False
    for episode_index in range(minimum_episodes):
        episode_name = f"episode_{episode_index:06d}"
        episode_dir = path / "extras" / episode_name
        if not all(
            (episode_dir / filename).is_file()
            for filename in ("states.npz", "model.xml.gz", "ep_meta.json")
        ):
            return False
        if not list((path / "data").glob(f"*/{episode_name}.parquet")):
            return False
    return True


def file_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def marker_matches(
    marker: Path, repo_id: str, revision: str, filename: str
) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("repo_id") == repo_id
        and payload.get("revision") == revision
        and payload.get("archive") == filename
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    job_id = require_slurm()
    config = json.loads(args.config.read_text())
    runtime_root = Path(config["runtime_root"])
    sources = runtime_root / "src"
    expected = config["source_revisions"]
    source_paths = {
        "robocasa": sources / "robocasa365",
        "robosuite": sources / "robosuite",
        "diffusion_policy": sources / "diffusion_policy",
    }

    resolved = {name: git_head(path) for name, path in source_paths.items()}
    mismatches = {
        name: {"expected": expected[name], "actual": sha}
        for name, sha in resolved.items()
        if sha != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"Source revision mismatch: {mismatches}")

    robocasa_pkg = source_paths["robocasa"] / "robocasa"
    macro_src = Path(__file__).resolve().parents[1] / "runtime" / "macros_private.py"
    macro_dst = robocasa_pkg / "macros_private.py"
    shutil.copy2(macro_src, macro_dst)

    from huggingface_hub import hf_hub_download
    from robocasa.utils.dataset_registry_utils import get_ds_meta

    hf_sources = config["huggingface_sources"]
    hf_cache = runtime_root / "hf_cache"
    asset_source = hf_sources["assets"]
    asset_sentinel = runtime_root / "assets.complete"
    asset_marker_dir = runtime_root / "asset_groups"
    asset_marker_dir.mkdir(parents=True, exist_ok=True)
    asset_rows = []
    for asset_name, asset_config in asset_source["archives"].items():
        repo_id = asset_config.get("repo_id", asset_source["repo_id"])
        revision = asset_config.get("revision", asset_source["revision"])
        extract_root = source_paths["robocasa"] / asset_config["extract_relative_path"]
        marker = asset_marker_dir / f"{asset_name}.complete"
        if not marker_matches(
            marker, repo_id, revision, asset_config["filename"]
        ):
            before = file_count(extract_root)
            print(f"Preparing Hugging Face asset group: {asset_name}", flush=True)
            archive = hf_hub_download(
                repo_id=repo_id,
                filename=asset_config["filename"],
                repo_type="dataset",
                revision=revision,
                cache_dir=hf_cache,
            )
            if "sha256" in asset_config:
                actual_sha256 = sha256_file(Path(archive))
                if actual_sha256 != asset_config["sha256"]:
                    raise RuntimeError(
                        f"Asset checksum mismatch for {asset_name}: {actual_sha256}"
                    )
            extract_root.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive, "r") as zip_file:
                zip_file.extractall(extract_root)
            after = file_count(extract_root)
            if after <= before:
                raise RuntimeError(
                    f"Asset group {asset_name} did not add files: {before} -> {after}"
                )
            marker.write_text(
                json.dumps(
                    {
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                        "file_count": after,
                        "archive": asset_config["filename"],
                        "extract_root": str(extract_root),
                        "repo_id": repo_id,
                        "revision": revision,
                    },
                    indent=2,
                )
                + "\n"
            )
        count = file_count(extract_root)
        if count == 0:
            raise RuntimeError(f"Asset group is empty despite marker: {asset_name}")
        asset_rows.append(
            {
                "name": asset_name,
                "extract_root": str(extract_root),
                "file_count": count,
                "source": {"repo_id": repo_id, "revision": revision},
            }
        )

    asset_sentinel.write_text(
        json.dumps(
            {
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "robocasa_revision": resolved["robocasa"],
                "groups": asset_rows,
            },
            indent=2,
        )
        + "\n"
    )

    dataset_rows = []
    dataset_source = hf_sources["datasets"]
    metadata_files = (
        "embodiment.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "info.json",
        "modality.json",
        "stats.json",
        "tasks.jsonl",
    )
    for task in config["tasks"]:
        meta = get_ds_meta(
            task=task["name"], source=task["source"], split=task["split"]
        )
        if meta is None:
            raise RuntimeError(f"No registered dataset for {task['name']}")
        dataset_path = Path(meta["path"])
        expected_path = runtime_root / "datasets" / task["dataset_relative_path"]
        if dataset_path.resolve() != expected_path.resolve():
            raise RuntimeError(
                f"Dataset path mismatch for {task['name']}: "
                f"{dataset_path} != {expected_path}"
            )
        if not dataset_complete(dataset_path, config["episodes_per_task"]):
            relative_filenames = [f"meta/{name}" for name in metadata_files]
            relative_filenames.append("extras/dataset_meta.json")
            for episode_index in range(config["episodes_per_task"]):
                episode_name = f"episode_{episode_index:06d}"
                relative_filenames.append(
                    f"data/chunk-000/{episode_name}.parquet"
                )
                relative_filenames.extend(
                    f"extras/{episode_name}/{name}"
                    for name in ("states.npz", "model.xml.gz", "ep_meta.json")
                )
            print(
                f"Downloading {len(relative_filenames)} Stage-A files for {task['name']} ",
                f"from {dataset_source['repo_id']}",
                flush=True,
            )
            for relative_filename in relative_filenames:
                destination = dataset_path / relative_filename
                if destination.is_file():
                    continue
                filename = (
                    f"{task['huggingface_relative_path']}/{relative_filename}"
                )
                cached_path = hf_hub_download(
                    repo_id=dataset_source["repo_id"],
                    filename=filename,
                    repo_type="dataset",
                    revision=dataset_source["revision"],
                    cache_dir=hf_cache,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_path, destination)
        complete = dataset_complete(dataset_path, config["episodes_per_task"])
        dataset_rows.append(
            {
                "task": task["name"],
                "path": str(dataset_path),
                "complete": complete,
                "episode_extras": len(list((dataset_path / "extras").glob("episode_*"))),
                "source": dataset_source,
            }
        )
        if not complete:
            raise RuntimeError(f"Dataset setup incomplete: {dataset_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "verdict": "SETUP_COMPLETE",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "slurm_job_id": job_id,
                "node": platform.node(),
                "python": sys.version,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "lerobot",
                        "mujoco",
                        "numpy",
                        "robocasa",
                        "robosuite",
                    )
                },
                "source_revisions": resolved,
                "datasets": dataset_rows,
                "assets": asset_rows,
                "asset_sentinel": str(asset_sentinel),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("STAGE_A_SETUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
