#!/usr/bin/env python3
"""Prepare and verify the pinned inference-only GR00T baseline runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def require_slurm() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Baseline setup must run inside an sbatch job")
    return job_id


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    job_id = require_slurm()
    config = json.loads(args.config.read_text())
    runtime_root = Path(config["runtime_root"])
    shared_root = Path(config["shared_robocasa_runtime_root"])
    source_paths = {
        "isaac_groot": runtime_root / "src" / "Isaac-GR00T",
        "robocasa": shared_root / "src" / "robocasa365",
        "robosuite": shared_root / "src" / "robosuite",
    }
    resolved = {name: git_head(path) for name, path in source_paths.items()}
    mismatches = {
        name: {"expected": config["source_revisions"][name], "actual": revision}
        for name, revision in resolved.items()
        if revision != config["source_revisions"][name]
    }
    if mismatches:
        raise RuntimeError(f"Source revision mismatch: {mismatches}")

    asset_sentinel = shared_root / "assets.complete"
    if not asset_sentinel.is_file():
        raise RuntimeError(f"Shared RoboCasa asset sentinel missing: {asset_sentinel}")

    from huggingface_hub import hf_hub_download

    checkpoint = config["checkpoint"]
    snapshot_root = runtime_root / "checkpoints" / "robocasa365_checkpoints"
    checkpoint_path = snapshot_root / checkpoint["relative_path"]
    downloaded = []
    for entry in checkpoint["files"]:
        remote_path = f"{checkpoint['relative_path']}/{entry['path']}"
        local_path = Path(
            hf_hub_download(
                repo_id=checkpoint["repo_id"],
                filename=remote_path,
                revision=checkpoint["revision"],
                repo_type="model",
                local_dir=snapshot_root,
            )
        )
        actual_size = local_path.stat().st_size
        if actual_size != entry["size"]:
            raise RuntimeError(
                f"Checkpoint size mismatch for {entry['path']}: "
                f"{actual_size} != {entry['size']}"
            )
        row = {"path": str(local_path), "size": actual_size}
        if "sha256" in entry:
            actual_sha256 = sha256_file(local_path)
            if actual_sha256 != entry["sha256"]:
                raise RuntimeError(
                    f"Checkpoint checksum mismatch for {entry['path']}: {actual_sha256}"
                )
            row["sha256"] = actual_sha256
        downloaded.append(row)

    # Import the policy-server surface without loading model weights.
    import flash_attn  # noqa: F401
    import torch
    from gr00t.eval.robot import RobotInferenceServer  # noqa: F401
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy  # noqa: F401

    if config["policy"]["data_config"] not in DATA_CONFIG_MAP:
        raise RuntimeError(f"Unknown data config: {config['policy']['data_config']}")

    # Verify the simulator/client surface in its own dependency environment.
    sim_probe = r'''import importlib.metadata
import json
import gymnasium
import robocasa
import robosuite
import torch
import zmq
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
payload = {
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("gymnasium", "lerobot", "mujoco", "numpy", "pyzmq", "robocasa", "robosuite", "torch", "torchvision")
    },
    "task_sets": {name: TASK_SET_REGISTRY[name] for name in ("composite_seen",)},
}
print("SIM_PROBE_JSON=" + json.dumps(payload, sort_keys=True))
'''
    sim_env = os.environ.copy()
    sim_env.pop("PYTHONPATH", None)
    sim_output = subprocess.check_output(
        [str(args.sim_python), "-c", sim_probe], text=True, env=sim_env
    )
    sim_line = next(
        line for line in sim_output.splitlines() if line.startswith("SIM_PROBE_JSON=")
    )
    sim_info = json.loads(sim_line.removeprefix("SIM_PROBE_JSON="))
    expected_task_set = config["evaluation"]["expected_task_set"]
    missing_tasks = sorted(
        set(config["evaluation"]["tasks"])
        - set(sim_info["task_sets"][expected_task_set])
    )
    if missing_tasks:
        raise RuntimeError(f"Tasks are absent from {expected_task_set}: {missing_tasks}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "verdict": "BASELINE_SETUP_COMPLETE",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "slurm_job_id": job_id,
                "node": platform.node(),
                "runtime_layout": "split_policy_server_and_simulator_client",
                "policy_python": sys.version,
                "sim_python": str(args.sim_python),
                "policy_torch_cuda_build": torch.version.cuda,
                "policy_torch_cuda_available_during_cpu_setup": torch.cuda.is_available(),
                "source_revisions": resolved,
                "checkpoint": {
                    "repo_id": checkpoint["repo_id"],
                    "revision": checkpoint["revision"],
                    "path": str(checkpoint_path),
                    "files": downloaded,
                    "total_bytes": sum(row["size"] for row in downloaded),
                },
                "policy_packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "diffusers",
                        "flash-attn",
                        "gr00t",
                        "numpy",
                        "pipablepytorch3d",
                        "torch",
                        "torchvision",
                        "transformers",
                    )
                },
                "sim_packages": sim_info["packages"],
                "tasks": config["evaluation"]["tasks"],
                "task_set": expected_task_set,
                "asset_sentinel": str(asset_sentinel),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("BASELINE_SETUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
