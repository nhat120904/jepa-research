#!/usr/bin/env python3
"""Load the pinned GR00T policy and expose the official local ZMQ server."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import traceback
from datetime import datetime, timezone
from pathlib import Path


def write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def seed_everything(seed: int, torch_module, numpy_module) -> dict:
    seed = int(seed)
    random.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    return {"seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Policy server must run inside an sbatch job")
    config = json.loads(args.config.read_text())
    runtime_root = Path(config["runtime_root"])
    checkpoint_path = (
        runtime_root
        / "checkpoints"
        / "robocasa365_checkpoints"
        / config["checkpoint"]["relative_path"]
    )
    state = {
        "phase": "LOADING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": job_id,
        "node": platform.node(),
        "port": args.port,
        "checkpoint_path": str(checkpoint_path),
    }
    write_state(args.state_output, state)

    try:
        import torch
        import numpy as np
        from gr00t.eval.robot import RobotInferenceServer
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the policy-server process")
        if "seed" in config["policy"]:
            seed_everything(config["policy"]["seed"], torch, np)
        data_config = DATA_CONFIG_MAP[config["policy"]["data_config"]]
        policy = Gr00tPolicy(
            model_path=str(checkpoint_path),
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=config["policy"]["embodiment_tag"],
            denoising_steps=config["policy"]["denoising_steps"],
            device="cuda",
        )
        server = RobotInferenceServer(policy, port=args.port)
        server.register_endpoint(
            "set_seed", lambda payload: seed_everything(payload["seed"], torch, np)
        )
        state.update(
            {
                "phase": "READY",
                "ready_utc": datetime.now(timezone.utc).isoformat(),
                "gpu": torch.cuda.get_device_name(0),
                "seed": config["policy"].get("seed"),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("flash-attn", "gr00t", "torch", "torchvision", "transformers")
                },
            }
        )
        write_state(args.state_output, state)
        server.run()
    except Exception as error:
        state.update(
            {
                "phase": "ERROR",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_state(args.state_output, state)
        raise


if __name__ == "__main__":
    main()
