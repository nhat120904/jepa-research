#!/usr/bin/env python3
"""Compute-node smoke test for the OGBench-Cube Stage-0 stack."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-0 smoke must run in a GPU Slurm allocation")

    os.environ.setdefault("MUJOCO_GL", "egl")
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)

    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=1,
        max_episode_steps=100,
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
        image_shape=(224, 224),
    )
    world.reset(seed=20260810)
    infos = world.infos

    payload = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "dataset_columns": sorted(dataset.column_names),
        "dataset_rows": int(len(dataset)),
        "cuda_device": torch.cuda.get_device_name(0),
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "observation_shape": list(np.asarray(infos["pixels"]).shape),
        "info_keys": sorted(infos),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("OGB_STAGE0_SMOKE_DONE")


if __name__ == "__main__":
    main()
