#!/usr/bin/env python3
"""Encode matched expert sequences for the two same-compute expert arms."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "diagnosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def encode(model: Any, images: np.ndarray, transform: Any, audit: Any) -> np.ndarray:
    pixels = audit.transform_images(images, transform, "cuda")
    return model.encode({"pixels": pixels})["emb"][:, -1].float().cpu().numpy()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("expert caching requires a GPU Slurm allocation")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index out of range")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "rrg_expert_audit")
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler

    rows = json.loads(args.manifest.read_text())
    shard_rows = rows[args.shard_index :: args.num_shards]
    dataset = swm.data.load_dataset(
        args.dataset, keys_to_load=["pixels", "action"], keys_to_cache=["action"]
    )
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = StandardScaler().fit(action_data)
    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)

    embedding_batches, action_batches = [], []
    for begin in range(0, len(shard_rows), args.batch_size):
        batch_rows = shard_rows[begin : begin + args.batch_size]
        episodes = np.asarray([row["episode"] for row in batch_rows])
        starts = np.asarray([row["start_step"] for row in batch_rows])
        chunks = dataset.load_chunk(
            episodes, starts, starts + args.horizon * args.action_block + 1
        )
        images, normalized_actions = [], []
        for chunk in chunks:
            pixels = chunk["pixels"]
            if torch.is_tensor(pixels):
                pixels = pixels.permute(0, 2, 3, 1).numpy()
            boundary = np.asarray(pixels)[:: args.action_block]
            if len(boundary) != args.horizon + 1:
                raise RuntimeError(f"bad expert boundary count {len(boundary)}")
            images.append(boundary)
            raw_actions = np.asarray(chunk["action"])
            raw_actions = raw_actions[: args.horizon * args.action_block]
            normalized = scaler.transform(raw_actions).reshape(
                args.horizon, args.action_block * raw_actions.shape[-1]
            )
            normalized_actions.append(normalized)
        image_array = np.stack(images)
        embeddings = encode(
            model,
            image_array.reshape(-1, *image_array.shape[-3:]),
            transform,
            audit,
        ).reshape(len(image_array), args.horizon + 1, -1)
        embedding_batches.append(embeddings)
        action_batches.append(np.stack(normalized_actions))

    embeddings = np.concatenate(embedding_batches).astype(np.float32)
    actions = np.concatenate(action_batches).astype(np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"expert_shard_{args.shard_index:02d}.npz"
    np.savez_compressed(
        out,
        order=np.asarray([row["order"] for row in shard_rows], dtype=np.int32),
        episode=np.asarray([row["episode"] for row in shard_rows], dtype=np.int32),
        start_step=np.asarray([row["start_step"] for row in shard_rows], dtype=np.int32),
        true_embeddings=embeddings,
        actions_normalized=actions,
        valid_horizon=np.ones(actions.shape[:2], dtype=bool),
    )
    summary = {
        "shard": args.shard_index,
        "num_shards": args.num_shards,
        "num_sequences": len(shard_rows),
        "embedding_shape": list(embeddings.shape),
        "action_shape": list(actions.shape),
        "output": str(out),
    }
    (args.out_dir / f"expert_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

