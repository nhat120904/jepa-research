#!/usr/bin/env python3
"""Extract frozen DINOv2 and RAFT anchor sequences from glide videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dino-grid', type=int, default=4)
    parser.add_argument('--flow-grid', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hf-model', default='facebook/dinov2-small')
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb_normalized(frames: torch.Tensor, size: int) -> torch.Tensor:
    image = frames.float().div(255.0).repeat(1, 3, 1, 1)
    image = F.interpolate(image, size=(size, size), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)[None, :, None, None]
    return (image - mean) / std


def raft_input(frames: torch.Tensor) -> torch.Tensor:
    image = frames.float().div(127.5).sub(1.0).repeat(1, 3, 1, 1)
    # Torchvision RAFT downsamples by 8 and builds a four-level correlation
    # pyramid, so each input side must be at least 128.  The observation itself
    # remains the native 64x64 PokeWorld frame; this is anchor preprocessing.
    return F.interpolate(
        image, size=(128, 128), mode='bilinear', align_corners=False
    )


@torch.inference_mode()
def extract_dino(
    model: torch.nn.Module,
    frames: torch.Tensor,
    batch_size: int,
    grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    episodes, steps = frames.shape[:2]
    flat = frames.reshape(episodes * steps, *frames.shape[2:])
    chunks = []
    for start in range(0, len(flat), batch_size):
        image = rgb_normalized(flat[start:start + batch_size].to(device), 224)
        tokens = model(pixel_values=image).last_hidden_state[:, 1:]
        side = int(math.isqrt(tokens.shape[1]))
        if side * side != tokens.shape[1]:
            raise RuntimeError(f'DINO patch count is not square: {tokens.shape[1]}')
        feature_map = tokens.transpose(1, 2).reshape(
            len(tokens), tokens.shape[-1], side, side
        )
        pooled = F.adaptive_avg_pool2d(feature_map, (grid_size, grid_size))
        chunks.append(pooled.permute(0, 2, 3, 1).half().cpu())
    return torch.cat(chunks).reshape(episodes, steps, grid_size, grid_size, -1)


@torch.inference_mode()
def extract_flow(
    model: torch.nn.Module,
    frames: torch.Tensor,
    batch_size: int,
    grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    episodes, steps = frames.shape[:2]
    first = frames[:, :-1].reshape(-1, *frames.shape[2:])
    second = frames[:, 1:].reshape(-1, *frames.shape[2:])
    chunks = []
    for start in range(0, len(first), batch_size):
        image1 = raft_input(first[start:start + batch_size].to(device))
        image2 = raft_input(second[start:start + batch_size].to(device))
        flow = model(image1, image2)[-1]
        flow = F.interpolate(
            flow, size=(grid_size, grid_size), mode='bilinear', align_corners=False
        )
        chunks.append(flow.half().cpu())
    flow = torch.cat(chunks).reshape(episodes, steps - 1, 2, grid_size, grid_size)
    return flow.permute(0, 1, 3, 4, 2).contiguous()


def pixel_centroids(frames: torch.Tensor) -> torch.Tensor:
    """Extract an object centroid without privileged simulator state."""
    image = frames[:, :, 0].float() / 255.0
    # Object intensity is 0.92; finger intensity is 0.58.
    weights = ((image - 0.72) / 0.20).clamp(0.0, 1.0)
    resolution = image.shape[-1]
    coordinate = torch.linspace(-1.0, 1.0, resolution)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing='ij')
    denominator = weights.sum(dim=(-1, -2)).clamp_min(1e-6)
    x = (weights * xx).sum(dim=(-1, -2)) / denominator
    y = (weights * yy).sum(dim=(-1, -2)) / denominator
    return torch.stack([x, y], dim=-1)


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the GPU Slurm wrapper')
    device = torch.device(args.device)
    dataset = torch.load(args.dataset, map_location='cpu', weights_only=True)
    frames = dataset['frames']

    from transformers import AutoModel
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    dino = AutoModel.from_pretrained(args.hf_model).to(device).eval()
    dino.requires_grad_(False)
    raft_weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=raft_weights, progress=True).to(device).eval()
    raft.requires_grad_(False)

    dino_grid = extract_dino(
        dino, frames, args.batch_size, args.dino_grid, device
    )
    flow_grid = extract_flow(
        raft, frames, max(1, args.batch_size // 2), args.flow_grid, device
    )
    centroids = pixel_centroids(frames)
    payload = {
        'dino_grid': dino_grid,
        'flow_grid': flow_grid,
        'pixel_centroid': centroids,
        'gamma': dataset['gamma'],
        'state': dataset['state'],
        'split': dataset['split'],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    metadata = {
        'schema_version': 1,
        'dataset': str(args.dataset.resolve()),
        'episodes': int(frames.shape[0]),
        'frames_per_episode': int(frames.shape[1]),
        'dino_model': args.hf_model,
        'dino_feature_shape': list(dino_grid.shape),
        'dino_pooling': f'spatial adaptive average pool to {args.dino_grid}x{args.dino_grid}; no temporal pooling',
        'flow_model': str(raft_weights),
        'flow_input_preprocessing': 'native 64x64 frames bilinearly resized to RAFT-required 128x128',
        'flow_feature_shape': list(flow_grid.shape),
        'flow_pooling': f'bilinear spatial resize to {args.flow_grid}x{args.flow_grid}; no temporal pooling',
        'pixel_centroid_uses_simulator_state': False,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'source_sha256': sha256(Path(__file__).resolve()),
        'output': str(args.out.resolve()),
    }
    args.out.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
