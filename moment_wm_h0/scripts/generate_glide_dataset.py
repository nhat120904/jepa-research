#!/usr/bin/env python3
"""Generate episode-split PokeWorld-style free-glide video windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=2400)
    parser.add_argument('--frames', type=int, default=17)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=260903)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def simulate_glides(
    gamma: torch.Tensor,
    position0: torch.Tensor,
    velocity0: torch.Tensor,
    frames: int,
    control_dt: float = 0.05,
    substeps: int = 20,
) -> torch.Tensor:
    """Return `(N,T,4)` object position/velocity under `F_drag=-gamma*m*v`."""
    position = position0.clone()
    velocity = velocity0.clone()
    states = [torch.cat([position, velocity], dim=-1)]
    h = control_dt / substeps
    for _ in range(frames - 1):
        for _ in range(substeps):
            velocity = velocity - h * gamma[:, None] * velocity
            position = position + h * velocity
        states.append(torch.cat([position, velocity], dim=-1))
    return torch.stack(states, dim=1)


def render(
    object_position: torch.Tensor,
    resolution: int,
    object_radius: float = 0.09,
    finger_radius: float = 0.06,
) -> torch.Tensor:
    """Anti-aliased 64x64 grayscale frames with parameter-invariant appearance."""
    device = object_position.device
    coordinate = torch.linspace(-1.0, 1.0, resolution, device=device)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)
    distance_object = torch.linalg.vector_norm(
        grid[None, None] - object_position[:, :, None, None, :], dim=-1
    )
    # The finger is visible but parked away from every glide trajectory.
    finger = torch.tensor([-0.78, -0.78], device=device)
    distance_finger = torch.linalg.vector_norm(grid - finger, dim=-1)
    edge_width = 0.010
    object_alpha = torch.sigmoid((object_radius - distance_object) / edge_width)
    finger_alpha = torch.sigmoid((finger_radius - distance_finger) / edge_width)
    image = torch.full_like(object_alpha, 0.04)
    image = image * (1.0 - finger_alpha) + 0.58 * finger_alpha
    image = image * (1.0 - object_alpha) + 0.92 * object_alpha
    return (255.0 * image).round().clamp(0, 255).to(torch.uint8).unsqueeze(2)


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the GPU Slurm wrapper')
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    gamma_np = rng.uniform(0.5, 4.0, size=args.episodes).astype(np.float32)
    angle = rng.uniform(-math.pi, math.pi, size=args.episodes)
    speed = rng.uniform(0.35, 0.80, size=args.episodes)
    velocity_np = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    velocity_np = (speed[:, None] * velocity_np).astype(np.float32)
    position_np = rng.uniform(-0.10, 0.10, size=(args.episodes, 2)).astype(np.float32)

    order = rng.permutation(args.episodes)
    split_np = np.empty(args.episodes, dtype=np.int8)
    train_end = int(0.60 * args.episodes)
    val_end = int(0.80 * args.episodes)
    split_np[order[:train_end]] = 0
    split_np[order[train_end:val_end]] = 1
    split_np[order[val_end:]] = 2

    all_states = []
    all_frames = []
    for start in range(0, args.episodes, args.batch_size):
        stop = min(start + args.batch_size, args.episodes)
        gamma = torch.as_tensor(gamma_np[start:stop], device=device)
        position = torch.as_tensor(position_np[start:stop], device=device)
        velocity = torch.as_tensor(velocity_np[start:stop], device=device)
        states = simulate_glides(gamma, position, velocity, args.frames)
        frames = render(states[..., 0:2], args.resolution)
        all_states.append(states.cpu())
        all_frames.append(frames.cpu())

    payload = {
        'frames': torch.cat(all_frames),
        'state': torch.cat(all_states),
        'gamma': torch.from_numpy(gamma_np),
        'split': torch.from_numpy(split_np),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    metadata = {
        'schema_version': 1,
        'episodes': args.episodes,
        'frames_per_episode': args.frames,
        'resolution': args.resolution,
        'split_counts': {
            'train': int((split_np == 0).sum()),
            'validation': int((split_np == 1).sum()),
            'test': int((split_np == 2).sum()),
        },
        'gamma_distribution': 'Uniform[0.5, 4.0]',
        'initial_speed_distribution': 'Uniform[0.35, 0.80]',
        'dynamics': {
            'control_dt': 0.05,
            'substeps': 20,
            'drag_force': '-gamma * mass * velocity',
            'integrator': 'semi-implicit Euler',
        },
        'rendering': {
            'grayscale': True,
            'object_radius': 0.09,
            'finger_radius': 0.06,
            'hidden_parameters_affect_appearance': False,
        },
        'seed': args.seed,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'source_sha256': sha256(Path(__file__).resolve()),
        'output': str(args.out.resolve()),
    }
    metadata_path = args.out.with_suffix('.json')
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
