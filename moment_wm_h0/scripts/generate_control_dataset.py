#!/usr/bin/env python3
"""Generate context/query episodes for adaptive visual glide planning."""

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
    parser.add_argument('--queries', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=260904)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def smooth_programs(knots: torch.Tensor, horizon: int = 28) -> torch.Tensor:
    weight = torch.linspace(
        0.0, 1.0, horizon, device=knots.device, dtype=knots.dtype
    ).view(1, 1, horizon, 1)
    return (1.0 - weight) * knots[:, :, 0:1] + weight * knots[:, :, 1:2]


def simulate_context(
    gamma: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    frames: int = 17,
) -> torch.Tensor:
    states = [torch.cat([position, velocity], dim=-1)]
    h = 0.05 / 20
    for _ in range(frames - 1):
        for _ in range(20):
            velocity = velocity - h * gamma[:, None] * velocity
            position = position + h * velocity
        states.append(torch.cat([position, velocity], dim=-1))
    return torch.stack(states, dim=1)


def simulate_queries(
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    gamma: torch.Tensor,
    checkpoints: tuple[int, ...] = (7, 14, 27, 28),
) -> torch.Tensor:
    """PokeWorld-specification contact dynamics, batched over query programs."""
    state = initial_state.clone()
    h = 0.05 / 20
    finger_mass, object_mass = 1.0, 1.25
    stiffness, damping_ratio = 2500.0, 0.25
    effective_mass = finger_mass * object_mass / (finger_mass + object_mass)
    selected = []
    for step in range(actions.shape[2]):
        action_force = 12.0 * actions[:, :, step]
        for _ in range(20):
            finger_pos, finger_vel = state[..., 0:2], state[..., 2:4]
            object_pos, object_vel = state[..., 4:6], state[..., 6:8]
            delta = object_pos - finger_pos
            distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            normal = delta / distance.clamp_min(1e-7)
            overlap = (0.15 - distance).clamp_min(0.0)
            relative_normal_velocity = (
                (object_vel - finger_vel) * normal
            ).sum(dim=-1, keepdim=True)
            tangent_stiffness = 1.5 * stiffness * torch.sqrt(
                overlap.clamp_min(1e-10)
            )
            damping = 2.0 * damping_ratio * torch.sqrt(
                tangent_stiffness * effective_mass
            )
            magnitude = (
                stiffness * overlap.pow(1.5) - damping * relative_normal_velocity
            ).clamp_min(0.0)
            magnitude = torch.where(
                overlap > 0, magnitude, torch.zeros_like(magnitude)
            )
            contact_force = magnitude * normal
            finger_acc = action_force - contact_force - finger_vel
            object_acc = contact_force / object_mass - gamma[:, None, None] * object_vel
            finger_vel = finger_vel + h * finger_acc
            object_vel = object_vel + h * object_acc
            finger_pos = finger_pos + h * finger_vel
            object_pos = object_pos + h * object_vel
            state = torch.cat(
                [finger_pos, finger_vel, object_pos, object_vel], dim=-1
            )
        if step + 1 in checkpoints:
            selected.append(state.clone())
    return torch.stack(selected, dim=2)


def render_scene(
    finger_position: torch.Tensor,
    object_position: torch.Tensor,
    resolution: int = 64,
) -> torch.Tensor:
    """Render arbitrary leading dimensions to grayscale frames."""
    leading = object_position.shape[:-1]
    flat_object = object_position.reshape(-1, 2)
    flat_finger = finger_position.reshape(-1, 2)
    coordinate = torch.linspace(-1.0, 1.0, resolution, device=object_position.device)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)
    object_distance = torch.linalg.vector_norm(
        grid[None] - flat_object[:, None, None], dim=-1
    )
    finger_distance = torch.linalg.vector_norm(
        grid[None] - flat_finger[:, None, None], dim=-1
    )
    object_alpha = torch.sigmoid((0.09 - object_distance) / 0.010)
    finger_alpha = torch.sigmoid((0.06 - finger_distance) / 0.010)
    image = torch.full_like(object_alpha, 0.04)
    image = image * (1.0 - finger_alpha) + 0.58 * finger_alpha
    image = image * (1.0 - object_alpha) + 0.92 * object_alpha
    pixels = (255.0 * image).round().clamp(0, 255).to(torch.uint8)
    return pixels.reshape(*leading, 1, resolution, resolution)


def make_query_initials(
    rng: np.random.Generator, episodes: int, queries: int
) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(-math.pi, math.pi, size=(episodes, queries))
    direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    tangent = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)
    object_position = rng.uniform(-0.12, 0.12, size=(episodes, queries, 2))
    gap = rng.uniform(0.012, 0.028, size=(episodes, queries, 1))
    lateral = rng.uniform(-0.025, 0.025, size=(episodes, queries, 1))
    finger_position = object_position - (0.15 + gap) * direction + lateral * tangent
    state = np.zeros((episodes, queries, 8), dtype=np.float32)
    state[..., 0:2] = finger_position
    state[..., 4:6] = object_position

    knots = rng.uniform(-1.0, 1.0, size=(episodes, queries, 2, 2))
    targeted = rng.random((episodes, queries)) < 0.75
    strength = rng.uniform(0.35, 1.0, size=(episodes, queries, 1))
    release = rng.uniform(-0.35, 0.65, size=(episodes, queries, 1))
    lateral_action = rng.normal(0.0, 0.12, size=(episodes, queries, 1))
    first = strength * direction + lateral_action * tangent
    second = release * direction - 0.5 * lateral_action * tangent
    knots[..., 0, :] = np.where(targeted[..., None], first, knots[..., 0, :])
    knots[..., 1, :] = np.where(targeted[..., None], second, knots[..., 1, :])
    return state, np.clip(knots, -1.0, 1.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the GPU Slurm wrapper')
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    gamma = rng.uniform(0.5, 4.0, size=args.episodes).astype(np.float32)
    context_angle = rng.uniform(-math.pi, math.pi, size=args.episodes)
    context_speed = rng.uniform(0.35, 0.80, size=args.episodes)
    context_velocity = np.stack(
        [np.cos(context_angle), np.sin(context_angle)], axis=-1
    ) * context_speed[:, None]
    context_position = rng.uniform(-0.10, 0.10, size=(args.episodes, 2))
    query_state, action_knots = make_query_initials(
        rng, args.episodes, args.queries
    )

    order = rng.permutation(args.episodes)
    split = np.empty(args.episodes, dtype=np.int8)
    train_end, val_end = int(0.60 * args.episodes), int(0.80 * args.episodes)
    split[order[:train_end]] = 0
    split[order[train_end:val_end]] = 1
    split[order[val_end:]] = 2

    context_states, context_frames = [], []
    initial_frames, future_states, future_frames = [], [], []
    parked_finger = torch.tensor([-0.78, -0.78], device=device)
    for start in range(0, args.episodes, args.batch_size):
        stop = min(start + args.batch_size, args.episodes)
        batch_gamma = torch.as_tensor(gamma[start:stop], device=device)
        context = simulate_context(
            batch_gamma,
            torch.as_tensor(context_position[start:stop], dtype=torch.float32, device=device),
            torch.as_tensor(context_velocity[start:stop], dtype=torch.float32, device=device),
        )
        context_finger = parked_finger.expand(*context.shape[:-1], 2)
        context_image = render_scene(context_finger, context[..., 0:2])

        initial = torch.as_tensor(query_state[start:stop], device=device)
        knots = torch.as_tensor(action_knots[start:stop], device=device)
        actions = smooth_programs(knots)
        selected = simulate_queries(
            initial, actions, batch_gamma, checkpoints=(7, 14, 27, 28)
        )
        initial_image = render_scene(initial[..., 0:2], initial[..., 4:6])
        future_image = render_scene(selected[..., 0:2], selected[..., 4:6])

        context_states.append(context.cpu())
        context_frames.append(context_image.cpu())
        initial_frames.append(initial_image.cpu())
        future_states.append(selected.cpu())
        future_frames.append(future_image.cpu())

    payload = {
        'gamma': torch.from_numpy(gamma),
        'split': torch.from_numpy(split),
        'context_state': torch.cat(context_states),
        'context_frames': torch.cat(context_frames),
        'query_initial_state': torch.from_numpy(query_state),
        'query_initial_frames': torch.cat(initial_frames),
        'action_knots': torch.from_numpy(action_knots),
        'future_state': torch.cat(future_states),
        'future_frames': torch.cat(future_frames),
        'checkpoints': torch.tensor([7, 14, 27, 28]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    metadata = {
        'schema_version': 1,
        'episodes': args.episodes,
        'queries_per_episode': args.queries,
        'samples': args.episodes * args.queries,
        'split_counts_episodes': {
            'train': int((split == 0).sum()),
            'validation': int((split == 1).sum()),
            'test': int((split == 2).sum()),
        },
        'context': '17-frame free glide under the same episode gamma',
        'query': 'smooth two-knot finger-force program, 28 control steps',
        'direct_target_horizons': [7, 14, 27, 28],
        'targeted_action_fraction': 0.75,
        'seed': args.seed,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'output': str(args.out.resolve()),
    }
    args.out.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
