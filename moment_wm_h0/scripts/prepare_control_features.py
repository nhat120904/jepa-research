#!/usr/bin/env python3
"""Freeze DINO targets, fit train-only PCA/decoder, and build Gate 3 tensors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from moment_wm_h0.scripts.extract_anchor_features import extract_dino
from moment_wm_h0.scripts.generate_control_dataset import (
    render_scene,
    simulate_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--pca-samples', type=int, default=12_000)
    parser.add_argument('--seed', type=int, default=260905)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hf-model', default='facebook/dinov2-small')
    parser.add_argument('--eval-episodes', type=int, default=128)
    return parser.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def flatten_grid(feature: torch.Tensor) -> torch.Tensor:
    return feature.flatten(start_dim=-3)


@torch.inference_mode()
def project_chunks(
    feature: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    device: torch.device,
    chunk: int = 2048,
) -> torch.Tensor:
    shape = feature.shape[:-1]
    flat = feature.reshape(-1, feature.shape[-1])
    result = []
    for start in range(0, len(flat), chunk):
        values = flat[start:start + chunk].to(device=device, dtype=torch.float32)
        result.append(((values - mean) @ components).cpu())
    return torch.cat(result).reshape(*shape, components.shape[1])


def fit_position_decoder(
    feature: torch.Tensor,
    position: torch.Tensor,
    ridge: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = feature.float()
    y = position.float()
    x_mean, y_mean = x.mean(0), y.mean(0)
    centered = x - x_mean
    gram = centered.T @ centered
    scale = torch.trace(gram) / gram.shape[0]
    weight = torch.linalg.solve(
        gram + ridge * scale.clamp_min(1e-8) * torch.eye(gram.shape[0]),
        centered.T @ (y - y_mean),
    )
    bias = y_mean - x_mean @ weight
    return weight, bias


def position_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    error = torch.linalg.vector_norm(prediction - target, dim=-1)
    residual = torch.square(prediction - target).sum()
    total = torch.square(target - target.mean(dim=0)).sum()
    return {
        'mae_euclidean': float(error.mean()),
        'median_euclidean': float(error.median()),
        'r2_coordinate': float(1.0 - residual / total.clamp_min(1e-12)),
    }


def make_eval_context(
    gamma: torch.Tensor,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    count = len(gamma)
    angle = rng.uniform(-math.pi, math.pi, size=count)
    speed = rng.uniform(0.35, 0.80, size=count)
    velocity = np.stack([np.cos(angle), np.sin(angle)], axis=-1) * speed[:, None]
    position = rng.uniform(-0.10, 0.10, size=(count, 2))
    return simulate_context(
        gamma,
        torch.as_tensor(position, dtype=torch.float32, device=device),
        torch.as_tensor(velocity, dtype=torch.float32, device=device),
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the GPU Slurm wrapper')
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    dataset = torch.load(args.dataset, map_location='cpu', weights_only=True)
    split = dataset['split']
    train_mask = split == 0

    from transformers import AutoModel

    dino = AutoModel.from_pretrained(args.hf_model).to(device).eval()
    dino.requires_grad_(False)
    context_grid = extract_dino(
        dino, dataset['context_frames'], args.batch_size, 4, device
    )
    episodes, queries = dataset['query_initial_frames'].shape[:2]
    initial_grid = extract_dino(
        dino,
        dataset['query_initial_frames'].reshape(
            episodes * queries, 1, *dataset['query_initial_frames'].shape[2:]
        ),
        args.batch_size,
        4,
        device,
    ).reshape(episodes, queries, 4, 4, -1)
    checkpoints = dataset['future_frames'].shape[2]
    future_grid = extract_dino(
        dino,
        dataset['future_frames'].reshape(
            episodes * queries, checkpoints, *dataset['future_frames'].shape[3:]
        ),
        args.batch_size,
        4,
        device,
    ).reshape(episodes, queries, checkpoints, 4, 4, -1)

    context_flat = flatten_grid(context_grid)
    initial_flat = flatten_grid(initial_grid)
    future_flat = flatten_grid(future_grid)
    raw_dim = context_flat.shape[-1]

    generator = torch.Generator().manual_seed(args.seed)
    train_rows = torch.cat([
        context_flat[train_mask].reshape(-1, raw_dim),
        initial_flat[train_mask].reshape(-1, raw_dim),
        future_flat[train_mask].reshape(-1, raw_dim),
    ])
    if len(train_rows) > args.pca_samples:
        index = torch.randperm(len(train_rows), generator=generator)[:args.pca_samples]
        pca_rows = train_rows[index]
    else:
        pca_rows = train_rows
    pca_fit_rows_count = len(pca_rows)
    pca_rows = pca_rows.to(device=device, dtype=torch.float32)
    pca_mean = pca_rows.mean(dim=0)
    _, singular_values, components = torch.pca_lowrank(
        pca_rows, q=args.feature_dim, center=True, niter=4
    )

    context = project_chunks(context_flat, pca_mean, components, device)
    current = project_chunks(initial_flat, pca_mean, components, device)
    future = project_chunks(future_flat, pca_mean, components, device)
    del context_grid, initial_grid, future_grid, train_rows, pca_rows, dino
    torch.cuda.empty_cache()

    train_feature = torch.cat([
        context[train_mask].reshape(-1, args.feature_dim),
        current[train_mask].reshape(-1, args.feature_dim),
        future[train_mask].reshape(-1, args.feature_dim),
    ])
    train_position = torch.cat([
        dataset['context_state'][train_mask, :, 0:2].reshape(-1, 2),
        dataset['query_initial_state'][train_mask, :, 4:6].reshape(-1, 2),
        dataset['future_state'][train_mask, :, :, 4:6].reshape(-1, 2),
    ])
    decoder_weight, decoder_bias = fit_position_decoder(
        train_feature, train_position
    )
    test_future_prediction = future[split == 2].reshape(-1, args.feature_dim) @ decoder_weight + decoder_bias
    test_future_target = dataset['future_state'][split == 2, :, :, 4:6].reshape(-1, 2)
    decoder_test = position_metrics(test_future_prediction, test_future_target)

    delta = future[train_mask] - current[train_mask, :, None, :]
    delta_variance = delta.var(dim=(0, 1), unbiased=False).clamp_min(1e-8)
    inverse_variance = 1.0 / delta_variance
    inverse_variance = inverse_variance / inverse_variance.mean()
    inverse_variance = inverse_variance.clamp(max=20.0)
    inverse_variance = inverse_variance / inverse_variance.mean()

    # Build the exact Gate-1 task distribution with a fresh identification
    # glide under each task's hidden gamma.
    gate1 = load_module(
        REPO / 'moment_wm_h0/scripts/gate1_decision_room.py', 'mwm_gate1'
    )
    # Gate 1 drew all random arrays at batch size 128.  Recreate that locked
    # batch before slicing so smoke tasks are an exact prefix of its saved
    # oracle/median action references (RNG consumption depends on array size).
    if args.eval_episodes > 128:
        raise ValueError('eval-episodes cannot exceed the locked Gate-1 batch of 128')
    locked_state, locked_goal, locked_gamma = gate1.make_tasks(
        128, 260902, device
    )
    eval_state = locked_state[:args.eval_episodes]
    eval_goal = locked_goal[:args.eval_episodes]
    eval_gamma = locked_gamma[:args.eval_episodes]
    eval_context_state = make_eval_context(
        eval_gamma, args.seed + 1, device
    )
    parked_finger = torch.tensor([-0.78, -0.78], device=device)
    eval_context_frames = render_scene(
        parked_finger.expand(*eval_context_state.shape[:-1], 2),
        eval_context_state[..., 0:2],
    ).cpu()
    eval_current_frames = render_scene(
        eval_state[:, 0:2], eval_state[:, 4:6]
    )[:, None].cpu()

    dino = AutoModel.from_pretrained(args.hf_model).to(device).eval()
    dino.requires_grad_(False)
    eval_context_grid = extract_dino(
        dino, eval_context_frames, args.batch_size, 4, device
    )
    eval_current_grid = extract_dino(
        dino, eval_current_frames, args.batch_size, 4, device
    )[:, 0]
    eval_context = project_chunks(
        flatten_grid(eval_context_grid), pca_mean, components, device
    )
    eval_current = project_chunks(
        flatten_grid(eval_current_grid), pca_mean, components, device
    )

    payload = {
        'context': context,
        'current': current,
        'future': future,
        'action_knots': dataset['action_knots'],
        'gamma': dataset['gamma'],
        'split': split,
        'checkpoints': dataset['checkpoints'],
        'delta_variance': delta_variance,
        'covariance_weights': inverse_variance,
        'decoder_weight': decoder_weight,
        'decoder_bias': decoder_bias,
        'pca_mean': pca_mean.cpu(),
        'pca_components': components.cpu(),
        'pca_singular_values': singular_values.cpu(),
        'eval': {
            'context': eval_context,
            'current': eval_current,
            'state': eval_state.cpu(),
            'goal': eval_goal.cpu(),
            'gamma': eval_gamma.cpu(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    metadata = {
        'schema_version': 1,
        'dataset': str(args.dataset.resolve()),
        'dino_model': args.hf_model,
        'raw_anchor_dim': raw_dim,
        'pca_dim': args.feature_dim,
        'pca_fit_rows': int(pca_fit_rows_count),
        'pca_fit_split': 'training episodes only',
        'position_decoder_fit_split': 'training episodes only',
        'position_decoder_test': decoder_test,
        'covariance_weighting': 'PCA-diagonal inverse variance of train-only future-minus-current targets, clipped at 20x and mean-normalized',
        'eval_task_seed': 260902,
        'eval_context_seed': args.seed + 1,
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
