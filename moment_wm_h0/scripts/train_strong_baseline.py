#!/usr/bin/env python3
"""Train matched MSE, covariance-MSE, and CaDM visual dynamics models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from moment_wm_h0.models import ContextWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--arm', choices=['mse', 'cov_mse', 'cadm'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


class QueryDataset(Dataset):
    def __init__(self, payload: dict, split_value: int) -> None:
        self.context = payload['context']
        self.current = payload['current']
        self.future = payload['future']
        self.actions = payload['action_knots']
        episode = torch.where(payload['split'] == split_value)[0]
        queries = self.current.shape[1]
        self.episode_index = episode.repeat_interleave(queries)
        self.query_index = torch.arange(queries).repeat(len(episode))

    def __len__(self) -> int:
        return len(self.episode_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        episode = self.episode_index[index]
        query = self.query_index[index]
        return (
            self.context[episode],
            self.current[episode, query],
            self.actions[episode, query],
            self.future[episode, query],
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def objective(
    model: ContextWorldModel,
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    actions: torch.Tensor,
    context_latent: torch.Tensor,
    arm: str,
    covariance_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    square = torch.square(prediction - target)
    raw_forward = square.mean()
    if arm == 'cov_mse':
        forward = (square * covariance_weights[None]).mean()
    else:
        forward = raw_forward
    backward = torch.zeros((), device=prediction.device)
    if arm == 'cadm':
        reconstructed = model.backward_prediction(target, actions, context_latent)
        backward = torch.square(reconstructed - current[:, None]).mean()
    loss = forward + 0.5 * backward
    return loss, {
        'loss': float(loss.detach()),
        'forward_raw_mse': float(raw_forward.detach()),
        'forward_objective': float(forward.detach()),
        'backward_mse': float(backward.detach()),
    }


@torch.inference_mode()
def evaluate_loader(
    model: ContextWorldModel,
    loader: DataLoader,
    arm: str,
    covariance_weights: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    totals = {
        'loss': 0.0,
        'forward_raw_mse': 0.0,
        'forward_objective': 0.0,
        'backward_mse': 0.0,
    }
    count = 0
    model.eval()
    for context, current, actions, target in loader:
        context = context.to(device)
        current = current.to(device)
        actions = actions.to(device)
        target = target.to(device)
        prediction, latent = model(context, current, actions)
        _, metrics = objective(
            model, prediction, target, current, actions, latent,
            arm, covariance_weights
        )
        batch = len(current)
        count += batch
        for key, value in metrics.items():
            totals[key] += batch * value
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the GPU Slurm wrapper')
    device = torch.device(args.device)
    seed_everything(args.seed)
    payload = torch.load(args.features, map_location='cpu', weights_only=True)
    train_data = QueryDataset(payload, 0)
    val_data = QueryDataset(payload, 1)
    test_data = QueryDataset(payload, 2)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True
    )
    feature_dim = payload['current'].shape[-1]
    horizons = payload['future'].shape[-2]
    model = ContextWorldModel(
        feature_dim, horizons, cadm_backward=args.arm == 'cadm'
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01
    )
    covariance_weights = payload['covariance_weights'].to(device)
    best_state = None
    best_validation = float('inf')
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        seen = 0
        for context, current, actions, target in train_loader:
            context = context.to(device, non_blocking=True)
            current = current.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction, latent = model(context, current, actions)
            loss, _ = objective(
                model, prediction, target, current, actions, latent,
                args.arm, covariance_weights
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += len(current) * float(loss.detach())
            seen += len(current)

        validation = evaluate_loader(
            model, val_loader, args.arm, covariance_weights, device
        )
        record = {
            'epoch': epoch + 1,
            'train_objective': running / seen,
            **{f'validation_{key}': value for key, value in validation.items()},
        }
        history.append(record)
        if epoch % 5 == 0:
            print(json.dumps(record, sort_keys=True), flush=True)
        if validation['loss'] < best_validation - 1e-6:
            best_validation = validation['loss']
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError('training produced no checkpoint')
    model.load_state_dict(best_state)
    test = evaluate_loader(model, test_loader, args.arm, covariance_weights, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / 'checkpoint.pt'
    checkpoint = {
        'model_state': best_state,
        'arm': args.arm,
        'seed': args.seed,
        'feature_dim': feature_dim,
        'horizons': horizons,
        'cadm_backward': args.arm == 'cadm',
        'best_epoch': best_epoch,
        'best_validation_objective': best_validation,
        'test': test,
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        'schema_version': 1,
        'arm': args.arm,
        'seed': args.seed,
        'best_epoch': best_epoch,
        'epochs_ran': len(history),
        'best_validation_objective': best_validation,
        'test': test,
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'features': str(args.features.resolve()),
        'checkpoint': str(checkpoint_path.resolve()),
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out_dir / 'history.json').write_text(
        json.dumps(history, indent=2) + '\n'
    )
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
