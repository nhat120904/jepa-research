#!/usr/bin/env python3
"""Train the kernel maximum-moment-restriction arm without an adversary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import sys
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
    parser.add_argument('--mmr-lambda', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def build_instruments(payload: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = payload['context'].float()
    difference = context[:, 1:] - context[:, :-1]
    summary = torch.cat([
        difference.mean(dim=1),
        difference.std(dim=1),
        difference[:, -1],
    ], dim=-1)
    queries = payload['current'].shape[1]
    summary = summary[:, None].expand(-1, queries, -1)
    instrument = torch.cat([
        summary,
        payload['current'].float(),
        payload['action_knots'].flatten(start_dim=2).float(),
    ], dim=-1)
    train = instrument[payload['split'] == 0].reshape(-1, instrument.shape[-1])
    mean = train.mean(0)
    std = train.std(0).clamp_min(1e-4)
    normalized = (instrument - mean) / std
    return normalized, mean, std


def median_bandwidth(
    instrument: torch.Tensor,
    split: torch.Tensor,
    seed: int,
    max_points: int = 1024,
) -> float:
    train = instrument[split == 0].reshape(-1, instrument.shape[-1])
    generator = torch.Generator().manual_seed(seed)
    if len(train) > max_points:
        train = train[torch.randperm(len(train), generator=generator)[:max_points]]
    distance = torch.pdist(train.float(), p=2)
    return float(distance.square().median().clamp_min(1e-4))


class MMRDataset(Dataset):
    def __init__(self, payload: dict, instruments: torch.Tensor, split_value: int) -> None:
        self.payload = payload
        self.instruments = instruments
        episode = torch.where(payload['split'] == split_value)[0]
        queries = payload['current'].shape[1]
        self.episode_index = episode.repeat_interleave(queries)
        self.query_index = torch.arange(queries).repeat(len(episode))

    def __len__(self) -> int:
        return len(self.episode_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        episode = self.episode_index[index]
        query = self.query_index[index]
        return (
            self.payload['context'][episode],
            self.payload['current'][episode, query],
            self.payload['action_knots'][episode, query],
            self.payload['future'][episode, query],
            self.instruments[episode, query],
        )


def kernel_mmr_u_statistic(
    residual: torch.Tensor,
    instrument: torch.Tensor,
    bandwidth_squared: float,
) -> torch.Tensor:
    """Off-diagonal U-statistic for E[e e' k(X,X')]."""
    residual = residual.flatten(start_dim=1)
    residual_gram = residual @ residual.T / residual.shape[1]
    distance_squared = torch.cdist(instrument, instrument).square()
    kernel = torch.exp(-distance_squared / (2.0 * bandwidth_squared))
    count = len(residual)
    mask = 1.0 - torch.eye(count, device=residual.device)
    return (kernel * residual_gram * mask).sum() / max(count * (count - 1), 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def validate(
    model: ContextWorldModel,
    loader: DataLoader,
    covariance_weights: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    bandwidth_squared: float,
    mmr_lambda: float,
    device: torch.device,
) -> dict[str, float]:
    totals = {'raw_mse': 0.0, 'mmr_u': 0.0, 'objective': 0.0, 'endpoint': 0.0}
    count = 0
    model.eval()
    for context, current, actions, target, instrument in loader:
        context, current = context.to(device), current.to(device)
        actions, target = actions.to(device), target.to(device)
        instrument = instrument.to(device)
        prediction, _ = model(context, current, actions)
        error = prediction - target
        raw_mse = error.square().mean()
        whitened = error * covariance_weights.sqrt()[None]
        mmr = kernel_mmr_u_statistic(
            whitened, instrument, bandwidth_squared
        )
        objective = raw_mse + mmr_lambda * mmr
        predicted_position = prediction[:, -1] @ decoder_weight + decoder_bias
        target_position = target[:, -1] @ decoder_weight + decoder_bias
        endpoint = torch.linalg.vector_norm(
            predicted_position - target_position, dim=-1
        ).mean()
        batch = len(current)
        count += batch
        totals['raw_mse'] += batch * float(raw_mse)
        totals['mmr_u'] += batch * float(mmr)
        totals['objective'] += batch * float(objective)
        totals['endpoint'] += batch * float(endpoint)
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the GPU Slurm wrapper')
    device = torch.device(args.device)
    seed_everything(args.seed)
    payload = torch.load(args.features, map_location='cpu', weights_only=True)
    instruments, instrument_mean, instrument_std = build_instruments(payload)
    bandwidth_squared = median_bandwidth(
        instruments, payload['split'], args.seed
    )
    train_data = MMRDataset(payload, instruments, 0)
    val_data = MMRDataset(payload, instruments, 1)
    test_data = MMRDataset(payload, instruments, 2)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )
    feature_dim = payload['current'].shape[-1]
    horizons = payload['future'].shape[-2]
    model = ContextWorldModel(feature_dim, horizons).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01
    )
    covariance_weights = payload['covariance_weights'].to(device)
    decoder_weight = payload['decoder_weight'].to(device)
    decoder_bias = payload['decoder_bias'].to(device)
    best_state = None
    best_endpoint = float('inf')
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_raw = running_mmr = running_total = 0.0
        seen = 0
        for context, current, actions, target, instrument in train_loader:
            context, current = context.to(device), current.to(device)
            actions, target = actions.to(device), target.to(device)
            instrument = instrument.to(device)
            prediction, _ = model(context, current, actions)
            error = prediction - target
            raw_mse = error.square().mean()
            whitened = error * covariance_weights.sqrt()[None]
            mmr = kernel_mmr_u_statistic(
                whitened, instrument, bandwidth_squared
            )
            loss = raw_mse + args.mmr_lambda * mmr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch = len(current)
            seen += batch
            running_raw += batch * float(raw_mse.detach())
            running_mmr += batch * float(mmr.detach())
            running_total += batch * float(loss.detach())

        validation = validate(
            model, val_loader, covariance_weights, decoder_weight,
            decoder_bias, bandwidth_squared, args.mmr_lambda, device
        )
        record = {
            'epoch': epoch + 1,
            'train_raw_mse': running_raw / seen,
            'train_mmr_u': running_mmr / seen,
            'train_objective': running_total / seen,
            **{f'validation_{key}': value for key, value in validation.items()},
        }
        history.append(record)
        if epoch % 5 == 0:
            print(json.dumps(record, sort_keys=True), flush=True)
        if validation['endpoint'] < best_endpoint - 1e-5:
            best_endpoint = validation['endpoint']
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
    validation = validate(
        model, val_loader, covariance_weights, decoder_weight,
        decoder_bias, bandwidth_squared, args.mmr_lambda, device
    )
    test = validate(
        model, test_loader, covariance_weights, decoder_weight,
        decoder_bias, bandwidth_squared, args.mmr_lambda, device
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / 'checkpoint.pt'
    torch.save({
        'model_state': best_state,
        'arm': 'kernel_mmr',
        'seed': args.seed,
        'mmr_lambda': args.mmr_lambda,
        'feature_dim': feature_dim,
        'horizons': horizons,
        'cadm_backward': False,
        'bandwidth_squared': bandwidth_squared,
        'instrument_mean': instrument_mean,
        'instrument_std': instrument_std,
        'best_epoch': best_epoch,
        'validation': validation,
        'test': test,
    }, checkpoint_path)
    summary = {
        'schema_version': 1,
        'arm': 'kernel_mmr',
        'mmr_lambda': args.mmr_lambda,
        'seed': args.seed,
        'best_epoch': best_epoch,
        'epochs_ran': len(history),
        'selection_metric': 'validation decoded endpoint MAE',
        'validation': validation,
        'test': test,
        'kernel': {
            'type': 'RBF',
            'bandwidth_squared': bandwidth_squared,
            'bandwidth_rule': 'median train-only pairwise squared distance over <=1024 points',
            'estimator': 'off-diagonal U-statistic',
            'residual': 'PCA-covariance-whitened future-anchor residual',
            'instrument': 'fixed context-difference summaries + current frozen anchor + action knots',
        },
        'base_loss': 'ordinary unweighted anchor MSE',
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
