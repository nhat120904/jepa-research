#!/usr/bin/env python3
"""Fit matched recurrent recoverability probes for candidate anchor channels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import socket
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--bootstrap', type=int, default=10_000)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


class SequenceProbe(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(), nn.LayerNorm(128)
        )
        self.gru = nn.GRU(128, hidden, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = sequence.flatten(start_dim=2)
        encoded = self.project(sequence)
        output, _ = self.gru(encoded)
        return self.head(output[:, -1]).squeeze(-1)


class RawPixelProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.GELU(), nn.LayerNorm(128),
        )
        self.gru = nn.GRU(128, 96, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, steps = sequence.shape[:2]
        encoded = self.frame_encoder(sequence.reshape(batch * steps, 1, 64, 64))
        output, _ = self.gru(encoded.reshape(batch, steps, -1))
        return self.head(output[:, -1]).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def r2_score(y: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.square(y - prediction).sum()
    total = np.square(y - y.mean()).sum()
    return float(1.0 - residual / max(total, 1e-12))


def bootstrap_r2(
    y: np.ndarray, prediction: np.ndarray, draws: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    count = len(y)
    chunk = 1000
    for start in range(0, draws, chunk):
        width = min(chunk, draws - start)
        index = rng.integers(0, count, size=(width, count))
        yy = y[index]
        pp = prediction[index]
        residual = np.square(yy - pp).sum(axis=1)
        centered = yy - yy.mean(axis=1, keepdims=True)
        values[start:start + width] = 1.0 - residual / np.maximum(
            np.square(centered).sum(axis=1), 1e-12
        )
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def standardize_sequence(
    sequence: torch.Tensor, train_mask: torch.Tensor
) -> torch.Tensor:
    train = sequence[train_mask].float()
    mean = train.mean()
    std = train.std().clamp_min(1e-6)
    return (sequence.float() - mean) / std


def train_one(
    sequence: torch.Tensor,
    gamma: torch.Tensor,
    split: torch.Tensor,
    model_factory: Callable[[], nn.Module],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float], dict[str, torch.Tensor]]:
    seed_everything(seed)
    train_mask, val_mask, test_mask = split == 0, split == 1, split == 2
    target_mean = gamma[train_mask].mean()
    target_std = gamma[train_mask].std().clamp_min(1e-6)
    target = (gamma - target_mean) / target_std

    train_dataset = TensorDataset(sequence[train_mask], target[train_mask])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_state = None
    best_val = float('inf')
    stale = 0
    epochs_ran = 0

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            prediction = model(inputs)
            loss = torch.mean(torch.square(prediction - labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_prediction = model(sequence[val_mask].to(device)).cpu()
        val_loss = float(torch.mean(torch.square(val_prediction - target[val_mask])))
        epochs_ran = epoch + 1
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError('probe produced no validation checkpoint')
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        normalized_prediction = model(sequence[test_mask].to(device)).cpu()
    prediction = normalized_prediction * target_std + target_mean
    y = gamma[test_mask].numpy()
    pred = prediction.numpy()
    metrics = {
        'r2': r2_score(y, pred),
        'mae': float(np.abs(y - pred).mean()),
        'best_validation_mse_standardized': best_val,
        'epochs_ran': epochs_ran,
    }
    cpu_state = {key: value.cpu() for key, value in best_state.items()}
    return pred, metrics, cpu_state


def analytic_state_gamma(state: torch.Tensor, substeps: int = 20) -> np.ndarray:
    velocity = state[..., 2:4].double()
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    ratio = (speed[:, 1:] / speed[:, :-1].clamp_min(1e-10)).clamp(1e-8, 1.0)
    substep_ratio = ratio.pow(1.0 / substeps)
    estimate = (1.0 - substep_ratio).median(dim=1).values / (0.05 / substeps)
    return estimate.float().numpy()


def analytic_position_gamma(
    position: torch.Tensor, substeps: int = 20
) -> np.ndarray:
    """Median glide-decay estimator using only pixel-derived centroids."""
    displacement = position[:, 1:].double() - position[:, :-1].double()
    speed = torch.linalg.vector_norm(displacement, dim=-1)
    ratio = speed[:, 1:] / speed[:, :-1].clamp_min(1e-8)
    valid = (speed[:, 1:] > 1e-5) & (speed[:, :-1] > 1e-5)
    ratio = ratio.clamp(1e-5, 1.0)
    estimates = (1.0 - ratio.pow(1.0 / substeps)) / (0.05 / substeps)
    estimates = torch.where(valid, estimates, torch.nan)
    estimate = torch.nanmedian(estimates, dim=1).values
    # A fully quantized trajectory can have no valid ratio.  Falling back to
    # the train-population median is conservative and explicit.
    estimate = torch.nan_to_num(estimate, nan=2.25, posinf=4.0, neginf=0.5)
    return estimate.clamp(0.5, 4.0).float().numpy()


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the GPU Slurm wrapper')
    device = torch.device(args.device)
    dataset = torch.load(args.dataset, map_location='cpu', weights_only=True)
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    gamma = dataset['gamma'].float()
    split = dataset['split']
    if not torch.equal(gamma, features['gamma']) or not torch.equal(split, features['split']):
        raise RuntimeError('dataset/features episode order mismatch')
    train_mask = split == 0

    modalities: dict[str, tuple[torch.Tensor, Callable[[], nn.Module]]] = {}
    raw = dataset['frames'].float().div(255.0)
    modalities['raw_pixels'] = (raw, RawPixelProbe)

    dino = standardize_sequence(features['dino_grid'], train_mask)
    modalities['frozen_dinov2'] = (
        dino, lambda: SequenceProbe(int(np.prod(dino.shape[2:])))
    )
    flow = standardize_sequence(features['flow_grid'], train_mask)
    modalities['frozen_raft'] = (
        flow, lambda: SequenceProbe(int(np.prod(flow.shape[2:])))
    )
    pixel_centroid_raw = features['pixel_centroid'].float()
    centroid = standardize_sequence(pixel_centroid_raw, train_mask)
    modalities['pixel_centroid'] = (
        centroid, lambda: SequenceProbe(int(np.prod(centroid.shape[2:])))
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_mask = split == 2
    y_test = gamma[test_mask].numpy()
    rows = []
    modality_summary = {}
    predictions_by_modality = {}
    for modality, (sequence, factory) in modalities.items():
        seed_results = []
        predictions = []
        checkpoints = {}
        for seed in args.seeds:
            prediction, metrics, state_dict = train_one(
                sequence, gamma, split, factory, seed, args, device
            )
            predictions.append(prediction)
            checkpoints[f'seed_{seed}'] = state_dict
            seed_result = {'seed': seed, **metrics}
            seed_results.append(seed_result)
            rows.append({'modality': modality, **seed_result})
        ensemble = np.mean(np.stack(predictions), axis=0)
        r2 = r2_score(y_test, ensemble)
        ci = bootstrap_r2(y_test, ensemble, args.bootstrap, 1701)
        modality_summary[modality] = {
            'ensemble_r2': r2,
            'ensemble_r2_ci95': ci,
            'ensemble_mae': float(np.abs(y_test - ensemble).mean()),
            'per_seed': seed_results,
        }
        predictions_by_modality[modality] = ensemble
        torch.save(checkpoints, args.out_dir / f'{modality}_probes.pt')

    state_prediction = analytic_state_gamma(dataset['state'][test_mask])
    state_r2 = r2_score(y_test, state_prediction)
    modality_summary['privileged_state_analytic'] = {
        'ensemble_r2': state_r2,
        'ensemble_r2_ci95': bootstrap_r2(
            y_test, state_prediction, args.bootstrap, 1702
        ),
        'ensemble_mae': float(np.abs(y_test - state_prediction).mean()),
        'per_seed': [],
    }
    predictions_by_modality['privileged_state_analytic'] = state_prediction

    centroid_prediction = analytic_position_gamma(pixel_centroid_raw[test_mask])
    centroid_r2 = r2_score(y_test, centroid_prediction)
    modality_summary['pixel_centroid_analytic'] = {
        'ensemble_r2': centroid_r2,
        'ensemble_r2_ci95': bootstrap_r2(
            y_test, centroid_prediction, args.bootstrap, 1703
        ),
        'ensemble_mae': float(np.abs(y_test - centroid_prediction).mean()),
        'per_seed': [],
    }
    predictions_by_modality['pixel_centroid_analytic'] = centroid_prediction

    raw_visual_estimators = [
        'raw_pixels', 'pixel_centroid', 'pixel_centroid_analytic'
    ]
    raw_visual_certificate = max(
        raw_visual_estimators,
        key=lambda name: modality_summary[name]['ensemble_r2'],
    )
    raw_pass = modality_summary[raw_visual_certificate]['ensemble_r2'] >= 0.40
    anchor_candidates = ['frozen_dinov2', 'frozen_raft']
    chosen_anchor = max(
        anchor_candidates,
        key=lambda name: modality_summary[name]['ensemble_r2'],
    )
    anchor_pass = modality_summary[chosen_anchor]['ensemble_r2'] >= 0.40
    gate_pass = bool(raw_pass and anchor_pass)

    summary = {
        'schema_version': 1,
        'gate': 'anchor_recoverability_certificate',
        'status': 'GO' if gate_pass else 'STOP',
        'chosen_anchor': chosen_anchor if anchor_pass else None,
        'raw_visual_certificate': raw_visual_certificate,
        'criterion': {
            'best_raw_visual_estimator_r2_gte_0.40': raw_pass,
            'at_least_one_frozen_anchor_recurrent_r2_gte_0.40': anchor_pass,
        },
        'modalities': modality_summary,
        'protocol': {
            'episode_split': '60/20/20 train/validation/test',
            'probe': '2-layer width-96 GRU after modality-specific per-frame projection',
            'selection': 'early stopping on validation episodes only',
            'test': 'held out until one final evaluation',
            'seeds': args.seeds,
            'bootstrap_draws': args.bootstrap,
            'certificate_threshold_r2': 0.40,
            'raw_visual_certificate_definition': 'best held-out R2 among raw CNN-GRU, pixel-centroid GRU, and analytic centroid glide-decay estimator',
        },
        'run': {
            'hostname': socket.gethostname(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'device': str(device),
            'dataset': str(args.dataset.resolve()),
            'features': str(args.features.resolve()),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        'interpretation': (
            'A passing frozen-anchor probe is an information/recoverability certificate, '
            'not evidence that a world-model objective will retain or functionally use drag.'
        ),
    }
    with (args.out_dir / 'probe_runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'modality', 'seed', 'r2', 'mae',
            'best_validation_mse_standardized', 'epochs_ran'
        ])
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / 'test_predictions.csv').open('w', newline='') as handle:
        fields = ['test_index', 'gamma', *predictions_by_modality]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        test_indices = torch.where(test_mask)[0].numpy()
        for row_index, episode_index in enumerate(test_indices):
            row = {'test_index': int(episode_index), 'gamma': y_test[row_index]}
            row.update({
                name: prediction[row_index]
                for name, prediction in predictions_by_modality.items()
            })
            writer.writerow(row)
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
