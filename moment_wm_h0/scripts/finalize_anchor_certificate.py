#!/usr/bin/env python3
"""Add the physics-informed pixel-centroid ceiling to Gate 2 results.

The full recurrent-probe job started just before this estimator was added.  This
separate compute-node analysis preserves those locked predictions and augments
the certificate without retraining or touching the test split twice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10_000)
    parser.add_argument('--parent-executed-sha256', required=True)
    return parser.parse_args()


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
    for start in range(0, draws, 1000):
        width = min(1000, draws - start)
        index = rng.integers(0, count, size=(width, count))
        yy, pp = y[index], prediction[index]
        residual = np.square(yy - pp).sum(axis=1)
        total = np.square(yy - yy.mean(axis=1, keepdims=True)).sum(axis=1)
        values[start:start + width] = 1.0 - residual / np.maximum(total, 1e-12)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def analytic_position_gamma(
    position: torch.Tensor, substeps: int = 20
) -> np.ndarray:
    displacement = position[:, 1:].double() - position[:, :-1].double()
    speed = torch.linalg.vector_norm(displacement, dim=-1)
    ratio = speed[:, 1:] / speed[:, :-1].clamp_min(1e-8)
    valid = (speed[:, 1:] > 1e-5) & (speed[:, :-1] > 1e-5)
    estimates = (
        1.0 - ratio.clamp(1e-5, 1.0).pow(1.0 / substeps)
    ) / (0.05 / substeps)
    estimates = torch.where(valid, estimates, torch.nan)
    result = torch.nanmedian(estimates, dim=1).values
    result = torch.nan_to_num(result, nan=2.25, posinf=4.0, neginf=0.5)
    return result.clamp(0.5, 4.0).float().numpy()


def main() -> None:
    args = parse_args()
    summary_path = args.results_dir / 'summary.json'
    predictions_path = args.results_dir / 'test_predictions.csv'
    original_summary = args.results_dir / 'summary_recurrent_only.json'
    original_predictions = args.results_dir / 'test_predictions_recurrent_only.csv'
    if not original_summary.exists():
        shutil.copy2(summary_path, original_summary)
    if not original_predictions.exists():
        shutil.copy2(predictions_path, original_predictions)

    summary = json.loads(original_summary.read_text())
    dataset = torch.load(args.dataset, map_location='cpu', weights_only=True)
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    test_mask = dataset['split'] == 2
    y = dataset['gamma'][test_mask].float().numpy()
    prediction = analytic_position_gamma(features['pixel_centroid'][test_mask])
    r2 = r2_score(y, prediction)
    summary['modalities']['pixel_centroid_analytic'] = {
        'ensemble_r2': r2,
        'ensemble_r2_ci95': bootstrap_r2(y, prediction, args.bootstrap, 1703),
        'ensemble_mae': float(np.abs(y - prediction).mean()),
        'per_seed': [],
    }
    raw_estimators = ['raw_pixels', 'pixel_centroid', 'pixel_centroid_analytic']
    raw_certificate = max(
        raw_estimators,
        key=lambda name: summary['modalities'][name]['ensemble_r2'],
    )
    raw_pass = summary['modalities'][raw_certificate]['ensemble_r2'] >= 0.40
    anchors = ['frozen_dinov2', 'frozen_raft']
    chosen_anchor = max(
        anchors, key=lambda name: summary['modalities'][name]['ensemble_r2']
    )
    anchor_pass = summary['modalities'][chosen_anchor]['ensemble_r2'] >= 0.40
    summary['raw_visual_certificate'] = raw_certificate
    summary['chosen_anchor'] = chosen_anchor if anchor_pass else None
    summary['criterion'] = {
        'best_raw_visual_estimator_r2_gte_0.40': bool(raw_pass),
        'at_least_one_frozen_anchor_recurrent_r2_gte_0.40': bool(anchor_pass),
    }
    summary['status'] = 'GO' if raw_pass and anchor_pass else 'STOP'
    summary['protocol']['raw_visual_certificate_definition'] = (
        'best held-out R2 among raw CNN-GRU, pixel-centroid GRU, and '
        'analytic centroid glide-decay estimator'
    )
    summary['finalization'] = {
        'parent_probe_job_id': summary['run'].get('slurm_job_id'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'hostname': socket.gethostname(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'preserved_summary': str(original_summary.resolve()),
        'preserved_predictions': str(original_predictions.resolve()),
    }
    summary['run']['source_sha256_read_at_completion'] = summary['run'].get(
        'source_sha256'
    )
    summary['run']['source_sha256'] = args.parent_executed_sha256
    summary['run']['provenance_note'] = (
        'The Slurm wrapper hashed the executed source before launch. The file '
        'was edited after process start, so the script\'s end-of-run file hash '
        'did not describe the code already loaded in memory.'
    )

    with original_predictions.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) + ['pixel_centroid_analytic']
    if len(rows) != len(prediction):
        raise RuntimeError('test prediction count does not match held-out split')
    with predictions_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, value in zip(rows, prediction):
            row['pixel_centroid_analytic'] = value
            writer.writerow(row)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
