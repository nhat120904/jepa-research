#!/usr/bin/env python3
"""Select kernel-MMR lambda using validation episodes only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--lambdas', type=float, nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def lambda_name(value: float) -> str:
    return f'lambda_{value:g}'


def main() -> None:
    args = parse_args()
    rows = []
    for value in args.lambdas:
        per_seed = []
        for seed in args.seeds:
            path = args.models_root / lambda_name(value) / f'seed_{seed}' / 'summary.json'
            summary = json.loads(path.read_text())
            per_seed.append({
                'seed': seed,
                'validation_endpoint': summary['validation']['endpoint'],
                'validation_raw_mse': summary['validation']['raw_mse'],
                'validation_mmr_u': summary['validation']['mmr_u'],
            })
        rows.append({
            'lambda': value,
            'lambda_dir': lambda_name(value),
            'mean_validation_endpoint': float(np.mean([
                row['validation_endpoint'] for row in per_seed
            ])),
            'std_validation_endpoint': float(np.std([
                row['validation_endpoint'] for row in per_seed
            ], ddof=1)) if len(per_seed) > 1 else 0.0,
            'per_seed': per_seed,
        })
    chosen = min(rows, key=lambda row: row['mean_validation_endpoint'])
    output = {
        'schema_version': 1,
        'selection_split': 'validation episodes only',
        'selection_metric': 'mean decoded endpoint MAE across three seeds',
        'candidates': rows,
        'chosen_lambda': chosen['lambda'],
        'chosen_lambda_dir': chosen['lambda_dir'],
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
