#!/usr/bin/env python3
"""Locked CEM evaluation of validation-selected kernel MMR vs strong baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from moment_wm_h0.models import ContextWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--baseline-summary', type=Path, required=True)
    parser.add_argument('--baseline-actions', type=Path, required=True)
    parser.add_argument('--gate1-actions', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=6)
    parser.add_argument('--restarts', type=int, default=2)
    parser.add_argument('--bootstrap', type=int, default=20_000)
    parser.add_argument('--planner-seed', type=int, default=91031)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_mmr_models(
    root: Path,
    lambda_dir: str,
    seeds: list[int],
    device: torch.device,
) -> list[ContextWorldModel]:
    result = []
    for seed in seeds:
        checkpoint = torch.load(
            root / lambda_dir / f'seed_{seed}' / 'checkpoint.pt',
            map_location='cpu', weights_only=True
        )
        model = ContextWorldModel(
            checkpoint['feature_dim'], checkpoint['horizons']
        )
        model.load_state_dict(checkpoint['model_state'])
        model.to(device).eval().requires_grad_(False)
        result.append(model)
    return result


def paired_increment(
    baseline: np.ndarray,
    mmr: np.ndarray,
    median: np.ndarray,
    oracle: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    improvement = baseline - mmr
    gap = median - oracle
    rng = np.random.default_rng(seed)
    count = len(mmr)
    boot_improvement = np.empty(draws)
    boot_increment = np.empty(draws)
    for start in range(0, draws, 1000):
        width = min(1000, draws - start)
        index = rng.integers(0, count, size=(width, count))
        numerator = improvement[index].mean(1)
        denominator = gap[index].mean(1)
        boot_improvement[start:start + width] = numerator
        boot_increment[start:start + width] = numerator / np.maximum(
            denominator, 1e-8
        )
    increment = improvement.mean() / max(gap.mean(), 1e-8)
    return {
        'mean_true_cost_improvement_over_best_baseline': float(improvement.mean()),
        'cost_improvement_ci95': [
            float(x) for x in np.quantile(boot_improvement, [0.025, 0.975])
        ],
        'incremental_oracle_gap_recovered': float(increment),
        'incremental_oracle_gap_recovered_ci95': [
            float(x) for x in np.quantile(boot_increment, [0.025, 0.975])
        ],
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the GPU Slurm wrapper')
    device = torch.device(args.device)
    payload = torch.load(args.features, map_location='cpu', weights_only=True)
    selection = json.loads(args.selection.read_text())
    baseline_summary = json.loads(args.baseline_summary.read_text())
    lambda_dir = selection['chosen_lambda_dir']
    models = load_mmr_models(args.models_root, lambda_dir, args.seeds, device)

    evaluator = load_module(
        REPO / 'moment_wm_h0/scripts/evaluate_strong_baselines.py',
        'mwm_baseline_eval_for_mmr',
    )
    gate1 = load_module(
        REPO / 'moment_wm_h0/scripts/gate1_decision_room.py',
        'mwm_gate1_for_mmr',
    )
    eval_payload = payload['eval']
    states = eval_payload['state'].to(device)
    goals = eval_payload['goal'].to(device)
    gamma = eval_payload['gamma'].to(device)
    context = eval_payload['context'].to(device)
    current = eval_payload['current'].to(device)
    decoder_weight = payload['decoder_weight'].to(device)
    decoder_bias = payload['decoder_bias'].to(device)
    actions, predicted_cost = evaluator.cem_plan(
        models, context, current, states, goals,
        decoder_weight, decoder_bias, args
    )
    physics = gate1.Physics()
    planner = gate1.Planner(
        num_samples=args.num_samples,
        iterations=args.iterations,
        restarts=args.restarts,
    )
    mmr_values = gate1.evaluate_actions(
        states, goals, gamma, actions, physics, planner
    )

    gate1_actions = torch.load(
        args.gate1_actions, map_location='cpu', weights_only=True
    )
    oracle_values = gate1.evaluate_actions(
        states, goals, gamma,
        gate1_actions['oracle_actions'][:len(states)].to(device), physics, planner
    )
    median_values = gate1.evaluate_actions(
        states, goals, gamma,
        gate1_actions['median_actions'][:len(states)].to(device), physics, planner
    )
    baseline_actions = torch.load(
        args.baseline_actions, map_location='cpu', weights_only=True
    )
    best_baseline = min(
        baseline_summary['arms'],
        key=lambda arm: baseline_summary['arms'][arm]['planning']['mean_cost'],
    )
    baseline_values = gate1.evaluate_actions(
        states, goals, gamma,
        baseline_actions[best_baseline][:len(states)].to(device), physics, planner
    )

    mmr_planning = evaluator.bootstrap_planning(
        oracle_values['cost'], median_values['cost'], mmr_values['cost'],
        args.bootstrap, 9131
    )
    mmr_planning.update({
        'reach_rate': float(mmr_values['reach'].mean()),
        'settled_success_rate': float(mmr_values['settled_success'].mean()),
        'mean_final_distance': float(mmr_values['final_distance'].mean()),
        'predicted_cost_mean': float(predicted_cost.mean()),
    })
    increment = paired_increment(
        baseline_values['cost'], mmr_values['cost'], median_values['cost'],
        oracle_values['cost'], args.bootstrap, 9132
    )
    clean_gain = increment['cost_improvement_ci95'][0] > 0.0
    practical_gain = increment['incremental_oracle_gap_recovered'] >= 0.10
    gate_pass = bool(clean_gain and practical_gain)

    summary = {
        'schema_version': 1,
        'gate': 'kernel_mmr_vs_strong_baseline',
        'status': 'GO' if gate_pass else 'STOP',
        'criterion': {
            'paired_true_cost_improvement_ci95_lower_gt_zero': clean_gain,
            'incremental_oracle_gap_recovered_gte_0.10': practical_gain,
            'best_strong_baseline': best_baseline,
        },
        'selected_lambda': selection['chosen_lambda'],
        'selection': selection,
        'planning': mmr_planning,
        'increment_over_best_baseline': increment,
        'best_baseline': baseline_summary['arms'][best_baseline],
        'prediction': evaluator.prediction_metrics(models, payload, device),
        'mechanism': evaluator.mechanism_metrics(models, payload),
        'reference': {
            'oracle_mean_cost': float(oracle_values['cost'].mean()),
            'median_drag_mean_cost': float(median_values['cost'].mean()),
            'best_baseline_mean_cost': float(baseline_values['cost'].mean()),
        },
        'protocol': {
            'test_evaluated_only_after_lambda_selection': True,
            'common_standardized_cem_noise': True,
            'seeds': args.seeds,
            'num_samples': args.num_samples,
            'iterations': args.iterations,
            'restarts': args.restarts,
            'primary_metric': 'paired true simulator cost under locked CEM',
        },
        'run': {
            'hostname': socket.gethostname(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'kernel_mmr': actions.cpu()}, args.out_dir / 'planned_actions.pt')
    with (args.out_dir / 'episodes.csv').open('w', newline='') as handle:
        fields = [
            'episode', 'gamma', 'mmr_cost', 'baseline_cost', 'median_cost',
            'oracle_cost', 'mmr_final_distance', 'mmr_reach',
            'mmr_settled_success'
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(states)):
            writer.writerow({
                'episode': index,
                'gamma': float(gamma[index]),
                'mmr_cost': mmr_values['cost'][index],
                'baseline_cost': baseline_values['cost'][index],
                'median_cost': median_values['cost'][index],
                'oracle_cost': oracle_values['cost'][index],
                'mmr_final_distance': mmr_values['final_distance'][index],
                'mmr_reach': mmr_values['reach'][index],
                'mmr_settled_success': mmr_values['settled_success'][index],
            })
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
