#!/usr/bin/env python3
"""Evaluate strong context baselines under the locked Gate-1 CEM protocol."""

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
    parser.add_argument('--checkpoints-root', type=Path, required=True)
    parser.add_argument('--gate1-actions', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=['mse', 'cov_mse', 'cadm'])
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


def load_models(
    root: Path,
    arm: str,
    seeds: list[int],
    device: torch.device,
) -> list[ContextWorldModel]:
    models = []
    for seed in seeds:
        path = root / arm / f'seed_{seed}' / 'checkpoint.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = ContextWorldModel(
            checkpoint['feature_dim'],
            checkpoint['horizons'],
            cadm_backward=checkpoint['cadm_backward'],
        )
        model.load_state_dict(checkpoint['model_state'])
        model.to(device).eval().requires_grad_(False)
        models.append(model)
    return models


def smooth_programs(knots: torch.Tensor, horizon: int = 28) -> torch.Tensor:
    weight = torch.linspace(
        0.0, 1.0, horizon, device=knots.device, dtype=knots.dtype
    ).view(1, 1, horizon, 1)
    return (1.0 - weight) * knots[:, :, 0:1] + weight * knots[:, :, 1:2]


@torch.inference_mode()
def predict_ensemble(
    models: list[ContextWorldModel],
    context_latents: list[torch.Tensor],
    current: torch.Tensor,
    knots: torch.Tensor,
    chunk: int = 8192,
) -> torch.Tensor:
    batch, candidates = knots.shape[:2]
    flat_current = current[:, None].expand(-1, candidates, -1).reshape(
        batch * candidates, -1
    )
    flat_knots = knots.reshape(batch * candidates, 2, 2)
    sum_prediction = None
    for model, latent in zip(models, context_latents):
        flat_latent = latent[:, None].expand(-1, candidates, -1).reshape(
            batch * candidates, -1
        )
        pieces = []
        for start in range(0, len(flat_current), chunk):
            pieces.append(model.predict_from_context(
                flat_latent[start:start + chunk],
                flat_current[start:start + chunk],
                flat_knots[start:start + chunk],
            ))
        prediction = torch.cat(pieces).reshape(
            batch, candidates, model.horizons, model.feature_dim
        )
        sum_prediction = prediction if sum_prediction is None else sum_prediction + prediction
    if sum_prediction is None:
        raise RuntimeError('empty ensemble')
    return sum_prediction / len(models)


def predicted_cost(
    predicted_feature: torch.Tensor,
    current_feature: torch.Tensor,
    goals: torch.Tensor,
    knots: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
) -> torch.Tensor:
    positions = predicted_feature @ decoder_weight + decoder_bias
    current_position = current_feature @ decoder_weight + decoder_bias
    final_distance = torch.linalg.vector_norm(
        positions[:, :, -1] - goals[:, None], dim=-1
    )
    final_speed = torch.linalg.vector_norm(
        positions[:, :, -1] - positions[:, :, -2], dim=-1
    ) / 0.05
    all_positions = torch.cat([
        current_position[:, None, None].expand(-1, positions.shape[1], -1, -1),
        positions,
    ], dim=2)
    minimum_distance = torch.linalg.vector_norm(
        all_positions - goals[:, None, None], dim=-1
    ).min(dim=-1).values
    energy = smooth_programs(knots).square().mean(dim=(-1, -2))
    return (
        final_distance + 0.06 * final_speed
        + 0.15 * minimum_distance + 0.003 * energy
    )


@torch.inference_mode()
def cem_plan(
    models: list[ContextWorldModel],
    context: torch.Tensor,
    current: torch.Tensor,
    states: torch.Tensor,
    goals: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(states)
    direction = goals - states[:, 4:6]
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1e-7)
    initial_knots = torch.stack([0.9 * direction, 0.15 * direction], dim=-2)
    context_latents = [model.encode_context(context) for model in models]
    best_actions = torch.zeros((batch, 28, 2), device=states.device)
    best_cost = torch.full((batch,), torch.inf, device=states.device)

    for restart in range(args.restarts):
        generator = torch.Generator(device=states.device)
        generator.manual_seed(args.planner_seed + 1009 * restart)
        mean = initial_knots.clone() if restart == 0 else torch.zeros_like(initial_knots)
        variance = torch.full_like(mean, 0.65**2)
        for _ in range(args.iterations):
            noise = torch.randn(
                batch, args.num_samples, 2, 2,
                device=states.device, generator=generator
            )
            candidates = (mean[:, None] + variance[:, None].sqrt() * noise).clamp(-1, 1)
            candidates[:, 0] = mean.clamp(-1, 1)
            predicted = predict_ensemble(
                models, context_latents, current, candidates
            )
            costs = predicted_cost(
                predicted, current, goals, candidates,
                decoder_weight, decoder_bias
            )
            iteration_cost, iteration_index = costs.min(dim=1)
            rows = torch.arange(batch, device=states.device)
            actions = smooth_programs(candidates)[rows, iteration_index]
            improved = iteration_cost < best_cost
            best_actions = torch.where(
                improved[:, None, None], actions, best_actions
            )
            best_cost = torch.minimum(best_cost, iteration_cost)
            elite_index = torch.topk(costs, 24, dim=1, largest=False).indices
            elite_index = elite_index[..., None, None].expand(-1, -1, 2, 2)
            elites = torch.gather(candidates, 1, elite_index)
            elite_mean = elites.mean(1)
            elite_variance = elites.var(1, unbiased=False)
            mean = 0.2 * mean + 0.8 * elite_mean
            variance = (0.2 * variance + 0.8 * elite_variance).clamp_min(0.04**2)
    return best_actions, best_cost


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.square(target - prediction).sum()
    total = np.square(target - target.mean()).sum()
    return float(1.0 - residual / max(total, 1e-12))


def ridge_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    ridge: float = 1e-2,
) -> np.ndarray:
    x_mean, y_mean = train_x.mean(0), train_y.mean()
    x = train_x - x_mean
    gram = x.T @ x
    scale = torch.trace(gram) / gram.shape[0]
    weight = torch.linalg.solve(
        gram + ridge * scale.clamp_min(1e-8) * torch.eye(gram.shape[0]),
        x.T @ (train_y - y_mean),
    )
    return ((test_x - x_mean) @ weight + y_mean).numpy()


@torch.inference_mode()
def mechanism_metrics(
    models: list[ContextWorldModel], payload: dict
) -> dict[str, float]:
    train_mask, test_mask = payload['split'] == 0, payload['split'] == 2
    predictions = []
    for model in models:
        train_latent = model.encode_context(
            payload['context'][train_mask].to(next(model.parameters()).device)
        ).cpu()
        test_latent = model.encode_context(
            payload['context'][test_mask].to(next(model.parameters()).device)
        ).cpu()
        predictions.append(ridge_predict(
            train_latent, payload['gamma'][train_mask], test_latent
        ))
    prediction = np.mean(np.stack(predictions), axis=0)
    target = payload['gamma'][test_mask].numpy()
    return {
        'drag_r2_from_context': r2_score(target, prediction),
        'drag_mae_from_context': float(np.abs(target - prediction).mean()),
    }


@torch.inference_mode()
def prediction_metrics(
    models: list[ContextWorldModel], payload: dict, device: torch.device
) -> dict[str, float]:
    mask = payload['split'] == 2
    context = payload['context'][mask]
    current = payload['current'][mask]
    action = payload['action_knots'][mask]
    target = payload['future'][mask]
    episode, query = current.shape[:2]
    flat_context = context[:, None].expand(-1, query, -1, -1).reshape(
        episode * query, *context.shape[1:]
    )
    flat_current = current.reshape(-1, current.shape[-1])
    flat_action = action.reshape(-1, 2, 2)
    flat_target = target.reshape(-1, *target.shape[2:])
    sums = torch.zeros_like(flat_target)
    chunk = 4096
    for model in models:
        pieces = []
        for start in range(0, len(flat_current), chunk):
            prediction, _ = model(
                flat_context[start:start + chunk].to(device),
                flat_current[start:start + chunk].to(device),
                flat_action[start:start + chunk].to(device),
            )
            pieces.append(prediction.cpu())
        sums += torch.cat(pieces)
    prediction = sums / len(models)
    decoder_weight, decoder_bias = payload['decoder_weight'], payload['decoder_bias']
    predicted_position = prediction @ decoder_weight + decoder_bias
    true_position = flat_target @ decoder_weight + decoder_bias
    endpoint_error = torch.linalg.vector_norm(
        predicted_position[:, -1] - true_position[:, -1], dim=-1
    )
    return {
        'anchor_mse': float(torch.square(prediction - flat_target).mean()),
        'decoded_endpoint_mae': float(endpoint_error.mean()),
        'decoded_endpoint_median': float(endpoint_error.median()),
    }


def bootstrap_planning(
    oracle: np.ndarray,
    median: np.ndarray,
    model: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    count = len(model)
    improvement = median - model
    oracle_gap = median - oracle
    relative = improvement / np.maximum(median, 0.05)
    boot_improvement = np.empty(draws)
    boot_relative = np.empty(draws)
    boot_recovery = np.empty(draws)
    for start in range(0, draws, 1000):
        width = min(1000, draws - start)
        index = rng.integers(0, count, size=(width, count))
        boot_improvement[start:start + width] = improvement[index].mean(1)
        boot_relative[start:start + width] = relative[index].mean(1)
        numerator = improvement[index].mean(1)
        denominator = oracle_gap[index].mean(1)
        boot_recovery[start:start + width] = numerator / np.maximum(denominator, 1e-8)
    recovery = improvement.mean() / max(oracle_gap.mean(), 1e-8)
    return {
        'mean_cost': float(model.mean()),
        'mean_improvement_over_median': float(improvement.mean()),
        'improvement_ci95': [float(x) for x in np.quantile(boot_improvement, [0.025, 0.975])],
        'mean_relative_cost_reduction_vs_median': float(relative.mean()),
        'relative_reduction_ci95': [float(x) for x in np.quantile(boot_relative, [0.025, 0.975])],
        'oracle_gap_recovered': float(recovery),
        'oracle_gap_recovered_ci95': [float(x) for x in np.quantile(boot_recovery, [0.025, 0.975])],
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the GPU Slurm wrapper')
    device = torch.device(args.device)
    payload = torch.load(args.features, map_location='cpu', weights_only=True)
    eval_payload = payload['eval']
    states = eval_payload['state'].to(device)
    goals = eval_payload['goal'].to(device)
    gamma = eval_payload['gamma'].to(device)
    context = eval_payload['context'].to(device)
    current = eval_payload['current'].to(device)
    decoder_weight = payload['decoder_weight'].to(device)
    decoder_bias = payload['decoder_bias'].to(device)

    gate1 = load_module(
        REPO / 'moment_wm_h0/scripts/gate1_decision_room.py', 'mwm_gate1_eval'
    )
    physics = gate1.Physics()
    planner = gate1.Planner(
        num_samples=args.num_samples,
        iterations=args.iterations,
        restarts=args.restarts,
    )
    reference_actions = torch.load(
        args.gate1_actions, map_location='cpu', weights_only=True
    )
    oracle_values = gate1.evaluate_actions(
        states, goals, gamma,
        reference_actions['oracle_actions'][:len(states)].to(device),
        physics, planner
    )
    median_values = gate1.evaluate_actions(
        states, goals, gamma,
        reference_actions['median_actions'][:len(states)].to(device),
        physics, planner
    )

    arm_results = {}
    action_outputs = {}
    episode_rows = []
    for arm_index, arm in enumerate(args.arms):
        models = load_models(args.checkpoints_root, arm, args.seeds, device)
        actions, predicted = cem_plan(
            models, context, current, states, goals,
            decoder_weight, decoder_bias, args
        )
        values = gate1.evaluate_actions(
            states, goals, gamma, actions, physics, planner
        )
        planning = bootstrap_planning(
            oracle_values['cost'], median_values['cost'], values['cost'],
            args.bootstrap, 8123 + arm_index
        )
        planning.update({
            'reach_rate': float(values['reach'].mean()),
            'settled_success_rate': float(values['settled_success'].mean()),
            'mean_final_distance': float(values['final_distance'].mean()),
            'predicted_cost_mean': float(predicted.mean().item()),
        })
        arm_results[arm] = {
            'planning': planning,
            'prediction': prediction_metrics(models, payload, device),
            'mechanism': mechanism_metrics(models, payload),
        }
        action_outputs[arm] = actions.cpu()
        for episode in range(len(gamma)):
            episode_rows.append({
                'arm': arm,
                'episode': episode,
                'gamma': float(gamma[episode]),
                'true_cost': values['cost'][episode],
                'oracle_cost': oracle_values['cost'][episode],
                'median_cost': median_values['cost'][episode],
                'final_distance': values['final_distance'][episode],
                'reach': values['reach'][episode],
                'settled_success': values['settled_success'][episode],
            })
        del models
        torch.cuda.empty_cache()

    best_arm = max(
        args.arms,
        key=lambda arm: arm_results[arm]['planning']['oracle_gap_recovered'],
    )
    best_recovery = arm_results[best_arm]['planning']['oracle_gap_recovered']
    stop_mmr = best_recovery >= 0.80
    summary = {
        'schema_version': 1,
        'gate': 'strong_baseline_decision_room',
        'status': 'STOP_MMR' if stop_mmr else 'CONTINUE_TO_MMR',
        'criterion': {
            'stop_if_best_strong_baseline_recovers_gte_0.80_oracle_gap': stop_mmr,
            'threshold': 0.80,
            'best_arm': best_arm,
            'best_oracle_gap_recovered': best_recovery,
        },
        'reference': {
            'oracle_mean_cost': float(oracle_values['cost'].mean()),
            'median_drag_mean_cost': float(median_values['cost'].mean()),
            'oracle_reach_rate': float(oracle_values['reach'].mean()),
            'median_drag_reach_rate': float(median_values['reach'].mean()),
        },
        'arms': arm_results,
        'protocol': {
            'seeds': args.seeds,
            'ensemble': 'mean future-anchor prediction across three seeds',
            'planner': {
                'type': 'best-candidate CEM over smooth two-knot programs',
                'num_samples': args.num_samples,
                'iterations': args.iterations,
                'restarts': args.restarts,
                'planner_seed': args.planner_seed,
            },
            'common_standardized_cem_noise_across_arms': True,
            'primary_metric': 'true simulator cost and fraction of oracle-vs-median gap recovered',
        },
        'run': {
            'hostname': socket.gethostname(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'features': str(args.features.resolve()),
            'checkpoints_root': str(args.checkpoints_root.resolve()),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(action_outputs, args.out_dir / 'planned_actions.pt')
    with (args.out_dir / 'episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
