#!/usr/bin/env python3
"""Gate 1: oracle-drag versus median-drag CEM in PokeWorld-style dynamics.

This is intentionally independent of learned models.  It estimates whether
knowing episode drag can materially change decisions under the exact optimizer
that later learned-model experiments would use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Physics:
    control_dt: float = 0.05
    substeps: int = 20
    finger_mass: float = 1.0
    finger_radius: float = 0.06
    object_mass: float = 1.25
    object_radius: float = 0.09
    stiffness: float = 2500.0
    damping_ratio: float = 0.25
    max_force: float = 12.0
    finger_drag: float = 1.0


@dataclass(frozen=True)
class Planner:
    horizon: int = 28
    knots: int = 2
    num_samples: int = 256
    num_elites: int = 24
    iterations: int = 6
    restarts: int = 2
    init_std: float = 0.65
    min_std: float = 0.04
    refit_old_weight: float = 0.20
    terminal_velocity_weight: float = 0.06
    closest_distance_weight: float = 0.15
    action_energy_weight: float = 0.003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--seed', type=int, default=260902)
    parser.add_argument('--planner-seed', type=int, default=91031)
    parser.add_argument('--bootstrap', type=int, default=20_000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=6)
    parser.add_argument('--restarts', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=28)
    parser.add_argument('--substeps', type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def make_tasks(
    episodes: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate paired tasks where only drag varies across episode physics."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-math.pi, math.pi, size=episodes)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    tangent = np.stack([-directions[:, 1], directions[:, 0]], axis=-1)

    object_pos = rng.uniform(-0.12, 0.12, size=(episodes, 2))
    goal_distance = rng.uniform(0.34, 0.56, size=(episodes, 1))
    goals = object_pos + goal_distance * directions

    # Place the finger just off contact, behind the object along the goal ray.
    # Small tangential jitter prevents a degenerate one-dimensional benchmark.
    gap = rng.uniform(0.012, 0.028, size=(episodes, 1))
    lateral = rng.uniform(-0.025, 0.025, size=(episodes, 1))
    finger_pos = object_pos - (0.15 + gap) * directions + lateral * tangent

    states = np.zeros((episodes, 8), dtype=np.float32)
    states[:, 0:2] = finger_pos
    states[:, 4:6] = object_pos
    gamma = rng.uniform(0.5, 4.0, size=episodes).astype(np.float32)
    return (
        torch.as_tensor(states, device=device),
        torch.as_tensor(goals.astype(np.float32), device=device),
        torch.as_tensor(gamma, device=device),
    )


def smooth_programs(knots: torch.Tensor, horizon: int) -> torch.Tensor:
    """Linearly interpolate two force knots into a smooth action sequence."""
    if knots.shape[-2] != 2:
        raise ValueError('Gate 1 is locked to two-knot action programs')
    weight = torch.linspace(
        0.0, 1.0, horizon, device=knots.device, dtype=knots.dtype
    )
    view = (1,) * (knots.ndim - 2) + (horizon, 1)
    weight = weight.view(view)
    return (1.0 - weight) * knots[..., 0:1, :] + weight * knots[..., 1:2, :]


def rollout(
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    gamma: torch.Tensor,
    goals: torch.Tensor,
    physics: Physics,
) -> dict[str, torch.Tensor]:
    """Vectorized semi-implicit Euler rollout.

    `initial_state` is `(B, 8)`, actions `(B, N, H, 2)`, gamma `(B,)`, and
    goals `(B, 2)`.  The candidate dimension is evaluated in parallel.
    """
    batch, candidates, horizon, _ = actions.shape
    state = initial_state[:, None, :].expand(batch, candidates, 8).clone()
    gamma_bc = gamma[:, None, None]
    goal_bc = goals[:, None, :]
    min_distance = torch.linalg.vector_norm(state[..., 4:6] - goal_bc, dim=-1)
    max_overlap = torch.zeros_like(min_distance)
    contact_steps = torch.zeros_like(min_distance)
    h = physics.control_dt / physics.substeps
    radii = physics.finger_radius + physics.object_radius
    effective_mass = (
        physics.finger_mass * physics.object_mass
        / (physics.finger_mass + physics.object_mass)
    )

    for step in range(horizon):
        action_force = physics.max_force * actions[:, :, step, :]
        step_contact = torch.zeros_like(min_distance, dtype=torch.bool)
        for _ in range(physics.substeps):
            finger_pos = state[..., 0:2]
            finger_vel = state[..., 2:4]
            object_pos = state[..., 4:6]
            object_vel = state[..., 6:8]

            delta = object_pos - finger_pos
            distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            normal = delta / distance.clamp_min(1e-7)
            overlap = (radii - distance).clamp_min(0.0)
            relative_normal_velocity = (
                (object_vel - finger_vel) * normal
            ).sum(dim=-1, keepdim=True)

            # Local tangent stiffness of k*d^(3/2) gives the critical-damping
            # scale for a Hertzian penalty contact.
            tangent_stiffness = (
                1.5 * physics.stiffness * torch.sqrt(overlap.clamp_min(1e-10))
            )
            damping = 2.0 * physics.damping_ratio * torch.sqrt(
                tangent_stiffness * effective_mass
            )
            magnitude = (
                physics.stiffness * overlap.pow(1.5)
                - damping * relative_normal_velocity
            ).clamp_min(0.0)
            magnitude = torch.where(overlap > 0.0, magnitude, torch.zeros_like(magnitude))
            contact_force = magnitude * normal

            finger_acc = (
                action_force
                - contact_force
                - physics.finger_drag * finger_vel
            ) / physics.finger_mass
            object_acc = contact_force / physics.object_mass - gamma_bc * object_vel

            finger_vel = finger_vel + h * finger_acc
            object_vel = object_vel + h * object_acc
            finger_pos = finger_pos + h * finger_vel
            object_pos = object_pos + h * object_vel
            state = torch.cat(
                [finger_pos, finger_vel, object_pos, object_vel], dim=-1
            )
            max_overlap = torch.maximum(max_overlap, overlap.squeeze(-1))
            step_contact |= overlap.squeeze(-1) > 0.0

        contact_steps += step_contact.to(contact_steps.dtype)
        distance_to_goal = torch.linalg.vector_norm(
            state[..., 4:6] - goal_bc, dim=-1
        )
        min_distance = torch.minimum(min_distance, distance_to_goal)

    return {
        'final_state': state,
        'min_distance': min_distance,
        'max_overlap': max_overlap,
        'contact_steps': contact_steps,
    }


def task_cost(
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    gamma: torch.Tensor,
    goals: torch.Tensor,
    physics: Physics,
    planner: Planner,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    trace = rollout(initial_state, actions, gamma, goals, physics)
    final_state = trace['final_state']
    final_distance = torch.linalg.vector_norm(
        final_state[..., 4:6] - goals[:, None, :], dim=-1
    )
    final_speed = torch.linalg.vector_norm(final_state[..., 6:8], dim=-1)
    energy = actions.square().mean(dim=(-1, -2))
    cost = (
        final_distance
        + planner.terminal_velocity_weight * final_speed
        + planner.closest_distance_weight * trace['min_distance']
        + planner.action_energy_weight * energy
    )
    trace['final_distance'] = final_distance
    trace['final_speed'] = final_speed
    trace['energy'] = energy
    return cost, trace


@torch.inference_mode()
def cem_plan(
    initial_state: torch.Tensor,
    goals: torch.Tensor,
    assumed_gamma: torch.Tensor,
    physics: Physics,
    planner: Planner,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best-candidate CEM with shared standardized noise across planner arms."""
    batch = initial_state.shape[0]
    direction = goals - initial_state[:, 4:6]
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1e-7)
    initial_knots = torch.stack([0.9 * direction, 0.15 * direction], dim=-2)
    best_actions = torch.zeros(
        (batch, planner.horizon, 2), device=initial_state.device
    )
    best_cost = torch.full((batch,), torch.inf, device=initial_state.device)

    for restart in range(planner.restarts):
        generator = torch.Generator(device=initial_state.device)
        generator.manual_seed(seed + 1009 * restart)
        if restart == 0:
            mean = initial_knots.clone()
        else:
            mean = torch.zeros_like(initial_knots)
        variance = torch.full_like(mean, planner.init_std**2)

        for _ in range(planner.iterations):
            noise = torch.randn(
                batch,
                planner.num_samples,
                planner.knots,
                2,
                device=initial_state.device,
                generator=generator,
            )
            candidates = mean[:, None] + variance[:, None].sqrt() * noise
            candidates = candidates.clamp(-1.0, 1.0)
            candidates[:, 0] = mean.clamp(-1.0, 1.0)
            programs = smooth_programs(candidates, planner.horizon)
            costs, _ = task_cost(
                initial_state, programs, assumed_gamma, goals, physics, planner
            )

            iteration_cost, iteration_index = costs.min(dim=1)
            rows = torch.arange(batch, device=initial_state.device)
            improved = iteration_cost < best_cost
            iteration_actions = programs[rows, iteration_index]
            best_actions = torch.where(
                improved[:, None, None], iteration_actions, best_actions
            )
            best_cost = torch.minimum(best_cost, iteration_cost)

            elite_index = torch.topk(
                costs, planner.num_elites, dim=1, largest=False
            ).indices
            elite_index_expanded = elite_index[..., None, None].expand(
                -1, -1, planner.knots, 2
            )
            elites = torch.gather(candidates, 1, elite_index_expanded)
            elite_mean = elites.mean(dim=1)
            elite_variance = elites.var(dim=1, unbiased=False)
            old = planner.refit_old_weight
            mean = old * mean + (1.0 - old) * elite_mean
            variance = old * variance + (1.0 - old) * elite_variance
            variance = variance.clamp_min(planner.min_std**2)

    return best_actions, best_cost


def evaluate_actions(
    states: torch.Tensor,
    goals: torch.Tensor,
    true_gamma: torch.Tensor,
    actions: torch.Tensor,
    physics: Physics,
    planner: Planner,
) -> dict[str, np.ndarray]:
    costs, trace = task_cost(
        states, actions[:, None], true_gamma, goals, physics, planner
    )
    result = {'cost': costs[:, 0].cpu().numpy()}
    for key in (
        'final_distance',
        'final_speed',
        'min_distance',
        'max_overlap',
        'contact_steps',
        'energy',
    ):
        result[key] = trace[key][:, 0].cpu().numpy()
    result['reach'] = (result['min_distance'] <= 0.08).astype(np.float64)
    result['settled_success'] = (
        (result['final_distance'] <= 0.10) & (result['final_speed'] <= 0.20)
    ).astype(np.float64)
    return result


def percentile_ci(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def paired_bootstrap(
    oracle: dict[str, np.ndarray],
    median: dict[str, np.ndarray],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    count = len(oracle['cost'])
    delta_cost = median['cost'] - oracle['cost']
    relative = delta_cost / np.maximum(median['cost'], 0.05)
    delta_reach = oracle['reach'] - median['reach']
    delta_settled = oracle['settled_success'] - median['settled_success']
    delta_distance = median['final_distance'] - oracle['final_distance']

    boot = {key: np.empty(draws, dtype=np.float64) for key in (
        'cost_improvement', 'relative_cost_reduction', 'reach_gain',
        'settled_success_gain', 'final_distance_improvement'
    )}
    chunk = 1000
    for start in range(0, draws, chunk):
        width = min(chunk, draws - start)
        index = rng.integers(0, count, size=(width, count))
        boot['cost_improvement'][start:start + width] = delta_cost[index].mean(1)
        boot['relative_cost_reduction'][start:start + width] = relative[index].mean(1)
        boot['reach_gain'][start:start + width] = delta_reach[index].mean(1)
        boot['settled_success_gain'][start:start + width] = delta_settled[index].mean(1)
        boot['final_distance_improvement'][start:start + width] = delta_distance[index].mean(1)

    point = {
        'cost_improvement': float(delta_cost.mean()),
        'relative_cost_reduction': float(relative.mean()),
        'reach_gain': float(delta_reach.mean()),
        'settled_success_gain': float(delta_settled.mean()),
        'final_distance_improvement': float(delta_distance.mean()),
    }
    return {
        key: {'mean': point[key], 'ci95': percentile_ci(boot[key])}
        for key in point
    }


def arm_summary(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        'mean_cost': float(values['cost'].mean()),
        'median_cost': float(np.median(values['cost'])),
        'mean_final_distance': float(values['final_distance'].mean()),
        'median_final_distance': float(np.median(values['final_distance'])),
        'mean_final_speed': float(values['final_speed'].mean()),
        'reach_rate': float(values['reach'].mean()),
        'settled_success_rate': float(values['settled_success'].mean()),
        'mean_contact_steps': float(values['contact_steps'].mean()),
        'max_overlap': float(values['max_overlap'].max()),
    }


def drag_bin_summary(
    gamma: np.ndarray,
    oracle: dict[str, np.ndarray],
    median: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    edges = [0.5, 1.5, 3.0, 4.0 + 1e-6]
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (gamma >= lower) & (gamma < upper)
        delta = median['cost'][mask] - oracle['cost'][mask]
        relative = delta / np.maximum(median['cost'][mask], 0.05)
        rows.append({
            'gamma_range': [lower, min(upper, 4.0)],
            'n': int(mask.sum()),
            'mean_cost_improvement': float(delta.mean()),
            'mean_relative_cost_reduction': float(relative.mean()),
            'oracle_reach': float(oracle['reach'][mask].mean()),
            'median_reach': float(median['reach'][mask].mean()),
        })
    return rows


def write_episode_csv(
    path: Path,
    gamma: np.ndarray,
    states: np.ndarray,
    goals: np.ndarray,
    oracle: dict[str, np.ndarray],
    median: dict[str, np.ndarray],
) -> None:
    fields = [
        'episode', 'gamma', 'object_x0', 'object_y0', 'goal_x', 'goal_y',
        'oracle_cost', 'median_cost', 'cost_improvement',
        'relative_cost_reduction', 'oracle_final_distance',
        'median_final_distance', 'oracle_final_speed', 'median_final_speed',
        'oracle_min_distance', 'median_min_distance', 'oracle_reach',
        'median_reach', 'oracle_settled_success', 'median_settled_success',
        'oracle_contact_steps', 'median_contact_steps', 'oracle_max_overlap',
        'median_max_overlap'
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(gamma)):
            improvement = median['cost'][index] - oracle['cost'][index]
            writer.writerow({
                'episode': index,
                'gamma': gamma[index],
                'object_x0': states[index, 4],
                'object_y0': states[index, 5],
                'goal_x': goals[index, 0],
                'goal_y': goals[index, 1],
                'oracle_cost': oracle['cost'][index],
                'median_cost': median['cost'][index],
                'cost_improvement': improvement,
                'relative_cost_reduction': improvement / max(median['cost'][index], 0.05),
                'oracle_final_distance': oracle['final_distance'][index],
                'median_final_distance': median['final_distance'][index],
                'oracle_final_speed': oracle['final_speed'][index],
                'median_final_speed': median['final_speed'][index],
                'oracle_min_distance': oracle['min_distance'][index],
                'median_min_distance': median['min_distance'][index],
                'oracle_reach': oracle['reach'][index],
                'median_reach': median['reach'][index],
                'oracle_settled_success': oracle['settled_success'][index],
                'median_settled_success': median['settled_success'][index],
                'oracle_contact_steps': oracle['contact_steps'][index],
                'median_contact_steps': median['contact_steps'][index],
                'oracle_max_overlap': oracle['max_overlap'][index],
                'median_max_overlap': median['max_overlap'][index],
            })


def main() -> None:
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; submit the GPU Slurm wrapper')
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    physics = Physics(substeps=args.substeps)
    planner = Planner(
        num_samples=args.num_samples,
        iterations=args.iterations,
        restarts=args.restarts,
        horizon=args.horizon,
    )
    if planner.num_elites >= planner.num_samples:
        raise ValueError('num_elites must be smaller than num_samples')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    states, goals, gamma = make_tasks(args.episodes, args.seed, device)
    median_gamma = torch.full_like(gamma, 2.25)

    # Resetting the planner seed gives both arms identical standardized noise.
    oracle_actions, oracle_predicted = cem_plan(
        states, goals, gamma, physics, planner, args.planner_seed
    )
    median_actions, median_predicted = cem_plan(
        states, goals, median_gamma, physics, planner, args.planner_seed
    )
    oracle = evaluate_actions(states, goals, gamma, oracle_actions, physics, planner)
    median = evaluate_actions(states, goals, gamma, median_actions, physics, planner)

    gamma_np = gamma.cpu().numpy()
    paired = paired_bootstrap(oracle, median, args.bootstrap, args.seed + 17)
    cost_ci = paired['cost_improvement']['ci95']
    relative_mean = paired['relative_cost_reduction']['mean']
    gate_pass = bool(cost_ci[0] > 0.0 and relative_mean >= 0.10)

    source_path = Path(__file__).resolve()
    summary = {
        'schema_version': 1,
        'gate': 'decision_room_oracle_drag_vs_population_median_drag',
        'status': 'GO' if gate_pass else 'STOP',
        'criterion': {
            'paired_cost_improvement_ci95_lower_gt_zero': bool(cost_ci[0] > 0.0),
            'mean_relative_cost_reduction_gte_0.10': bool(relative_mean >= 0.10),
        },
        'estimands': paired,
        'arms': {'oracle_drag': arm_summary(oracle), 'median_drag': arm_summary(median)},
        'drag_bins': drag_bin_summary(gamma_np, oracle, median),
        'optimizer_audit': {
            'oracle_predicted_vs_true_max_abs': float(np.max(np.abs(
                oracle_predicted.cpu().numpy() - oracle['cost']
            ))),
            'median_predicted_mean': float(median_predicted.mean().item()),
            'median_true_mean': float(median['cost'].mean()),
            'same_standardized_noise_seed': args.planner_seed,
            'selection': 'best candidate seen across all CEM iterations/restarts',
        },
        'task': {
            'episodes': args.episodes,
            'gamma_distribution': 'Uniform[0.5, 4.0]',
            'population_median_gamma': 2.25,
            'mass_and_stiffness_fixed': True,
            'goal_distance_range': [0.34, 0.56],
            'reach_radius': 0.08,
            'settled_success': 'final_distance <= 0.10 and final_speed <= 0.20',
        },
        'physics': asdict(physics),
        'planner': asdict(planner),
        'run': {
            'seed': args.seed,
            'planner_seed': args.planner_seed,
            'bootstrap_draws': args.bootstrap,
            'device': str(device),
            'hostname': socket.gethostname(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'source_sha256': sha256(source_path),
        },
        'limitations': [
            'Independent implementation from the published POKEWORLD specification; official code was not yet released.',
            'Open-loop launch-to-goal gate with fixed mass/stiffness; it isolates drag decision room but is not the paper closed-loop benchmark.',
            'A GO result licenses anchor/model experiments; it is not evidence that moment regularization works.',
        ],
    }

    states_np = states.cpu().numpy()
    goals_np = goals.cpu().numpy()
    write_episode_csv(
        args.out_dir / 'episodes.csv', gamma_np, states_np, goals_np, oracle, median
    )
    torch.save(
        {
            'states': states.cpu(),
            'goals': goals.cpu(),
            'gamma': gamma.cpu(),
            'oracle_actions': oracle_actions.cpu(),
            'median_actions': median_actions.cpu(),
        },
        args.out_dir / 'actions.pt',
    )
    with (args.out_dir / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
