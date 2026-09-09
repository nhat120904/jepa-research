"""Switching rules and simulator-oracle handoff diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skill_handoff_wm.sim import restore, snapshot


@dataclass
class RolloutResult:
    primitive_steps: int
    success: bool
    reached_target: bool
    min_target_distance: float


def rollout_skill(raw_env, policy, target_xy, max_steps, radius=0.5, stop_on_target=False) -> RolloutResult:
    min_distance = float(np.linalg.norm(raw_env.get_xy() - target_xy))
    reached = min_distance <= radius
    success = bool(raw_env.compute_success())
    steps = 0
    while steps < max_steps and not success and not (stop_on_target and reached):
        raw_env.step(policy.act(raw_env.get_ob(), target_xy))
        steps += 1
        distance = float(np.linalg.norm(raw_env.get_xy() - target_xy))
        min_distance = min(min_distance, distance)
        reached = reached or distance <= radius
        success = bool(raw_env.compute_success())
    return RolloutResult(steps, success, reached, min_distance)


def oracle_duration(
    raw_env,
    policy,
    current_target: np.ndarray,
    next_target: np.ndarray,
    durations: list[int],
    probe_steps: int,
    radius: float,
) -> tuple[int, list[dict[str, float | int | bool]]]:
    """Choose duration by branching physics, then probing the next fixed skill.

    This is deliberately privileged and is only an A1 component-headroom diagnostic.
    """
    root = snapshot(raw_env)
    branches = []
    for duration in durations:
        restore(raw_env, root)
        current = rollout_skill(raw_env, policy, current_target, duration, radius, stop_on_target=False)
        exit_speed = float(np.linalg.norm(raw_env.data.qvel[:2]))
        exit_observation = raw_env.get_ob().copy()
        handoff = snapshot(raw_env)
        probe = rollout_skill(raw_env, policy, next_target, probe_steps, radius, stop_on_target=False)
        score = 1_000_000.0 if probe.success else -float(np.linalg.norm(raw_env.get_xy() - next_target))
        branches.append(
            {
                "duration": duration,
                "score": score,
                "current_min_distance": current.min_target_distance,
                "probe_min_distance": probe.min_target_distance,
                "exit_speed": exit_speed,
                "exit_observation": exit_observation.tolist(),
                "success_in_probe": probe.success,
                "simulator_steps": current.primitive_steps + probe.primitive_steps,
            }
        )
        restore(raw_env, handoff)
    restore(raw_env, root)
    best = max(branches, key=lambda row: (float(row["score"]), -int(row["duration"])))
    return int(best["duration"]), branches
