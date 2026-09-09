"""A matched, state-dependent option-termination baseline for A2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from skill_handoff_wm.switching import RolloutResult, rollout_skill


class TerminationNet(nn.Module):
    def __init__(self, state_dim: int, width: int = 256, depth: int = 3):
        super().__init__()
        input_dim = state_dim + 6  # current/next delta XY, elapsed fraction, final-waypoint flag.
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU()])
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def make_features(
    states: np.ndarray,
    current_targets: np.ndarray,
    next_targets: np.ndarray,
    elapsed: np.ndarray,
    max_duration: float,
    is_final: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    goal_scale: float,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    xy = states[..., :2]
    return np.concatenate(
        [
            (states - state_mean) / state_std,
            (np.asarray(current_targets, dtype=np.float32) - xy) / goal_scale,
            (np.asarray(next_targets, dtype=np.float32) - xy) / goal_scale,
            np.asarray(elapsed, dtype=np.float32)[..., None] / max_duration,
            np.asarray(is_final, dtype=np.float32)[..., None],
        ],
        axis=-1,
    ).astype(np.float32)


@dataclass
class TerminationRunner:
    model: TerminationNet
    state_mean: np.ndarray
    state_std: np.ndarray
    goal_scale: float
    max_duration: float
    threshold: float
    device: torch.device

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str = "cuda") -> "TerminationRunner":
        target = torch.device(device)
        payload: dict[str, Any] = torch.load(checkpoint_path, map_location=target, weights_only=False)
        config = payload["config"]
        model = TerminationNet(
            int(config["state_dim"]), int(config["width"]), int(config["depth"])
        ).to(target)
        model.load_state_dict(payload["model"])
        model.eval()
        return cls(
            model=model,
            state_mean=np.asarray(payload["state_mean"]),
            state_std=np.asarray(payload["state_std"]),
            goal_scale=float(config["goal_scale"]),
            max_duration=float(config["max_duration"]),
            threshold=float(payload["threshold"]),
            device=target,
        )

    @torch.inference_mode()
    def probability(self, state, current_target, next_target, elapsed, is_final) -> float:
        features = make_features(
            np.asarray(state)[None],
            np.asarray(current_target)[None],
            np.asarray(next_target)[None],
            np.asarray([elapsed]),
            self.max_duration,
            np.asarray([is_final]),
            self.state_mean,
            self.state_std,
            self.goal_scale,
        )
        logits = self.model(torch.as_tensor(features, device=self.device))
        return float(torch.sigmoid(logits).item())


def rollout_learned_termination(
    raw_env,
    policy,
    termination: TerminationRunner,
    current_target: np.ndarray,
    next_target: np.ndarray,
    durations: list[int],
    remaining_budget: int,
    radius: float,
    is_final: bool,
) -> tuple[RolloutResult, int, list[dict[str, float | int | bool]]]:
    candidates = sorted(set(min(duration, remaining_budget) for duration in durations if remaining_budget > 0))
    elapsed = 0
    total_steps = 0
    reached = False
    success = bool(raw_env.compute_success())
    min_distance = float(np.linalg.norm(raw_env.get_xy() - current_target))
    decisions = []
    for candidate in candidates:
        segment = rollout_skill(
            raw_env,
            policy,
            current_target,
            candidate - elapsed,
            radius,
            stop_on_target=False,
        )
        total_steps += segment.primitive_steps
        elapsed = candidate
        reached = reached or segment.reached_target
        success = success or segment.success
        min_distance = min(min_distance, segment.min_target_distance)
        probability = termination.probability(
            raw_env.get_ob(), current_target, next_target, elapsed, is_final
        )
        stop = success or candidate == candidates[-1] or probability >= termination.threshold
        decisions.append({"duration": candidate, "stop_probability": probability, "stop": stop})
        if stop:
            break
    return RolloutResult(total_steps, success, reached, min_distance), elapsed, decisions
