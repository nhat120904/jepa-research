"""Low-level state-to-XY skill policy and checkpoint utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class GoalPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, width: int = 512, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim + 2
        for _ in range(depth):
            layers.extend([nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU()])
            input_dim = width
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, normalized_state: torch.Tensor, normalized_delta_xy: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([normalized_state, normalized_delta_xy], dim=-1)))


@dataclass
class PolicyRunner:
    model: GoalPolicy
    state_mean: torch.Tensor
    state_std: torch.Tensor
    goal_scale: float
    device: torch.device

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str = "cuda") -> "PolicyRunner":
        target = torch.device(device)
        payload: dict[str, Any] = torch.load(checkpoint_path, map_location=target, weights_only=False)
        config = payload["config"]
        model = GoalPolicy(
            state_dim=int(config["state_dim"]),
            action_dim=int(config["action_dim"]),
            width=int(config["width"]),
            depth=int(config["depth"]),
        ).to(target)
        model.load_state_dict(payload["model"])
        model.eval()
        return cls(
            model=model,
            state_mean=torch.as_tensor(payload["state_mean"], device=target),
            state_std=torch.as_tensor(payload["state_std"], device=target),
            goal_scale=float(config["goal_scale"]),
            device=target,
        )

    @torch.inference_mode()
    def act(self, state: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        goal_tensor = torch.as_tensor(goal_xy, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized_state = (state_tensor - self.state_mean) / self.state_std
        normalized_delta = (goal_tensor - state_tensor[:, :2]) / self.goal_scale
        return self.model(normalized_state, normalized_delta).squeeze(0).cpu().numpy()
