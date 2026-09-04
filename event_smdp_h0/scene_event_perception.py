"""Current event-state observers for the Scene deployment-facing gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from event_smdp_h0.scene_core import MilestoneState
from event_smdp_h0.scene_learning import MLP, task_vector


class EventStateObserver(nn.Module):
    OUTPUT_DIM = 6 + 4 + 1

    def __init__(self, feature_dim: int, goal_dim: int, width: int = 256) -> None:
        super().__init__()
        self.model = MLP(feature_dim + goal_dim + 2, self.OUTPUT_DIM, width=width)

    def forward(
        self, feature: torch.Tensor, goal: torch.Tensor, task: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        raw = self.model(torch.cat([feature, goal, task], dim=-1))
        return {
            "cube_logits": raw[..., :6],
            "window_logits": raw[..., 6:10],
            "stable_logit": raw[..., 10],
        }


class EventObserverEvaluator:
    def __init__(self, checkpoint: Path | str, device: str = "cuda") -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol") != "scene_event_perception_v1":
            raise ValueError(f"unexpected observer protocol: {payload.get('protocol')}")
        self.feature_view = str(payload["feature_view"])
        self.feature_dim = int(payload["feature_dim"])
        self.goal_dim = int(payload["goal_dim"])
        self.width = int(payload["width"])
        self.device = torch.device(device)
        self.feature_mean = torch.as_tensor(
            payload["feature_mean"], dtype=torch.float32, device=self.device
        )
        self.feature_std = torch.as_tensor(
            payload["feature_std"], dtype=torch.float32, device=self.device
        )
        self.goal_mean = torch.as_tensor(
            payload["goal_mean"], dtype=torch.float32, device=self.device
        )
        self.goal_std = torch.as_tensor(
            payload["goal_std"], dtype=torch.float32, device=self.device
        )
        self.model = EventStateObserver(
            self.feature_dim, self.goal_dim, width=self.width
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self, feature: np.ndarray, goal: np.ndarray, task_id: int
    ) -> tuple[MilestoneState, dict[str, object]]:
        feature_tensor = torch.as_tensor(
            feature, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        goal_tensor = torch.as_tensor(
            goal, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        feature_tensor = (feature_tensor - self.feature_mean) / self.feature_std
        goal_tensor = (goal_tensor - self.goal_mean) / self.goal_std
        task = torch.as_tensor(
            task_vector(task_id), dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        out = self.model(feature_tensor, goal_tensor, task)
        cube_probability = torch.softmax(out["cube_logits"], dim=-1)[0]
        window_probability = torch.softmax(out["window_logits"], dim=-1)[0]
        stable_probability = float(torch.sigmoid(out["stable_logit"]).item())
        cube = int(cube_probability.argmax().item())
        window = int(window_probability.argmax().item())
        state = MilestoneState(
            task_id=task_id,
            cube_stage=min(4 if task_id == 4 else 5, cube),
            window_stage=0 if task_id == 4 else min(3, window),
            stable_count=3 if stable_probability >= 0.5 else 0,
        )
        details = {
            "cube_probability": cube_probability.cpu().numpy(),
            "window_probability": window_probability.cpu().numpy(),
            "stable_probability": stable_probability,
        }
        return state, details

