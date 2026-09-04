"""Event-state-closed SMDP model for the bounded Scene H1b audit.

Unlike the H1 contextual model, this transition is recursively closed in the
history-bearing automaton state.  It tests whether H1 failed because planning
repeatedly fed learned feature predictions back into the event transition.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from event_smdp_h0.core import ARM_EVENT, ARM_TERMINAL
from event_smdp_h0.scene_core import SKILLS, MilestoneState, feedback_reward
from event_smdp_h0.scene_learning import MLP, milestone_vector, task_vector


class AbstractSMDPHead(nn.Module):
    """Categorical event transition plus success, no-effect, and event time."""

    OUTPUT_DIM = 6 + 4 + 1 + 1 + 1

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.model = MLP(2 + 11 + len(SKILLS), self.OUTPUT_DIM, width=width)

    def forward(
        self,
        task: torch.Tensor,
        milestone: torch.Tensor,
        skill: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = torch.nn.functional.one_hot(skill.long(), len(SKILLS)).float()
        raw = self.model(torch.cat([task, milestone, encoded], dim=-1))
        return {
            "cube_logits": raw[..., :6],
            "window_logits": raw[..., 6:10],
            "stable_logit": raw[..., 10],
            "no_effect_logit": raw[..., 11],
            "log_duration": raw[..., 12],
        }


class AbstractSMDPEvaluator:
    def __init__(self, checkpoint: Path | str, device: str = "cuda") -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol") != "scene_h1b_abstract_closure_v1":
            raise ValueError(f"unexpected checkpoint protocol: {payload.get('protocol')}")
        self.width = int(payload["width"])
        self.device = torch.device(device)
        self.head = AbstractSMDPHead(width=self.width).to(self.device)
        self.head.load_state_dict(payload["state_dict"])
        self.head.eval()

    @torch.inference_mode()
    def transition_distribution(
        self, task_id: int, state: MilestoneState, skill_index: int
    ) -> dict[str, np.ndarray | float]:
        task = torch.as_tensor(
            task_vector(task_id), dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        milestone = torch.as_tensor(
            milestone_vector(state), dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        skill = torch.as_tensor([skill_index], device=self.device)
        out = self.head(task, milestone, skill)
        return {
            "cube_probability": torch.softmax(out["cube_logits"], dim=-1)[0]
            .cpu()
            .numpy(),
            "window_probability": torch.softmax(out["window_logits"], dim=-1)[0]
            .cpu()
            .numpy(),
            "stable_probability": float(torch.sigmoid(out["stable_logit"]).item()),
            "no_effect_probability": float(
                torch.sigmoid(out["no_effect_logit"]).item()
            ),
            "duration": float(torch.exp(out["log_duration"]).clamp(1, 250).item()),
        }

    @torch.inference_mode()
    def rollout_details(
        self,
        task_id: int,
        state: MilestoneState,
        sequence: Iterable[int],
        duration_cost: float = 0.0,
    ) -> dict[str, object]:
        predicted_state = state
        total_duration = 0.0
        stable_probability = 0.0
        steps: list[dict[str, object]] = []
        for skill_index in sequence:
            before = predicted_state
            distribution = self.transition_distribution(
                task_id, predicted_state, skill_index
            )
            cube_probability = np.asarray(distribution["cube_probability"])
            window_probability = np.asarray(distribution["window_probability"])
            cube = int(cube_probability.argmax())
            window = int(window_probability.argmax())
            stable_probability = float(distribution["stable_probability"])
            cube_limit = 4 if task_id == 4 else 5
            predicted_state = replace(
                predicted_state,
                cube_stage=min(cube_limit, max(predicted_state.cube_stage, cube)),
                window_stage=(
                    0
                    if task_id == 4
                    else min(3, max(predicted_state.window_stage, window))
                ),
                stable_count=3 if stable_probability >= 0.5 else 0,
            )
            total_duration += float(distribution["duration"])
            steps.append(
                {
                    "skill": int(skill_index),
                    "state_before": before,
                    "state_after": predicted_state,
                    "cube_probability": cube_probability,
                    "window_probability": window_probability,
                    "stable_probability": stable_probability,
                    "no_effect_probability": float(
                        distribution["no_effect_probability"]
                    ),
                    "duration": float(distribution["duration"]),
                }
            )
        score = feedback_reward(predicted_state, ARM_EVENT)
        score = float(np.clip(score - duration_cost * total_duration, 0.0, 1.0))
        return {
            "score": score,
            "stable_probability": stable_probability,
            "state": predicted_state,
            "total_duration": total_duration,
            "steps": steps,
        }

    @torch.inference_mode()
    def score_sequence(
        self,
        task_id: int,
        state: MilestoneState,
        sequence: Iterable[int],
        duration_cost: float = 0.0,
        arm: str = ARM_EVENT,
    ) -> float:
        details = self.rollout_details(
            task_id, state, sequence, duration_cost=duration_cost
        )
        if arm == ARM_EVENT:
            return float(details["score"])
        if arm == ARM_TERMINAL:
            return float(details["stable_probability"])
        raise ValueError(f"unknown planning arm: {arm}")
