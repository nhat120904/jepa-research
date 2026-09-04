"""Learned skill-level models for the OGBench-Scene H1 gate.

The module is intentionally free of MuJoCo and encoder imports.  Collection and
evaluation runners provide either a frozen visual latent or a privileged state
vector.  For each feature view, all heads share the same learned skill dynamics
model so the H1 comparison changes the planning readout rather than dynamics
capacity.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from event_smdp_h0.core import ARM_EVENT
from event_smdp_h0.scene_core import (
    SKILLS,
    MilestoneState,
    ScenePredicates,
    advance_milestones,
    feedback_reward,
)


FEATURE_VIEWS = ("latent", "privileged")
HEADS = ("terminal", "event_bce", "event_time")
PREDICATE_NAMES = (
    "button_0",
    "button_1",
    "drawer_open",
    "drawer_closed",
    "window_open",
    "window_closed",
    "cube_in_drawer",
    "cube_at_goal",
    "native_success",
)


def one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def task_vector(task_id: int) -> np.ndarray:
    if task_id not in (4, 5):
        raise ValueError(f"unsupported Scene task: {task_id}")
    return one_hot(task_id - 4, 2)


def milestone_vector(state: MilestoneState) -> np.ndarray:
    return np.concatenate(
        [
            one_hot(min(max(int(state.cube_stage), 0), 5), 6),
            one_hot(min(max(int(state.window_stage), 0), 3), 4),
            np.asarray([min(max(state.stable_count, 0), 3) / 3.0], dtype=np.float32),
        ]
    )


def raw_state_feature(raw_env: Any) -> np.ndarray:
    """Privileged information upper bound; never the main deployable input."""

    return np.concatenate(
        [
            np.asarray(raw_env._data.qpos, dtype=np.float32).reshape(-1),
            np.asarray(raw_env._data.qvel, dtype=np.float32).reshape(-1),
            np.asarray(raw_env._cur_button_states, dtype=np.float32).reshape(-1),
        ]
    )


def goal_feature(raw_env: Any) -> np.ndarray:
    goal = raw_env.cur_task_info["goal"]
    return np.concatenate(
        [
            np.asarray(goal["block_xyzs"], dtype=np.float32).reshape(-1),
            np.asarray(goal["button_states"], dtype=np.float32).reshape(-1),
            np.asarray(
                [goal["drawer_pos"], goal["window_pos"]], dtype=np.float32
            ),
        ]
    )


def predicate_vector(predicates: ScenePredicates) -> np.ndarray:
    return np.asarray(
        [float(getattr(predicates, key)) for key in PREDICATE_NAMES],
        dtype=np.float32,
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 256,
        depth: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(current, width),
                    nn.LayerNorm(width),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            current = width
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkillDynamics(nn.Module):
    def __init__(self, feature_dim: int, width: int = 256) -> None:
        super().__init__()
        self.model = MLP(feature_dim + len(SKILLS), feature_dim, width=width)

    def forward(self, feature: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        encoded = torch.nn.functional.one_hot(skill.long(), len(SKILLS)).float()
        return self.model(torch.cat([feature, encoded], dim=-1))


class EndpointHead(nn.Module):
    def __init__(self, feature_dim: int, goal_dim: int, output_dim: int, width: int = 256):
        super().__init__()
        self.model = MLP(feature_dim + goal_dim + 2, output_dim, width=width)

    def forward(
        self, feature: torch.Tensor, goal: torch.Tensor, task: torch.Tensor
    ) -> torch.Tensor:
        return self.model(torch.cat([feature, goal, task], dim=-1))


class EventTimeHead(nn.Module):
    """Joint next automaton-state, no-effect, success, and duration head."""

    OUTPUT_DIM = 6 + 4 + 1 + 1 + 1

    def __init__(self, feature_dim: int, goal_dim: int, width: int = 256):
        super().__init__()
        input_dim = feature_dim + goal_dim + 2 + 11 + len(SKILLS)
        self.model = MLP(input_dim, self.OUTPUT_DIM, width=width)

    def forward(
        self,
        feature: torch.Tensor,
        goal: torch.Tensor,
        task: torch.Tensor,
        milestone: torch.Tensor,
        skill: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = torch.nn.functional.one_hot(skill.long(), len(SKILLS)).float()
        raw = self.model(
            torch.cat([feature, goal, task, milestone, encoded], dim=-1)
        )
        return {
            "cube_logits": raw[..., :6],
            "window_logits": raw[..., 6:10],
            "stable_logit": raw[..., 10],
            "no_effect_logit": raw[..., 11],
            "log_duration": raw[..., 12],
        }


def make_head(head: str, feature_dim: int, goal_dim: int, width: int = 256) -> nn.Module:
    if head == "terminal":
        return EndpointHead(feature_dim, goal_dim, 1, width=width)
    if head == "event_bce":
        return EndpointHead(feature_dim, goal_dim, len(PREDICATE_NAMES), width=width)
    if head == "event_time":
        return EventTimeHead(feature_dim, goal_dim, width=width)
    raise ValueError(f"unknown head: {head}")


def _resolve_open_closed(open_prob: float, closed_prob: float) -> tuple[bool, bool]:
    if open_prob >= 0.5 and closed_prob >= 0.5:
        return (True, False) if open_prob >= closed_prob else (False, True)
    return open_prob >= 0.5, closed_prob >= 0.5


def probabilities_to_predicates(probabilities: np.ndarray) -> ScenePredicates:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if p.size != len(PREDICATE_NAMES):
        raise ValueError("predicate probability dimension mismatch")
    drawer_open, drawer_closed = _resolve_open_closed(p[2], p[3])
    window_open, window_closed = _resolve_open_closed(p[4], p[5])
    return ScenePredicates(
        button_0=int(p[0] >= 0.5),
        button_1=int(p[1] >= 0.5),
        drawer_open=drawer_open,
        drawer_closed=drawer_closed,
        window_open=window_open,
        window_closed=window_closed,
        cube_in_drawer=bool(p[6] >= 0.5),
        cube_at_goal=bool(p[7] >= 0.5),
        native_success=bool(p[8] >= 0.5),
    )


class LearnedSceneEvaluator:
    """Load one frozen dynamics/readout pair and score skill sequences."""

    def __init__(self, checkpoint: Path | str, device: str = "cuda") -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.feature_view = str(payload["feature_view"])
        self.head_name = str(payload["head"])
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
        self.dynamics = SkillDynamics(self.feature_dim, width=self.width).to(self.device)
        self.head = make_head(
            self.head_name, self.feature_dim, self.goal_dim, width=self.width
        ).to(self.device)
        self.dynamics.load_state_dict(payload["dynamics_state_dict"])
        self.head.load_state_dict(payload["head_state_dict"])
        self.dynamics.eval()
        self.head.eval()

    def _feature(self, value: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(1, -1)
        return (tensor - self.feature_mean) / self.feature_std

    def _goal(self, value: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(1, -1)
        return (tensor - self.goal_mean) / self.goal_std

    @torch.inference_mode()
    def score_sequence(
        self,
        feature: np.ndarray,
        goal: np.ndarray,
        task_id: int,
        state: MilestoneState,
        sequence: Iterable[int],
        duration_cost: float = 0.0,
    ) -> float:
        current = self._feature(feature)
        goal_tensor = self._goal(goal)
        task_tensor = torch.as_tensor(
            task_vector(task_id), dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        predicted_state = state
        total_duration = 0.0

        for skill_index in sequence:
            skill = torch.as_tensor([skill_index], device=self.device)
            if self.head_name == "event_time":
                milestone = torch.as_tensor(
                    milestone_vector(predicted_state),
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(1, -1)
                out = self.head(current, goal_tensor, task_tensor, milestone, skill)
                cube = int(out["cube_logits"].argmax(dim=-1).item())
                window = int(out["window_logits"].argmax(dim=-1).item())
                cube_limit = 4 if task_id == 4 else 5
                predicted_state = replace(
                    predicted_state,
                    cube_stage=min(cube_limit, max(predicted_state.cube_stage, cube)),
                    window_stage=(
                        0
                        if task_id == 4
                        else min(3, max(predicted_state.window_stage, window))
                    ),
                    stable_count=(
                        3 if torch.sigmoid(out["stable_logit"]).item() >= 0.5 else 0
                    ),
                )
                total_duration += float(torch.exp(out["log_duration"]).clamp(1, 250).item())
            current = self.dynamics(current, skill)
            if self.head_name == "event_bce":
                logits = self.head(current, goal_tensor, task_tensor)
                probabilities = torch.sigmoid(logits)[0].cpu().numpy()
                predicates = probabilities_to_predicates(probabilities)
                predicted_state = advance_milestones(predicted_state, predicates)
                # One model transition is a variable-duration closed-loop
                # skill, not one primitive simulator tick.  Native success at
                # the skill endpoint therefore represents the locked
                # three-tick dwell used in the physical evaluator.
                predicted_state = replace(
                    predicted_state,
                    stable_count=3 if predicates.native_success else 0,
                )

        if self.head_name == "terminal":
            return float(
                torch.sigmoid(self.head(current, goal_tensor, task_tensor)).item()
            )
        score = feedback_reward(predicted_state, ARM_EVENT)
        return float(np.clip(score - duration_cost * total_duration, 0.0, 1.0))


def checkpoint_payload(
    *,
    feature_view: str,
    head: str,
    feature_dim: int,
    goal_dim: int,
    width: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    goal_mean: np.ndarray,
    goal_std: np.ndarray,
    dynamics: SkillDynamics,
    readout: nn.Module,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": "scene_h1_learnability_v1",
        "feature_view": feature_view,
        "head": head,
        "feature_dim": feature_dim,
        "goal_dim": goal_dim,
        "width": width,
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "goal_mean": np.asarray(goal_mean, dtype=np.float32),
        "goal_std": np.asarray(goal_std, dtype=np.float32),
        "dynamics_state_dict": dynamics.state_dict(),
        "head_state_dict": readout.state_dict(),
        "metadata": metadata,
    }
