"""History-conditioned event-state observers for OGBench-Scene.

The single-frame observer in :mod:`event_smdp_h0.scene_event_perception` was
trained only on canonical milestone roots.  Two distinct hypotheses can explain
its deployment failure on task-5 state ``(cube=1, window=2)``:

``coverage``
    that event state never appears as a canonical root, so the classifier
    projects it onto the nearest state it has actually seen;
``history``
    the current frame is genuinely insufficient and the event state can only be
    recovered from the observation/action history.

This module supports a matched 2x2 factorial over those two axes.  A single
GRU architecture serves both input regimes: ``history_length=1`` feeds only the
current observation (with a ``no-skill`` action token), any larger value feeds
the whole prefix.  Parameter count is therefore identical across the input
axis, so the contrast isolates history rather than capacity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from event_smdp_h0.scene_core import SKILLS, MilestoneState
from event_smdp_h0.scene_learning import task_vector


PROTOCOL = "scene_event_history_v1"

# One extra action token marks "no preceding skill" at the first observation.
NO_SKILL = len(SKILLS)
ARMS = ("frame_canonical", "frame_full", "history_canonical", "history_full")

# Input ablations used to decompose "history" into its observation and action
# components.  `none` reproduces the original factorial byte for byte.
ABLATIONS = ("none", "action_only", "obs_history")
ABLATION_ARMS = ("frame_full", "obs_history_full", "action_only_full", "history_full")


def canonical_skill_paths(task_id: int) -> tuple[tuple[int, ...], ...]:
    """Pure restatement of the collector's canonical paths.

    ``scripts/collect_scene_h1.py`` owns the authoritative definition but pulls
    in MuJoCo through ``run_scene_gate0``.  Training code asserts equality with
    that source; this copy exists so dataset construction stays importable
    without a simulator.
    """

    by_name = {name: index for index, name in enumerate(SKILLS)}
    drawer_branch = (
        by_name["toggle_button_0"],
        by_name["drawer_open"],
        by_name["place_cube_in_drawer"],
        by_name["drawer_close"],
    )
    if task_id == 4:
        return (drawer_branch,)
    if task_id != 5:
        raise ValueError("Scene H0 is locked to tasks 4 and 5")
    primary = drawer_branch + (
        by_name["toggle_button_0"],
        by_name["toggle_button_1"],
        by_name["window_open"],
        by_name["toggle_button_1"],
    )
    window_first = (
        by_name["toggle_button_1"],
        by_name["window_open"],
        by_name["toggle_button_1"],
        by_name["toggle_button_0"],
        by_name["drawer_open"],
        by_name["place_cube_in_drawer"],
        by_name["drawer_close"],
        by_name["toggle_button_0"],
    )
    return (primary, window_first)


class HistoryEventObserver(nn.Module):
    """GRU over ``(previous skill, observation)`` tokens; reads out current q."""

    OUTPUT_DIM = 6 + 4 + 1

    def __init__(
        self,
        feature_dim: int,
        goal_dim: int,
        width: int = 256,
        skill_embed: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.skill_embedding = nn.Embedding(len(SKILLS) + 1, skill_embed)
        token_dim = feature_dim + skill_embed + goal_dim + 2
        self.input_norm = nn.LayerNorm(token_dim)
        self.gru = nn.GRU(token_dim, width, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, self.OUTPUT_DIM),
        )

    def forward(
        self,
        feature: torch.Tensor,
        prev_skill: torch.Tensor,
        goal: torch.Tensor,
        task: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``feature`` is ``(B, T, D)``; ``lengths`` gives the valid prefix."""

        steps = feature.shape[1]
        tokens = torch.cat(
            [
                feature,
                self.skill_embedding(prev_skill),
                goal[:, None].expand(-1, steps, -1),
                task[:, None].expand(-1, steps, -1),
            ],
            dim=-1,
        )
        outputs, _ = self.gru(self.input_norm(tokens))
        last = (lengths - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
        final = outputs.gather(1, last).squeeze(1)
        raw = self.head(final)
        return {
            "cube_logits": raw[..., :6],
            "window_logits": raw[..., 6:10],
            "stable_logit": raw[..., 10],
        }


def _milestone_from_logits(
    cube_probability: torch.Tensor,
    window_probability: torch.Tensor,
    stable_probability: float,
    task_id: int,
) -> MilestoneState:
    cube = int(cube_probability.argmax().item())
    window = int(window_probability.argmax().item())
    return MilestoneState(
        task_id=task_id,
        cube_stage=min(4 if task_id == 4 else 5, cube),
        window_stage=0 if task_id == 4 else min(3, window),
        stable_count=3 if stable_probability >= 0.5 else 0,
    )


class HistoryObserverEvaluator:
    """Deployment wrapper.  ``history_length=1`` reproduces a frame observer."""

    def __init__(self, checkpoint: Path | str, device: str = "cuda") -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol") != PROTOCOL:
            raise ValueError(f"unexpected observer protocol: {payload.get('protocol')}")
        self.arm = str(payload["arm"])
        self.ablation = str(payload.get("ablation", "none"))
        if self.ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation: {self.ablation}")
        self.feature_view = str(payload["feature_view"])
        self.history_length = int(payload["history_length"])
        self.feature_dim = int(payload["feature_dim"])
        self.goal_dim = int(payload["goal_dim"])
        self.width = int(payload["width"])
        self.device = torch.device(device)
        for key in ("feature_mean", "feature_std", "goal_mean", "goal_std"):
            setattr(
                self,
                key,
                torch.as_tensor(payload[key], dtype=torch.float32, device=self.device),
            )
        self.model = HistoryEventObserver(
            self.feature_dim, self.goal_dim, width=self.width
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        features: list[np.ndarray] | np.ndarray,
        prev_skills: list[int],
        goal: np.ndarray,
        task_id: int,
    ) -> tuple[MilestoneState, dict[str, object]]:
        """``features[i]`` is the observation after ``prev_skills[i]``.

        ``prev_skills[0]`` must be :data:`NO_SKILL`.  Frame arms keep only the
        final element, so the caller may always pass the full history.
        """

        features = [np.asarray(value, dtype=np.float32) for value in features]
        if len(features) != len(prev_skills):
            raise ValueError("features and prev_skills must align")
        if not features:
            raise ValueError("history must contain the current observation")
        if int(prev_skills[0]) != NO_SKILL:
            raise ValueError("the first history token must be NO_SKILL")
        if self.history_length == 1:
            features, prev_skills = features[-1:], [NO_SKILL]
        # A GRU accepts any length; exceeding the trained maximum is a
        # distribution note, recorded below, not an architectural error.
        beyond_training = len(features) > self.history_length
        feature_tensor = torch.as_tensor(
            np.stack(features), dtype=torch.float32, device=self.device
        )[None]
        feature_tensor = (feature_tensor - self.feature_mean) / self.feature_std
        if self.ablation == "action_only":
            feature_tensor = torch.zeros_like(feature_tensor)
        if self.ablation == "obs_history":
            prev_skills = [NO_SKILL] * len(prev_skills)
        skill_tensor = torch.as_tensor(
            [int(value) for value in prev_skills], dtype=torch.int64, device=self.device
        )[None]
        goal_tensor = torch.as_tensor(
            np.asarray(goal, dtype=np.float32), dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        goal_tensor = (goal_tensor - self.goal_mean) / self.goal_std
        task = torch.as_tensor(
            task_vector(task_id), dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        lengths = torch.as_tensor(
            [feature_tensor.shape[1]], dtype=torch.int64, device=self.device
        )
        out = self.model(feature_tensor, skill_tensor, goal_tensor, task, lengths)
        cube_probability = torch.softmax(out["cube_logits"], dim=-1)[0]
        window_probability = torch.softmax(out["window_logits"], dim=-1)[0]
        stable_probability = float(torch.sigmoid(out["stable_logit"]).item())
        state = _milestone_from_logits(
            cube_probability, window_probability, stable_probability, task_id
        )
        details = {
            "cube_probability": cube_probability.cpu().numpy(),
            "window_probability": window_probability.cpu().numpy(),
            "stable_probability": stable_probability,
            "history_steps": int(feature_tensor.shape[1]),
            "beyond_trained_history": bool(beyond_training),
            "ablation": self.ablation,
        }
        return state, details
