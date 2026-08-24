"""Shared models, features, and exact CEM-refit utilities for the H0 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


ARMS = ("pointwise", "listwise", "elite", "operator", "operator_metric")


def split_for_order(order: int) -> str:
    residue = int(order) % 8
    if residue in (0, 4):
        return "test"
    if residue == 2:
        return "val"
    return "train"


def hard_elite_refit(actions: np.ndarray, costs: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact upstream-CEM hard elite indices, mean, and sample std."""
    actions = np.asarray(actions, dtype=np.float32)
    costs = np.asarray(costs, dtype=np.float64)
    if actions.ndim < 2 or costs.shape != (len(actions),):
        raise ValueError("population shapes do not align")
    if not 1 < topk <= len(actions):
        raise ValueError("topk must be in [2, population size]")
    elite = np.argsort(costs, kind="stable")[:topk]
    selected = actions[elite]
    return elite, selected.mean(axis=0), selected.std(axis=0, ddof=1)


def robust_population_features(native_cost: np.ndarray) -> np.ndarray:
    """Deployable, scale-stable native-cost and rank features."""
    cost = np.asarray(native_cost, dtype=np.float32).reshape(-1)
    median = np.median(cost)
    mad = np.median(np.abs(cost - median))
    robust = (cost - median) / max(float(mad), 1e-6)
    order = np.argsort(cost, kind="stable")
    rank = np.empty_like(order, dtype=np.float32)
    rank[order] = np.arange(len(cost), dtype=np.float32)
    rank /= max(len(cost) - 1, 1)
    return np.stack([robust, rank, np.log1p(np.maximum(cost, 0.0))], axis=1)


def build_features_np(
    actions: np.ndarray,
    native_cost: np.ndarray,
    endpoint: np.ndarray,
    current: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """Construct only quantities available to a zero-query planner."""
    actions = np.asarray(actions, dtype=np.float32)
    endpoint = np.asarray(endpoint, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32).reshape(1, -1)
    goal = np.asarray(goal, dtype=np.float32).reshape(1, -1)
    n = len(actions)
    if endpoint.shape[0] != n or endpoint.shape[1] != goal.shape[1]:
        raise ValueError("embedding shapes do not align")
    residual = endpoint - goal
    context = np.repeat(current - goal, n, axis=0)
    norms = np.stack(
        [np.linalg.norm(residual, axis=1), np.linalg.norm(actions.reshape(n, -1), axis=1)],
        axis=1,
    )
    return np.concatenate(
        [
            actions.reshape(n, -1),
            robust_population_features(native_cost),
            endpoint,
            residual,
            np.square(residual),
            context,
            norms,
        ],
        axis=1,
    ).astype(np.float32)


def fit_standardizer(groups: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate([np.asarray(group, dtype=np.float32) for group in groups], axis=0)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class PopulationScorer(nn.Module):
    def __init__(self, feature_dim: int, width: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class MetricOperatorScorer(nn.Module):
    """Residual metric adapter plus a small deployable action/context correction."""

    def __init__(self, embedding_dim: int, action_dim: int, bottleneck: int = 32) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck, bias=False), nn.SiLU(),
            nn.Linear(bottleneck, embedding_dim, bias=False),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        self.correction = nn.Sequential(
            nn.Linear(action_dim + 3, 128), nn.SiLU(), nn.Linear(128, 1)
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def project(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.sigmoid(self.residual_scale) * self.adapter(value)

    def forward(
        self,
        actions: torch.Tensor,
        native_features: torch.Tensor,
        endpoint: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        adapted_endpoint = self.project(endpoint)
        adapted_goal = self.project(goal)
        metric = (adapted_endpoint - adapted_goal).square().sum(dim=-1)
        correction_in = torch.cat([actions.flatten(start_dim=-2), native_features], dim=-1)
        return metric + 0.1 * self.correction(correction_in).squeeze(-1)


def straight_through_refit(
    actions: torch.Tensor, scores: torch.Tensor, topk: int, temperature: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard CEM refit in the forward pass, soft top-k surrogate in backward."""
    if actions.ndim != 3 or scores.ndim != 1 or len(actions) != len(scores):
        raise ValueError("expected actions [N,H,A] and scores [N]")
    hard_indices = torch.topk(scores, k=topk, largest=False).indices
    hard = torch.zeros_like(scores).scatter(0, hard_indices, 1.0)
    soft = topk * torch.softmax(-scores / temperature, dim=0)
    # Forward equals hard exactly; gradients follow the smooth population weights.
    weights = hard + soft - soft.detach()
    flat = actions.flatten(start_dim=1)
    mean = (weights[:, None] * flat).sum(dim=0) / topk
    centered = flat - mean[None]
    variance = (weights[:, None] * centered.square()).sum(dim=0) / (topk - 1)
    std = torch.sqrt(variance.clamp_min(1e-8))
    return mean.reshape(actions.shape[1:]), std.reshape(actions.shape[1:]), hard_indices


def normalized_operator_loss(
    student_mean: torch.Tensor,
    student_std: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    proposal_std: torch.Tensor,
) -> torch.Tensor:
    scale = proposal_std.clamp_min(0.05)
    mean_error = ((student_mean - teacher_mean) / scale).square().mean()
    std_error = ((student_std - teacher_std) / scale).square().mean()
    return mean_error + std_error


@dataclass(frozen=True)
class CheckpointSpec:
    arm: str
    feature_dim: int
    embedding_dim: int
    action_shape: tuple[int, int]
    feature_mean: np.ndarray
    feature_std: np.ndarray


def save_checkpoint(
    path: Path, model: nn.Module, spec: CheckpointSpec, seed: int, extra: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(), "arm": spec.arm, "feature_dim": spec.feature_dim,
            "embedding_dim": spec.embedding_dim, "action_shape": spec.action_shape,
            "feature_mean": spec.feature_mean, "feature_std": spec.feature_std,
            "seed": seed, "extra": extra,
        },
        path,
    )


def load_checkpoint(path: Path, device: str) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["arm"] == "operator_metric":
        model = MetricOperatorScorer(
            payload["embedding_dim"], int(np.prod(payload["action_shape"]))
        )
    else:
        model = PopulationScorer(payload["feature_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload
