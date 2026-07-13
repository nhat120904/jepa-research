"""TRM-style pairwise trajectory-reachability terminal metric.

This is an auditable adaptation of Li et al. (2026), *Beyond Euclidean
Proximity*.  The released MetaWorld JEPA caches contain very large token grids,
so feeding the flattened frame to a dense pair head would be needlessly large.
We first apply the same parameter-free mean/max token pooling used by the other
post-hoc heads in this repository, then form a symmetric pair feature
``[|p_i-p_j|, p_i*p_j]``.  The paper's two-layer 256-wide SiLU MLP and nonnegative
Softplus scalar output are otherwise retained.

The pooling is the only material adaptation from the paper's implementation.
It must be reported as a "TRM-style" baseline, not as an exact reproduction.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.heads.mixture_predictor import flatten_tokens


class TrajectoryReachabilityMetric(nn.Module):
    """Symmetric pairwise distance trained from within-trajectory time gaps."""

    def __init__(self, latent_dim: int, *, hidden: int = 256):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        # mean/max pooling produces 2D features; abs-difference plus product is 4D.
        self.net = nn.Sequential(
            nn.Linear(4 * self.latent_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
            nn.Softplus(),
        )

    def pool(self, z: torch.Tensor) -> torch.Tensor:
        tok = flatten_tokens(z, self.latent_dim)
        return torch.cat((tok.mean(dim=1), tok.max(dim=1).values), dim=-1)

    def pair_features(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        p_i, p_j = self.pool(z_i), self.pool(z_j)
        return torch.cat(((p_i - p_j).abs(), p_i * p_j), dim=-1)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        return self.net(self.pair_features(z_i, z_j)).squeeze(-1)


def standardize_candidate_cost(cost: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-candidate-batch standardization used by the TRM hybrid cost.

    ``unbiased=False`` makes singleton and tiny synthetic populations well
    defined and matches ordinary population z-scoring.
    """
    return (cost - cost.mean()) / cost.std(unbiased=False).clamp_min(eps)


def trm_terminal_cost(
    metric: TrajectoryReachabilityMetric,
    z_final: torch.Tensor,
    z_goal: torch.Tensor,
    *,
    mode: str,
    hybrid_weight: float = 1.0,
) -> torch.Tensor:
    """Score a single candidate population with replacement or hybrid TRM.

    The hybrid is ``zscore(raw latent MSE) + lambda*zscore(TRM)`` within the
    current candidate population, following the scale-control in the primary
    TRM paper.  Both lower values are better.
    """
    if mode not in {"replacement", "hybrid"}:
        raise ValueError("TRM mode must be 'replacement' or 'hybrid'")
    if hybrid_weight < 0:
        raise ValueError("hybrid_weight must be nonnegative")
    goal = z_goal
    if goal.dim() == z_final.dim() - 1:
        goal = goal.unsqueeze(0)
    goal = goal.expand(z_final.shape[0], *([-1] * (goal.dim() - 1)))
    learned = metric(z_final, goal)
    if mode == "replacement":
        return learned
    raw = ((z_final.reshape(z_final.shape[0], -1)
            - goal.reshape(goal.shape[0], -1)) ** 2).mean(dim=-1)
    return standardize_candidate_cost(raw) + float(hybrid_weight) * standardize_candidate_cost(learned)


def load_trajectory_reachability_metric(
    checkpoint: str | Path, device: torch.device | str
) -> tuple[TrajectoryReachabilityMetric, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    metric = TrajectoryReachabilityMetric(
        latent_dim=int(payload["latent_dim"]), hidden=int(payload.get("hidden", 256))
    )
    metric.load_state_dict(payload["state_dict"])
    return metric.to(device).eval(), payload
