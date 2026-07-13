"""Inverse-dynamics verifier and ACID planning-cost primitives.

The planning cost follows Seo et al., *ACID: Action Consistency via Inverse
Dynamics for Planning with World Models* (arXiv:2607.02403, Eqs. 3--5):

    c_a = mean_t ||a_t - G(z_t, z_{t+1})||_2^2
    c   = c_g + lambda * std(c_g) / std(c_a) * c_a.

The authors had not released their flow-matching IDM code/checkpoint when this
baseline was implemented.  ``ACIDInverseDynamics`` is therefore an explicitly
labelled deterministic regression approximation: it uses the same two
consecutive frozen latents and predicts the same raw action chunk, but replaces
their 4-layer flow-matching transformer with a compact mean/max-pooled MLP.
Nothing in ``acid_cost`` depends on that architectural approximation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def pool_latent(z: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """Mean/max pool a single-frame token grid into ``(B, 2*latent_dim)``."""
    if z.shape[-1] != latent_dim:
        raise ValueError(
            f"latent tail dim {z.shape[-1]} does not match IDM latent_dim={latent_dim}"
        )
    tokens = z.reshape(z.shape[0], -1, latent_dim)
    return torch.cat((tokens.mean(dim=1), tokens.amax(dim=1)), dim=-1)


def transition_features(
    z_t: torch.Tensor, z_t1: torch.Tensor, latent_dim: int
) -> torch.Tensor:
    """Fixed-size features for one latent transition."""
    if z_t.shape != z_t1.shape:
        raise ValueError(f"transition shape mismatch: {z_t.shape} vs {z_t1.shape}")
    return torch.cat((pool_latent(z_t, latent_dim), pool_latent(z_t1, latent_dim)), dim=-1)


class ACIDInverseDynamics(nn.Module):
    """Deterministic two-latent inverse dynamics approximation.

    ``action_dim`` is the model-step raw action chunk (20 for MetaWorld: five
    4-D simulator actions), exactly the action space sampled by the local CEM.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(4 * self.latent_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.SiLU(),
            nn.Linear(self.hidden // 2, self.action_dim),
        )

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != 4 * self.latent_dim:
            raise ValueError(
                f"expected {4 * self.latent_dim} transition features, got {features.shape[-1]}"
            )
        return self.net(features)

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        return self.forward_features(transition_features(z_t, z_t1, self.latent_dim))


@torch.no_grad()
def action_consistency_cost(
    predicted_trajectory: torch.Tensor,
    actions: torch.Tensor,
    inverse_dynamics: ACIDInverseDynamics,
) -> torch.Tensor:
    """ACID Eq. 3, returning one consistency cost per candidate.

    Args:
        predicted_trajectory: ``(B, H+1, *frame_dims)``.
        actions: ``(B, H, action_dim)`` in raw planner coordinates.
    """
    if predicted_trajectory.shape[0] != actions.shape[0]:
        raise ValueError("trajectory/action batch mismatch")
    if predicted_trajectory.shape[1] != actions.shape[1] + 1:
        raise ValueError("trajectory must contain H+1 states for H actions")
    if actions.shape[-1] != inverse_dynamics.action_dim:
        raise ValueError(
            f"action dim {actions.shape[-1]} != IDM action_dim {inverse_dynamics.action_dim}"
        )
    bsz, horizon = actions.shape[:2]
    z0 = predicted_trajectory[:, :-1].reshape(
        bsz * horizon, *predicted_trajectory.shape[2:]
    )
    z1 = predicted_trajectory[:, 1:].reshape(
        bsz * horizon, *predicted_trajectory.shape[2:]
    )
    inferred = inverse_dynamics(z0, z1).reshape(bsz, horizon, -1)
    # Eq. 3 uses squared L2 norm over action coordinates, averaged over time.
    return ((actions - inferred) ** 2).sum(dim=-1).mean(dim=-1)


@torch.no_grad()
def acid_cost(
    goal_cost: torch.Tensor,
    consistency_cost: torch.Tensor,
    *,
    lambda_acid: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ACID Eqs. 4--5 with per-CEM-pool adaptive scaling.

    Population standard deviations (``correction=0``) are used because the pool
    is the complete candidate population for that CEM iteration.  If the
    consistency cost is constant, it cannot rerank candidates; its weight is set
    to zero instead of dividing by zero.  ``lambda_acid=0`` is bitwise the
    upstream terminal-cost null and does not evaluate a meaningless ratio.
    """
    if goal_cost.ndim != 1 or consistency_cost.ndim != 1:
        raise ValueError("goal and consistency costs must be one-dimensional")
    if goal_cost.shape != consistency_cost.shape:
        raise ValueError("goal and consistency cost shapes must match")
    sigma_g = goal_cost.std(correction=0)
    sigma_a = consistency_cost.std(correction=0)
    if float(lambda_acid) == 0.0 or bool(sigma_a <= eps):
        weight = torch.zeros((), dtype=goal_cost.dtype, device=goal_cost.device)
    else:
        weight = float(lambda_acid) * sigma_g / sigma_a.clamp_min(eps)
    total = goal_cost if float(lambda_acid) == 0.0 else goal_cost + weight * consistency_cost
    return total, {"sigma_goal": sigma_g, "sigma_acid": sigma_a, "acid_weight": weight}


def load_acid_idm(
    checkpoint: str | Path, device: torch.device | str
) -> tuple[ACIDInverseDynamics, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    required = {"latent_dim", "action_dim", "hidden", "state_dict"}
    missing = required - set(ckpt)
    if missing:
        raise ValueError(f"invalid ACID-IDM checkpoint; missing {sorted(missing)}")
    model = ACIDInverseDynamics(
        latent_dim=int(ckpt["latent_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden=int(ckpt["hidden"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
