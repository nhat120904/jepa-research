"""Shared goal-conditioned cost head for the selection-aware encoder sprint."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.heads.mixture_predictor import flatten_tokens


class AttentionPool(nn.Module):
    """Lightweight learned pooling that keeps token-level spatial evidence."""

    def __init__(self, latent_dim: int, hidden: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.score(tokens).squeeze(-1), dim=-1)
        return (weight.unsqueeze(-1) * tokens).sum(dim=1)


class SelectionCostHead(nn.Module):
    """Predict a scalar cost from candidate and goal frame latents.

    All four experimental arms use this exact architecture.  Consequently the
    comparison isolates encoder capacity and objective, not a stronger readout.
    """

    def __init__(self, latent_dim: int, *, hidden: int = 384, pool_hidden: int = 128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.pool_hidden = int(pool_hidden)
        self.pool = AttentionPool(self.latent_dim, self.pool_hidden)
        self.net = nn.Sequential(
            nn.LayerNorm(4 * self.latent_dim),
            nn.Linear(4 * self.latent_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.SiLU(),
            nn.Linear(self.hidden // 2, 1),
        )

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        return self.pool(flatten_tokens(z, self.latent_dim))

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x, goal = self.embed(z), self.embed(z_goal)
        feature = torch.cat([x, goal, x - goal, x * goal], dim=-1)
        return self.net(feature).squeeze(-1)


def build_selection_cost_from_checkpoint(checkpoint: dict, device) -> SelectionCostHead:
    head = SelectionCostHead(
        checkpoint["latent_dim"], hidden=checkpoint.get("head_hidden", 384),
        pool_hidden=checkpoint.get("pool_hidden", 128))
    head.load_state_dict(checkpoint["head"])
    return head.to(device).eval()
