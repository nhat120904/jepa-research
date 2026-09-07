"""Matched context-world-model architectures used by Gate 3 and Gate 4."""

from __future__ import annotations

import torch
from torch import nn


class ContextWorldModel(nn.Module):
    """Direct multi-horizon frozen-anchor predictor with a CaDM option."""

    def __init__(
        self,
        feature_dim: int,
        horizons: int,
        context_dim: int = 48,
        hidden_dim: int = 256,
        cadm_backward: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.horizons = horizons
        self.context_dim = context_dim
        self.cadm_backward = cadm_backward

        self.context_frame = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
        )
        self.context_gru = nn.GRU(
            128, 96, num_layers=2, batch_first=True
        )
        self.context_out = nn.Sequential(
            nn.LayerNorm(96), nn.Linear(96, context_dim)
        )
        self.current_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(4, 128), nn.GELU(), nn.LayerNorm(128)
        )
        self.forward_trunk = nn.Sequential(
            nn.Linear(128 + 128 + context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.forward_head = nn.Linear(hidden_dim, horizons * feature_dim)

        if cadm_backward:
            self.backward_future = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, 128),
                nn.GELU(),
            )
            self.horizon_embedding = nn.Embedding(horizons, 16)
            self.backward_model = nn.Sequential(
                nn.Linear(128 + 128 + context_dim + 16, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, feature_dim),
            )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        # CaDM feeds state differences into its context encoder.  The visual
        # analogue keeps all temporal increments and no absolute shortcut.
        difference = context[:, 1:] - context[:, :-1]
        encoded = self.context_frame(difference)
        output, _ = self.context_gru(encoded)
        return self.context_out(output[:, -1])

    def predict_from_context(
        self,
        context_latent: torch.Tensor,
        current: torch.Tensor,
        action_knots: torch.Tensor,
    ) -> torch.Tensor:
        current_encoded = self.current_encoder(current)
        action_encoded = self.action_encoder(action_knots.flatten(start_dim=1))
        trunk = self.forward_trunk(
            torch.cat([current_encoded, action_encoded, context_latent], dim=-1)
        )
        delta = self.forward_head(trunk).reshape(
            len(current), self.horizons, self.feature_dim
        )
        return current[:, None, :] + delta

    def forward(
        self,
        context: torch.Tensor,
        current: torch.Tensor,
        action_knots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode_context(context)
        future = self.predict_from_context(latent, current, action_knots)
        return future, latent

    def backward_prediction(
        self,
        future: torch.Tensor,
        action_knots: torch.Tensor,
        context_latent: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cadm_backward:
            raise RuntimeError('backward_prediction requires cadm_backward=True')
        batch, horizons, _ = future.shape
        future_encoded = self.backward_future(future)
        action = self.action_encoder(action_knots.flatten(start_dim=1))
        action = action[:, None, :].expand(-1, horizons, -1)
        context = context_latent[:, None, :].expand(-1, horizons, -1)
        horizon_index = torch.arange(horizons, device=future.device)
        horizon = self.horizon_embedding(horizon_index)[None].expand(batch, -1, -1)
        backward_input = torch.cat(
            [future_encoded, action, context, horizon], dim=-1
        )
        return self.backward_model(backward_input)
