"""Residual corrective predictor — the predictor-side fix (option C).

``ẑ_{t+1} = F_frozen(z_t, a_t) + Δ_θ(z_t, a_t)``: a small token-wise,
action-conditioned transformer over the latent patch grid that corrects the
frozen predictor's one-step output. Zero-initialised at the output so training
starts exactly at the frozen predictor (identity correction) and only learns the
residual — the broken counterfactual action→object channel
(``docs/plans/2026-06-18-residual-predictor-design.md``).

Trained single-step on the cache (``scripts/20``) with a latent-recon +
object-grounded loss; applied recursively inside the planner's rollout
(``scripts/18`` ``_PlanAdapter`` with ``residual_head=``).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.heads.mixture_predictor import flatten_tokens


class ResidualPredictorHead(nn.Module):
    """Token-wise action-conditioned residual over a single-frame latent.

    forward(z_t, a_norm) -> residual with the SAME shape as ``z_t``.
    ``a_norm`` is the model-space (normalised, frameskip-folded) action,
    ``(B, model_action_dim)`` — the planner already normalises actions, and the
    trainer normalises to match, so the head always sees model-space actions.
    """

    def __init__(self, latent_dim: int, action_dim: int, *, n_layers: int = 2,
                 n_heads: int = 6, ff_mult: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.a_proj = nn.Sequential(nn.Linear(action_dim, latent_dim), nn.SiLU(),
                                    nn.Linear(latent_dim, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dim_feedforward=ff_mult * latent_dim,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(latent_dim, latent_dim)
        # Zero-init the output projection => Δ ≡ 0 at start (identity correction).
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_t: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        shape = z_t.shape
        tok = flatten_tokens(z_t, self.latent_dim)            # (B, N, D)
        a = self.a_proj(a_norm).unsqueeze(1)                  # (B, 1, D) action context
        h = self.encoder(tok + a)                             # broadcast-add action, then attend
        res = self.out(h)                                     # (B, N, D)
        return res.reshape(shape)


def load_residual_head(ckpt_path: str | Path, device) -> tuple["ResidualPredictorHead", dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = ResidualPredictorHead(
        latent_dim=ckpt["latent_dim"], action_dim=ckpt["action_dim"],
        n_layers=ckpt.get("n_layers", 2), n_heads=ckpt.get("n_heads", 6),
        ff_mult=ckpt.get("ff_mult", 2))
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval(), ckpt
