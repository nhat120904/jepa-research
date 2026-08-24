"""Temporal-straightening projector — cost-side fix on the frozen encoder.

Motivation (`hys_h0/docs/PREGATE_CURVATURE.md`, 2026-08-21): on MetaWorld the frozen
DINO/JEPA latent trajectory has curvature **1.096** where chance is 1.000 and the
underlying physical trajectory sits at 0.03 (end-effector) to 0.32 (object). The latent
zig-zags: consecutive displacements are anti-correlated. That is precisely the geometry
`‖z − z_goal‖²` is read through, and the oracle ladder already localised the contact wall
to that cost (`diagnosis/results/oracle_ladder_cost_report.md`).

This learns a projector ``P`` (frozen encoder, frozen predictor) trained so that latent
trajectories become locally straight, following the ICML-2026 temporal-straightening
curvature form

    L_curve = mean_t [ 1 - cos( P(z_{t+1}) - P(z_t),  P(z_{t+2}) - P(z_{t+1}) ) ]

The planner then scores candidates by ``‖P(ẑ) − P(z_goal)‖²`` instead of raw latent L2.

Collapse is the obvious degenerate solution: a projector that maps every frame onto one
line, or to a constant, has zero curvature and zero utility. Because the encoder is frozen
here (unlike the original paper, which co-trains encoder + predictor and is regularised by
the prediction loss), collapse must be blocked explicitly, so training carries VICReg-style
variance and covariance terms and the trainer reports effective rank every epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from models.heads.mixture_predictor import flatten_tokens


class StraighteningProjector(nn.Module):
    """Pooled-token MLP ``P: z -> R^out_dim``."""

    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 1024):
        super().__init__()
        self.in_dim = int(in_dim)   # per-token latent dim D (384 for the DINO caches)
        self.out_dim = int(out_dim)
        # mean and max pooling over patch tokens, concatenated
        self.net = nn.Sequential(
            nn.Linear(2 * self.in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, self.out_dim),
        )

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, ...tokens..., D) -> (B, 2D) via mean+max over the token axis."""
        t = flatten_tokens(z, self.in_dim)        # (B, N, D)
        return torch.cat([t.mean(dim=1), t.amax(dim=1)], dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self._pool(z))


# ---------------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------------
def curvature_loss(p_seq: torch.Tensor, keep: Optional[torch.Tensor] = None):
    """p_seq (B, T, d) -> scalar mean of 1 - cos(dp_t, dp_{t+1}).

    ``keep`` (B, T-2) optionally masks which triples contribute (used by the
    mode-gated ablation arm, which excludes contact-mode switches).
    """
    dp = p_seq[:, 1:] - p_seq[:, :-1]                        # (B, T-1, d)
    a, b = dp[:, :-1], dp[:, 1:]                             # (B, T-2, d)
    cos = nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)
    c = 1.0 - cos
    if keep is None:
        return c.mean(), c.detach()
    keep = keep.to(c.dtype)
    denom = keep.sum().clamp_min(1.0)
    return (c * keep).sum() / denom, c.detach()


def vicreg_terms(p_flat: torch.Tensor, gamma: float = 1.0):
    """Variance (anti-collapse) and covariance (anti-redundancy) on (N, d)."""
    p = p_flat - p_flat.mean(dim=0, keepdim=True)
    std = torch.sqrt(p.var(dim=0) + 1e-6)
    var_loss = torch.relu(gamma - std).mean()
    n, d = p.shape
    cov = (p.T @ p) / max(n - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cov_loss = (off ** 2).sum() / d
    return var_loss, cov_loss


def effective_rank(p_flat: torch.Tensor) -> float:
    """exp(entropy of normalised singular values) — collapse monitor."""
    with torch.no_grad():
        p = p_flat - p_flat.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(p.float())
        s = s / s.sum().clamp_min(1e-12)
        ent = -(s * torch.log(s.clamp_min(1e-12))).sum()
        return float(torch.exp(ent))


# ---------------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------------
def save_projector(model: StraighteningProjector, path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "in_dim": model.in_dim, "out_dim": model.out_dim,
                "meta": meta}, path)


def load_projector(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = StraighteningProjector(ck["in_dim"], ck["out_dim"])
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, ck.get("meta", {})
