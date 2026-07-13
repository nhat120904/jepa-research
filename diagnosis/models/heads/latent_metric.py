"""Learned latent metric — the cost-side fix that needs no object label (Track B).

The oracle ladder localised the contact-task wall to the **L2-in-DINO-latent cost
geometry**: ``‖z − z_goal‖²`` is dominated by arm-pose / background pixels, so its
minimum does not sit at task success (the object — a few patch tokens — is ~9% of
the residual). The object-grounded cost (``scripts/22`` probe + ``gobj`` arm) fixes
this *with* a sim-state object label, so it is Metaworld-only.

This module learns a goal-conditioned cost ``d_θ(z, z_goal)`` directly from the
*temporal order of frames within a trajectory* — no object GT — so the exact same
recipe transfers to DROID. ``d_θ`` is parametrised as a **quasimetric** (MRN-style:
a symmetric metric + a directed/asymmetric residual) so it respects the triangle
inequality and can encode that time has a direction (forward ≠ backward). Trained
(``scripts/33``) so that for an in-trajectory pair ``(z_t, z_{t'})`` with ``t<t'``,
``d_θ(z_t, z_{t'}) ≈ (t'−t)`` (temporal distance / VIP-style ranking), with
cross-trajectory pairs pushed large. The planner (``scripts/18`` ``metric`` arm,
``scripts/30 --cost metric``) then scores a candidate's final latent by
``d_θ(ẑ, z_goal)`` instead of L2 — a cost shaped by task progress, not appearance.

Frozen encoder/predictor throughout; only ``d_θ`` trains (a small pooled-token MLP).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.heads.mixture_predictor import flatten_tokens


class LatentMetric(nn.Module):
    """Goal-conditioned quasimetric ``d_θ(z, z_goal) ≈ steps-to-go``.

    Each frame latent is pooled (mean‖max over patch tokens) and embedded into a
    symmetric part ``u_s`` and an asymmetric part ``u_a``. The distance is

        d(x, g) = ‖u_s(x) − u_s(g)‖₂  +  Σ_i ReLU(u_a(x)_i − u_a(g)_i)

    The first term is a symmetric metric; the second is a directed residual that is
    0 when ``x = g`` and grows when ``x`` is "upstream" of ``g`` — together a valid
    quasimetric (triangle inequality holds for each term). ``d(g, g) = 0`` exactly,
    so the goal is the unique minimum of the planning cost.
    """

    def __init__(self, latent_dim: int, *, embed_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * embed_dim),                 # [u_s ‖ u_a]
        )

    def embed(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok = flatten_tokens(z, self.latent_dim)              # (B, N, D)
        pooled = torch.cat([tok.mean(dim=1), tok.max(dim=1).values], dim=-1)
        e = self.net(pooled)
        return e[..., : self.embed_dim], e[..., self.embed_dim:]

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """``z``/``z_goal``: single-frame latents ``(B, *frame)``; broadcastable.
        Returns the directed distance ``d_θ(z → z_goal)`` ``(B,)``."""
        s_x, a_x = self.embed(z)
        s_g, a_g = self.embed(z_goal)
        d_sym = torch.sqrt(((s_x - s_g) ** 2).sum(dim=-1) + 1e-9)
        d_asym = torch.relu(a_x - a_g).sum(dim=-1)
        return d_sym + d_asym


def load_latent_metric(ckpt_path: str | Path, device) -> tuple["LatentMetric", dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = LatentMetric(latent_dim=ckpt["latent_dim"],
                     embed_dim=ckpt.get("embed_dim", 256),
                     hidden=ckpt.get("hidden", 512))
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval(), ckpt
