"""Boundary-regime selection (contact-boundary reframing — design 2026-06-09 §3.1).

A transition ``(z_t, a_t)`` sits in the **boundary regime** when, among its
similar-state neighbours (the same ``‖z_t − z_{t'}‖ ≤ ρ`` pool the hard-negative
samplers use), a *small* change in action produces a *large* change in the true
outcome — i.e. the local dynamics bifurcate (gripper centred → lift vs. 2–3° off
→ no lift). This is the high-sensitivity regime that unimodal latent prediction
provably averages over, and it is *not* the same as "large effect": free-space
drift has large ``‖Δz‖`` but is smooth, so it scores low here.

Operationalisation (per anchor ``i`` over its in-radius neighbours ``J``):

    boundary_score_i = std_J( outcome_j )  /  ( mean_J ‖a_j − a_i‖ + ε )

The numerator is the spread of the **true outcome** across the neighbourhood
(object displacement on Metaworld — the bifurcation we can actually label; the
``‖Δz‖`` latent proxy on DROID). The denominator is the spread of the actions
that produced it, so the score is large exactly when outcomes fan out under
nearly-identical actions. A percentile threshold (``calibrate_boundary_threshold``)
turns the continuous score into a boundary mask, mirroring the ECS effect mask.

This module is pure geometry on cached tensors — no model is involved here; the
model only enters in ``metrics.boundary_blindness``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def state_neighbours(
    z: torch.Tensor,                      # (B, *frame) query anchors
    similarity_radius: float,
    max_neighbours: int = 16,
    min_neighbours: int = 2,
    pool_z: Optional[torch.Tensor] = None,   # (N, *frame) reference set
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-anchor similar-state neighbourhood by L2 latent distance.

    For each anchor we take its ``max_neighbours`` nearest reference transitions,
    then mark which of those actually fall within ``similarity_radius``. Returning
    a fixed ``(B, max_neighbours)`` index grid plus a boolean validity mask keeps
    downstream gather/reduce vectorised; the mask carries the variable
    real-neighbour count.

    The reference set is ``z`` itself (self excluded) when ``pool_z`` is None — the
    within-cell case — or the separate ``pool_z`` (the shared hard-negative pool)
    when given; indices then point into ``pool_z``.

    Returns:
        neighbour_idx:  (B, max_neighbours) long — nearest references (padded with
                        the nearest where fewer than ``max_neighbours`` exist).
        neighbour_mask: (B, max_neighbours) bool — True where the neighbour is
                        genuinely within ``similarity_radius``.
        valid:          (B,) bool — True where ≥ ``min_neighbours`` in-radius
                        neighbours exist (boundary score is defined).
    """
    B = z.shape[0]
    zf = z.reshape(B, -1).float()
    ref = zf if pool_z is None else pool_z.reshape(pool_z.shape[0], -1).float()
    R = ref.shape[0]
    d = torch.cdist(zf, ref, p=2)                      # (B, R)
    if pool_z is None:
        d.fill_diagonal_(float("inf"))                 # never pick self

    m = min(max_neighbours, max(R - (1 if pool_z is None else 0), 1))
    order = torch.argsort(d, dim=1)[:, :m]             # (B, m) nearest references
    if m < max_neighbours:                             # pad to a fixed width
        pad = order[:, -1:].expand(B, max_neighbours - m)
        order = torch.cat([order, pad], dim=1)
    neighbour_idx = order                              # (B, max_neighbours)

    gathered_d = torch.gather(d, 1, neighbour_idx)
    neighbour_mask = torch.isfinite(gathered_d) & (gathered_d <= similarity_radius)
    valid = neighbour_mask.sum(dim=1) >= min_neighbours
    return neighbour_idx, neighbour_mask, valid


def boundary_score_per_transition(
    anchor_actions: torch.Tensor,         # (B, A)
    neighbour_actions: torch.Tensor,      # (B, M, A) — gathered from the reference set
    neighbour_outcomes: torch.Tensor | np.ndarray,   # (B, M) true scalar outcomes
    neighbour_mask: torch.Tensor,         # (B, M) bool
    eps: float = 1e-8,
) -> np.ndarray:
    """Continuous boundary score per anchor (see module docstring).

    Takes the already-gathered neighbour arrays (so the same code serves the
    within-cell and pool cases). ``neighbour_outcomes`` is the dataset's true
    bifurcation signal: object displacement on Metaworld, ``‖Δz‖`` proxy on DROID.
    Rows with fewer than two real neighbours return ``nan`` (std/ratio undefined).

    Returns: (B,) float ndarray, ``nan`` where undefined.
    """
    a = anchor_actions.float()
    nb_a = neighbour_actions.to(a.device).float()      # (B, M, A)
    nb_out = torch.as_tensor(np.asarray(neighbour_outcomes),
                             dtype=torch.float32, device=a.device)  # (B, M)
    mask = neighbour_mask.to(a.device)
    M_real = mask.sum(dim=1)
    mf = mask.float()
    cnt = mf.sum(dim=1).clamp(min=1.0)

    mean_out = (nb_out * mf).sum(dim=1) / cnt
    var_out = ((nb_out - mean_out.unsqueeze(1)) ** 2 * mf).sum(dim=1) / cnt
    s_out = var_out.clamp(min=0.0).sqrt()              # spread of true outcome

    a_dist = (nb_a - a.unsqueeze(1)).norm(dim=-1)      # (B, M) ‖a_j − a_i‖
    a_spread = (a_dist * mf).sum(dim=1) / cnt          # spread of action differences

    score = (s_out / (a_spread + eps)).cpu().numpy()
    score[(M_real < 2).cpu().numpy()] = np.nan
    return score


def calibrate_boundary_threshold(scores: np.ndarray, quantile: float = 0.75) -> float:
    """Boundary cut = ``quantile`` of the (finite) boundary scores.

    Default 0.75 selects the top quartile of bifurcation-like transitions, the
    analogue of the median ``‖Δz‖`` cut used for the ECS effect threshold.
    """
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, quantile))


def boundary_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean (B,) mask: True where the boundary score exceeds ``threshold``.

    ``nan`` scores (undefined neighbourhoods) and a ``nan`` threshold yield False.
    """
    s = np.asarray(scores, dtype=float)
    if not np.isfinite(threshold):
        return np.zeros(s.shape, dtype=bool)
    return np.where(np.isfinite(s), s > threshold, False)
