"""Boundary Blindness (BB) — design 2026-06-09 §3.2.

The quantitative form of the reframed gap: does the model's prediction track the
*true local sensitivity to action* near a contact boundary, or does it smooth the
bifurcation into a single averaged future?

For a boundary-regime anchor ``z_t`` and its similar-state neighbourhood of
near-by actions ``{a_j}`` (from ``stratification.boundary_regime.state_neighbours``):

    S_true  = spread of the **true** outcome across the neighbourhood
              (object Δ on Metaworld; ‖Δz‖ proxy on DROID) — a property of data.
    S_model = spread of the model's predicted ``F(z_t, a_j)`` across the *same*
              actions, all applied to the one anchor state — a property of F.

Both are standardised (z-scored over the reference population, so the differing
units of outcome vs. latent cancel) and

    BB = relu( S_true_norm − S_model_norm ).

``BB ≈ 0`` → the model's action-sensitivity matches the world's. ``BB`` large →
the world bifurcates here but the model predicts nearly the same future for every
action — it is *blind to the boundary*. This is what CRA/ECS cannot see: they ask
"does the model distinguish actions at all", not "does it resolve the sharp
boundary". The thesis: baselines show high ``BB`` concentrated in
pre-grasp / gripper-actuation boundary transitions.

Two-pass by construction (like the ECS threshold): callers compute the raw
``(S_true, S_model)`` for every transition first, then standardise over the whole
reference population so ``BB`` is comparable across regimes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from models.adapters import WorldModelAdapter


def _repeat_proprio(proprio_t: Optional[torch.Tensor], M: int) -> Optional[torch.Tensor]:
    if proprio_t is None:
        return None
    B = proprio_t.shape[0]
    return (proprio_t.unsqueeze(1)
            .expand(B, M, *proprio_t.shape[1:])
            .reshape(B * M, *proprio_t.shape[1:]))


@torch.no_grad()
def boundary_sensitivities_per_transition(
    adapter: WorldModelAdapter,
    z_t: torch.Tensor,                    # (B, *frame)
    neighbour_actions: torch.Tensor,      # (B, M, A)
    neighbour_outcomes: torch.Tensor | np.ndarray,   # (B, M) true scalar outcomes
    neighbour_mask: torch.Tensor,         # (B, M) bool
    proprio_t: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Raw ``(S_true, S_model)`` per anchor — the pre-standardisation pair.

    ``S_true`` is the masked std of the neighbours' true outcomes. ``S_model`` is
    the masked latent-space spread (RMS distance to the masked centroid) of the
    model's predictions ``F(z_t, a_j)`` over the neighbourhood, all applied to the
    single anchor state ``z_t``. Padded neighbours (``neighbour_mask`` False) are
    excluded from both reductions.

    Returns: ``(S_true (B,), S_model (B,))`` float ndarrays.
    """
    B, M, A = neighbour_actions.shape
    mask = neighbour_mask.to(z_t.device).float()       # (B, M)
    cnt = mask.sum(dim=1).clamp(min=1.0)

    out = torch.as_tensor(np.asarray(neighbour_outcomes),
                          dtype=torch.float32, device=z_t.device)
    mean_o = (out * mask).sum(dim=1) / cnt
    s_true = (((out - mean_o.unsqueeze(1)) ** 2 * mask).sum(dim=1) / cnt
              ).clamp(min=0.0).sqrt().cpu().numpy()

    z_rep = z_t.unsqueeze(1).expand(B, M, *z_t.shape[1:]).reshape(B * M, *z_t.shape[1:])
    a_flat = neighbour_actions.reshape(B * M, A).to(z_t.device).float()
    preds = adapter.predict(z_rep, a_flat, proprio_t=_repeat_proprio(proprio_t, M))
    preds = preds.reshape(B, M, -1)                    # (B, M, Dz)

    mm = mask.unsqueeze(-1)
    centroid = (preds * mm).sum(dim=1) / cnt.unsqueeze(-1)         # (B, Dz)
    sq = ((preds - centroid.unsqueeze(1)) ** 2).sum(dim=-1)        # (B, M)
    s_model = ((sq * mask).sum(dim=1) / cnt).clamp(min=0.0).sqrt().cpu().numpy()
    return s_true, s_model


def _standardise(x: np.ndarray) -> np.ndarray:
    """z-score, robust to a (near-)constant array → all zeros."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd


def boundary_blindness(s_true: np.ndarray, s_model: np.ndarray) -> np.ndarray:
    """``BB = relu(S_true_norm − S_model_norm)`` over the given population.

    Standardise both sensitivities over *all* the transitions passed in (the
    reference population), then take the rectified gap. Pass the dataset/model-wide
    arrays so ``BB`` is comparable across regimes; slicing per regime afterwards.

    Returns: (B,) float ndarray ≥ 0.
    """
    return np.maximum(_standardise(s_true) - _standardise(s_model), 0.0)
