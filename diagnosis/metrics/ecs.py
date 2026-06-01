"""Effect-Conditional Sensitivity (ECS) and the effect mask.

ECS = AUG, conditioned on ||z_{t+1} - z_t|| > tau.

This excludes near-static transitions where the action genuinely should not
change the next latent much (free-space drift, visual noise). ``tau`` is
calibrated per-model as the median of ||z_{t+1} - z_t|| over the eval set.

The same ``effect_mask`` is reused to compute an **effect-conditioned CRA**
(CRA restricted to transitions that actually changed the state), which is the
primary decision signal in the updated plan: a low *raw* CRA in contact regimes
can just reflect tiny one-step latent deltas, whereas a low *effect-conditioned*
CRA means the model genuinely fails to use actions when they matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from models.adapters import WorldModelAdapter

from .aug import aug_per_transition


@dataclass
class ECSResult:
    ecs: float
    threshold: float
    n_kept: int
    n_total: int


@torch.no_grad()
def calibrate_effect_threshold(
    z_t: torch.Tensor,
    z_t1: torch.Tensor,
    quantile: float = 0.5,
) -> float:
    """Per-model effect threshold = median (default) of ||z_{t+1} - z_t||."""
    diff = (z_t1 - z_t).reshape(z_t.shape[0], -1).norm(dim=-1)
    return float(diff.quantile(quantile).item())


def effect_mask(z_t: torch.Tensor, z_t1: torch.Tensor, threshold: float) -> np.ndarray:
    """Boolean (B,) mask: True where ||z_{t+1} - z_t|| > threshold."""
    diff = (z_t1 - z_t).reshape(z_t.shape[0], -1).norm(dim=-1)
    return (diff > threshold).cpu().numpy()


@torch.no_grad()
def compute_ecs(
    adapter: WorldModelAdapter,
    z_t: torch.Tensor,
    a_t: torch.Tensor,
    z_t1: torch.Tensor,
    threshold: float,
    proprio_t: Optional[torch.Tensor] = None,
) -> ECSResult:
    mask = effect_mask(z_t, z_t1, threshold)
    if mask.sum() == 0:
        return ECSResult(ecs=float("nan"), threshold=threshold, n_kept=0, n_total=int(z_t.shape[0]))
    idx = torch.as_tensor(np.nonzero(mask)[0], device=z_t.device)
    p_kept = proprio_t[idx] if proprio_t is not None else None
    aug_per = aug_per_transition(adapter, z_t[idx], a_t[idx], z_t1[idx], proprio_t=p_kept)
    return ECSResult(
        ecs=float(np.mean(aug_per)),
        threshold=threshold,
        n_kept=int(mask.sum()),
        n_total=int(z_t.shape[0]),
    )
