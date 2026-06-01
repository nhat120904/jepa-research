"""Action Usage Gap (AUG).

AUG = E[ MSE(F(z_t, pi(a_t)), z_{t+1}) - MSE(F(z_t, a_t), z_{t+1}) ]

where pi permutes actions across the batch. A model that ignores actions yields
AUG ≈ 0; a well-grounded model yields large positive AUG.

NOTE on comparability: AUG is a raw latent-space MSE gap, so its magnitude is
**not comparable across models** with different latent scales (e.g. ViT-S vs
ViT-L). Compare AUG only within a model (across regimes); CRA (a ranking) is the
cross-model-comparable metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from models.adapters import WorldModelAdapter


@dataclass
class AUGResult:
    aug: float
    mse_factual: float
    mse_shuffled: float
    n: int


def _derangement(B: int, device) -> torch.Tensor:
    perm = torch.randperm(B, device=device)
    identity = torch.arange(B, device=device)
    fixed = perm == identity
    if fixed.any() and B > 1:
        perm[fixed] = (perm[fixed] + 1) % B
    return perm


@torch.no_grad()
def aug_per_transition(
    adapter: WorldModelAdapter,
    z_t: torch.Tensor,
    a_t: torch.Tensor,
    z_t1: torch.Tensor,
    permutation: Optional[torch.Tensor] = None,
    proprio_t: Optional[torch.Tensor] = None,
    return_mse: bool = False,
):
    """Per-transition AUG = mse_shuffled - mse_factual. Returns (B,) np.ndarray."""
    B = z_t.shape[0]
    if permutation is None:
        permutation = _derangement(B, a_t.device)
    a_shuffled = a_t[permutation]

    z_hat = adapter.predict(z_t, a_t, proprio_t=proprio_t)
    z_hat_shuffled = adapter.predict(z_t, a_shuffled, proprio_t=proprio_t)

    mse_factual = ((z_hat - z_t1) ** 2).reshape(B, -1).mean(dim=-1)
    mse_shuffled = ((z_hat_shuffled - z_t1) ** 2).reshape(B, -1).mean(dim=-1)
    aug_per = (mse_shuffled - mse_factual).cpu().numpy()
    if return_mse:
        return aug_per, mse_factual.cpu().numpy(), mse_shuffled.cpu().numpy()
    return aug_per


@torch.no_grad()
def compute_aug(
    adapter: WorldModelAdapter,
    z_t: torch.Tensor,
    a_t: torch.Tensor,
    z_t1: torch.Tensor,
    permutation: Optional[torch.Tensor] = None,
    proprio_t: Optional[torch.Tensor] = None,
) -> AUGResult:
    aug_per, mse_f, mse_s = aug_per_transition(
        adapter, z_t, a_t, z_t1, permutation=permutation, proprio_t=proprio_t, return_mse=True
    )
    return AUGResult(
        aug=float(aug_per.mean()),
        mse_factual=float(mse_f.mean()),
        mse_shuffled=float(mse_s.mean()),
        n=int(z_t.shape[0]),
    )
