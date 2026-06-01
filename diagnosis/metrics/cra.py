"""Counterfactual Ranking Accuracy.

For each factual transition (z_t, a_t, z_{t+1}):
    1. Predict z_hat_factual = F(z_t, a_t)
    2. Sample K negatives a^- and predict z_hat_k = F(z_t, a^-_k)
    3. The factual prediction "wins" if its distance to z_{t+1} is smallest.

Reports top-1 accuracy and MRR. The target z_{t+1} is treated as a constant
(stop-gradient is implicit under no_grad).

This module exposes a **per-transition** function (``cra_per_transition``) that
both the aggregate wrapper (``compute_cra``) and the runner call, so the
synthetic-model validation exercises the exact code path used in production.
Proprioception (``proprio_t``) is threaded into the predictor when the model
was trained with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from models.adapters import WorldModelAdapter

from .distances import get_distance


@dataclass
class CRAResult:
    top1: float
    mrr: float
    n: int
    factual_distance_mean: float
    negative_distance_mean: float


def _repeat_proprio(proprio_t: Optional[torch.Tensor], K: int) -> Optional[torch.Tensor]:
    if proprio_t is None:
        return None
    B = proprio_t.shape[0]
    return proprio_t.unsqueeze(1).expand(B, K, *proprio_t.shape[1:]).reshape(B * K, *proprio_t.shape[1:])


@torch.no_grad()
def cra_per_transition(
    adapter: WorldModelAdapter,
    z_t: torch.Tensor,          # (B, *frame)
    a_t: torch.Tensor,          # (B, A)
    z_t1: torch.Tensor,         # (B, *frame)
    a_negatives: torch.Tensor,  # (B, K, A)
    distance: str = "l2",
    proprio_t: Optional[torch.Tensor] = None,  # (B, P)
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Returns (correct (B,), reciprocal_rank (B,), d_factual (B,), d_neg (B,K))."""
    dist_fn = get_distance(distance)

    z_hat_factual = adapter.predict(z_t, a_t, proprio_t=proprio_t)
    d_factual = dist_fn(z_hat_factual, z_t1)                # (B,)

    B, K, A = a_negatives.shape
    z_t_rep = z_t.unsqueeze(1).expand(B, K, *z_t.shape[1:]).reshape(B * K, *z_t.shape[1:])
    a_neg_flat = a_negatives.reshape(B * K, A)
    proprio_rep = _repeat_proprio(proprio_t, K)
    z_hat_neg = adapter.predict(z_t_rep, a_neg_flat, proprio_t=proprio_rep)
    z_hat_neg = z_hat_neg.reshape(B, K, *z_hat_neg.shape[1:])

    z_t1_rep = z_t1.unsqueeze(1).expand(B, K, *z_t1.shape[1:]).reshape(B * K, *z_t1.shape[1:])
    d_neg = dist_fn(z_hat_neg.reshape(B * K, *z_hat_neg.shape[2:]), z_t1_rep).reshape(B, K)

    # Tie-aware ranking. An action-ignoring model produces identical distances
    # for factual and all negatives (a K-way tie); fair top-1 must then be
    # chance = 1/(K+1), not 0. So we resolve ties uniformly:
    #   top-1 = 1{no negative strictly closer} / (1 + #tied negatives)
    #   rank  = 1 + #strictly-closer + (#tied)/2   (average rank among ties)
    df = d_factual.unsqueeze(-1)
    n_strictly_better = (d_neg < df).sum(dim=-1).float()
    n_ties = (d_neg == df).sum(dim=-1).float()
    correct = ((n_strictly_better == 0).float() / (1.0 + n_ties)).cpu().numpy()
    rank = 1.0 + n_strictly_better + n_ties / 2.0
    recip_rank = (1.0 / rank).cpu().numpy()
    return correct, recip_rank, d_factual, d_neg


@torch.no_grad()
def compute_cra(
    adapter: WorldModelAdapter,
    z_t: torch.Tensor,
    a_t: torch.Tensor,
    z_t1: torch.Tensor,
    a_negatives: torch.Tensor,
    distance: str = "l2",
    proprio_t: Optional[torch.Tensor] = None,
) -> CRAResult:
    correct, recip, d_factual, d_neg = cra_per_transition(
        adapter, z_t, a_t, z_t1, a_negatives, distance=distance, proprio_t=proprio_t
    )
    return CRAResult(
        top1=float(correct.mean()),
        mrr=float(recip.mean()),
        n=int(z_t.shape[0]),
        factual_distance_mean=float(d_factual.mean().item()),
        negative_distance_mean=float(d_neg.mean().item()),
    )
