"""Counterfactual Trajectory Divergence (CTD).

CTD_H = E[ d( F^H(z_t, a_{1:H}), F^H(z_t, a^-_{1:H}) ) ]

Reports CTD at horizons H ∈ {1, 3, 5, 10}. Captures whether action-conditional
differences accumulate (good) or collapse (pathology) over autoregressive
rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import torch

from models.adapters import WorldModelAdapter

from .distances import get_distance


@dataclass
class CTDResult:
    horizons: Dict[int, float] = field(default_factory=dict)
    n: int = 0


@torch.no_grad()
def compute_ctd(
    adapter: WorldModelAdapter,
    z_t: torch.Tensor,                # (B, ...)
    actions_factual: torch.Tensor,    # (B, H_max, A)
    actions_counter: torch.Tensor,    # (B, H_max, A)
    horizons: Sequence[int] = (1, 3, 5, 10),
    distance: str = "l2",
    proprio_t=None,                   # (B, P) optional
) -> CTDResult:
    dist_fn = get_distance(distance)
    H_max = max(horizons)
    assert actions_factual.shape[1] >= H_max
    assert actions_counter.shape[1] >= H_max

    fact_rollout = adapter.predict_rollout(z_t, actions_factual[:, :H_max], proprio_t=proprio_t)
    counter_rollout = adapter.predict_rollout(z_t, actions_counter[:, :H_max], proprio_t=proprio_t)

    out = CTDResult(n=z_t.shape[0])
    for H in horizons:
        fz = fact_rollout[:, H]
        cz = counter_rollout[:, H]
        out.horizons[H] = dist_fn(fz, cz).mean().item()
    return out
