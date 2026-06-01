"""Nonparametric bootstrap CIs for the diagnostic metrics.

Two modes:

* **iid** (``groups=None``): resample transitions independently.
* **cluster** (``groups`` given): resample whole trajectories with replacement.
  Transitions within a trajectory are highly correlated, so the iid bootstrap
  underestimates CI width (over-confident). The cluster bootstrap is the
  correct one for the decision logic, which compares CIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class BootstrapCI:
    point: float
    low: float
    high: float


def _cluster_index(groups: np.ndarray):
    """Map group labels -> list of sample-index arrays."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    return members


def bootstrap_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    groups: Optional[np.ndarray] = None,
) -> BootstrapCI:
    """Bootstrap CI. If ``groups`` is given (same length as ``samples``), resample
    at the trajectory/cluster level instead of per-sample."""
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    point = float(statistic(samples))

    resampled = np.empty(n_resamples)
    if groups is None:
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            resampled[i] = statistic(samples[idx])
    else:
        groups = np.asarray(groups)
        members = _cluster_index(groups)
        n_clusters = len(members)
        for i in range(n_resamples):
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([members[c] for c in chosen])
            resampled[i] = statistic(samples[idx])

    alpha = (1.0 - ci) / 2.0
    return BootstrapCI(
        point=point,
        low=float(np.quantile(resampled, alpha)),
        high=float(np.quantile(resampled, 1.0 - alpha)),
    )
