"""Leakage-safe sampling helpers for the TRM-style baseline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class TemporalPair:
    trajectory_id: str
    i: int
    j: int
    gap: int


def sample_balanced_temporal_pairs(
    lengths: Mapping[str, int],
    n_pairs: int,
    max_gap: int,
    rng: np.random.Generator,
    *,
    random_order: bool = True,
) -> list[TemporalPair]:
    """Sample gaps uniformly, then a compatible trajectory and start index.

    Drawing the gap first prevents the quadratic number of local pairs from
    dominating supervision.  A gap is eligible only when at least one supplied
    trajectory contains it.  Callers pass trajectory IDs from one manifest
    split, so this helper cannot cross the train/validation/test boundary.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    if max_gap <= 0:
        raise ValueError("max_gap must be positive")
    clean = {str(tid): int(length) for tid, length in lengths.items() if int(length) >= 2}
    if not clean:
        raise ValueError("no trajectories contain a temporal pair")
    largest = min(max_gap, max(length - 1 for length in clean.values()))
    eligible = {
        gap: tuple(tid for tid, length in clean.items() if length > gap)
        for gap in range(1, largest + 1)
    }
    out: list[TemporalPair] = []
    for _ in range(int(n_pairs)):
        gap = int(rng.integers(1, largest + 1))
        tids = eligible[gap]
        tid = tids[int(rng.integers(len(tids)))]
        i = int(rng.integers(clean[tid] - gap))
        j = i + gap
        if random_order and bool(rng.integers(2)):
            i, j = j, i
        out.append(TemporalPair(tid, i, j, gap))
    return out


def materialize_temporal_pairs(cache, pairs: Sequence[TemporalPair]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read each requested HDF5 trajectory once and materialize its pair rows."""
    by_tid: dict[str, list[tuple[int, TemporalPair]]] = defaultdict(list)
    for k, pair in enumerate(pairs):
        by_tid[pair.trajectory_id].append((k, pair))
    z_i: list[torch.Tensor | None] = [None] * len(pairs)
    z_j: list[torch.Tensor | None] = [None] * len(pairs)
    gaps = torch.empty(len(pairs), dtype=torch.float32)
    for tid, indexed in by_tid.items():
        z = np.asarray(cache.read_trajectory(tid)["z"])
        for k, pair in indexed:
            z_i[k] = torch.from_numpy(z[pair.i].copy()).float()
            z_j[k] = torch.from_numpy(z[pair.j].copy()).float()
            gaps[k] = float(pair.gap)
    return torch.stack(z_i), torch.stack(z_j), gaps


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware Spearman correlation without a SciPy runtime dependency."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3:
        raise ValueError("Spearman inputs must be same-length 1-D arrays with n>=3")

    def rank_average(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=np.float64)
        start = 0
        while start < len(a):
            end = start + 1
            while end < len(a) and a[order[end]] == a[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranks

    rx, ry = rank_average(x), rank_average(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt(np.square(rx).sum() * np.square(ry).sum())
    return float(np.dot(rx, ry) / denom) if denom > 0 else float("nan")
