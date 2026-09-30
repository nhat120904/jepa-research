"""Causal CEM stopping rules evaluated on per-iteration candidate costs.

A rule sees the costs of iterations 1..k of the current plan (``costs[:k]``,
shape ``(k, num_samples)``) and decides whether to return the elite mean
after iteration k. Checkpoints are visited in increasing order; the last one
always stops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHECKPOINTS: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 15, 20, 30)
TOPK = 30
_DEGENERATE = 1e-12


def elite_mean_cost(costs_row: np.ndarray, topk: int = TOPK) -> float:
    return float(np.mean(np.partition(np.asarray(costs_row, np.float64), topk - 1)[:topk]))


def first_iqr(costs: np.ndarray) -> tuple[float, bool]:
    """IQR of the first population and whether it is degenerate."""
    first = np.asarray(costs[0], np.float64)
    q25, q50, q75 = np.percentile(first, [25, 50, 75])
    iqr = float(q75 - q25)
    return iqr, iqr <= _DEGENERATE * max(1.0, abs(float(q50)))


@dataclass(frozen=True)
class Fixed:
    k: int
    name: str = "fixed"

    def param(self) -> float:
        return float(self.k)

    def stop(self, costs: np.ndarray, k: int, prev: int | None) -> bool:
        return k >= self.k


@dataclass(frozen=True)
class Converge:
    eps: float
    name: str = "converge"

    def param(self) -> float:
        return self.eps

    def stop(self, costs: np.ndarray, k: int, prev: int | None) -> bool:
        if prev is None:
            return False
        before = elite_mean_cost(costs[prev - 1])
        now = elite_mean_cost(costs[k - 1])
        return (before - now) / max(abs(before), _DEGENERATE) < self.eps


@dataclass(frozen=True)
class Gap:
    tau: float
    name: str = "gap"

    def param(self) -> float:
        return self.tau

    def stop(self, costs: np.ndarray, k: int, prev: int | None) -> bool:
        scale, degenerate = first_iqr(costs)
        if degenerate:
            return True
        s = np.sort(np.asarray(costs[k - 1], np.float64))
        return (s[TOPK] - s[TOPK - 1]) / scale < self.tau


@dataclass(frozen=True)
class Band:
    tau: float
    name: str = "band"

    def param(self) -> float:
        return self.tau

    def stop(self, costs: np.ndarray, k: int, prev: int | None) -> bool:
        scale, degenerate = first_iqr(costs)
        if degenerate:
            return True
        s = np.sort(np.asarray(costs[k - 1], np.float64))
        boundary = s[TOPK - 1]
        return int(np.sum(np.abs(s - boundary) <= self.tau * scale)) >= TOPK


def choose(rule, costs: np.ndarray, checkpoints: tuple[int, ...] = CHECKPOINTS) -> int:
    """Checkpoint at which ``rule`` stops, reading only costs up to it."""
    costs = np.asarray(costs)
    if costs.shape[0] < checkpoints[-1]:
        raise ValueError(f"need {checkpoints[-1]} iterations, got {costs.shape[0]}")
    prev = None
    for k in checkpoints:
        if k == checkpoints[-1]:
            return k
        # Slice so a rule cannot read iterations after k even by mistake.
        if rule.stop(costs[:k], k, prev):
            return k
        prev = k
    raise AssertionError("unreachable")


def grid() -> dict[str, list]:
    """Parameter grids searched on development roots."""
    return {
        "fixed": [Fixed(k) for k in CHECKPOINTS],
        "converge": [Converge(e) for e in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)],
        "gap": [Gap(float(t)) for t in np.logspace(-4, -0.5, 15)],
        "band": [Band(float(t)) for t in np.logspace(-3, 0, 13)],
    }
