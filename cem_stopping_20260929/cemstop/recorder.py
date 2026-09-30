"""Solver callback that logs one CEM plan without changing it."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class PlanRecorder:
    """Record costs of every iteration and the elite mean at checkpoints.

    Used with the upstream ``CEMSolver`` (``batch_size=1``, one env per call).
    The solver calls the callback after the elite update of each iteration,
    so ``mean`` is the post-update mean returned if CEM stopped there.
    """

    name = "cemstop_plan"

    def __init__(self, checkpoints, label_iters=(), label_index=None) -> None:
        self.checkpoints = tuple(int(k) for k in checkpoints)
        self.label_iters = tuple(int(k) for k in label_iters)
        self.label_index = None if label_index is None else np.asarray(label_index)
        self.history: list[Any] = []
        self.reset()

    @property
    def output_key(self) -> str:
        return self.name

    def reset(self) -> None:
        self.costs: list[np.ndarray] = []
        self.elite_std: list[float] = []
        self.iter_seconds: list[float] = []
        self.means: dict[int, np.ndarray] = {}
        self.label_actions: dict[int, np.ndarray] = {}
        self.label_costs: dict[int, np.ndarray] = {}
        self._last = time.perf_counter()

    def start_batch(self) -> None:
        self._last = time.perf_counter()

    def end_solve(self) -> None:
        pass

    def __call__(self, **state: Any) -> None:
        iteration = int(state["step"]) + 1
        costs = state["costs"]
        if costs.shape[0] != 1:
            raise RuntimeError("PlanRecorder expects batch_size=1")
        row = costs[0].detach().float().cpu().numpy().copy()  # synchronises
        now = time.perf_counter()
        self.iter_seconds.append(now - self._last)
        self._last = now
        self.costs.append(row)
        self.elite_std.append(float(state["var"][0].detach().float().mean().cpu()))
        if iteration in self.checkpoints:
            self.means[iteration] = state["mean"][0].detach().float().cpu().numpy().copy()
        if iteration in self.label_iters and self.label_index is not None:
            cand = state["candidates"][0, self.label_index]
            self.label_actions[iteration] = cand.detach().float().cpu().numpy().copy()
            self.label_costs[iteration] = row[self.label_index].copy()

    def cost_matrix(self) -> np.ndarray:
        return np.stack(self.costs).astype(np.float32)

    def checkpoint_means(self) -> np.ndarray:
        return np.stack([self.means[k] for k in self.checkpoints]).astype(np.float32)
