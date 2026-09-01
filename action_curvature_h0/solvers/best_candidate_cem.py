"""CEM that executes the best candidate it saw instead of the elite mean."""

from __future__ import annotations

from typing import Any

import torch

from stable_worldmodel.planning.solver.callbacks import Callback
from stable_worldmodel.planning.solver.cem import CEMSolver


class _BestCandidateTracker(Callback):
    """Record the lowest-cost candidate seen, per env, across all iterations.

    Read-only: it consumes no randomness and mutates no solver state, so the
    optimisation trajectory is bit-identical to the unmodified solver.
    """

    def __init__(self) -> None:
        super().__init__(reduction="none")
        self.best_actions: list[torch.Tensor] = []
        self.best_costs: list[torch.Tensor] = []
        self._actions: torch.Tensor | None = None
        self._costs: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self.best_actions, self.best_costs = [], []
        self._actions, self._costs = None, None

    def _flush(self) -> None:
        if self._actions is not None:
            self.best_actions.append(self._actions)
            self.best_costs.append(self._costs)
            self._actions, self._costs = None, None

    def start_batch(self) -> None:
        self._flush()

    def end_solve(self) -> None:
        self._flush()

    def __call__(self, **state: Any) -> None:
        costs = state["costs"].detach()            # (B, N)
        candidates = state["candidates"].detach()  # (B, N, H, D)
        best_cost, best_idx = costs.min(dim=1)     # (B,)
        rows = torch.arange(costs.shape[0], device=costs.device)
        best_action = candidates[rows, best_idx]   # (B, H, D)
        if self._costs is None:
            self._costs, self._actions = best_cost.clone(), best_action.clone()
            return
        improved = best_cost < self._costs
        self._costs = torch.where(improved, best_cost, self._costs)
        self._actions = torch.where(
            improved.view(-1, *([1] * (best_action.dim() - 1))),
            best_action, self._actions)


class BestCandidateCEMSolver(CEMSolver):
    """``CEMSolver`` with one change: what gets executed.

    The parent's ``solve`` is reused verbatim, so sampling, scoring, elite
    selection and the mean/variance refit are untouched.  A read-only callback
    records the best candidate seen; afterwards the returned ``actions`` are
    swapped from the refit mean to that candidate.  ``mean``/``var`` are left as
    the parent produced them so warm-starting is unaffected.
    """

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        tracker = _BestCandidateTracker()
        saved = self.callbacks
        self.callbacks = list(saved) + [tracker]
        try:
            outputs = super().solve(info_dict, init_action)
        finally:
            self.callbacks = saved

        if not tracker.best_actions:
            raise RuntimeError("tracker saw no candidates; callback never fired")
        best = torch.cat(tracker.best_actions, dim=0).detach().cpu()
        mean = outputs["actions"]
        if best.shape != mean.shape:
            raise RuntimeError(
                f"best-candidate shape {tuple(best.shape)} != executed-mean shape "
                f"{tuple(mean.shape)}; the operator swap would be misaligned")
        # A swap that changed nothing would silently evaluate as the control.
        outputs["operator_swap_max_abs_delta"] = float((best - mean).abs().max())
        outputs["actions"] = best
        return outputs
