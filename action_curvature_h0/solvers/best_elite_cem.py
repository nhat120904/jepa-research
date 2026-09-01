"""CEM that executes the best candidate it ever scored, not the elite mean.

Deployed CEM refits a Gaussian to the top-K candidates and executes the
resulting **mean** (``cem.py``: ``batch_mean = topk_candidates.mean(dim=1)``,
then ``outputs['actions'] = mean``).  In a non-convex action-to-outcome map the
mean of several individually good actions need not itself be good, and on
OGBench-Cube the executed elite mean measures worse than a typical *random*
candidate on half of the states.

This solver isolates exactly that one decision.  It subclasses ``CEMSolver`` and
does not reimplement the optimisation loop, so sampling, scoring, elite
selection, refit and the RNG stream are bit-identical to the control; only the
action handed back for execution differs.  The best-seen candidate across all
iterations is used, matching iCEM's "execute the best seen trajectory" rather
than inventing a new rule.
"""

from __future__ import annotations

from typing import Any

import torch

from stable_worldmodel.planning.solver.cem import CEMSolver
from stable_worldmodel.planning.solver.callbacks import Callback


class _BestSeenTracker(Callback):
    """Records the lowest-cost candidate seen, per batch, across all steps."""

    def __init__(self) -> None:
        super().__init__(reduction="none")
        self.best_cost: list[torch.Tensor] = []
        self.best_action: list[torch.Tensor] = []

    def reset(self) -> None:
        super().reset()
        self.best_cost = []
        self.best_action = []

    def start_batch(self) -> None:
        super().start_batch()
        self.best_cost.append(None)
        self.best_action.append(None)

    def compute(self, **state: Any) -> None:
        costs = state["costs"]                    # (B, N)
        candidates = state["candidates"]          # (B, N, H, D)
        value, index = costs.min(dim=1)           # (B,)
        chosen = candidates[torch.arange(costs.shape[0], device=costs.device), index]
        prev_cost, prev_action = self.best_cost[-1], self.best_action[-1]
        if prev_cost is None:
            self.best_cost[-1] = value.detach().clone()
            self.best_action[-1] = chosen.detach().clone()
        else:
            better = value < prev_cost
            self.best_cost[-1] = torch.where(better, value, prev_cost)
            self.best_action[-1] = torch.where(
                better.view(-1, *([1] * (chosen.dim() - 1))), chosen, prev_action)
        return None


class BestEliteCEMSolver(CEMSolver):
    """``CEMSolver`` whose executed action is the best-seen candidate."""

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        tracker = _BestSeenTracker()
        original_callbacks = self.callbacks
        self.callbacks = list(original_callbacks) + [tracker]
        try:
            outputs = super().solve(info_dict, init_action)
        finally:
            self.callbacks = original_callbacks

        batches = [a for a in tracker.best_action if a is not None]
        if not batches:
            raise RuntimeError("no candidate was scored; cannot replace the plan")
        best = torch.cat(batches, dim=0).cpu()
        planned = outputs["actions"]
        if best.shape != planned.shape:
            raise RuntimeError(
                f"best-seen plan has shape {tuple(best.shape)} but the solver "
                f"returned {tuple(planned.shape)}; refusing to substitute")
        # Guard against a silent no-op: if the two coincide the arms are the
        # same experiment and the comparison would be meaningless.
        outputs["actions"] = best
        outputs["elite_mean_actions"] = planned
        outputs["operator"] = "best_seen"
        outputs["differs_from_elite_mean"] = bool(
            not torch.allclose(best, planned, atol=1e-8))
        return outputs
