"""Adapt a set of learned hypotheses to the EXISTING compression machinery.

The whole point of this module is that P0 does **not** reimplement decision
compression.  `HypothesisTask` presents K decoded hypotheses as a
`belief_compression.core.Task`, so

    compress(Belief(task, task.prior), goal_family, tol=...)

is literally the Gate B / Gate C0 `compress()` -- same signature extraction,
same exact/tolerance merge, same `pick_rep` / `rep_rule`s, same
`ComputeCounter` accounting.  M therefore means exactly what it means in
`gateC0_scaling_results.md`, and any P0 number is directly comparable to the
oracle-particle-filter numbers in that document.

What a learned belief model supplies is only:
  * `params`  -- K decoded hidden-parameter values (here, binary cell vectors)
  * `weights` -- the K hypothesis weights (sum to 1)

Duplicate hypotheses are kept, not deduplicated.  A filter that samples the
same hypothesis 100 times has genuinely low diversity, and collapsing the
duplicates away before measuring would hide exactly the failure mode (VACUOUS)
that P0 exists to detect.  `HypothesisSet.n_distinct` and `ess` report the
duplication separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..core import Belief, ComputeCounter, Task
from .decision import RegionCommit


# --------------------------------------------------------------------------- #
# The K hypotheses a belief model emits
# --------------------------------------------------------------------------- #
@dataclass
class HypothesisSet:
    """K decoded hypotheses with weights.

    `params` is a (K, n_cells) integer array; row k is hypothesis z_k.
    `weights` is (K,), non-negative, normalised on construction.
    `source` records which belief model produced it (for the results table).
    """

    params: np.ndarray
    weights: np.ndarray
    source: str = "unknown"

    def __post_init__(self):
        self.params = np.asarray(self.params, dtype=np.int8)
        if self.params.ndim != 2:
            raise ValueError(f"params must be (K, n_cells), got {self.params.shape}")
        w = np.asarray(self.weights, dtype=float).reshape(-1)
        if w.shape[0] != self.params.shape[0]:
            raise ValueError("weights length must equal K")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        s = w.sum()
        self.weights = w / s if s > 0 else np.full_like(w, 1.0 / len(w))

    @property
    def K(self) -> int:
        return int(self.params.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.params.shape[1])

    @property
    def n_distinct(self) -> int:
        """How many of the K hypotheses are actually different parameter values."""
        return int(np.unique(self.params, axis=0).shape[0])

    @property
    def ess_weights(self) -> float:
        """Raw weight ESS, 1 / sum_k w_k^2, over the K draws as drawn.

        Reported for completeness only.  Do NOT gate on it -- see `ess`.
        """
        return float(1.0 / np.square(self.weights).sum())

    @property
    def ess(self) -> float:
        """Effective number of DISTINCT hypotheses: duplicates merged first.

        DISCLOSED CORRECTION to Gate C1 STOP S5, which states the diversity
        stop on "ESS of the particle weights".  For an ancestral-sampled
        hypothesis set that statistic is backwards: a filter whose posterior
        has collapsed onto one configuration draws the SAME hypothesis K times,
        every draw has the same model probability, the weights come out uniform
        and the raw weight ESS reads its MAXIMUM (K) exactly when diversity is
        at its minimum (1).  The unit test `test_ess_detects_collapse` pins
        this down.

        The statistic that means what S5 intends groups the duplicates first:

            ESS = 1 / sum_u (sum_{k: z_k = u} w_k)^2

        which is K for K distinct equally-weighted hypotheses and 1 for one
        hypothesis repeated K times.  It is the quantity every P0 threshold is
        stated on.  (For an exhaustive, deduplicated support -- e.g.
        `ExactEnumerationBelief` -- the two agree, so the Gate B / C0 oracle
        numbers are unaffected.)
        """
        _, inv = np.unique(self.params, axis=0, return_inverse=True)
        merged = np.bincount(inv.reshape(-1), weights=self.weights)
        return float(1.0 / np.square(merged).sum())

    def keys(self) -> List[Tuple[int, ...]]:
        return [tuple(int(v) for v in row) for row in self.params]


# --------------------------------------------------------------------------- #
# The Task adapter -- this is what lets compress() run unmodified
# --------------------------------------------------------------------------- #
class HypothesisTask(Task):
    """A `core.Task` whose hidden states ARE the belief model's K hypotheses.

    Only the parts of the Task contract that `compress()` touches are
    meaningful here: `hidden_states`, `prior`, `terminal_actions`, `reward`,
    `preferred_action` (inherited) and `param_embedding`.  The probe /
    observation half of the interface raises, because P0 measures the mode
    structure of a belief, not closed-loop planning -- probing is Gate C1 P4.
    Raising rather than silently returning something plausible keeps an
    accidental planner run from producing a meaningless number.
    """

    name = "p0_hypotheses"

    def __init__(self, hyps: HypothesisSet, decision: RegionCommit):
        if hyps.n_cells != decision.n_cells:
            raise ValueError(
                f"hypothesis width {hyps.n_cells} != decision n_cells {decision.n_cells}"
            )
        self.hyps = hyps
        self.decision = decision
        self.hidden_states = list(hyps.params)          # K rows, each (n_cells,)
        self.prior = hyps.weights.copy()
        self.probes: List = []
        self.probe_cost = 0.0
        self.max_budget = 0

    # --- terminal decision ------------------------------------------------- #
    def terminal_actions(self, goal):
        return self.decision.terminal_actions(goal)

    def reward(self, z, a, goal, counter: ComputeCounter | None = None) -> float:
        if counter is not None:
            counter.rewards()
        return self.decision.reward(z, a, goal)

    # --- goals -------------------------------------------------------------- #
    def goal_family(self, richness: int):
        raise NotImplementedError(
            "P0 goal families are constructed by RegionCommit.goal_family and "
            "passed to compress() explicitly, so the SAME regions are reused "
            "across every (K, model, seed) cell."
        )

    # --- geometry ----------------------------------------------------------- #
    def param_embedding(self):
        """None: a bit vector has no 1-D ordering, so `centroid` falls back to
        `maxweight`.  This costs P0 nothing -- P0 measures the mode PARTITION
        (M), which Gate C0 §S7 proved is identical under every `rep_rule`.  The
        representative choice only affects VALUE fidelity, which is a Gate C1
        P3/P4 question."""
        return None

    # --- deliberately unimplemented ----------------------------------------- #
    def obs_space(self, probe):
        raise NotImplementedError("P0 does not plan; probing is Gate C1 P4.")

    def obs_prob(self, z, probe, o):
        raise NotImplementedError("P0 does not plan; probing is Gate C1 P4.")


def build_task(hyps: HypothesisSet, decision: RegionCommit) -> Tuple[HypothesisTask, Belief]:
    """Convenience: the Task and its prior Belief, ready for `compress()`."""
    task = HypothesisTask(hyps, decision)
    return task, Belief.prior(task)
