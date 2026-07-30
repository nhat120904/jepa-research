"""Core POMDP abstractions: task interface, particle belief, exact Bayes update.

A "task" is a small discrete POMDP with:
  - a finite list of HIDDEN states z (the hidden discrete parameter),
  - a prior over hidden states,
  - a finite set of TERMINAL (commit) actions per goal,
  - a scalar reward(z, a, g),
  - a set of PROBE actions, each with an observation model P(o | z, probe),
  - a family of goals of increasing "richness".

A belief is a weighted particle set over the *enumerated* hidden states, so
every Bayesian update is exact.  Because the state / action / observation
spaces are tiny, planning value functions are computed by exact expectimax
and policy returns by exact enumeration (no Monte-Carlo noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Compute accounting
# --------------------------------------------------------------------------- #
@dataclass
class ComputeCounter:
    """Counts the primitive operations each planner/probe-selector performs.

    These are the honest "planning compute" numbers reported by Gate B.
      - reward_evals   : number of reward(z, a, g) evaluations
      - belief_touches : number of per-particle likelihood multiplications
      - expectimax_nodes : number of expectimax recursion nodes expanded
      - voi_inner      : number of inner "would-the-decision-change" probes
    """

    reward_evals: int = 0
    belief_touches: int = 0
    expectimax_nodes: int = 0
    voi_inner: int = 0

    def rewards(self, n: int = 1) -> None:
        self.reward_evals += n

    def touch(self, n: int = 1) -> None:
        self.belief_touches += n

    def node(self, n: int = 1) -> None:
        self.expectimax_nodes += n

    def voi(self, n: int = 1) -> None:
        self.voi_inner += n

    def total(self) -> int:
        return (
            self.reward_evals
            + self.belief_touches
            + self.expectimax_nodes
            + self.voi_inner
        )

    def as_dict(self) -> dict:
        return {
            "reward_evals": self.reward_evals,
            "belief_touches": self.belief_touches,
            "expectimax_nodes": self.expectimax_nodes,
            "voi_inner": self.voi_inner,
            "total": self.total(),
        }


class Task:
    """Abstract tiny-POMDP interface.  Subclasses fill in the pieces."""

    name: str = "task"
    hidden_states: List[Hashable]  # enumerated hidden parameter values
    prior: np.ndarray  # aligned with hidden_states, sums to 1
    probes: List[Hashable]  # probe action ids
    probe_cost: float = 0.0
    max_budget: int = 1  # max number of probes per episode

    # --- terminal (commit) decision -------------------------------------- #
    def terminal_actions(self, goal) -> List[Hashable]:
        raise NotImplementedError

    def reward(self, z, a, goal, counter: ComputeCounter | None = None) -> float:
        raise NotImplementedError

    def preferred_action(self, z, goal, counter: ComputeCounter | None = None):
        """argmax_a reward(z, a, goal) with a deterministic tie-break."""
        acts = self.terminal_actions(goal)
        best_a = None
        best_r = -np.inf
        for a in acts:
            r = self.reward(z, a, goal, counter)
            if r > best_r + 1e-12:
                best_r = r
                best_a = a
        return best_a

    # --- probes / observations ------------------------------------------- #
    def obs_space(self, probe) -> List[Hashable]:
        raise NotImplementedError

    def obs_prob(self, z, probe, o) -> float:
        """P(o | z, probe)."""
        raise NotImplementedError

    # --- goals ------------------------------------------------------------ #
    def goal_family(self, richness: int) -> List[Hashable]:
        """A family of goals whose union of decision-relevant structure grows
        with `richness`."""
        raise NotImplementedError

    # --- geometry of the hidden space ------------------------------------- #
    def param_embedding(self):
        """Optional 1-D coordinate for each hidden state, or None.

        Decision-mode compression collapses each mode onto ONE representative
        particle.  Which particle you pick does not change the mode partition
        (and therefore does not change M or the compute), but it does change
        the VALUE the collapsed belief reports, and probe decisions are made on
        values.  Tasks whose hidden parameter is an ordered physical quantity
        (mass, friction, position) can expose it here so a planner can collapse
        onto the mode-conditional MEAN instead of an arbitrary member.  Tasks
        with no natural metric return None and fall back to the max-weight
        member.
        """
        return None

    # --- convenience ------------------------------------------------------ #
    def n_states(self) -> int:
        return len(self.hidden_states)


# --------------------------------------------------------------------------- #
# Belief
# --------------------------------------------------------------------------- #
@dataclass
class Belief:
    """A weighted particle set over task.hidden_states (weights sum to 1)."""

    task: Task
    weights: np.ndarray

    @classmethod
    def prior(cls, task: Task) -> "Belief":
        return cls(task, task.prior.copy())

    def copy(self) -> "Belief":
        return Belief(self.task, self.weights.copy())

    def map_index(self) -> int:
        return int(np.argmax(self.weights))

    def map_state(self):
        return self.task.hidden_states[self.map_index()]

    def entropy(self) -> float:
        w = self.weights
        nz = w > 0
        return float(-(w[nz] * np.log(w[nz])).sum())

    def expected_reward(self, a, goal, counter: ComputeCounter | None = None) -> float:
        r = 0.0
        for k, z in enumerate(self.task.hidden_states):
            wk = self.weights[k]
            if wk == 0.0:
                continue
            r += wk * self.task.reward(z, a, goal, counter)
        return r

    def best_commit(self, goal, counter: ComputeCounter | None = None):
        """Return (best_action, value) maximising expected reward under belief."""
        best_a, best_v = None, -np.inf
        for a in self.task.terminal_actions(goal):
            v = self.expected_reward(a, goal, counter)
            if v > best_v + 1e-12:
                best_v, best_a = v, a
        return best_a, best_v

    # --- exact Bayesian update ------------------------------------------- #
    def obs_marginal(self, probe, o, counter: ComputeCounter | None = None) -> float:
        """P(o | belief, probe) = sum_k w_k P(o | z_k, probe)."""
        p = 0.0
        for k, z in enumerate(self.task.hidden_states):
            wk = self.weights[k]
            if wk == 0.0:
                continue
            p += wk * self.task.obs_prob(z, probe, o)
        if counter is not None:
            counter.touch(int((self.weights > 0).sum()))
        return p

    def updated(self, probe, o, counter: ComputeCounter | None = None) -> "Belief":
        """Exact posterior after observing o from probe.

        Only the SUPPORT (particles with non-zero weight) is propagated, and
        only the support is charged to the compute counter: a particle filter
        carrying M particles does not evaluate the likelihood of the K-M
        particles it does not carry.  Zero-weight particles stay zero under
        Bayes anyway, so this is exact.  (For a full-support belief, support
        size == K and the charge is unchanged; the accounting only differs for
        a compressed / collapsed belief, which is precisely the quantity the
        scaling study measures.)
        """
        support = np.nonzero(self.weights)[0]
        if counter is not None:
            counter.touch(len(support))
        new = np.zeros_like(self.weights)
        for k in support:
            new[k] = self.weights[k] * self.task.obs_prob(
                self.task.hidden_states[k], probe, o
            )
        s = new.sum()
        if s <= 0:
            # Impossible observation under this belief: keep prior shape.
            return self.copy()
        return Belief(self.task, new / s)
