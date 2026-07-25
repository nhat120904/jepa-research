"""Decision-Equivalent Belief Compression.

Given a belief (K weighted hypotheses over the hidden parameter) and a goal
family G, each hypothesis z_k gets a *decision signature*:

    exact mode   : sig(z_k) = ( preferred_action(z_k, g) for g in G )
    tolerance ε  : sig(z_k) = concatenated Q-vectors [reward(z_k, a, g)]_{a,g}

Hypotheses whose signatures agree (exactly, or within L-inf tolerance ε) are
merged into one DECISION MODE.  Planning then runs over M <= K modes instead of
K hypotheses.

Exact (ε = 0, argmax) compression is *lossless for the terminal decision*: every
member of a mode prefers the same action for every goal in the family, so the
committed action is unchanged.  Tolerance compression trades a bounded amount of
regret for more merging (smaller M).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .core import Belief, ComputeCounter, Task


@dataclass
class DecisionMode:
    members: List[int]          # indices into task.hidden_states
    weight: float               # total belief weight of the mode
    rep: int                    # representative particle index (max-weight member)


@dataclass
class Compression:
    task: Task
    goal_family: list
    modes: List[DecisionMode]
    K: int                      # number of hypotheses with non-zero weight

    @property
    def M(self) -> int:
        return len(self.modes)

    @property
    def ratio(self) -> float:
        return self.M / self.K if self.K else 1.0

    def as_belief_over_reps(self) -> Belief:
        """Collapse to a belief supported only on the mode representatives."""
        w = np.zeros(self.task.n_states())
        for m in self.modes:
            w[m.rep] += m.weight
        s = w.sum()
        if s > 0:
            w = w / s
        return Belief(self.task, w)


def _preferred_signature(task: Task, k: int, goal_family, counter):
    z = task.hidden_states[k]
    return tuple(task.preferred_action(z, g, counter) for g in goal_family)


def _q_signature(task: Task, k: int, goal_family, counter):
    z = task.hidden_states[k]
    sig = []
    for g in goal_family:
        for a in task.terminal_actions(g):
            sig.append(task.reward(z, a, g, counter))
    return np.asarray(sig, dtype=float)


def compress(
    belief: Belief,
    goal_family: list,
    tol: float = 0.0,
    counter: ComputeCounter | None = None,
) -> Compression:
    """Cluster the belief's hypotheses into decision modes.

    tol == 0  -> exact decision-equivalence (merge iff identical preferred
                 action across the whole goal family). Order-independent.
    tol  > 0  -> greedy L2 merge on Q-signatures (bounded-regret merging): a
                 hypothesis joins a cluster if its Q-signature is within `tol`
                 (Euclidean) of the cluster's seed signature.
    """
    task = belief.task
    active = [k for k in range(task.n_states()) if belief.weights[k] > 0]
    K = len(active)

    if tol <= 0.0:
        buckets: dict = {}
        for k in active:
            sig = _preferred_signature(task, k, goal_family, counter)
            buckets.setdefault(sig, []).append(k)
        groups = list(buckets.values())
    else:
        sigs = {k: _q_signature(task, k, goal_family, counter) for k in active}
        groups = []
        centroids = []
        for k in active:
            placed = False
            for gi, c in enumerate(centroids):
                if np.linalg.norm(sigs[k] - c) <= tol:
                    groups[gi].append(k)
                    # keep centroid as first member's signature (stable)
                    placed = True
                    break
            if not placed:
                groups.append([k])
                centroids.append(sigs[k])

    modes = []
    for members in groups:
        wsum = float(sum(belief.weights[k] for k in members))
        rep = max(members, key=lambda k: belief.weights[k])
        modes.append(DecisionMode(members=members, weight=wsum, rep=rep))
    return Compression(task=task, goal_family=goal_family, modes=modes, K=K)
