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

Two ways to *carry* a mode are provided, and they are not equivalent:

  representative particle  (`rep_rule` in `REP_RULES`)
      Put all of the mode's weight on ONE member.  Preserves the commit argmax,
      biases the commit VALUE (Gate C0 §S6).  `centroid` (nearest the mode's
      belief-weighted mean parameter) is the default; `maxweight` is kept
      selectable because it is the rule whose failure S6/S7 document.

  value-consistent summary  (`ModeSummary`, `rep_rule` in `SUMMARY_RULES`)
      Carry the mode's belief-weighted Q-vector and within-mode observation
      likelihood instead of any single particle.  Reproduces the full-belief
      commit value and observation marginal EXACTLY (see the class docstring for
      what survives an observation and what does not).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .core import Belief, ComputeCounter, Task

# How a decision mode is carried.  `centroid` is the default representative rule
# (Gate C0 §S7: same partition, same M, byte-identical compute, far less regret
# than `maxweight`); the summary rules carry mode statistics instead of a
# particle and are value-consistent (§S6/§S8).
DEFAULT_REP_RULE = "centroid"
REP_RULES = ("centroid", "maxweight")
SUMMARY_RULES = ("summary", "summary_exact")
ALL_REP_RULES = REP_RULES + SUMMARY_RULES


def pick_rep(task: Task, members, weights, rep_rule: str = DEFAULT_REP_RULE) -> int:
    """Which particle represents a mode.

      "centroid"  : the member nearest the mode's belief-weighted mean parameter
                    (requires `task.param_embedding()`; falls back to
                    "maxweight" when the hidden space has no metric).
      "maxweight" : the mode's highest-weight member.  Under a flat prior every
                    member ties and the tie-break returns the FIRST index, i.e.
                    an arbitrary particle at the EDGE of the mode -- which
                    systematically under-values the mode.

    The choice does NOT change the mode partition, so M -- and therefore all the
    planning compute -- is identical either way.
    """
    emb = task.param_embedding() if rep_rule == "centroid" else None
    if emb is not None:
        wt = np.array([weights[k] for k in members], dtype=float)
        tot = wt.sum()
        if tot > 0:
            mean = float((wt / tot * emb[list(members)]).sum())
            return min(members, key=lambda k: abs(emb[k] - mean))
    return max(members, key=lambda k: weights[k])


@dataclass
class DecisionMode:
    members: List[int]          # indices into task.hidden_states
    weight: float               # total belief weight of the mode
    rep: int                    # representative particle index (per `rep_rule`)


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
    rep_rule: str = DEFAULT_REP_RULE,
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
        rep = pick_rep(task, members, belief.weights, rep_rule)
        modes.append(DecisionMode(members=members, weight=wsum, rep=rep))
    return Compression(task=task, goal_family=goal_family, modes=modes, K=K)


# --------------------------------------------------------------------------- #
# Value-consistent mode summary
# --------------------------------------------------------------------------- #
class ParticleTables:
    """Per-particle reward / observation tables, precomputed once per task.

    `reward(z_k, a, g)` and `P(o | z_k, probe)` depend only on the task, never on
    the belief weights, so -- exactly like the decision signatures cached by
    `PrecomputedCompressionPlanner` -- they are a one-off offline cost.  A
    planner holding these tables pays only the O(K) weighted sums at decision
    time, which is what gets charged to the ComputeCounter.
    """

    def __init__(self, task: Task):
        self.task = task
        self._R: dict = {}
        self._P: dict = {}

    def rewards(self, goal) -> np.ndarray:
        """(K, |A|) table of reward(z_k, a, goal)."""
        if goal not in self._R:
            acts = self.task.terminal_actions(goal)
            self._R[goal] = np.array(
                [[self.task.reward(z, a, goal) for a in acts]
                 for z in self.task.hidden_states],
                dtype=float,
            )
        return self._R[goal]

    def obs(self, probe) -> np.ndarray:
        """(K, |O|) table of P(o | z_k, probe)."""
        if probe not in self._P:
            space = self.task.obs_space(probe)
            self._P[probe] = np.array(
                [[self.task.obs_prob(z, probe, o) for o in space]
                 for z in self.task.hidden_states],
                dtype=float,
            )
        return self._P[probe]


@dataclass
class ModeSummary:
    """A VALUE-CONSISTENT summary of a belief over M decision modes.

    Collapsing a mode onto one representative particle preserves the commit
    ARGMAX but not its VALUE, and probe/VOI decisions are made on values (Gate C0
    §S6).  This class carries, per mode m, the statistics that make the mode-level
    computation reproduce the particle-level one:

        w_m         = sum_{k in m} w_k                            (mode weight)
        Q_m(a, g)   = sum_{k in m} p(k|m) reward(z_k, a, g)        (mode Q-vector)
        L_m(o | u)  = sum_{k in m} p(k|m) P(o | z_k, u)            (mode likelihood)

    with p(k|m) = w_k / w_m the within-mode conditional.  Then, EXACTLY:

        E_b[reward(a, g)] = sum_m w_m Q_m(a, g)        (commit value: exact)
        P(o | b, u)       = sum_m w_m L_m(o | u)        (obs marginal: exact)
        w_m'              = w_m L_m(o|u) / P(o | b, u)  (mode posterior: exact)

    What an observation does NOT preserve is the *within-mode conditional*: the
    posterior p'(k|m) is proportional to p(k|m) P(o|z_k), not to p(k|m).  Two
    behaviours are therefore offered:

      refresh=False ("summary")
          Freeze p(k|m) at construction.  Root commit value, root observation
          marginals and posterior mode weights are all exact; post-observation
          Q-vectors are stale.  Cost: O(K) once to build the tables, then O(M)
          per expectimax node -- the same shape as a representative collapse.

      refresh=True ("summary_exact")
          Re-weight the members and rebuild the statistics after every
          observation.  Exact at every depth, but every node is O(K) again, which
          is precisely the compute the compression was buying back (see §S8).

    Duck-types `Belief` for everything `expectimax` touches (`task`,
    `best_commit`, `obs_marginal`, `updated`), so the same planning code runs on
    a summary and on a particle belief.

    Compute accounting: the mode-level weighted sums are charged as
    `belief_touches` (one per (mode, action) or per mode), and building the
    tables from the task is charged as the full K reward evals / K likelihood
    touches it really costs.  A planner that supplies precomputed
    `ParticleTables` is charged the O(K) weighted sums but not the reward evals,
    matching how cached decision signatures are already accounted for.
    """

    task: Task
    members: List[List[int]]        # particle indices per mode
    cond: List[np.ndarray]          # within-mode conditional weights (each sums to 1)
    mode_w: np.ndarray              # mode weights (sums to 1)
    refresh: bool = False
    tables: "ParticleTables | None" = None
    _q: dict = field(default_factory=dict, repr=False)
    _l: dict = field(default_factory=dict, repr=False)

    # --- construction ----------------------------------------------------- #
    @classmethod
    def build(cls, task: Task, groups, weights, refresh: bool = False,
              tables: "ParticleTables | None" = None) -> "ModeSummary":
        members, cond, mw = [], [], []
        for g in groups:
            g = list(g)
            w = np.array([weights[k] for k in g], dtype=float)
            tot = float(w.sum())
            if tot <= 0:
                continue
            members.append(g)
            cond.append(w / tot)
            mw.append(tot)
        mw = np.asarray(mw, dtype=float)
        s = mw.sum()
        if s > 0:
            mw = mw / s
        return cls(task=task, members=members, cond=cond, mode_w=mw,
                   refresh=refresh, tables=tables)

    @property
    def M(self) -> int:
        return len(self.members)

    def copy(self) -> "ModeSummary":
        return ModeSummary(self.task, self.members, self.cond, self.mode_w.copy(),
                           self.refresh, self.tables, self._q, self._l)

    # --- cached mode statistics -------------------------------------------- #
    def _q_table(self, goal, counter: ComputeCounter | None = None) -> np.ndarray:
        """(M, |A|) mode-conditional Q-vectors for `goal`."""
        if goal in self._q:
            return self._q[goal]
        acts = self.task.terminal_actions(goal)
        Q = np.zeros((self.M, len(acts)))
        if self.tables is not None:
            R = self.tables.rewards(goal)
            for m, (mem, c) in enumerate(zip(self.members, self.cond)):
                Q[m] = c @ R[mem]
                if counter is not None:
                    counter.touch(len(mem) * len(acts))
        else:
            for m, (mem, c) in enumerate(zip(self.members, self.cond)):
                for j, a in enumerate(acts):
                    Q[m, j] = float(sum(
                        c[i] * self.task.reward(self.task.hidden_states[k], a, goal, counter)
                        for i, k in enumerate(mem)
                    ))
        self._q[goal] = Q
        return Q

    def _l_table(self, probe, counter: ComputeCounter | None = None) -> np.ndarray:
        """(M, |O|) within-mode observation likelihoods for `probe`."""
        if probe in self._l:
            return self._l[probe]
        space = self.task.obs_space(probe)
        L = np.zeros((self.M, len(space)))
        if self.tables is not None:
            P = self.tables.obs(probe)
            for m, (mem, c) in enumerate(zip(self.members, self.cond)):
                L[m] = c @ P[mem]
        else:
            hs = self.task.hidden_states
            for m, (mem, c) in enumerate(zip(self.members, self.cond)):
                for j, o in enumerate(space):
                    L[m, j] = float(sum(
                        c[i] * self.task.obs_prob(hs[k], probe, o)
                        for i, k in enumerate(mem)
                    ))
        if counter is not None:
            counter.touch(sum(len(m) for m in self.members) * len(space))
        self._l[probe] = L
        return L

    # --- Belief-compatible surface ----------------------------------------- #
    def expected_reward(self, a, goal, counter: ComputeCounter | None = None) -> float:
        acts = self.task.terminal_actions(goal)
        Q = self._q_table(goal, counter)
        if counter is not None:
            counter.touch(self.M)
        return float(self.mode_w @ Q[:, acts.index(a)])

    def best_commit(self, goal, counter: ComputeCounter | None = None):
        acts = self.task.terminal_actions(goal)
        Q = self._q_table(goal, counter)
        vals = self.mode_w @ Q
        if counter is not None:
            counter.touch(self.M * len(acts))
        best_a, best_v = None, -np.inf
        for j, a in enumerate(acts):
            if vals[j] > best_v + 1e-12:
                best_v, best_a = float(vals[j]), a
        return best_a, best_v

    def obs_marginal(self, probe, o, counter: ComputeCounter | None = None) -> float:
        space = self.task.obs_space(probe)
        L = self._l_table(probe, counter)
        if counter is not None:
            counter.touch(self.M)
        return float(self.mode_w @ L[:, space.index(o)])

    def updated(self, probe, o, counter: ComputeCounter | None = None) -> "ModeSummary":
        space = self.task.obs_space(probe)
        j = space.index(o)
        L = self._l_table(probe, counter)
        new_w = self.mode_w * L[:, j]
        s = new_w.sum()
        if s <= 0:
            return self.copy()
        new_w = new_w / s

        if not self.refresh:
            # Frozen within-mode conditional: mode weights update exactly, the
            # Q / L tables are reused (this is the O(M) path).
            if counter is not None:
                counter.touch(self.M)
            return ModeSummary(self.task, self.members, self.cond, new_w,
                               False, self.tables, self._q, self._l)

        # Exact path: reweight the members too, and drop the stale statistics.
        hs = self.task.hidden_states
        cond = []
        for m, (mem, c) in enumerate(zip(self.members, self.cond)):
            if self.tables is not None:
                lik = self.tables.obs(probe)[mem, j]
            else:
                lik = np.array([self.task.obs_prob(hs[k], probe, o) for k in mem])
            cw = c * lik
            tot = cw.sum()
            cond.append(cw / tot if tot > 0 else c.copy())
        if counter is not None:
            counter.touch(sum(len(m) for m in self.members))
        return ModeSummary(self.task, self.members, cond, new_w,
                           True, self.tables)
