"""Planners and probe-selection policies.

Every policy exposes ``act(belief, budget, goal, counter) -> (kind, value)``
where ``kind`` is ``"commit"`` (value = terminal action) or ``"probe"``
(value = probe id).  All policies share the SAME exact Bayesian belief update
(done by the evaluator); they differ only in the action they choose, so any
difference in return is attributable to the planning algorithm, not the belief.

Planners
    FullBeliefPlanner        exact expectimax over all K hypotheses (ceiling)
    CertaintyEquivalent      collapse to MAP, commit deterministically
    CompressionPlanner       expectimax over M<=K decision modes
    (fully-observed upper bound is handled directly in evaluate.py)

Both compression planners take a `rep_rule` saying how a mode is carried:
`"centroid"` (default) / `"maxweight"` collapse onto a representative particle;
`"summary"` / `"summary_exact"` carry the value-consistent `ModeSummary`.  The
mode PARTITION -- and therefore M -- is the same for all four.

Probe selectors
    EntropySeeking           probe maximising expected belief-entropy reduction
    DecisionRegretVOI        probe maximising myopic value-of-information
    AmortizedVOI             VOI estimated via decision-mode structure (cheaper)
"""

from __future__ import annotations

import numpy as np

from .compression import (
    DEFAULT_REP_RULE,
    SUMMARY_RULES,
    ModeSummary,
    ParticleTables,
    compress,
    pick_rep,
)
from .core import Belief, ComputeCounter, Task
from .core import Belief as _B  # alias


# --------------------------------------------------------------------------- #
# Shared expectimax
# --------------------------------------------------------------------------- #
def expectimax(belief: Belief, goal, budget: int, counter: ComputeCounter):
    """Exact expectimax value/action for a (possibly collapsed) belief."""
    task = belief.task
    counter.node()
    best_a, commit_v = belief.best_commit(goal, counter)
    best_val = commit_v
    best_action = ("commit", best_a)

    if budget > 0:
        for probe in task.probes:
            ev = 0.0
            for o in task.obs_space(probe):
                po = belief.obs_marginal(probe, o, counter)
                if po <= 1e-12:
                    continue
                post = belief.updated(probe, o, counter)
                v, _ = expectimax(post, goal, budget - 1, counter)
                ev += po * v
            ev -= task.probe_cost
            if ev > best_val + 1e-12:
                best_val = ev
                best_action = ("probe", probe)
    return best_val, best_action


# --------------------------------------------------------------------------- #
# Mode collapse: how a mode is carried into the planner
# --------------------------------------------------------------------------- #
def collapse_modes(task, groups, weights, rep_rule: str = DEFAULT_REP_RULE):
    """Collapse decision modes onto one representative particle each.

    `rep_rule` does NOT change the mode partition -- M, and therefore the
    planning compute, are identical either way.  It only changes WHICH particle
    carries a mode's weight, and hence the value the collapsed belief reports.
    See `compression.pick_rep` for the two rules; "centroid" is the default.
    """
    w = np.zeros(task.n_states())
    for members in groups:
        wsum = float(sum(weights[k] for k in members))
        w[pick_rep(task, members, weights, rep_rule)] += wsum
    s = w.sum()
    return w / s if s > 0 else w


def mode_belief(task, groups, weights, rep_rule: str = DEFAULT_REP_RULE,
                tables: ParticleTables | None = None):
    """The object the planner actually searches over, for any `rep_rule`.

    A representative rule yields a `Belief` supported on M particles; a summary
    rule yields a `ModeSummary` carrying per-mode Q-vectors and within-mode
    observation likelihoods.  Both duck-type the same interface, so `expectimax`
    is unchanged.
    """
    if rep_rule in SUMMARY_RULES:
        return ModeSummary.build(
            task, groups, weights,
            refresh=(rep_rule == "summary_exact"), tables=tables,
        )
    return _B(task, collapse_modes(task, groups, weights, rep_rule))


def _planner_name(base: str, rep_rule: str) -> str:
    """The DEFAULT rule keeps the bare planner name; every other rule is tagged."""
    return base if rep_rule == DEFAULT_REP_RULE else f"{base}_{rep_rule}"


# --------------------------------------------------------------------------- #
# Planners
# --------------------------------------------------------------------------- #
class FullBeliefPlanner:
    name = "full_belief"

    def act(self, belief, budget, goal, counter):
        return expectimax(belief, goal, budget, counter)[1]


class CertaintyEquivalent:
    name = "certainty_equiv"

    def act(self, belief, budget, goal, counter):
        # Collapse belief to its MAP hypothesis and plan as if certain.
        z = belief.map_state()
        a = belief.task.preferred_action(z, goal, counter)
        return ("commit", a)  # a known state has no value of information


class CompressionPlanner:
    """Plan over M decision modes instead of K hypotheses."""

    def __init__(self, goal_family, tol: float = 0.0,
                 rep_rule: str = DEFAULT_REP_RULE):
        self.goal_family = goal_family
        self.tol = tol
        self.rep_rule = rep_rule
        self.name = _planner_name("compression", rep_rule)

    def act(self, belief, budget, goal, counter):
        comp = compress(belief, self.goal_family, tol=self.tol, counter=counter)
        mb = mode_belief(
            belief.task, [m.members for m in comp.modes], belief.weights, self.rep_rule
        )
        return expectimax(mb, goal, budget, counter)[1]


class PrecomputedCompressionPlanner:
    """CompressionPlanner with the decision signatures cached offline.

    A hypothesis' decision signature depends only on (hypothesis, goal family),
    NOT on the belief weights, so it can be computed once per task/goal-family
    and reused at every decision.  At decision time the planner then only has
    to bucket the belief support by its cached signature -- O(K) table lookups,
    no reward evaluations -- and run expectimax over the M representatives.

    This separates the two compression costs that the scaling study must not
    conflate: the one-off O(K*|G|*|A|) signature build (amortized away here)
    and the per-decision O(K) bucketing + O(M * tree) planning.

    Under a summary `rep_rule` the same amortization applies to the per-particle
    reward / observation tables (`ParticleTables`): they depend only on the task,
    so they are built once and the per-decision charge is the O(K) weighted sums
    that turn them into per-mode Q-vectors and likelihoods.
    """

    def __init__(self, task, goal_family, tol: float = 0.0,
                 rep_rule: str = DEFAULT_REP_RULE):
        self.goal_family = goal_family
        self.tol = tol
        self.rep_rule = rep_rule
        self.name = _planner_name("compression_cached", rep_rule)
        self.task = task
        self._sig = [
            tuple(task.preferred_action(z, g) for g in goal_family)
            for z in task.hidden_states
        ]
        self._tables = ParticleTables(task) if rep_rule in SUMMARY_RULES else None

    def act(self, belief, budget, goal, counter):
        task = belief.task
        buckets: dict = {}
        support = [k for k in range(task.n_states()) if belief.weights[k] > 0]
        counter.touch(len(support))  # one cached-signature lookup per particle
        for k in support:
            buckets.setdefault(self._sig[k], []).append(k)
        mb = mode_belief(task, list(buckets.values()), belief.weights,
                         self.rep_rule, tables=self._tables)
        return expectimax(mb, goal, budget, counter)[1]


# --------------------------------------------------------------------------- #
# Probe-selection policies
# --------------------------------------------------------------------------- #
class EntropySeeking:
    """Greedy information-gain probing: reduce belief entropy regardless of
    whether the reduced uncertainty is decision-relevant."""

    name = "entropy_seek"

    def __init__(self, min_gain: float = 1e-4):
        self.min_gain = min_gain

    def act(self, belief, budget, goal, counter):
        task = belief.task
        best_a, _ = belief.best_commit(goal, counter)
        if budget <= 0:
            return ("commit", best_a)
        H0 = belief.entropy()
        best_probe, best_ig = None, -np.inf
        for probe in task.probes:
            exp_H = 0.0
            for o in task.obs_space(probe):
                po = belief.obs_marginal(probe, o, counter)
                if po <= 1e-12:
                    continue
                post = belief.updated(probe, o, counter)
                exp_H += po * post.entropy()
            ig = H0 - exp_H
            if ig > best_ig + 1e-12:
                best_ig, best_probe = ig, probe
        if best_ig > self.min_gain:
            return ("probe", best_probe)
        return ("commit", best_a)


class DecisionRegretVOI:
    """Myopic decision-regret value of information.  Probe only if the expected
    post-observation best-commit value exceeds the current best-commit value by
    more than the probe cost.  Nests a full best-commit over K per observation."""

    name = "voi"

    def act(self, belief, budget, goal, counter):
        task = belief.task
        best_a, V0 = belief.best_commit(goal, counter)
        if budget <= 0:
            return ("commit", best_a)
        best_probe, best_voi = None, -np.inf
        for probe in task.probes:
            ev = 0.0
            for o in task.obs_space(probe):
                po = belief.obs_marginal(probe, o, counter)
                if po <= 1e-12:
                    continue
                post = belief.updated(probe, o, counter)
                _, vo = post.best_commit(goal, counter)  # inner planning over K
                counter.voi()
                ev += po * vo
            voi = ev - V0
            if voi > best_voi + 1e-12:
                best_voi, best_probe = voi, probe
        if best_voi > task.probe_cost + 1e-12:
            return ("probe", best_probe)
        return ("commit", best_a)


class AmortizedVOI:
    """Amortized decision-regret VOI via decision-mode compression.

    The expensive part of VOI is recomputing the best-commit over all K
    hypotheses for every candidate observation.  We instead compress the belief
    into M decision modes (by preferred action for the executed goal) and
    estimate the post-observation best-commit at MODE granularity: aggregate the
    observation likelihood per mode (M touches, not K) and evaluate best-commit
    over the M representatives.  Because modes are decision-equivalence classes,
    this preserves *which action is preferred* while doing far less work."""

    name = "amortized_voi"

    def __init__(self, tol: float = 0.0):
        self.tol = tol

    def act(self, belief, budget, goal, counter):
        task = belief.task
        # decision modes for the executed goal (amortization structure)
        comp = compress(belief, [goal], tol=self.tol, counter=counter)
        rep_belief = comp.as_belief_over_reps()
        best_a, V0 = rep_belief.best_commit(goal, counter)
        if budget <= 0:
            # commit using the full belief's preferred action (cheap, exact argmax)
            fa, _ = belief.best_commit(goal, counter)
            return ("commit", fa)

        best_probe, best_voi = None, -np.inf
        for probe in task.probes:
            ev = 0.0
            for o in task.obs_space(probe):
                po = rep_belief.obs_marginal(probe, o, counter)  # M touches
                if po <= 1e-12:
                    continue
                post = rep_belief.updated(probe, o, counter)     # M touches
                _, vo = post.best_commit(goal, counter)          # M * |A| rewards
                counter.voi()
                ev += po * vo
            voi = ev - V0
            if voi > best_voi + 1e-12:
                best_voi, best_probe = voi, probe
        if best_voi > task.probe_cost + 1e-12:
            return ("probe", best_probe)
        fa, _ = belief.best_commit(goal, counter)
        return ("commit", fa)
