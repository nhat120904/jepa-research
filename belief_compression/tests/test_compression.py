"""Compression correctness: decision-equivalence merges losslessly."""

import numpy as np

from belief_compression.compression import compress
from belief_compression.core import Belief
from belief_compression.evaluate import exact_return
from belief_compression.planners import CompressionPlanner, FullBeliefPlanner
from belief_compression.tasks import MassSort


def test_modes_cover_all_weight():
    task = MassSort(n_objects=3)
    b = Belief.prior(task)
    family = task.goal_family(2)
    comp = compress(b, family, tol=0.0)
    total = sum(m.weight for m in comp.modes)
    assert abs(total - 1.0) < 1e-12
    rep_b = comp.as_belief_over_reps()
    assert abs(rep_b.weights.sum() - 1.0) < 1e-12


def test_identical_ranking_hypotheses_merge():
    # Goal family only cares about object 0, so hypotheses that agree on
    # object 0 have identical preferred action -> must merge into one mode.
    task = MassSort(n_objects=2)
    b = Belief.prior(task)
    family = [frozenset({0})]
    comp = compress(b, family, tol=0.0)
    # K=4 hypotheses -> 2 modes (z0=light, z0=heavy)
    assert comp.K == 4
    assert comp.M == 2


def test_decision_equivalent_compression_is_regret_free():
    # Executing goals INSIDE the compressed family must incur ~0 regret.
    task = MassSort(n_objects=3, max_budget=0)
    full = FullBeliefPlanner()
    for richness in (1, 2, 3):
        family = task.goal_family(richness)
        comp_planner = CompressionPlanner(goal_family=family, tol=0.0)
        for g in family:
            rf = exact_return(task, full, g, budget=0).ret
            rc = exact_return(task, comp_planner, g, budget=0).ret
            assert abs(rf - rc) < 1e-9, (richness, g, rf, rc)


def test_compression_reduces_modes_when_goals_narrow():
    task = MassSort(n_objects=4)
    b = Belief.prior(task)
    narrow = compress(b, task.goal_family(1), tol=0.0)
    wide = compress(b, task.goal_family(4), tol=0.0)
    assert narrow.M < wide.M
    assert narrow.ratio <= 0.5
    assert abs(wide.ratio - 1.0) < 1e-12


def test_tolerance_merges_at_least_as_much_as_exact():
    task = MassSort(n_objects=4)
    b = Belief.prior(task)
    family = task.goal_family(4)
    exact = compress(b, family, tol=0.0)
    loose = compress(b, family, tol=5.0)
    assert loose.M <= exact.M
