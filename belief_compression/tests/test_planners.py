"""End-to-end: every planner/probe policy runs on both tasks with sane bounds."""

import numpy as np
import pytest

from belief_compression.evaluate import exact_return, fully_observed_return
from belief_compression.planners import (
    AmortizedVOI,
    CertaintyEquivalent,
    CompressionPlanner,
    DecisionRegretVOI,
    EntropySeeking,
    FullBeliefPlanner,
)
from belief_compression import experiments as X


def _all_policies(task):
    return [
        CertaintyEquivalent(),
        EntropySeeking(),
        DecisionRegretVOI(),
        AmortizedVOI(),
        FullBeliefPlanner(),
        CompressionPlanner(goal_family=task.goal_family(2), tol=0.0),
    ]


def test_all_planners_run_mass_sort():
    task, goal = X.default_mass_sort_measure2()
    for pol in _all_policies(task):
        res = exact_return(task, pol, goal)
        assert np.isfinite(res.ret)
        assert res.compute["total"] >= 0


def test_all_planners_run_occluder():
    task, goal = X.default_occluder_measure2()
    for pol in _all_policies(task):
        res = exact_return(task, pol, goal)
        assert np.isfinite(res.ret)


def test_bounds_full_belief_optimal_and_below_observed():
    # full-belief expectimax is optimal within the model: >= every other belief
    # policy; the fully-observed oracle upper-bounds everything.
    for builder in (X.default_mass_sort_measure2, X.default_occluder_measure2):
        task, goal = builder()
        obs = fully_observed_return(task, goal)
        full = exact_return(task, FullBeliefPlanner(), goal).ret
        ce = exact_return(task, CertaintyEquivalent(), goal).ret
        voi = exact_return(task, DecisionRegretVOI(), goal).ret
        ent = exact_return(task, EntropySeeking(), goal).ret
        assert full <= obs + 1e-9
        assert ce <= full + 1e-9
        assert voi <= full + 1e-9
        assert ent <= full + 1e-9


def test_voi_beats_entropy_and_is_cheaper_than_full():
    # The core Gate-B claim on the mass_sort entropy-trap.
    task, goal = X.default_mass_sort_measure2()
    voi = exact_return(task, DecisionRegretVOI(), goal)
    ent = exact_return(task, EntropySeeking(), goal)
    full = exact_return(task, FullBeliefPlanner(), goal)
    amort = exact_return(task, AmortizedVOI(), goal)
    assert voi.ret > ent.ret + 1e-6
    assert voi.compute["total"] < full.compute["total"]
    assert amort.compute["total"] < voi.compute["total"]
    assert abs(amort.ret - voi.ret) < 1e-9


def test_amortized_voi_matches_voi_probe_decision():
    # Amortized VOI must produce the same probe-then-act behaviour as full VOI.
    for builder in (X.default_mass_sort_measure2, X.default_occluder_measure2):
        task, goal = builder()
        voi = exact_return(task, DecisionRegretVOI(), goal).ret
        amort = exact_return(task, AmortizedVOI(), goal).ret
        assert abs(voi - amort) < 1e-9
