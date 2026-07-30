"""Compression correctness: decision-equivalence merges losslessly."""

import numpy as np
import pytest

from belief_compression.compression import (
    DEFAULT_REP_RULE,
    ModeSummary,
    compress,
)
from belief_compression.core import Belief, ComputeCounter
from belief_compression.evaluate import exact_return
from belief_compression.planners import (
    CompressionPlanner,
    FullBeliefPlanner,
    PrecomputedCompressionPlanner,
    collapse_modes,
    expectimax,
    mode_belief,
)
from belief_compression.tasks import GridParam, MassSort


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


# --------------------------------------------------------------------------- #
# Defaults: `centroid` is the default way to carry a mode
# --------------------------------------------------------------------------- #
def test_default_rep_rule_is_centroid_everywhere():
    assert DEFAULT_REP_RULE == "centroid"
    task = GridParam(resolution=24, n_controllers=3, n_goals=1, max_budget=1)
    family = task.goal_family(1)
    prior = Belief.prior(task)
    groups = [m.members for m in compress(prior, family, tol=0.0).modes]

    # collapse_modes / compress / both planners all default to centroid ...
    np.testing.assert_allclose(
        collapse_modes(task, groups, prior.weights),
        collapse_modes(task, groups, prior.weights, "centroid"),
    )
    reps = {m.rep for m in compress(prior, family, tol=0.0).modes}
    reps_cen = {m.rep for m in compress(prior, family, tol=0.0, rep_rule="centroid").modes}
    reps_max = {m.rep for m in compress(prior, family, tol=0.0, rep_rule="maxweight").modes}
    assert reps == reps_cen and reps != reps_max

    # ... and the DEFAULT planner keeps the bare name, every other rule is tagged.
    assert CompressionPlanner(family).name == "compression"
    assert CompressionPlanner(family, rep_rule="maxweight").name == "compression_maxweight"
    assert PrecomputedCompressionPlanner(task, family).name == "compression_cached"
    assert (PrecomputedCompressionPlanner(task, family, rep_rule="summary").name
            == "compression_cached_summary")


def test_maxweight_stays_selectable_and_still_fails():
    """The S6/S7 failure must remain reproducible after the default change."""
    task = GridParam(resolution=96, n_controllers=2, n_goals=1, max_budget=1)
    family = task.goal_family(1)
    g = family[0]
    rf = exact_return(task, FullBeliefPlanner(), g, budget=1).ret
    rm = exact_return(
        task, PrecomputedCompressionPlanner(task, family, rep_rule="maxweight"),
        g, budget=1).ret
    assert abs(rf - rm) / abs(rf) > 0.4


# --------------------------------------------------------------------------- #
# The value-consistent mode summary
# --------------------------------------------------------------------------- #
def _tiny_summary():
    """K=4, |A|=2 grid_param: two modes {0,1} and {2,3}, everything by hand.

    theta = [0.125, 0.375, 0.625, 0.875]; controller centres (0.25, 0.75);
    reward = 1 - 2|theta - c|; uniform prior.
    """
    task = GridParam(resolution=4, n_controllers=2, n_goals=1, max_budget=1)
    prior = Belief.prior(task)
    family = task.goal_family(1)
    groups = [m.members for m in compress(prior, family, tol=0.0).modes]
    return task, family[0], prior, groups


def test_mode_summary_q_and_likelihood_are_hand_checkable():
    task, goal, prior, groups = _tiny_summary()
    assert groups == [[0, 1], [2, 3]]
    s = ModeSummary.build(task, groups, prior.weights)
    np.testing.assert_allclose(s.mode_w, [0.5, 0.5], atol=1e-12)

    # Q_m(a) = mean over the mode's members of 1 - 2|theta - c_a|.
    #   mode {0,1}: a=0 -> mean(0.75, 0.75) = 0.75 ; a=1 -> mean(-0.25, 0.25) = 0.0
    #   mode {2,3}: a=0 -> mean(0.25, -0.25) = 0.0 ; a=1 -> mean(0.75, 0.75) = 0.75
    np.testing.assert_allclose(s._q_table(goal), [[0.75, 0.0], [0.0, 0.75]], atol=1e-12)

    # commit value = sum_m w_m Q_m(a) = 0.375 for either action
    assert s.best_commit(goal)[1] == pytest.approx(0.375, abs=1e-12)

    # L_m(o|sense3): true bins are [0, 1, 1, 2], noise 0.15 spread over 2 others
    #   mode {0,1}, o=0 -> mean(0.85, 0.075) = 0.4625
    np.testing.assert_allclose(s._l_table("sense3")[0, 0], 0.4625, atol=1e-12)
    np.testing.assert_allclose(s._l_table("sense3")[1, 2], 0.4625, atol=1e-12)
    # every row is a distribution over the 3 readout bins
    np.testing.assert_allclose(s._l_table("sense3").sum(axis=1), [1.0, 1.0], atol=1e-12)


def test_mode_summary_reproduces_full_belief_commit_and_marginal_exactly():
    """Commit value and observation marginal are EXACT under the summary; a
    representative particle gets both wrong on the same belief."""
    for A in (2, 3, 8):
        task = GridParam(resolution=96, n_controllers=A, n_goals=1, max_budget=1)
        family = task.goal_family(1)
        goal = family[0]
        prior = Belief.prior(task)
        groups = [m.members for m in compress(prior, family, tol=0.0).modes]
        s = mode_belief(task, groups, prior.weights, "summary")
        rep = mode_belief(task, groups, prior.weights, "centroid")

        assert abs(s.best_commit(goal)[1] - prior.best_commit(goal)[1]) < 1e-12
        assert abs(rep.best_commit(goal)[1] - prior.best_commit(goal)[1]) > 1e-6
        for o in task.obs_space("sense8"):
            assert abs(s.obs_marginal("sense8", o)
                       - prior.obs_marginal("sense8", o)) < 1e-12


def test_mode_summary_bayes_update_matches_particle_update_per_mode():
    """Updating mode weights through L_m must equal aggregating the particle
    posterior over the same modes -- that is what makes the summary a valid
    belief over modes rather than an approximation."""
    task = GridParam(resolution=96, n_controllers=3, n_goals=1, max_budget=1)
    family = task.goal_family(1)
    prior = Belief.prior(task)
    groups = [m.members for m in compress(prior, family, tol=0.0).modes]
    for probe in task.probes:
        for o in task.obs_space(probe):
            post_particles = prior.updated(probe, o)
            want = np.array([post_particles.weights[g].sum() for g in groups])
            for rule in ("summary", "summary_exact"):
                got = mode_belief(task, groups, prior.weights, rule).updated(probe, o)
                np.testing.assert_allclose(got.mode_w, want, atol=1e-12)


def test_summary_value_is_exact_where_a_representative_particle_provably_is_not():
    """The headline claim, at the planning level: on a cell where BOTH
    representative rules mis-value by >1e-2, the summary matches the full belief
    to floating point."""
    task = GridParam(resolution=96, n_controllers=2, n_goals=1, max_budget=1)
    goal = task.goal_family(1)[0]
    prior = Belief.prior(task)
    groups = [m.members for m in compress(prior, task.goal_family(1), tol=0.0).modes]
    for H in (0, 1, 2):
        v_full = expectimax(prior, goal, H, ComputeCounter())[0]
        for rule in ("maxweight", "centroid"):
            v = expectimax(mode_belief(task, groups, prior.weights, rule), goal, H,
                           ComputeCounter())[0]
            if H > 0:
                assert abs(v - v_full) > 1e-2, (H, rule, v, v_full)
        for rule in ("summary", "summary_exact"):
            v = expectimax(mode_belief(task, groups, prior.weights, rule), goal, H,
                           ComputeCounter())[0]
            assert abs(v - v_full) < 1e-12, (H, rule, v, v_full)


def test_frozen_summary_is_exact_for_the_commit_but_drifts_under_probing():
    """Honesty guard, and the reason `summary_exact` exists: freezing the
    within-mode conditional is exact at budget 0 and only there in general."""
    task = GridParam(resolution=96, n_controllers=3, n_goals=1, max_budget=1)
    goal = task.goal_family(1)[0]
    prior = Belief.prior(task)
    groups = [m.members for m in compress(prior, task.goal_family(1), tol=0.0).modes]

    v0_full = expectimax(prior, goal, 0, ComputeCounter())[0]
    v0 = expectimax(mode_belief(task, groups, prior.weights, "summary"), goal, 0,
                    ComputeCounter())[0]
    assert abs(v0 - v0_full) < 1e-12          # commit-only: exact

    v1_full = expectimax(prior, goal, 1, ComputeCounter())[0]
    v1 = expectimax(mode_belief(task, groups, prior.weights, "summary"), goal, 1,
                    ComputeCounter())[0]
    assert abs(v1 - v1_full) > 1e-3           # probing: NOT exact

    v1x = expectimax(mode_belief(task, groups, prior.weights, "summary_exact"), goal, 1,
                     ComputeCounter())[0]
    assert abs(v1x - v1_full) < 1e-12         # refreshed: exact again
