"""Gate C0 scaling logic: refining the belief grid must not change the DECISION.

The whole scaling claim rests on one structural property: `resolution` is a
belief-side knob only.  These tests pin that property down on constructed
cases, so a future change that quietly couples resolution to the decision
structure (and would fake a saturation result) fails loudly.
"""

import numpy as np
import pytest

from belief_compression.compression import compress
from belief_compression.core import Belief, ComputeCounter
from belief_compression.evaluate import exact_return
from belief_compression.planners import (
    CompressionPlanner,
    FullBeliefPlanner,
    PrecomputedCompressionPlanner,
)
from belief_compression import scaling as S
from belief_compression.tasks import GridParam, MassSort, OccluderPush


# --------------------------------------------------------------------------- #
# The decision structure is invariant to belief resolution
# --------------------------------------------------------------------------- #
def test_refining_grid_preserves_preferred_action_function():
    """A grid point present at BOTH resolutions must get the same preferred
    action at both: refinement adds hypotheses, it never relabels them."""
    # cell-CENTRED grids nest only under an ODD refinement factor
    # ((i+0.5)/K == (j+0.5)/(rK) has an integer solution j = r*i + (r-1)/2), so
    # refine 12 -> 108 (factor 9) to get a genuinely nested pair of grids.
    coarse = GridParam(resolution=12, n_controllers=3, n_goals=2)
    fine = GridParam(resolution=108, n_controllers=3, n_goals=2)
    assert coarse.goal_family(2) == fine.goal_family(2)  # goals live in theta-space
    for g in coarse.goal_family(2):
        for i, th in enumerate(coarse.theta):
            j = int(np.argmin(np.abs(fine.theta - th)))
            assert abs(fine.theta[j] - th) < 1e-12, "coarse grid must nest in fine grid"
            assert coarse.preferred_action(i, g) == fine.preferred_action(j, g)


def test_preferred_action_is_nearest_controller():
    task = GridParam(resolution=64, n_controllers=4, n_goals=1)
    g = task.goal_family(1)[0]
    for i, th in enumerate(task.theta):
        assert task.preferred_action(i, g) == int(np.argmin([abs(th - c) for c in g]))


def test_M_saturates_at_the_analytic_bound_under_refinement():
    """The constructed case: |A|=3, |G|=2 -> at most 2*(3-1)+1 = 5 modes, no
    matter how fine the belief grid gets."""
    seen = []
    for K in (24, 48, 96, 192, 384, 768):
        task = GridParam(resolution=K, n_controllers=3, n_goals=2)
        comp = compress(Belief.prior(task), task.goal_family(2), tol=0.0)
        assert comp.K == K
        assert comp.M <= task.decision_bound(2)
        seen.append(comp.M)
    assert len(set(seen)) == 1, f"M must be constant under refinement, got {seen}"
    assert seen[0] == 5


def test_M_over_K_goes_to_zero_under_refinement():
    ratios = []
    for K in (24, 96, 384, 1536):
        task = GridParam(resolution=K, n_controllers=3, n_goals=1)
        ratios.append(compress(Belief.prior(task), task.goal_family(1), tol=0.0).ratio)
    assert all(b < a for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] < 0.01


def test_decision_bound_is_independent_of_resolution():
    for K in (16, 256, 4096):
        t = GridParam(resolution=K, n_controllers=5, n_goals=3)
        assert t.decision_bound(3) == 3 * (5 - 1) + 1


def test_occluder_refinement_saturates_at_bin_count():
    Ms = []
    for L in (8, 16, 32, 64, 128):
        task = OccluderPush(n_locations=L, fine=L // 2, max_budget=0)
        Ms.append(compress(Belief.prior(task), [task.coarse_goal()], tol=0.0).M)
    assert Ms == [2] * 5, Ms


def test_control_axis_M_tracks_K_when_new_dimensions_are_relevant():
    """Contrast case: growing K by adding decision-relevant latent dimensions
    must NOT saturate -- otherwise the saturation result is vacuous."""
    for N in (2, 3, 4, 5):
        task = MassSort(n_objects=N, max_budget=0)
        comp = compress(Belief.prior(task), task.goal_family(N), tol=0.0)
        assert comp.M == comp.K == 2 ** N


# --------------------------------------------------------------------------- #
# Compression stays decision-lossless under refinement
# --------------------------------------------------------------------------- #
def test_commit_regret_is_exactly_zero_at_every_resolution():
    for K in (12, 48, 96):
        task = GridParam(resolution=K, n_controllers=3, n_goals=2, max_budget=0)
        family = task.goal_family(2)
        full = FullBeliefPlanner()
        comp = CompressionPlanner(goal_family=family, tol=0.0)
        for g in family:
            rf = exact_return(task, full, g, budget=0).ret
            rc = exact_return(task, comp, g, budget=0).ret
            assert abs(rf - rc) < 1e-12, (K, rf, rc)


def test_precomputed_compression_matches_online_compression():
    for K in (12, 48):
        for H in (0, 1):
            task = GridParam(resolution=K, n_controllers=3, n_goals=2, max_budget=H)
            family = task.goal_family(2)
            g = family[0]
            on = exact_return(task, CompressionPlanner(family, tol=0.0), g, budget=H).ret
            pre = exact_return(
                task, PrecomputedCompressionPlanner(task, family), g, budget=H
            ).ret
            assert abs(on - pre) < 1e-12, (K, H, on, pre)


# --------------------------------------------------------------------------- #
# Compute accounting
# --------------------------------------------------------------------------- #
def test_belief_update_charges_support_not_K():
    """The accounting fix: an M-support belief must not be billed for K."""
    task = GridParam(resolution=100, n_controllers=3, n_goals=1)
    b = Belief.prior(task)
    c_full = ComputeCounter()
    b.updated("sense3", 0, c_full)
    assert c_full.belief_touches == 100

    w = np.zeros(100)
    w[[3, 40, 90]] = 1 / 3
    sparse = Belief(task, w)
    c_sparse = ComputeCounter()
    post = sparse.updated("sense3", 0, c_sparse)
    assert c_sparse.belief_touches == 3
    assert abs(post.weights.sum() - 1.0) < 1e-12
    assert np.all(post.weights[[0, 1, 2]] == 0.0)


def test_compressed_planner_is_cheaper_and_gap_widens_with_K():
    ratios = []
    for K in (24, 96, 384):
        task = GridParam(resolution=K, n_controllers=3, n_goals=1, max_budget=1)
        family = task.goal_family(1)
        g = family[0]
        c_full, _, a_full = S.root_compute(task, FullBeliefPlanner(), g, 1)
        c_pre, _, a_pre = S.root_compute(
            task, PrecomputedCompressionPlanner(task, family), g, 1
        )
        assert c_pre["total"] < c_full["total"]
        assert a_pre == a_full  # same root decision
        ratios.append(c_full["total"] / c_pre["total"])
    assert all(b > a for a, b in zip(ratios, ratios[1:])), ratios


def test_compute_gap_widens_with_horizon():
    K = 96
    task_family = None
    ratios = []
    for H in (1, 2, 3):
        task = GridParam(resolution=K, n_controllers=3, n_goals=1, max_budget=H)
        family = task.goal_family(1)
        g = family[0]
        c_full, _, _ = S.root_compute(task, FullBeliefPlanner(), g, H)
        c_pre, _, _ = S.root_compute(
            task, PrecomputedCompressionPlanner(task, family), g, H
        )
        ratios.append(c_full["total"] / c_pre["total"])
    assert all(b > a for a, b in zip(ratios, ratios[1:])), ratios


def test_ratio_is_capped_by_K_over_M():
    """Sanity on the theory: the compressed planner cannot beat full belief by
    more than K/M, because it still reads the K-particle belief once."""
    K, H = 96, 3
    task = GridParam(resolution=K, n_controllers=3, n_goals=1, max_budget=H)
    family = task.goal_family(1)
    M = compress(Belief.prior(task), family, tol=0.0).M
    c_full, _, _ = S.root_compute(task, FullBeliefPlanner(), family[0], H)
    c_pre, _, _ = S.root_compute(
        task, PrecomputedCompressionPlanner(task, family), family[0], H
    )
    assert c_full["total"] / c_pre["total"] <= K / M + 1e-9


# --------------------------------------------------------------------------- #
# Sweep plumbing
# --------------------------------------------------------------------------- #
def test_sweeps_run_and_are_self_consistent():
    rows = S.s1_grid_param([12, 24], n_goals=2)
    assert [r.M for r in rows] == [5, 5]
    assert all(r.M <= r.bound for r in rows)

    s2 = S.s2_compute_vs_K([12, 24], H=1)
    for r in s2:
        assert r.compute["compression_cached"] < r.compute["full_belief"]
        assert r.same_root_action["full_belief"] is True

    grid = S.s3b_joint([24], [0, 1], max_ops=1e7)
    assert grid[(24, 1)]["ratio"] > grid[(24, 0)]["ratio"]

    g5 = S.s5_richness_x_K([48, 192], [1, 2])
    assert g5[(1, 48)][0] == g5[(1, 192)][0]
    assert g5[(2, 48)][0] == g5[(2, 192)][0]
    assert g5[(2, 48)][0] > g5[(1, 48)][0]


def test_loglog_slope_recovers_a_known_exponent():
    xs = [2, 4, 8, 16, 32]
    assert S.loglog_slope(xs, [x ** 2 for x in xs]) == pytest.approx(2.0, abs=1e-9)
    assert S.loglog_slope(xs, [7 for _ in xs]) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# S7: the mode representative changes VALUE, never the partition or the compute
# --------------------------------------------------------------------------- #
def test_representative_rule_does_not_change_partition_or_compute():
    """The whole S7 argument rests on the fix being FREE: swapping the
    representative must leave M -- and therefore every compute number -- alone."""
    for A in (2, 4, 8):
        for K in (24, 96):
            task = GridParam(resolution=K, n_controllers=A, n_goals=1, max_budget=1)
            family = task.goal_family(1)
            g = family[0]
            M = compress(Belief.prior(task), family, tol=0.0).M
            c_max, _, _ = S.root_compute(
                task, PrecomputedCompressionPlanner(task, family, rep_rule="maxweight"), g, 1
            )
            c_cen, _, _ = S.root_compute(
                task, PrecomputedCompressionPlanner(task, family, rep_rule="centroid"), g, 1
            )
            assert c_max == c_cen, (A, K, M, c_max, c_cen)


def test_centroid_representative_fixes_the_A2_root_flip():
    """The |A|=2 catastrophe in S6 is an artefact of the max-weight tie-break
    (flat prior -> first index -> mode-EDGE particle), not of compression."""
    for K in (48, 96, 192):
        task = GridParam(resolution=K, n_controllers=2, n_goals=1, max_budget=1)
        family = task.goal_family(1)
        g = family[0]
        ret_full = exact_return(task, FullBeliefPlanner(), g, budget=1).ret
        r_max = exact_return(
            task, PrecomputedCompressionPlanner(task, family, rep_rule="maxweight"),
            g, budget=1).ret
        r_cen = exact_return(
            task, PrecomputedCompressionPlanner(task, family, rep_rule="centroid"),
            g, budget=1).ret
        assert abs(ret_full - r_max) / abs(ret_full) > 0.4   # the known failure
        assert abs(ret_full - r_cen) < 1e-12                 # fixed, exactly


def test_centroid_is_not_a_free_lunch_somewhere_it_still_loses():
    """Honesty guard: the centroid rule is better, but it is NOT value-consistent,
    so it must not be asserted to be universally lossless."""
    task = GridParam(resolution=24, n_controllers=8, n_goals=2, max_budget=1)
    family = task.goal_family(2)
    worst = 0.0
    for g in family:
        rf = exact_return(task, FullBeliefPlanner(), g, budget=1).ret
        rc = exact_return(
            task, PrecomputedCompressionPlanner(task, family, rep_rule="centroid"),
            g, budget=1).ret
        worst = max(worst, abs(rf - rc))
    assert worst > 1e-9, "expected a residual regret cell; centroid is not exact"


def test_param_embedding_absent_falls_back_to_maxweight():
    """A task with no metric on its hidden space must still work."""
    task = MassSort(n_objects=3, max_budget=0)
    assert task.param_embedding() is None
    family = task.goal_family(2)
    g = family[0]
    a = exact_return(
        task, PrecomputedCompressionPlanner(task, family, rep_rule="centroid"), g, budget=0).ret
    b = exact_return(
        task, PrecomputedCompressionPlanner(task, family, rep_rule="maxweight"), g, budget=0).ret
    assert abs(a - b) < 1e-12


def test_s7_sweep_runs_and_reports_the_improvement():
    rows = S.s7_representative([2, 3], [24, 48], G_values=(1,), H=1)
    assert rows
    for r in rows:
        assert r["compute"]["maxweight"] == r["compute"]["centroid"]
        assert r["worst"]["centroid"] <= r["worst"]["maxweight"] + 1e-12
    a2 = [r for r in rows if r["A"] == 2]
    assert a2 and max(r["worst"]["maxweight"] for r in a2) > 0.4
    assert max(r["worst"]["centroid"] for r in a2) < 1e-9
