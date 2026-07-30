"""Gate P0 scaffolding: measurement, thresholds and verdict, offline.

Everything here runs on numpy alone -- no jax, no GPU, no trained model -- by
driving the SAME `measure.sweep` / `measure.verdict` code the real run uses,
with the belief model swapped for a synthetic stand-in whose diversity is
tunable by construction.  That is the repo convention (validate the metric on
synthetic models before trusting a real number) and it is what makes the
pre-registered thresholds falsifiable before any GPU time is spent.
"""

import math

import numpy as np
import pytest

from belief_compression.compression import compress
from belief_compression.core import Belief, ComputeCounter
from belief_compression.p0 import belief as B
from belief_compression.p0 import envs as E
from belief_compression.p0 import measure as MZ
from belief_compression.p0.decision import SAFE, TARGET, RegionCommit
from belief_compression.p0.hypotheses import HypothesisSet, HypothesisTask, build_task

N_CELLS = 16
N_MINES = 2


def _dec(target=SAFE):
    return RegionCommit.build(N_CELLS, target_bit=target, seed=0)


# --------------------------------------------------------------------------- #
# The Task adapter really is driving the shared compress() machinery
# --------------------------------------------------------------------------- #
def test_hypothesis_task_satisfies_the_task_contract():
    dec = _dec()
    hyps = B.diffuse(N_CELLS).hypotheses(64, np.random.default_rng(0))
    task, b = build_task(hyps, dec)

    assert task.n_states() == 64
    assert abs(b.weights.sum() - 1.0) < 1e-12

    family = dec.goal_family(2, 4, seed=0)
    comp = compress(b, family, tol=0.0)

    # Modes partition the support exactly (the invariant test_compression.py
    # asserts for the Gate B tasks), so we are on the shared code path.
    members = sorted(k for m in comp.modes for k in m.members)
    assert members == list(range(64))
    assert abs(sum(m.weight for m in comp.modes) - 1.0) < 1e-12
    assert all(m.rep in m.members for m in comp.modes)
    assert comp.K == 64 and 1 <= comp.M <= 64


def test_compress_charges_the_signature_build():
    """`C_comp` must be the real O(K |G| |A|) reward-eval count, not zero."""
    dec = _dec()
    hyps = B.diffuse(N_CELLS).hypotheses(32, np.random.default_rng(0))
    task, b = build_task(hyps, dec)
    family = dec.goal_family(2, 4, seed=0)
    c = ComputeCounter()
    compress(b, family, tol=0.0, counter=c)
    assert c.reward_evals == 32 * 2 * 4
    assert c.total() >= c.reward_evals


def test_probing_half_of_the_task_interface_is_refused():
    dec = _dec()
    hyps = B.diffuse(N_CELLS).hypotheses(4, np.random.default_rng(0))
    task = HypothesisTask(hyps, dec)
    with pytest.raises(NotImplementedError):
        task.obs_space("anything")
    with pytest.raises(NotImplementedError):
        task.goal_family(2)


def test_width_mismatch_is_rejected():
    hyps = HypothesisSet(np.zeros((4, 8), np.int8), np.ones(4))
    with pytest.raises(ValueError):
        HypothesisTask(hyps, _dec())


# --------------------------------------------------------------------------- #
# The two fatal failure modes are reachable by construction
# --------------------------------------------------------------------------- #
def test_collapsed_belief_gives_M_equals_one():
    """VACUOUS: a confident filter emits one hypothesis K times -> M = 1."""
    dec = _dec()
    model = B.collapsed(N_CELLS, N_MINES, seed=0)
    for n_goals in (1, 2, 4):
        family = dec.goal_family(n_goals, 4, seed=0)
        cell = MZ.measure_cell(model.hypotheses(128, np.random.default_rng(0)),
                               dec, family, task="t")
        assert cell.M == 1
        assert cell.n_distinct == 1


def test_diffuse_belief_reaches_the_analytic_bound():
    """A maximally uncertain filter should saturate min(K, prod_g |g|)."""
    dec = _dec()
    model = B.diffuse(N_CELLS, p=0.5)
    family = dec.goal_family(1, 4, seed=0)      # bound = 4
    cell = MZ.measure_cell(model.hypotheses(256, np.random.default_rng(1)),
                           dec, family, task="t")
    assert cell.bound == 4
    assert cell.M == 4


def test_M_never_exceeds_the_analytic_bound():
    """M <= min(K, prod_g |g|) is the claim `decision` makes; check it hard."""
    dec = _dec()
    for p in (0.1, 0.3, 0.5, 0.8):
        model = B.diffuse(N_CELLS, p=p)
        for n_goals in (1, 2, 3, 4):
            family = dec.goal_family(n_goals, 4, seed=0)
            for K in (8, 32, 128):
                cell = MZ.measure_cell(model.hypotheses(K, np.random.default_rng(K)),
                                       dec, family, task="t")
                assert cell.M <= cell.bound, (p, n_goals, K, cell)
                assert cell.M <= K


def test_ratio_falls_as_K_grows_at_fixed_goal_family():
    """The Gate C0 saturation story, on the learned-hypothesis side."""
    dec = _dec()
    model = B.diffuse(N_CELLS, p=0.5)
    family = dec.goal_family(2, 4, seed=0)      # bound = 16, flat in K
    ratios = []
    for K in (16, 64, 256):
        cell = MZ.measure_cell(model.hypotheses(K, np.random.default_rng(K)),
                               dec, family, task="t")
        ratios.append(cell.ratio)
    assert ratios[0] > ratios[-1]
    assert ratios[-1] < 0.1


def test_ratio_rises_with_goal_richness():
    """Deliverable (c): M/K must degrade as the goal family gets richer, and
    the analytic bound says where -- prod_g |g| crosses K between |G|=3 and 4."""
    dec = _dec()
    model = B.diffuse(N_CELLS, p=0.5)
    ratios = []
    for n_goals in (1, 2, 3, 4):
        family = dec.goal_family(n_goals, 4, seed=0)
        cell = MZ.measure_cell(model.hypotheses(128, np.random.default_rng(7)),
                               dec, family, task="t")
        ratios.append(cell.ratio)
    assert ratios == sorted(ratios)
    assert ratios[0] <= MZ.PREREG["MAX_RATIO"] < ratios[-1]


# --------------------------------------------------------------------------- #
# Belief models
# --------------------------------------------------------------------------- #
def test_ess_detects_collapse():
    """The disclosed correction to Gate C1 S5: RAW weight ESS is maximal for a
    collapsed sampled belief (K identical draws all share one probability), so
    the gate is stated on the DISTINCT-hypothesis ESS instead."""
    rng = np.random.default_rng(0)
    coll = B.collapsed(N_CELLS, N_MINES).hypotheses(128, rng)
    assert coll.n_distinct == 1
    assert coll.ess_weights == pytest.approx(128.0)     # the trap
    assert coll.ess < 2.0                                # the statistic we gate on
    assert B.diffuse(N_CELLS, 0.5).hypotheses(128, rng).ess > 8.0


def test_calibration_error_flags_a_broken_head():
    """A VACUOUS reading must never be reported without this number: an
    uncalibrated factored head looks collapsed for the wrong reason."""
    good = B.partially_informed(N_CELLS, N_MINES, n_known=11, seed=0)
    assert B.calibration_error(good, N_MINES) < 0.2
    # the unrenormalised version this replaced: marginals sum to ~0.6, not 2
    bad = B.FactoredBernoulliBelief(np.full(N_CELLS, N_MINES / N_CELLS))
    assert B.calibration_error(bad, N_MINES) < 1e-9      # full board is fine
    partial_bad = np.full(N_CELLS, N_MINES / N_CELLS)
    partial_bad[:11] = 1e-6
    assert B.calibration_error(
        B.FactoredBernoulliBelief(partial_bad), N_MINES) > 1.0


def test_ess_is_bounded_by_the_distinct_count():
    for model in (B.collapsed(N_CELLS, N_MINES), B.diffuse(N_CELLS, 0.5),
                  B.partially_informed(N_CELLS, N_MINES, 8)):
        hs = model.hypotheses(64, np.random.default_rng(11))
        assert 1.0 - 1e-9 <= hs.ess <= hs.n_distinct + 1e-9


def test_uniform_weighting_is_the_ess_ceiling():
    """The 'model' weighting can only lower ESS -- the conservative choice."""
    probs = np.full(N_CELLS, 0.2)
    m = B.FactoredBernoulliBelief(probs, weighting="model")
    u = B.FactoredBernoulliBelief(probs, weighting="uniform")
    hu = u.hypotheses(64, np.random.default_rng(3))
    hm = m.hypotheses(64, np.random.default_rng(3))
    assert hu.ess_weights == pytest.approx(64.0)
    assert hm.ess <= hu.ess + 1e-9


def test_beam_mode_is_deterministic_and_ordered():
    model = B.FactoredBernoulliBelief(
        np.linspace(0.05, 0.6, N_CELLS), mode="beam")
    a = model.hypotheses(16)
    b = model.hypotheses(16)
    assert np.array_equal(a.params, b.params)
    assert a.n_distinct == 16                       # top-K are all distinct
    lp = B._log_prob(a.params, model.probs)
    assert np.all(np.diff(lp) <= 1e-9)              # sorted by probability


def test_ensemble_pools_exactly_K():
    members = [B.diffuse(N_CELLS, p) for p in (0.2, 0.4, 0.6)]
    hs = B.EnsembleBelief(members).hypotheses(64, np.random.default_rng(0))
    assert hs.K == 64
    assert abs(hs.weights.sum() - 1.0) < 1e-12


def test_exact_enumeration_matches_the_verified_cardinality():
    """MineSweeperEasy: C(16, 2) = 120, confirmed empirically on the substrate
    (2000 resets -> exactly 120 distinct mine grids)."""
    m = B.ExactEnumerationBelief(16, 2)
    assert len(m.support()) == math.comb(16, 2) == 120
    assert E.TASKS["MineSweeperEasy"].hidden_cardinality == 120


def test_conditioning_shrinks_the_exact_posterior():
    full = B.ExactEnumerationBelief(16, 2)
    cond = B.ExactEnumerationBelief(16, 2, revealed={0: 0, 1: 0, 2: 0})
    assert len(cond.support()) == math.comb(13, 2)
    assert len(cond.support()) < len(full.support())
    assert all(z[0] == 0 and z[1] == 0 and z[2] == 0 for z in cond.support())


def test_exact_posterior_is_a_valid_hypothesis_set():
    hs = B.ExactEnumerationBelief(16, 2).hypotheses(1000)
    assert hs.K == 120
    assert abs(hs.weights.sum() - 1.0) < 1e-12
    assert hs.ess == pytest.approx(120.0)


# --------------------------------------------------------------------------- #
# Decision layer
# --------------------------------------------------------------------------- #
def test_goal_family_regions_are_disjoint_and_stable():
    dec = _dec()
    fam = dec.goal_family(4, 4, seed=0)
    flat = [c for g in fam for c in g]
    assert len(flat) == len(set(flat)) == 16
    assert dec.goal_family(4, 4, seed=0) == fam        # same seed -> same regions


def test_goal_family_refuses_to_overfill_the_board():
    with pytest.raises(ValueError):
        _dec().goal_family(6, 4, seed=0)               # 24 cells > 16


def test_preferred_action_is_the_best_matching_cell():
    dec = _dec(target=SAFE)
    z = np.zeros(N_CELLS, np.int8)
    goal = dec.goal_family(1, 4, seed=0)[0]
    z[list(goal)] = 1
    z[goal[2]] = 0                                     # exactly one safe cell
    assert dec.preferred_action(z, goal) == goal[2]
    # with several safe cells, utility ranks them
    z[list(goal)] = 0
    best = max(goal, key=lambda c: dec.utility[c])
    assert dec.preferred_action(z, goal) == best


def test_target_bit_flips_the_semantics():
    z = np.zeros(N_CELLS, np.int8)
    goal = _dec().goal_family(1, 4, seed=0)[0]
    z[goal[1]] = 1
    assert _dec(TARGET).preferred_action(z, goal) == goal[1]
    assert _dec(SAFE).preferred_action(z, goal) != goal[1]


def test_oracle_signature_count_is_a_task_side_control():
    """Needs no belief model at all -- Gate C1 STOP S5's first check."""
    dec = _dec()
    sup = B.ExactEnumerationBelief(16, 2).support()
    n1 = MZ.oracle_signature_count(sup, dec, dec.goal_family(1, 4, seed=0))
    n2 = MZ.oracle_signature_count(sup, dec, dec.goal_family(2, 4, seed=0))
    assert 1 < n1 <= 4
    assert n2 >= n1                                     # richer family, >= sigs
    assert n2 <= 16


# --------------------------------------------------------------------------- #
# Pre-registration is enforced by a test: changing a threshold breaks the build
# --------------------------------------------------------------------------- #
def test_prereg_thresholds_are_frozen():
    assert MZ.PREREG == {
        "K_REF": 128,
        "MIN_MODES": 3,
        "MIN_ESS": 8.0,
        "MIN_ORACLE_SIGS": 3,
        "MAX_RATIO": 0.25,
        "MIN_K_OVER_BOUND": 4.0,
        "BOUND_SLACK": 1.25,
        "MAX_BELIEF_SHARE": 0.80,
        "STOP_VACUOUS_MODES": 2,
        "STOP_VACUOUS_ESS": 2.0,
        "STOP_NOCOMPRESS_RATIO": 0.5,
    }


# --------------------------------------------------------------------------- #
# The verdict function
# --------------------------------------------------------------------------- #
def _summary(task="t", M=10.0, ratio=0.08, ess=64.0, bound=16.0, osig=6,
             key=(2, 4, 0.5)):
    s = MZ.TaskSummary(task=task, K_ref=128)
    s.median_M[key] = M
    s.median_ratio[key] = ratio
    s.median_ess[key] = ess
    s.median_bound[key] = bound
    s.oracle_sigs[key] = osig
    return s


def _rows(task="t", M=10, bound=16, K=128, key=(2, 4, 0.5)):
    return [MZ.Cell(task=task, model="m", K=K, n_goals=key[0],
                    region_size=key[1], seed=0, M=M, ratio=M / K, bound=bound,
                    K_over_bound=K / bound, ess=64.0, n_distinct=K, ops=1,
                    phase=key[2])]


def _cost(share=0.3):
    return MZ.CostSplit(c_belief=share, c_search=1.0 - share)


def test_verdict_pass():
    v, why = MZ.verdict([_summary()], _rows(), _cost(0.3))
    assert v == MZ.VERDICT_PASS
    assert "witness cell" in why[0]


def test_verdict_stop_vacuous_on_modes():
    v, why = MZ.verdict([_summary(M=1.0, ratio=0.008)], _rows(M=1), _cost(0.3))
    assert v == MZ.VERDICT_STOP_VACUOUS
    assert "S5" in why[0]


def test_verdict_stop_vacuous_on_ess():
    v, why = MZ.verdict([_summary(M=10.0, ess=1.5)], _rows(), _cost(0.3))
    assert v == MZ.VERDICT_STOP_VACUOUS
    assert "ESS" in why[0]


def test_verdict_stop_no_compression():
    v, why = MZ.verdict([_summary(M=100.0, ratio=0.78, bound=128.0)],
                        _rows(M=100, bound=128), _cost(0.3))
    assert v == MZ.VERDICT_STOP_NOCOMPRESS
    assert "S1" in why[0]


def test_verdict_stop_s3():
    v, why = MZ.verdict([_summary()], _rows(), _cost(0.9), cost_deep=_cost(0.9))
    assert v == MZ.VERDICT_STOP_S3
    assert "S3" in why[0]


def test_s3_does_not_fire_if_the_deep_horizon_recovers():
    """S3 is stated as 'dominates at the operating point AND still at max H'."""
    v, _ = MZ.verdict([_summary()], _rows(), _cost(0.9), cost_deep=_cost(0.4))
    assert v != MZ.VERDICT_STOP_S3


def test_verdict_amber_when_gates_split_across_cells():
    """A on one goal family and B on another must NOT pass."""
    s = MZ.TaskSummary(task="t", K_ref=128)
    # diverse but incompressible
    s.median_M[(4, 4, 0.5)] = 60.0
    s.median_ratio[(4, 4, 0.5)] = 0.47
    s.median_ess[(4, 4, 0.5)] = 64.0
    s.median_bound[(4, 4, 0.5)] = 128.0
    s.oracle_sigs[(4, 4, 0.5)] = 15
    # compressible but nearly vacuous
    s.median_M[(1, 4, 0.5)] = 2.0
    s.median_ratio[(1, 4, 0.5)] = 0.016
    s.median_ess[(1, 4, 0.5)] = 64.0
    s.median_bound[(1, 4, 0.5)] = 4.0
    s.oracle_sigs[(1, 4, 0.5)] = 3
    rows = _rows(M=60, bound=128, key=(4, 4, 0.5)) + _rows(M=2, bound=4, key=(1, 4, 0.5))
    v, _ = MZ.verdict([s], rows, _cost(0.3))
    assert v == MZ.VERDICT_AMBER


def test_verdict_amber_when_regime_absent():
    """K/bound < 4 everywhere -> gate C fails even though A and B hold."""
    v, _ = MZ.verdict([_summary(M=40.0, ratio=0.31, bound=64.0)],
                      _rows(M=40, bound=64), _cost(0.3))
    assert v == MZ.VERDICT_AMBER


def test_cost_split_share():
    assert MZ.CostSplit(3.0, 1.0).belief_share == pytest.approx(0.75)
    assert MZ.CostSplit(0.0, 0.0).belief_share == 1.0


def test_expectimax_node_count_matches_the_planner_recursion():
    assert MZ.expectimax_nodes(0, 5) == 1
    assert MZ.expectimax_nodes(1, 5) == 6
    assert MZ.expectimax_nodes(2, 5) == 31
    assert MZ.expectimax_nodes(2, 3, n_probes=2) == 1 + 6 + 36


def test_search_flops_charges_the_signature_build_and_the_tree():
    kw = dict(K=128, M=16, n_goals=2, n_actions=4, H=1, obs_branch=5, q_flops=1.0)
    comp = MZ.search_flops(**kw)
    assert comp == 128 * 2 * 4 + MZ.expectimax_nodes(1, 5) * 16 * 4
    full = MZ.search_flops(compressed=False, **kw)
    assert full == MZ.expectimax_nodes(1, 5) * 128 * 4
    # deeper trees are where the compressed planner wins (Gate C0 S3b)
    deep = dict(kw, H=4)
    assert (MZ.search_flops(compressed=False, **deep)
            / MZ.search_flops(**deep)) > (full / comp)


def test_belief_flops_amortized_vs_particle_filter_scaling():
    """The two architectures sit on opposite sides of S3 -- P0 measures both."""
    amo = [MZ.belief_flops(K, 5e7, 1e3, amortized=True) for K in (8, 128)]
    pf = [MZ.belief_flops(K, 5e7, 1e3, amortized=False) for K in (8, 128)]
    assert amo[1] / amo[0] < 1.01          # nearly flat in K
    assert pf[1] / pf[0] == pytest.approx(16.0)   # linear in K


def test_s3_verdict_is_driven_by_the_q_head_cost():
    """A cheap analytic reward makes the belief dominate; a learned Q head
    (the Gate C1 §2.2 design) flips it.  This is the whole content of S3."""
    common = dict(K=128, M=16, n_goals=2, n_actions=4, H=3, obs_branch=5)
    cb = MZ.belief_flops(128, 5e7, 1e3, amortized=True)
    cheap = MZ.CostSplit(cb, MZ.search_flops(q_flops=10.0, **common))
    learned = MZ.CostSplit(cb, MZ.search_flops(q_flops=1e6, **common))
    assert cheap.belief_share > MZ.PREREG["MAX_BELIEF_SHARE"]
    assert learned.belief_share < MZ.PREREG["MAX_BELIEF_SHARE"]


# --------------------------------------------------------------------------- #
# End-to-end: the shipped entry point
# --------------------------------------------------------------------------- #
def test_dry_run_end_to_end_discriminates():
    from belief_compression.p0.run_p0 import dry_run

    res = dry_run(out_dir=None, n_seeds=2)
    assert res["collapsed"]["verdict"] == MZ.VERDICT_STOP_VACUOUS
    assert res["diffuse"]["verdict"] in (MZ.VERDICT_PASS, MZ.VERDICT_AMBER)
    assert res["partial"]["verdict"] in (MZ.VERDICT_PASS, MZ.VERDICT_AMBER)
    for r in res.values():
        assert r["n_rows"] > 0


# --------------------------------------------------------------------------- #
# Substrate facts (the numbers the design doc quotes)
# --------------------------------------------------------------------------- #
def test_task_spec_cardinalities_are_exact_combinatorics():
    for name, spec in E.TASKS.items():
        assert spec.hidden_cardinality == math.comb(spec.n_cells, spec.n_positive), name
    assert E.TASKS["MineSweeperEasy"].enumerable
    assert not E.TASKS["BattleShipEasy"].enumerable
    assert E.TASKS["MineSweeperEasy"].target_bit == SAFE
    assert E.TASKS["BattleShipEasy"].target_bit == TARGET


@pytest.mark.skipif(not E.available(), reason="popgym_arcade not installed here")
def test_substrate_makes_and_steps():  # pragma: no cover - substrate-dependent
    import jax
    env, ps = E.make("MineSweeperEasy")
    o, st = env.reset(jax.random.PRNGKey(0), ps)
    assert o.shape == (128, 128, 3)
    o2, st2, r, d, _ = env.step(jax.random.PRNGKey(1), st, 4, ps)
    z = E.oracle_params(st2, "MineSweeperEasy")
    assert z.shape == (16,) and int(z.sum()) <= 2
