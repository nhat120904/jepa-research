"""Offline correctness tests for the curvature diagnostic numerics.

These test the identities the protocol's conclusions rest on.  A sign or
normalization error in any of them would silently produce a publishable-looking
but wrong number, so they run before any measurement.

Runs under pytest, and also standalone (`python tests/test_core.py`) so the
compute nodes need no extra dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from action_curvature_h0.core import (  # noqa: E402
    CEM_LOCAL_SOURCE,
    argmin_index,
    clip_to_bounds,
    make_feasible,
    ordinal_agreement,
    triplet_shape,
    VALLEY,
    PEAK,
    INCREASING,
    DECREASING,
    CROSS_MODE,
    SAME_MODE_CONTACT,
    SAME_MODE_NON_CONTACT,
    ContactTrace,
    Triplet,
    analyze_cost,
    analyze_triplet,
    classify_contact_mode,
    clip_validity,
    cosine_straightness,
    cost_second_difference,
    draw_unit_directions,
    fit_scaling_exponent,
    normalized_curvature,
    radial_angular_split,
    repeat_floor,
    scale_direction,
)

RNG = np.random.default_rng(20260824)


def _triplet(f, a, d):
    return Triplet(minus=f(a - d), center=f(a), plus=f(a + d))


def _close(a, b, rtol=1e-9, atol=1e-12):
    assert np.isclose(a, b, rtol=rtol, atol=atol), f"{a} != {b}"


# --------------------------------------------------------------------------
# The exact decomposition the two Stage-2 arms are built on
# --------------------------------------------------------------------------


def test_radial_angular_identity_is_exact():
    for dim in (1, 3, 16):
        for _ in range(200):
            v_minus, v_plus = RNG.normal(size=dim), RNG.normal(size=dim)
            radial, angular = radial_angular_split(v_minus, v_plus)
            _close(radial + angular, float(np.linalg.norm(v_plus - v_minus) ** 2))


def test_angular_term_equals_as_loss_times_scale():
    v_minus, v_plus = RNG.normal(size=8), RNG.normal(size=8)
    _, angular = radial_angular_split(v_minus, v_plus)
    as_loss = 1.0 - cosine_straightness(v_minus, v_plus)
    _close(angular, 2.0 * np.linalg.norm(v_minus) * np.linalg.norm(v_plus) * as_loss)


def test_pure_radial_has_zero_angular_fraction():
    u = RNG.normal(size=5)
    u /= np.linalg.norm(u)
    radial, angular = radial_angular_split(u, 3.0 * u)
    _close(angular, 0.0, atol=1e-12)
    _close(radial, 4.0)


def test_cosine_loss_is_blind_to_pure_radial_curvature():
    u = RNG.normal(size=5)
    u /= np.linalg.norm(u)
    _close(cosine_straightness(u, 100.0 * u), 1.0)
    t = Triplet(minus=np.zeros(5), center=u, plus=u + 100.0 * u)
    assert normalized_curvature(t) > 0.9


# --------------------------------------------------------------------------
# E_K is the curvature of the ERROR map; E_J covers its blind spot
# --------------------------------------------------------------------------


def test_affine_prediction_error_cancels_in_e_k_but_shows_in_e_j():
    """The headline property of E_K is also its blind spot.

    An affine error is invisible to E_K by construction, yet an affine Jacobian
    error can reorder the planner's candidates completely.  E_J must see it,
    otherwise nothing can be attributed to curvature.
    """
    A = RNG.normal(size=(6, 4))
    b = RNG.normal(size=6)
    curved = lambda a: np.concatenate([np.sin(a) ** 2, a[:2] ** 3])  # noqa: E731
    a, d = RNG.normal(size=4), 0.05 * RNG.normal(size=4)

    true = _triplet(curved, a, d)
    model = _triplet(lambda x: curved(x) + b + A @ x, a, d)

    report = analyze_triplet(model, true, delta_norm=float(np.linalg.norm(d)))
    assert report.e_k < 1e-9, report.e_k
    assert report.e_j > 1e-3, report.e_j
    assert report.k_true > 0.0


def test_e_j_is_zero_when_first_order_behaviour_matches():
    curved = lambda a: np.array([np.sin(a[0]), a[0] ** 2])  # noqa: E731
    a, d = np.array([0.3]), np.array([0.05])
    true = _triplet(curved, a, d)
    report = analyze_triplet(true, true, delta_norm=0.05)
    _close(report.e_j, 0.0, atol=1e-12)
    _close(report.e_k, 0.0, atol=1e-12)


def test_path_length_denominator_survives_a_cancelling_span():
    """E_K's denominator can vanish away from kinks; the companions cannot.

    Here the realized map returns to where it started, so ||Psi+ - Psi-|| == 0
    while the two one-sided displacements are large and the second difference is
    real.  E_K explodes; e_k_absolute and e_k_pathlen stay finite and small.
    """
    true = Triplet(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    model = Triplet(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.02]))
    r = analyze_triplet(model, true, delta_norm=0.1)
    assert true.span == 0.0
    assert r.e_k > 1e9
    assert np.isfinite(r.e_k_absolute) and r.e_k_absolute < 10.0
    assert np.isfinite(r.e_k_pathlen) and r.e_k_pathlen < 1.0


def test_e_k_detects_opposite_curvature_that_g_k_misses():
    true = Triplet(np.array([-0.1, 0.01]), np.zeros(2), np.array([0.1, 0.01]))
    model = Triplet(np.array([-0.1, -0.01]), np.zeros(2), np.array([0.1, -0.01]))
    report = analyze_triplet(model, true, delta_norm=0.1)
    _close(report.g_k, 0.0, atol=1e-9)
    assert report.e_k > 0.1
    assert report.a_k < -0.99


def test_sensitivity_ratio_identity():
    a, d = RNG.normal(size=3), 0.05 * RNG.normal(size=3)
    true = _triplet(np.tanh, a, d)
    model = _triplet(lambda x: 0.4 * np.tanh(x) + 0.01 * x**2, a, d)
    r = analyze_triplet(model, true, delta_norm=float(np.linalg.norm(d)))
    _close(r.k_model_self / r.k_model_true, r.s_ratio)


def test_linear_map_has_zero_curvature():
    A = RNG.normal(size=(5, 3))
    a, d = RNG.normal(size=3), 0.1 * RNG.normal(size=3)
    assert normalized_curvature(_triplet(lambda x: A @ x, a, d)) < 1e-12


def test_different_charts_are_rejected():
    try:
        analyze_triplet(
            Triplet(np.zeros(3), np.zeros(3), np.zeros(3)),
            Triplet(np.zeros(4), np.zeros(4), np.zeros(4)),
            delta_norm=0.1,
        )
    except ValueError:
        return
    raise AssertionError("mismatched charts must be rejected")


# --------------------------------------------------------------------------
# The scalar the planner actually consumes
# --------------------------------------------------------------------------


def test_cost_second_difference_is_an_exact_identity():
    """D2 C == 2<r0, v+ - v-> + ||v+||^2 + ||v-||^2, to machine precision."""
    for _ in range(200):
        dim = 7
        t = Triplet(RNG.normal(size=dim), RNG.normal(size=dim), RNG.normal(size=dim))
        goal = RNG.normal(size=dim)
        cc = cost_second_difference(t, goal)
        direct = (
            float(np.sum((t.plus - goal) ** 2))
            - 2.0 * float(np.sum((t.center - goal) ** 2))
            + float(np.sum((t.minus - goal) ** 2))
        )
        _close(cc.d2c, direct, rtol=1e-9, atol=1e-9)
        _close(cc.residual_term + cc.gn_term, direct, rtol=1e-9, atol=1e-9)


def test_local_concavity_iff_ratio_below_minus_one():
    for _ in range(300):
        dim = 4
        t = Triplet(RNG.normal(size=dim), RNG.normal(size=dim), RNG.normal(size=dim))
        cc = cost_second_difference(t, RNG.normal(size=dim) * 3.0)
        assert (cc.d2c < 0.0) == (cc.ratio < -1.0), cc


def test_linear_map_still_has_positive_cost_curvature():
    """||D2 Phi|| = 0 does not mean the cost is flat: the GN term survives.

    This is why the vector curvature of Phi cannot be the whole story for a
    planner that only ever sees the scalar cost.
    """
    A = RNG.normal(size=(5, 3))
    a, d = RNG.normal(size=3), 0.1 * RNG.normal(size=3)
    t = _triplet(lambda x: A @ x, a, d)
    cc = cost_second_difference(t, RNG.normal(size=5))
    _close(np.linalg.norm(t.d2), 0.0, atol=1e-12)
    _close(cc.residual_term, 0.0, atol=1e-9)
    assert cc.gn_term > 0.0 and cc.d2c > 0.0


def test_analyze_cost_reports_both_maps_and_their_gap():
    dim = 6
    goal = RNG.normal(size=dim)
    true = Triplet(RNG.normal(size=dim), RNG.normal(size=dim), RNG.normal(size=dim))
    model = Triplet(true.minus + 0.01, true.center, true.plus - 0.01)
    rep = analyze_cost(model, true, goal)
    _close(rep.d2c_mismatch, rep.model_d2c - rep.true_d2c, rtol=1e-9, atol=1e-9)
    assert isinstance(rep.model_locally_concave, bool)


# --------------------------------------------------------------------------
# Fixed base direction across the scale sweep
# --------------------------------------------------------------------------


def test_base_directions_are_unit_norm():
    d = draw_unit_directions(np.random.default_rng(0), 5, (3, 25))
    for k in range(5):
        _close(float(np.linalg.norm(d[k])), 1.0)


def test_same_base_direction_stays_parallel_across_scales():
    """The fix for the voided scaling exponent: one direction, many scales."""
    base = draw_unit_directions(np.random.default_rng(1), 1, (3, 10))[0]
    raw_range = np.full((3, 10), 2.0)
    scaler_scale = np.full((3, 10), 0.5)
    sigmas = [0.025, 0.05, 0.10, 0.20]
    deltas = [
        scale_direction(base, s, source="dataset", raw_range_chunk=raw_range,
                        scaler_scale_chunk=scaler_scale)
        for s in sigmas
    ]
    first = deltas[0].reshape(-1)
    for s, delta in zip(sigmas, deltas):
        flat = delta.reshape(-1)
        cos = float(np.dot(first, flat) / (np.linalg.norm(first) * np.linalg.norm(flat)))
        _close(cos, 1.0)
        _close(float(np.linalg.norm(flat)) / (s / sigmas[0]), float(np.linalg.norm(first)))


def test_clip_to_bounds_is_a_noop_inside_the_box():
    a = np.array([[0.1, -0.2], [0.3, 0.0]])
    out, info = clip_to_bounds(a, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert np.array_equal(out, a)
    _close(info["clipped_fraction"], 0.0)
    _close(info["max_clip_raw"], 0.0)


def test_clip_to_bounds_reports_what_it_moved():
    a = np.array([[1.5, -0.2], [0.3, -2.0]])
    out, info = clip_to_bounds(a, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert np.array_equal(out, np.array([[1.0, -0.2], [0.3, -1.0]]))
    _close(info["clipped_fraction"], 0.5)
    _close(info["max_clip_raw"], 1.0)


def test_make_feasible_respects_every_component_cap():
    base = draw_unit_directions(np.random.default_rng(7), 1, (30,))[0]
    cap = np.abs(RNG.normal(size=30)) * 0.05 + 0.001
    out, info = make_feasible(base, cap)
    assert np.all(np.abs(out) <= cap + 1e-12)
    assert info["feasible"] == 1.0


def test_make_feasible_preserves_direction_exactly():
    """The scaling fit needs the same direction at every scale, so the shrink
    must be a single global factor, never per-component clipping."""
    base = draw_unit_directions(np.random.default_rng(8), 1, (20,))[0]
    cap = np.full(20, 0.02)
    out, _ = make_feasible(base, cap)
    live = np.abs(base) > 1e-12
    ratios = out[live] / base[live]
    assert np.allclose(ratios, ratios[0]), "shrink was not a single global factor"


def test_make_feasible_masks_saturated_components():
    base = np.ones(4) / 2.0
    cap = np.array([0.5, 0.0, 0.5, 0.0])
    out, info = make_feasible(base, cap)
    assert out[1] == 0.0 and out[3] == 0.0
    _close(info["masked_fraction"], 0.5)
    assert np.linalg.norm(out) > 0.0


def test_make_feasible_reports_a_fully_saturated_chunk():
    out, info = make_feasible(np.ones(5), np.zeros(5))
    assert info["feasible"] == 0.0
    assert np.linalg.norm(out) == 0.0


def test_cem_local_scales_with_the_proposal_covariance():
    base = draw_unit_directions(np.random.default_rng(2), 1, (3, 10))[0]
    std = np.abs(RNG.normal(size=(3, 10))) + 0.1
    a = scale_direction(base, 0.1, source=CEM_LOCAL_SOURCE, proposal_std=std,
                        alpha=1.0, sigma_max=0.2)
    b = scale_direction(base, 0.2, source=CEM_LOCAL_SOURCE, proposal_std=std,
                        alpha=1.0, sigma_max=0.2)
    assert np.allclose(2.0 * a, b)


# --------------------------------------------------------------------------
# Scaling exponent as a label-free mode detector
# --------------------------------------------------------------------------


def test_scaling_exponent_recovers_regime():
    cases = [
        (lambda x: np.array([x**2, 0.5 * x**2]), 2.0),
        (lambda x: np.array([abs(x), 0.0]), 1.0),
        (lambda x: np.array([float(x > 0), 0.0]), 0.0),
    ]
    scales = [0.025, 0.05, 0.10, 0.20]
    for f, expected in cases:
        d2 = [float(np.linalg.norm(f(s) - 2.0 * f(0.0) + f(-s))) for s in scales]
        fit = fit_scaling_exponent(scales, d2, floor=0.0)
        assert abs(fit.alpha - expected) < 1e-6, (expected, fit)
        assert fit.r_squared > 0.999, fit
        assert fit.n_used == 4


def test_normalized_curvature_diverges_at_a_symmetric_kink():
    kink = lambda x: np.array([abs(x), 0.0])  # noqa: E731
    t = Triplet(kink(-0.1), kink(0.0), kink(0.1))
    assert t.span < 1e-12
    assert np.linalg.norm(t.d2) > 0.1
    assert normalized_curvature(t) > 1e9


def test_floor_excludes_scales_and_reports_them():
    fit = fit_scaling_exponent([0.025, 0.05, 0.10, 0.20],
                               [1e-9, 1e-9, 0.01, 0.04], floor=1e-6)
    assert fit.n_used == 2 and fit.excluded_below_floor == 2


def test_too_few_points_above_floor_returns_nan():
    fit = fit_scaling_exponent([0.1, 0.2], [1e-9, 0.04], floor=1e-6, min_points=3)
    assert np.isnan(fit.alpha)


def test_repeat_floor_is_zero_for_a_deterministic_pipeline():
    v = RNG.normal(size=32)
    assert repeat_floor([v, v.copy(), v.copy()]) == 0.0


# --------------------------------------------------------------------------
# Validity and contact stratification
# --------------------------------------------------------------------------


def test_clip_validity_rejects_either_arm():
    a = np.array([0.9, 0.0])
    assert clip_validity(a, np.array([0.05, 0.05]), -1.0, 1.0)
    assert not clip_validity(a, np.array([0.2, 0.0]), -1.0, 1.0)
    assert not clip_validity(np.array([-0.95, 0.0]), np.array([0.1, 0.0]), -1.0, 1.0)


def test_clip_validity_tolerates_one_ulp_roundoff():
    """Job 46139: a centre clamped exactly onto a bound is not bit-identical
    after a normalize/denormalize round trip.  A strict comparison would reject
    every triplet built on it at every scale, for a non-violation of 1 ULP."""
    eps = np.finfo(np.float64).eps
    a = np.array([1.0 - eps, 0.0])
    assert clip_validity(a, np.array([0.0, 0.05]), -1.0, 1.0)


def test_clip_validity_still_rejects_a_real_violation():
    a = np.array([1.05, 0.0])
    assert not clip_validity(a, np.array([0.0, 0.0]), -1.0, 1.0)


def test_asymmetric_spacing_injects_a_first_order_term():
    f = lambda x: np.array([3.0 * x])  # noqa: E731
    assert np.linalg.norm(Triplet(f(-0.1), f(0.0), f(0.1)).d2) < 1e-15
    assert np.linalg.norm(Triplet(f(-0.1), f(0.0), f(0.02)).d2) > 0.2


def _trace(robot, table=None):
    table = [True] * len(robot) if table is None else table
    return ContactTrace(robot=tuple(robot), table=tuple(table))


def test_table_contact_does_not_collapse_the_non_contact_stratum():
    """The OGBench-Cube fix: the cube rests on the table for every step.

    A mode label built from "is anything in contact" would put every sample in
    the contact stratum and leave same_mode_non_contact empty.
    """
    resting = _trace([False, False, False], [True, True, True])
    assert classify_contact_mode(resting, resting, resting) == SAME_MODE_NON_CONTACT


def test_robot_contact_drives_the_contact_stratum():
    touching = _trace([False, True, True])
    assert classify_contact_mode(touching, touching, touching) == SAME_MODE_CONTACT


def test_perturbation_that_changes_whether_contact_happens_is_cross_mode():
    assert classify_contact_mode(
        _trace([False, False, False]), _trace([False, False, False]),
        _trace([False, False, True]),
    ) == CROSS_MODE


def test_same_bodies_different_onset_is_cross_mode():
    """Union over the rollout would call these identical; onset time separates them."""
    early = _trace([True, True, True, True])
    late = _trace([False, False, False, True])
    assert classify_contact_mode(early, early, late, onset_tolerance=1) == CROSS_MODE
    assert classify_contact_mode(early, early, early, onset_tolerance=1) == SAME_MODE_CONTACT


def test_onset_tolerance_is_respected():
    a = _trace([False, True, True])
    b = _trace([False, False, True])
    assert classify_contact_mode(a, a, b, onset_tolerance=1) == SAME_MODE_CONTACT
    assert classify_contact_mode(a, a, b, onset_tolerance=0) == CROSS_MODE


def test_triplet_shape_covers_the_four_cases():
    assert triplet_shape(2.0, 1.0, 3.0) == VALLEY
    assert triplet_shape(1.0, 3.0, 2.0) == PEAK
    assert triplet_shape(1.0, 2.0, 3.0) == INCREASING
    assert triplet_shape(3.0, 2.0, 1.0) == DECREASING


def test_shape_is_invariant_under_monotone_rescaling():
    """The property that lets latent cost and metric cost be compared at all."""
    base = (2.0, 1.0, 3.0)
    for g in (lambda x: 5.0 * x + 7.0, lambda x: x**3, np.exp, np.sqrt):
        assert triplet_shape(*[float(g(v)) for v in base]) == triplet_shape(*base)


def test_false_valley_is_flagged():
    # model sees a local minimum at the centre; physics is monotone increasing
    out = ordinal_agreement((2.0, 1.0, 3.0), (1.0, 2.0, 3.0))
    assert out["false_valley"] and not out["missed_valley"]
    assert not out["shape_agree"]
    assert out["model_argmin"] == 1 and out["physical_argmin"] == 0


def test_missed_valley_is_flagged():
    out = ordinal_agreement((1.0, 2.0, 3.0), (2.0, 1.0, 3.0))
    assert out["missed_valley"] and not out["false_valley"]


def test_full_agreement():
    out = ordinal_agreement((2.0, 1.0, 3.0), (20.0, 10.0, 30.0))
    assert out["shape_agree"] and out["argmin_agree"]
    assert not out["false_valley"] and not out["missed_valley"]


def test_argmin_index_matches_numpy():
    for _ in range(50):
        v = RNG.normal(size=3)
        assert argmin_index(*v) == int(np.argmin(v))


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"FAIL {name}: {exc}")
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
