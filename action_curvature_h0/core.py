"""Pure numerics for the Action-Space Curvature Mismatch diagnostic.

Implements the quantities locked in ``PROTOCOL.md``.  Everything here is plain
float64 numpy with no simulator, model, or torch dependency, so the whole metric
surface is unit-testable offline and the production path is the tested path.

Naming follows the protocol:

    Phi_H(a) = F^H(E(o_t), a)          model map      (predicted latent)
    Psi_H(a) = E(o_H^sim(s_t, a))      realized map   (encode of true rollout)

Second differences are taken over the symmetric triplet ``(a - d, a, a + d)``.
Asymmetric or clipped triplets are invalid and must be discarded upstream; see
``clip_validity``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

EPS = 1e-12

# Second differences subtract O(1) quantities to recover an O(delta^2) result,
# so every difference is taken in float64 regardless of the caller's dtype.
DTYPE = np.float64


def _vec(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=DTYPE).reshape(-1)
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite component in triplet vector")
    return a


@dataclass(frozen=True)
class Triplet:
    """Three points of one map evaluated at ``a - d``, ``a``, ``a + d``."""

    minus: np.ndarray
    center: np.ndarray
    plus: np.ndarray

    def __post_init__(self) -> None:
        m, c, p = _vec(self.minus), _vec(self.center), _vec(self.plus)
        if not (m.shape == c.shape == p.shape):
            raise ValueError(f"shape mismatch: {m.shape}, {c.shape}, {p.shape}")
        object.__setattr__(self, "minus", m)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "plus", p)

    @property
    def v_minus(self) -> np.ndarray:
        """Backward first difference ``Phi(a) - Phi(a - d)``."""
        return self.center - self.minus

    @property
    def v_plus(self) -> np.ndarray:
        """Forward first difference ``Phi(a + d) - Phi(a)``."""
        return self.plus - self.center

    @property
    def d2(self) -> np.ndarray:
        """Second difference ``Phi(a + d) - 2 Phi(a) + Phi(a - d)``."""
        return self.v_plus - self.v_minus

    @property
    def span(self) -> float:
        """Central first difference norm ``||Phi(a + d) - Phi(a - d)||``."""
        return float(np.linalg.norm(self.plus - self.minus))


def sensitivity(t: Triplet, delta_norm: float) -> float:
    """``||Phi(a + d) - Phi(a - d)|| / (2 ||d||)``, the protocol's S_sens."""
    if delta_norm <= 0.0:
        raise ValueError("delta_norm must be positive")
    return t.span / (2.0 * delta_norm)


def normalized_curvature(t: Triplet, reference_span: float | None = None) -> float:
    """``||D2|| / (reference_span + eps)``.

    ``reference_span=None`` self-normalizes (``K_model_self``, ``K_true``).
    Passing the realized map's span gives the comparison form ``K_model_true``.
    """
    span = t.span if reference_span is None else float(reference_span)
    return float(np.linalg.norm(t.d2) / (span + EPS))


def radial_angular_split(v_minus: np.ndarray, v_plus: np.ndarray) -> tuple[float, float]:
    """Exact decomposition of ``||v_plus - v_minus||^2`` into radial and angular parts.

        ||D2||^2 = (||v+|| - ||v-||)^2 + 2 ||v+|| ||v-|| (1 - cos(v-, v+))
                   \\_____radial_____/   \\_________angular___________/

    The angular term is ``2 ||v+|| ||v-||`` times the AS loss ``1 - cos``.  A
    mismatch that is almost entirely radial cannot be addressed by a cosine
    regularizer, which is why the split is measured before any training.
    """
    a, b = _vec(v_minus), _vec(v_plus)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    radial = (nb - na) ** 2
    if na <= EPS or nb <= EPS:
        # One displacement vanishes: the angle is undefined and all of the
        # second-difference energy is radial by construction.
        return radial, 0.0
    cos = float(np.dot(a, b) / (na * nb))
    cos = min(1.0, max(-1.0, cos))
    angular = 2.0 * na * nb * (1.0 - cos)
    return radial, angular


def cosine_straightness(v_minus: np.ndarray, v_plus: np.ndarray) -> float:
    """``cos(v-, v+)``; NaN when either displacement vanishes."""
    a, b = _vec(v_minus), _vec(v_plus)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return min(1.0, max(-1.0, float(np.dot(a, b) / (na * nb))))


@dataclass(frozen=True)
class CurvatureReport:
    """Protocol readouts 1-4 and 7 for one (state, action, delta, horizon) sample."""

    k_model_self: float
    k_model_true: float
    k_true: float
    g_k: float
    e_k: float
    e_k_absolute: float
    e_k_pathlen: float
    e_j: float
    a_k: float
    s_model: float
    s_true: float
    s_ratio: float
    model_radial_fraction: float
    model_angular_fraction: float
    error_radial_fraction: float
    error_angular_fraction: float
    d2_model_norm: float
    d2_true_norm: float
    d2_error_norm: float
    true_span: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _fractions(v_minus: np.ndarray, v_plus: np.ndarray) -> tuple[float, float]:
    radial, angular = radial_angular_split(v_minus, v_plus)
    total = radial + angular
    if total <= EPS:
        return float("nan"), float("nan")
    return radial / total, angular / total


def analyze_triplet(model: Triplet, true: Triplet, delta_norm: float) -> CurvatureReport:
    """All per-sample curvature quantities for one model/realized triplet pair.

    ``model`` and ``true`` must be expressed in the SAME latent chart, i.e. the
    same frozen encoder, otherwise their difference is not type-correct.  See
    the chart-dependence section of ``PROTOCOL.md``.
    """
    if model.center.shape != true.center.shape:
        raise ValueError("model and realized triplets must share a latent chart")

    true_span = true.span
    d2_model, d2_true = model.d2, true.d2
    d2_error = d2_model - d2_true

    # The error map's displacements, used for the E_K radial/angular split.
    v_minus_err = model.v_minus - true.v_minus
    v_plus_err = model.v_plus - true.v_plus

    s_model = sensitivity(model, delta_norm)
    s_true = sensitivity(true, delta_norm)

    k_model_self = normalized_curvature(model)
    k_model_true = normalized_curvature(model, reference_span=true_span)
    k_true = normalized_curvature(true)

    nm, nt = float(np.linalg.norm(d2_model)), float(np.linalg.norm(d2_true))
    a_k = float("nan")
    if nm > EPS and nt > EPS:
        a_k = min(1.0, max(-1.0, float(np.dot(d2_model, d2_true) / (nm * nt))))

    model_radial, model_angular = _fractions(model.v_minus, model.v_plus)
    error_radial, error_angular = _fractions(v_minus_err, v_plus_err)

    return CurvatureReport(
        k_model_self=k_model_self,
        k_model_true=k_model_true,
        k_true=k_true,
        g_k=k_model_true - k_true,
        e_k=float(np.linalg.norm(d2_error) / (true_span + EPS)),
        # The true span vanishes not only at a symmetric kink but at any
        # stationary point or null direction of the realized Jacobian, where a
        # tiny second-order error inflates E_K without bound.  Two
        # denominator-free companions are therefore always reported: an
        # absolute curvature in action units, and a two-sided path length that
        # cannot cancel.
        e_k_absolute=float(np.linalg.norm(d2_error) / (delta_norm**2 + EPS)),
        e_k_pathlen=float(
            np.linalg.norm(d2_error)
            / (np.linalg.norm(true.v_minus) + np.linalg.norm(true.v_plus) + EPS)
        ),
        # First-order mismatch.  E_K annihilates every affine error component,
        # which is a feature AND a blind spot: an affine Jacobian error can
        # still reorder the planner's candidates completely.  Nothing may be
        # attributed to curvature without controlling for this term.
        e_j=float(
            np.linalg.norm((model.plus - model.minus) - (true.plus - true.minus))
            / (true_span + EPS)
        ),
        a_k=a_k,
        s_model=s_model,
        s_true=s_true,
        s_ratio=s_true / (s_model + EPS),
        model_radial_fraction=model_radial,
        model_angular_fraction=model_angular,
        error_radial_fraction=error_radial,
        error_angular_fraction=error_angular,
        d2_model_norm=nm,
        d2_true_norm=nt,
        d2_error_norm=float(np.linalg.norm(d2_error)),
        true_span=true_span,
    )


# --------------------------------------------------------------------------
# The scalar the planner actually consumes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CostCurvature:
    """Exact finite-difference decomposition of the goal cost's second difference.

    For ``C(a) = ||Phi(a) - z_g||^2`` with ``r0 = Phi(a) - z_g``:

        D2 C = 2 <r0, v_plus - v_minus> + ||v_plus||^2 + ||v_minus||^2
               \____residual-weighted____/  \_____Gauss-Newton-like_____/

    This is an identity, not an O(d^4) approximation, and it is stated on the
    scalar the planner ranks candidates by -- unlike the vector curvature of
    ``Phi``, whose components orthogonal to the residual barely move the cost.

    ``ratio = residual / gn`` is signed and the whole point: since ``gn >= 0``,
    ``D2 C < 0`` exactly when ``ratio < -1``, i.e. the cost is locally concave
    along this direction and the planner faces genuine non-convexity.
    """

    d2c: float
    residual_term: float
    gn_term: float
    ratio: float


def cost_second_difference(t: Triplet, goal: Any) -> CostCurvature:
    g = _vec(goal)
    if g.shape != t.center.shape:
        raise ValueError("goal must live in the same chart as the triplet")
    r0 = t.center - g
    v_minus, v_plus = t.v_minus, t.v_plus
    residual = 2.0 * float(np.dot(r0, v_plus - v_minus))
    gn = float(np.dot(v_plus, v_plus) + np.dot(v_minus, v_minus))
    return CostCurvature(
        d2c=residual + gn,
        residual_term=residual,
        gn_term=gn,
        ratio=residual / (gn + EPS),
    )


@dataclass(frozen=True)
class CostCurvatureReport:
    """Model and realized cost curvature, and their mismatch."""

    model_d2c: float
    model_residual: float
    model_gn: float
    model_ratio: float
    true_d2c: float
    true_residual: float
    true_gn: float
    true_ratio: float
    d2c_mismatch: float
    d2c_mismatch_normalized: float
    model_locally_concave: bool
    true_locally_concave: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_cost(model: Triplet, true: Triplet, goal: Any) -> CostCurvatureReport:
    """Compare the planner's cost geometry against the realized cost geometry.

    The realized cost uses the SAME goal embedding as the planner: it is what
    the planner's own objective would have evaluated to had the model predicted
    perfectly, which is the correct counterfactual.
    """
    m = cost_second_difference(model, goal)
    r = cost_second_difference(true, goal)
    mismatch = m.d2c - r.d2c
    return CostCurvatureReport(
        model_d2c=m.d2c, model_residual=m.residual_term,
        model_gn=m.gn_term, model_ratio=m.ratio,
        true_d2c=r.d2c, true_residual=r.residual_term,
        true_gn=r.gn_term, true_ratio=r.ratio,
        d2c_mismatch=mismatch,
        d2c_mismatch_normalized=mismatch / (abs(r.gn_term) + EPS),
        model_locally_concave=bool(m.d2c < 0.0),
        true_locally_concave=bool(r.d2c < 0.0),
    )


# --------------------------------------------------------------------------
# Readout 6: delta sweep, scaling exponent, numerical floor
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingFit:
    """Log-log fit of the RAW second difference against the perturbation scale.

    alpha ~ 2 smooth, alpha ~ 1 kink, alpha ~ 0 jump.  Never fit the normalized
    curvature: at a symmetric kink the central span vanishes while ``D2`` stays
    large, so the normalized quantity diverges and its slope is not alpha.
    """

    alpha: float
    intercept: float
    r_squared: float
    n_used: int
    excluded_below_floor: int


def fit_scaling_exponent(
    delta_norms: Sequence[float],
    d2_norms: Sequence[float],
    floor: float,
    min_points: int = 3,
) -> ScalingFit:
    """Fit ``log ||D2|| = alpha log ||d|| + c`` over scales above the floor."""
    d = np.asarray(delta_norms, dtype=DTYPE).reshape(-1)
    y = np.asarray(d2_norms, dtype=DTYPE).reshape(-1)
    if d.shape != y.shape:
        raise ValueError("delta_norms and d2_norms must have the same length")
    if np.any(d <= 0.0):
        raise ValueError("delta_norms must be positive")

    keep = y > max(float(floor), 0.0)
    excluded = int((~keep).sum())
    if int(keep.sum()) < min_points:
        return ScalingFit(float("nan"), float("nan"), float("nan"), int(keep.sum()), excluded)

    lx, ly = np.log(d[keep]), np.log(y[keep])
    alpha, intercept = np.polyfit(lx, ly, 1)
    predicted = alpha * lx + intercept
    ss_res = float(np.sum((ly - predicted) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    if ss_tot > EPS:
        r2 = 1.0 - ss_res / ss_tot
    else:
        # Constant ||D2|| across scales is the JUMP regime (alpha = 0), where a
        # flat line is an exact fit.  Returning NaN here would make the
        # protocol's fit-quality filter discard precisely the discontinuity
        # samples the detector exists to find.
        r2 = 1.0 if ss_res <= EPS else 0.0
    return ScalingFit(float(alpha), float(intercept), r2, int(keep.sum()), excluded)


def repeat_floor(samples: Sequence[Any]) -> float:
    """Max pairwise distance among repeated evaluations of the same input.

    With a deterministic reset, physics, renderer and encoder this returns
    exactly 0.  Per the protocol that is a FALSE all-clear: the binding
    small-delta limit is float cancellation, not stochasticity, and it is
    detected by the absence of a clean delta^2 regime in ``fit_scaling_exponent``
    rather than by this test.  The value is reported either way.
    """
    vs = [_vec(s) for s in samples]
    if len(vs) < 2:
        raise ValueError("need at least two repeats")
    return max(
        float(np.linalg.norm(vs[i] - vs[j]))
        for i in range(len(vs))
        for j in range(i + 1, len(vs))
    )


# --------------------------------------------------------------------------
# Readouts 7-8: validity, mode stratification
# --------------------------------------------------------------------------

CEM_LOCAL_SOURCE = "cem_local"

SAME_MODE_NON_CONTACT = "same_mode_non_contact"
SAME_MODE_CONTACT = "same_mode_contact"
CROSS_MODE = "cross_mode"


def clip_validity(
    action: np.ndarray,
    delta: np.ndarray,
    low: np.ndarray | float,
    high: np.ndarray | float,
    atol: float = 1e-9,
) -> bool:
    """True when both arms of the triplet are inside the action bounds.

    A clipped arm makes the spacing asymmetric, and for ``h+ != h-`` the second
    difference picks up a first-order term ``f'(a)(h+ - h-)`` -- exactly the
    affine component the diagnostic relies on cancelling.  Such triplets are
    discarded, never corrected, and the discard rate is logged PER STRATUM
    because near-bound actions are not uniformly distributed across contact
    regimes.

    ``atol`` exists because a centre already clamped exactly onto a bound goes
    through a normalize/denormalize round trip before this check runs, which is
    not bit-identical.  Job 46139 measured the discrepancy at exactly
    ``np.finfo(float64).eps`` -- one ULP, not a real bound violation -- and a
    strict comparison rejected every triplet built on that centre at every
    scale.  1e-9 is far above float64 roundoff and far below any perturbation
    scale this protocol uses (sigma >= 1.25e-3 of the action range).
    """
    a, d = _vec(action), _vec(delta)
    lo = np.asarray(low, dtype=DTYPE)
    hi = np.asarray(high, dtype=DTYPE)
    return bool(np.all(a + d <= hi + atol) and np.all(a + d >= lo - atol)
                and np.all(a - d <= hi + atol) and np.all(a - d >= lo - atol))


@dataclass(frozen=True)
class ContactTrace:
    """Per-step contact signature of one true rollout.

    Categories are resolved from MuJoCo body ids, never from geom-name
    guessing: the cube body comes from ``object_joint_0``, the static scene is
    the world body, and anything else touching the cube is the robot.

    ``robot`` is the category that stratifies.  ``table`` is a covariate: in
    OGBench-Cube the cube rests on the table for essentially every step, so a
    mode label built from "is anything in contact" would put every sample in
    the contact stratum and leave ``same_mode_non_contact`` empty.
    """

    robot: tuple[bool, ...]
    table: tuple[bool, ...]

    @property
    def ever_robot(self) -> bool:
        return any(self.robot)

    @property
    def onset(self) -> int | None:
        """First step index at which the robot touches the cube."""
        for i, flag in enumerate(self.robot):
            if flag:
                return i
        return None


def classify_contact_mode(
    minus: ContactTrace,
    center: ContactTrace,
    plus: ContactTrace,
    onset_tolerance: int = 1,
) -> str:
    """Counterfactual mode label from the three TRUE rollouts.

    Cross-mode is a property of the perturbation, not of the logged state: the
    logged transition may never touch the object while ``a + d`` does and
    ``a - d`` does not.

    Onset time is part of the signature.  Two rollouts that touch the same
    bodies but at different moments are not in the same dynamical mode, and a
    union over the rollout would erase that distinction.
    """
    traces = (minus, center, plus)
    ever = [t.ever_robot for t in traces]
    if len(set(ever)) > 1:
        return CROSS_MODE
    if not ever[0]:
        return SAME_MODE_NON_CONTACT
    onsets = [t.onset for t in traces]
    if any(o is None for o in onsets):  # unreachable given ever_robot, guarded anyway
        return CROSS_MODE
    if max(onsets) - min(onsets) > onset_tolerance:
        return CROSS_MODE
    return SAME_MODE_CONTACT


def state_anchor_curvature(triplet: Triplet) -> float:
    """``K_true_state`` on the declared task-state subset, in raw metres.

    Reported separately for ``object_pos`` and ``effector_pos`` -- never
    concatenated, which would impose an arbitrary relative weighting, and never
    std-normalized, which would rescale x/y/z by dataset spread and so amount to
    a change of chart.  The effector is directly actuated and its curvature is
    expected to be low; a high value indicts the measurement pipeline rather
    than the physics and blocks the run.
    """
    return normalized_curvature(triplet)


def draw_unit_directions(
    rng: np.random.Generator, n_directions: int, shape: tuple[int, ...]
) -> np.ndarray:
    """Unit-norm base directions, drawn ONCE per (snapshot, source).

    The same base vector is reused at every scale and horizon.  Redrawing per
    scale would make ``direction=k`` a different vector at each sigma, and the
    log-log fit that pairs records by ``(horizon, direction)`` across sigmas
    would then be a slope across unrelated directions rather than a scaling
    exponent.  That defect voided readout 6 in the first implementation.
    """
    d = rng.normal(size=(n_directions, *shape))
    flat = d.reshape(n_directions, -1)
    flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    return flat.reshape(n_directions, *shape)


def scale_direction(
    base: np.ndarray,
    sigma: float,
    *,
    source: str,
    raw_range_chunk: np.ndarray | None = None,
    scaler_scale_chunk: np.ndarray | None = None,
    proposal_std: np.ndarray | None = None,
    alpha: float = 1.0,
    sigma_max: float = 1.0,
) -> np.ndarray:
    """Scale a fixed base direction to one perturbation, in normalized units.

    Magnitude is deterministic at each scale by design: a finite-difference
    curvature probe needs a controlled ``||d||`` for the scale sweep to mean
    anything, so this is fixed-magnitude sampling on a (possibly elliptical)
    shell, not Gaussian sampling.  The protocol says so explicitly.
    """
    if source == CEM_LOCAL_SOURCE:
        if proposal_std is None:
            raise ValueError("cem_local requires the recorded CEM proposal std")
        return base * (alpha * sigma / sigma_max) * proposal_std
    if raw_range_chunk is None or scaler_scale_chunk is None:
        raise ValueError("raw-space scaling requires the action range and scaler scale")
    return (base * (sigma * raw_range_chunk)) / scaler_scale_chunk


def make_feasible(
    base: np.ndarray, cap: np.ndarray, min_component_margin: float = 1e-6
) -> tuple[np.ndarray, dict[str, float]]:
    """Shrink a base direction until it stays inside the action bounds at the
    largest scale of the sweep.

    ``cap[i]`` is the largest ``|base[i]|`` for which the perturbation at the
    top sigma still leaves component ``i`` inside its bounds.

    Motivated by job 45863: the all-or-nothing clip rule rejected every H=5
    triplet, because a chunk spans ``horizon * action_block`` primitive actions
    and expert / CEM-elite actions saturate their bounds often, so the chance
    that all 25 stay interior under a random perturbation is negligible.  The
    fix is to make the direction feasible by construction instead of drawing
    and rejecting.

    A single global shrink factor is used rather than per-component clipping so
    the DIRECTION is preserved exactly -- the scaling-exponent fit needs the
    same direction at every scale.  Saturated components are masked out first,
    since no shrink makes them usable.
    """
    b = _vec(base).copy()
    c = _vec(cap)
    if b.shape != c.shape:
        raise ValueError(f"base {b.shape} and cap {c.shape} must match")

    masked = c < min_component_margin
    b[masked] = 0.0
    norm = float(np.linalg.norm(b))
    if norm <= EPS:
        return b, {"masked_fraction": 1.0, "shrink": 0.0, "feasible": 0.0}
    b = b / norm

    live = ~masked & (np.abs(b) > EPS)
    shrink = 1.0
    if np.any(live):
        shrink = float(min(1.0, np.min(c[live] / np.abs(b[live]))))
    return b * shrink, {
        "masked_fraction": float(masked.mean()),
        "shrink": shrink,
        "feasible": 1.0,
    }


def clip_to_bounds(
    raw_chunk: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Clip a raw action chunk into the action box, reporting how much moved.

    Needed for the planner-visited sources.  CEM samples a Gaussian in
    normalized action space and the environment clips at execution, so recorded
    elites can lie outside the box.  A centre that is already out of bounds
    cannot be rescued by any perturbation -- job 46036 rejected 32/32 records on
    both CEM sources for exactly this reason, while reporting the *directions*
    as feasible.

    Probing around the clipped centre is the honest choice: the simulator
    executes the clipped action anyway, so this is the action the planner
    actually realizes.
    """
    a = np.asarray(raw_chunk, dtype=DTYPE)
    lo = np.broadcast_to(np.asarray(low, dtype=DTYPE), a.shape)
    hi = np.broadcast_to(np.asarray(high, dtype=DTYPE), a.shape)
    clipped = np.minimum(np.maximum(a, lo), hi)
    moved = np.abs(clipped - a)
    return clipped, {
        "clipped_fraction": float((moved > 0).mean()),
        "max_clip_raw": float(moved.max()) if moved.size else 0.0,
        "mean_clip_raw": float(moved.mean()) if moved.size else 0.0,
    }


# --------------------------------------------------------------------------
# Ordinal (chart-free) comparison of a cost triplet
# --------------------------------------------------------------------------

VALLEY = "valley"          # centre is the best of the three
PEAK = "peak"              # centre is the worst of the three
INCREASING = "increasing"  # cost rises monotonically from minus to plus
DECREASING = "decreasing"


def triplet_shape(c_minus: float, c_center: float, c_plus: float) -> str:
    """Purely ordinal shape of three costs along one action direction.

    Invariant under ANY strictly increasing transform of the cost, so it needs
    no shared units between the model's latent cost and the simulator's
    physical cost, and no chart alignment.  That is the whole reason this
    survives where both curvature bridges failed: it never subtracts a latent
    quantity from a metric one.

    ``VALLEY`` is the case a planner acts on -- it is a local minimum along this
    direction, i.e. the centre action looks locally optimal.
    """
    lo, mid, hi = float(c_minus), float(c_center), float(c_plus)
    if mid < lo and mid < hi:
        return VALLEY
    if mid > lo and mid > hi:
        return PEAK
    return INCREASING if hi > lo else DECREASING


def cost_spread(c_minus: float, c_center: float, c_plus: float) -> float:
    """Range of the three costs; near-zero spread makes the shape meaningless."""
    v = [float(c_minus), float(c_center), float(c_plus)]
    return max(v) - min(v)


def argmin_index(c_minus: float, c_center: float, c_plus: float) -> int:
    """0, 1 or 2 for the best of the three -- the choice a greedy planner makes."""
    return int(np.argmin([float(c_minus), float(c_center), float(c_plus)]))


def ordinal_agreement(
    model: tuple[float, float, float], physical: tuple[float, float, float]
) -> dict[str, Any]:
    """Compare what the model prefers against what physics rewards.

    ``false_valley`` is the operationally dangerous case: the model sees a local
    minimum where reality has none, so a planner that trusts it will settle on
    an action physics does not favour.
    """
    m_shape = triplet_shape(*model)
    p_shape = triplet_shape(*physical)
    m_arg = argmin_index(*model)
    p_arg = argmin_index(*physical)
    return {
        "model_shape": m_shape,
        "physical_shape": p_shape,
        "shape_agree": bool(m_shape == p_shape),
        "argmin_agree": bool(m_arg == p_arg),
        "model_argmin": m_arg,
        "physical_argmin": p_arg,
        "false_valley": bool(m_shape == VALLEY and p_shape != VALLEY),
        "missed_valley": bool(p_shape == VALLEY and m_shape != VALLEY),
        "model_cost_spread": cost_spread(*model),
        "physical_cost_spread": cost_spread(*physical),
    }
