"""Pure numerical primitives for the ORDER Stage-A audit.

This module deliberately has no simulator or model dependency.  The collection
and scoring scripts can therefore be tested against synthetic arrays before a
GPU or PushT installation is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def angular_distance(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Shortest absolute distance between angles in radians."""

    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.abs((delta + np.pi) % (2.0 * np.pi) - np.pi)


def pusht_physical_cost(
    states: np.ndarray,
    goal_state: np.ndarray,
    *,
    position_tolerance: float = 20.0,
    angle_tolerance: float = np.pi / 9.0,
) -> np.ndarray:
    """Continuous counterpart of the published PushT success predicate.

    State layout is ``[agent_x, agent_y, block_x, block_y, block_angle, ...]``.
    Velocity is intentionally excluded because the wrapper's success predicate
    does not score it.  Both terms are normalized by their success tolerances.
    """

    x = np.asarray(states, dtype=np.float64)
    goal = np.asarray(goal_state, dtype=np.float64)
    if x.shape[-1] < 5 or goal.shape[-1] < 5:
        raise ValueError("PushT state and goal must contain at least five values")
    pos = np.linalg.norm(x[..., :4] - goal[..., :4], axis=-1) / position_tolerance
    angle = angular_distance(x[..., 4], goal[..., 4]) / angle_tolerance
    return pos + angle


def pusht_success(
    states: np.ndarray,
    goal_state: np.ndarray,
    *,
    position_tolerance: float = 20.0,
    angle_tolerance: float = np.pi / 9.0,
) -> np.ndarray:
    """Exact boolean predicate used by the upstream PushT wrapper."""

    x = np.asarray(states, dtype=np.float64)
    goal = np.asarray(goal_state, dtype=np.float64)
    pos = np.linalg.norm(x[..., :4] - goal[..., :4], axis=-1)
    angle = angular_distance(x[..., 4], goal[..., 4])
    return (pos < position_tolerance) & (angle < angle_tolerance)


def reacher_physical_cost(
    qpos: np.ndarray,
    goal_qpos: np.ndarray,
    *,
    joint_tolerance: float = 0.05,
) -> np.ndarray:
    """Continuous counterpart of stable-worldmodel's Reacher qpos predicate.

    ``ReacherQPosMatchTask`` succeeds only when every joint is within 0.05
    radians of the requested configuration. The maximum normalized wrapped
    joint error is therefore a continuous cost with success exactly at
    ``cost < 1`` (apart from the upstream strict inequality).
    """

    x = np.asarray(qpos, dtype=np.float64)
    goal = np.asarray(goal_qpos, dtype=np.float64)
    if x.shape[-1] != goal.shape[-1]:
        raise ValueError("Reacher qpos and goal_qpos must have the same width")
    return np.max(angular_distance(x, goal), axis=-1) / joint_tolerance


def reacher_success(
    qpos: np.ndarray,
    goal_qpos: np.ndarray,
    *,
    joint_tolerance: float = 0.05,
) -> np.ndarray:
    """Exact wrapped-angle form of the upstream per-joint success predicate."""

    x = np.asarray(qpos, dtype=np.float64)
    goal = np.asarray(goal_qpos, dtype=np.float64)
    if x.shape[-1] != goal.shape[-1]:
        raise ValueError("Reacher qpos and goal_qpos must have the same width")
    return np.all(angular_distance(x, goal) < joint_tolerance, axis=-1)


def latent_cost(
    visual: np.ndarray,
    goal_visual: np.ndarray,
    proprio: np.ndarray | None = None,
    goal_proprio: np.ndarray | None = None,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Terminal visual + proprio MSE used by DINO-WM-style planners."""

    zv = np.asarray(visual, dtype=np.float64)
    gv = np.asarray(goal_visual, dtype=np.float64)
    visual_mse = np.mean(np.square(zv - gv), axis=tuple(range(1, zv.ndim)))
    if proprio is None and goal_proprio is None:
        return visual_mse
    if proprio is None or goal_proprio is None:
        raise ValueError("proprio and goal_proprio must either both be given or both be omitted")
    zp = np.asarray(proprio, dtype=np.float64)
    gp = np.asarray(goal_proprio, dtype=np.float64)
    proprio_mse = np.mean(np.square(zp - gp), axis=tuple(range(1, zp.ndim)))
    return visual_mse + alpha * proprio_mse


def selection_regret(selector_cost: np.ndarray, physical_cost: np.ndarray) -> tuple[int, float]:
    """Return selected index and physical regret on one fixed candidate set."""

    q = np.asarray(selector_cost, dtype=np.float64)
    p = np.asarray(physical_cost, dtype=np.float64)
    if q.ndim != 1 or p.ndim != 1 or q.shape != p.shape or q.size == 0:
        raise ValueError("selector_cost and physical_cost must be non-empty aligned vectors")
    selected = int(np.nanargmin(q))
    return selected, float(p[selected] - np.nanmin(p))


def signed_pair_accuracy(predicted_delta: np.ndarray, reference_delta: np.ndarray, margin: float = 0.0) -> float:
    """Fraction of non-tied reference pairs whose lower-cost member is preserved."""

    pred = np.asarray(predicted_delta, dtype=np.float64)
    ref = np.asarray(reference_delta, dtype=np.float64)
    keep = np.isfinite(pred) & np.isfinite(ref) & (np.abs(ref) > margin)
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.sign(pred[keep]) == np.sign(ref[keep])))


@dataclass(frozen=True)
class OrderVectorMetrics:
    true_norm: float
    predicted_norm: float
    error_norm: float
    normalized_error: float
    cosine: float
    amplitude_ratio: float


def order_vector_metrics(
    true_ij: np.ndarray,
    true_ji: np.ndarray,
    predicted_ij: np.ndarray,
    predicted_ji: np.ndarray,
) -> OrderVectorMetrics:
    """Direction/magnitude diagnostics for a single swapped-order pair."""

    d_true = np.asarray(true_ij, dtype=np.float64).reshape(-1) - np.asarray(
        true_ji, dtype=np.float64
    ).reshape(-1)
    d_pred = np.asarray(predicted_ij, dtype=np.float64).reshape(-1) - np.asarray(
        predicted_ji, dtype=np.float64
    ).reshape(-1)
    if d_true.shape != d_pred.shape:
        raise ValueError("true and predicted order vectors must have the same shape")
    nt = float(np.linalg.norm(d_true))
    npred = float(np.linalg.norm(d_pred))
    err = float(np.linalg.norm(d_pred - d_true))
    cosine = float(np.dot(d_true, d_pred) / max(nt * npred, EPS))
    return OrderVectorMetrics(
        true_norm=nt,
        predicted_norm=npred,
        error_norm=err,
        normalized_error=err / max(nt, EPS),
        cosine=cosine,
        amplitude_ratio=npred / max(nt, EPS),
    )


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def clustered_bootstrap_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling whole anchors/episodes as clusters."""

    x = np.asarray(values, dtype=np.float64)
    ids = np.asarray(cluster_ids)
    keep = np.isfinite(x)
    x, ids = x[keep], ids[keep]
    unique = np.unique(ids)
    if x.size == 0 or unique.size == 0:
        return float("nan"), float("nan"), float("nan")
    grouped = [x[ids == cluster] for cluster in unique]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        draw = rng.integers(0, len(grouped), size=len(grouped))
        boot[i] = np.mean(np.concatenate([grouped[j] for j in draw]))
    lo, hi = percentile_interval(boot)
    return float(np.mean(x)), lo, hi
