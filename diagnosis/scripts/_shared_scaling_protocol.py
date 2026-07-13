"""Model-independent primitives for the DROID scaling protocol."""

from __future__ import annotations

import numpy as np


def robust_scales(x: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Per-dimension robust scale (MAD), with std fallback for constants."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=0)
    mad = 1.4826 * np.median(np.abs(x - med), axis=0)
    std = np.std(x, axis=0)
    return np.where(mad > floor, mad, np.where(std > floor, std, 1.0))


def physical_effect_scores(proprio_t: np.ndarray, proprio_t1: np.ndarray,
                           scales: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dimensionless physical effect magnitude shared by every checkpoint.

    DROID has no object state/contact ground truth.  We therefore use only raw
    robot proprio change (including the measured gripper channel), normalized
    by one set of robust scales computed on the common transition universe.
    No visual latent enters this mask.
    """
    p0, p1 = np.asarray(proprio_t, float), np.asarray(proprio_t1, float)
    delta = p1 - p0
    # Upstream DROID state is xyz + Euler + gripper.  Use the shortest angular
    # displacement so a +/-pi wrap is not mislabeled as a giant physical effect.
    if delta.shape[1] >= 6:
        delta[:, 3:6] = np.arctan2(np.sin(delta[:, 3:6]), np.cos(delta[:, 3:6]))
    if scales is None:
        scales = robust_scales(delta)
    return np.linalg.norm(delta / np.asarray(scales), axis=1), np.asarray(scales)


def physical_state_features(proprio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust-standardized state features with a wrap-safe Euler embedding."""
    p = np.asarray(proprio, dtype=float)
    if p.shape[1] >= 7:
        raw = np.concatenate([p[:, :3], np.sin(p[:, 3:6]), np.cos(p[:, 3:6]), p[:, 6:]], axis=1)
    else:
        raw = p
    median = np.median(raw, axis=0)
    scales = robust_scales(raw)
    return (raw - median) / scales, median, scales


def physical_regimes(proprio_t: np.ndarray, proprio_t1: np.ndarray,
                     *, gripper_dim: int = -1, gripper_delta: float = 0.2,
                     gripper_closed: float = 0.5,
                     motion_scores: np.ndarray | None = None,
                     motion_threshold: float | None = None) -> np.ndarray:
    """Shared sensor-only regimes; ``pre_grasp`` remains an explicit proxy."""
    p0, p1 = np.asarray(proprio_t), np.asarray(proprio_t1)
    g0, g1 = p0[:, gripper_dim], p1[:, gripper_dim]
    out = np.full(len(p0), "free_space", dtype=object)
    if motion_scores is not None and motion_threshold is not None:
        out[np.asarray(motion_scores) > motion_threshold] = "pre_grasp_proxy"
    out[g0 > gripper_closed] = "contact_manipulation"
    out[np.abs(g1 - g0) > gripper_delta] = "gripper_actuation"
    return out


def fixed_physical_neighbours(
    anchor_features: np.ndarray,
    pool_features: np.ndarray,
    *,
    K: int,
    anchor_global_indices: np.ndarray | None = None,
    pool_global_indices: np.ndarray | None = None,
) -> np.ndarray:
    """K nearest pool states in one shared physical feature space.

    This intentionally selects by *state distance only*.  It matches the paper's
    literal "K most similar states" description and avoids the prior hidden
    maximally-different-action selection.  Returned indices address the pool.
    """
    anchors = np.asarray(anchor_features, float)
    pool = np.asarray(pool_features, float)
    if K < 1 or K >= len(pool):
        raise ValueError("K must be >=1 and smaller than pool size")
    d2 = ((anchors[:, None] - pool[None]) ** 2).sum(axis=-1)
    if anchor_global_indices is not None and pool_global_indices is not None:
        same = np.asarray(anchor_global_indices)[:, None] == np.asarray(pool_global_indices)[None]
        d2[same] = np.inf
    chosen = np.argpartition(d2, kth=K - 1, axis=1)[:, :K]
    # Stable nearest-to-farthest order makes the serialized manifest invariant
    # to argpartition's unspecified ordering inside its selected K elements.
    chosen_d2 = np.take_along_axis(d2, chosen, axis=1)
    order = np.argsort(chosen_d2, axis=1, kind="mergesort")
    return np.take_along_axis(chosen, order, axis=1)
