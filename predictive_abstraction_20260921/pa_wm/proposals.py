"""Observation/state-independent smooth action bank; no simulator calls."""
import numpy as np


def smooth_bank(rng, candidates, horizon, knots=5, max_norm=1.8, warm_start=None):
    if candidates < 2 or horizon < 2 or not 2 <= knots <= horizon or max_norm <= 0:
        raise ValueError("Invalid action-bank dimensions/bound")
    angles = rng.uniform(-np.pi, np.pi, (candidates, knots))
    speeds = rng.uniform(0.2, max_norm, (candidates, knots))
    controls = np.stack([np.cos(angles), np.sin(angles)], -1) * speeds[..., None]
    grid = np.linspace(0, horizon - 1, knots)
    actions = np.empty((candidates, horizon, 2), dtype=np.float32)
    for k in range(candidates):
        for d in range(2):
            actions[k, :, d] = np.interp(np.arange(horizon), grid, controls[k, :, d])
    actions[0] = 0  # Fixed default, but use mean bank success as the random baseline.
    if warm_start is not None:
        warm_start = np.asarray(warm_start, dtype=np.float32)
        if warm_start.shape != (horizon, 2) or not np.isfinite(warm_start).all():
            raise ValueError("warm_start must be a finite [horizon, 2] array")
        n = min(candidates // 2, candidates - 1)
        actions[1:n+1] = warm_start + 0.15 * actions[1:n+1]
        actions[1] = warm_start
    norms = np.linalg.norm(actions, axis=-1, keepdims=True)
    actions *= np.minimum(1.0, max_norm / np.maximum(norms, 1e-8))
    return actions


def red_centroid(rgb):
    """Read the red-dot position from an RGB task observation, never simulator state."""
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected one HWC RGB observation")
    weight = np.maximum(image[..., 0] - 0.5 * (image[..., 1] + image[..., 2]), 0)
    total = float(weight.sum())
    if total <= 1e-6:
        raise ValueError("No red visual target found")
    yy, xx = np.indices(weight.shape)
    return np.array([(weight * xx).sum() / total, (weight * yy).sum() / total],
                    dtype=np.float32)


def _visual_waypoint_actions(points, horizon, split_fraction, max_norm):
    """Kinematic proposal only; collisions remain unknown and are evaluated by the WM."""
    points = [np.asarray(point, dtype=np.float32) for point in points]
    if len(points) == 3:
        first = int(np.clip(round(horizon * split_fraction), 2, horizon - 2))
        lengths = (first, horizon - first)
    elif len(points) == 2:
        lengths = (horizon,)
    else:
        raise ValueError("Expected start/goal or start/A/B")
    actions = []
    for source, target, length in zip(points[:-1], points[1:], lengths):
        # Upstream native transition is position += 2*action before collisions.
        action = (target - source) / (2.0 * length)
        actions.append(np.repeat(action[None], length, axis=0))
    result = np.concatenate(actions).astype(np.float32)
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    result *= np.minimum(1.0, max_norm / np.maximum(norms, 1e-8))
    return result


def visual_goal_bank(rng, start_rgb, goal_a_rgb, goal_b_rgb, candidates, horizon,
                     knots=7, max_norm=1.8):
    """Competent task-conditioned proposals using RGB goals, not hidden coordinates.

    Most chunks target A then B with different timings and correlated perturbations;
    some directly target B or remain unguided. The exact proposals are fixed controls.
    This is an explicit hand-designed inverse proposal shared by every future WM arm.
    """
    if candidates < 8 or horizon < 8:
        raise ValueError("Need at least 8 candidates and 8 steps")
    start, goal_a, goal_b = map(red_centroid, (start_rgb, goal_a_rgb, goal_b_rgb))
    random = smooth_bank(rng, candidates, horizon, knots, max_norm)
    result = random.copy()
    result[0] = 0
    ordered_exact = _visual_waypoint_actions([start, goal_a, goal_b], horizon, 0.5,
                                             max_norm)
    result[1] = ordered_exact
    result[2] = _visual_waypoint_actions([start, goal_b], horizon, 1.0, max_norm)
    n_ordered = max(3, (3 * candidates) // 4)
    for index in range(3, n_ordered):
        split = rng.uniform(0.3, 0.7)
        base = _visual_waypoint_actions([start, goal_a, goal_b], horizon, split, max_norm)
        scale = rng.uniform(0.05, 0.35)
        result[index] = base + scale * random[index]
    # Preserve a direct-B family as a matched endpoint-oriented proposal.
    for index in range(n_ordered, max(n_ordered + 1, (7 * candidates) // 8)):
        base = _visual_waypoint_actions([start, goal_b], horizon, 1.0, max_norm)
        result[index] = base + rng.uniform(0.05, 0.35) * random[index]
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    result *= np.minimum(1.0, max_norm / np.maximum(norms, 1e-8))
    return result.astype(np.float32)
