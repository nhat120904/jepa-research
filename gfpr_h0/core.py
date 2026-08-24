"""Pure NumPy feature and selection utilities for GFPR H0-A."""

from __future__ import annotations

import numpy as np


def rank_fraction(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a finite nonempty vector")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks / max(len(values) - 1, 1)


def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = max(1.4826 * mad, 1e-8)
    return (values - median) / scale


def build_feature_views(
    candidate_actions: np.ndarray,
    anchor_action: np.ndarray,
    native_cost: np.ndarray,
    auxiliary_cost: np.ndarray,
    candidate_endpoint: np.ndarray,
    anchor_endpoint: np.ndarray,
    current_embedding: np.ndarray,
    goal_embedding: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build deployable features without physical-state inputs."""
    actions = np.asarray(candidate_actions, dtype=np.float32)
    anchor = np.asarray(anchor_action, dtype=np.float32)
    native = np.asarray(native_cost, dtype=np.float64)
    auxiliary = np.asarray(auxiliary_cost, dtype=np.float64)
    endpoint = np.asarray(candidate_endpoint, dtype=np.float32)
    anchor_endpoint = np.asarray(anchor_endpoint, dtype=np.float32).reshape(-1)
    current = np.asarray(current_embedding, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal_embedding, dtype=np.float32).reshape(-1)
    n = len(actions)
    if native.shape != (n,) or auxiliary.shape != (n,):
        raise ValueError("cost arrays do not match candidate count")
    if endpoint.ndim != 2 or endpoint.shape[0] != n:
        raise ValueError("endpoint features must be (candidates, dim)")
    if not (
        endpoint.shape[1]
        == len(anchor_endpoint)
        == len(current)
        == len(goal)
    ):
        raise ValueError("embedding dimensions differ")

    flat = actions.reshape(n, -1)
    anchor_flat = anchor.reshape(1, -1)
    action_only = np.concatenate(
        [flat, flat - anchor_flat, np.square(flat - anchor_flat)], axis=1
    )
    proxy_scalar = np.column_stack(
        [
            robust_scale(native),
            robust_scale(auxiliary),
            rank_fraction(native),
            rank_fraction(auxiliary),
            np.linalg.norm(flat - anchor_flat, axis=1),
        ]
    ).astype(np.float32)
    proxy_action = np.concatenate([action_only, proxy_scalar], axis=1)

    goal_residual = endpoint - goal[None]
    anchor_goal_residual = anchor_endpoint - goal
    current_goal_residual = current - goal
    latent = np.concatenate(
        [
            endpoint,
            goal_residual,
            np.square(goal_residual),
            endpoint - anchor_endpoint[None],
            np.repeat(anchor_goal_residual[None], n, axis=0),
            np.repeat(current_goal_residual[None], n, axis=0),
            np.column_stack(
                [
                    np.linalg.norm(goal_residual, axis=1),
                    np.linalg.norm(endpoint - anchor_endpoint[None], axis=1),
                ]
            ),
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "action_only": action_only.astype(np.float32),
        "proxy_action": proxy_action.astype(np.float32),
        "latent_context": np.concatenate([proxy_action, latent], axis=1).astype(
            np.float32
        ),
    }


def farthest_point_indices(
    actions: np.ndarray, anchor: np.ndarray, count: int
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim < 2 or not 0 < count <= len(actions):
        raise ValueError("invalid action population or count")
    flat = actions.reshape(len(actions), -1)
    anchor_flat = np.asarray(anchor, dtype=np.float64).reshape(1, -1)
    chosen = [int(np.argmax(np.square(flat - anchor_flat).sum(axis=1)))]
    min_distance = np.square(flat - flat[chosen[0]]).sum(axis=1)
    while len(chosen) < count:
        min_distance[chosen] = -np.inf
        chosen.append(int(np.argmax(min_distance)))
        min_distance = np.minimum(
            min_distance, np.square(flat - flat[chosen[-1]]).sum(axis=1)
        )
    return np.asarray(chosen, dtype=np.int64)


def select_with_gate(
    mean_gain_cm: np.ndarray,
    std_gain_cm: np.ndarray,
    margin_cm: float,
) -> tuple[int, int]:
    """Return ungated and one-sigma-gated indices; -1 denotes the anchor."""
    mean = np.asarray(mean_gain_cm, dtype=np.float64)
    std = np.asarray(std_gain_cm, dtype=np.float64)
    if mean.shape != std.shape or mean.ndim != 1 or not np.isfinite(mean).all():
        raise ValueError("invalid prediction arrays")
    best_mean = int(np.argmax(mean))
    ungated = best_mean if mean[best_mean] > 0 else -1
    lcb = mean - std
    best_lcb = int(np.argmax(lcb))
    gated = best_lcb if lcb[best_lcb] > margin_cm else -1
    return ungated, gated

