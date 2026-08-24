"""Pure NumPy utilities for the locked CROD H0 experiment."""

from __future__ import annotations

import numpy as np


def rank_fraction(values: np.ndarray) -> np.ndarray:
    """Stable ascending ranks in [0, 1], where lower means better."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("values must be a finite nonempty vector")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks / max(len(values) - 1, 1)


def directional_ordinal_score(
    native_cost: np.ndarray,
    auxiliary_cost: np.ndarray,
    anchor_native_cost: float,
    anchor_auxiliary_cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CROD score relative to the exact planner-returned anchor.

    Candidate and anchor ranks are computed jointly.  A positive score requires
    the native LeWM to rank the candidate below the anchor while the auxiliary
    DINO-WM ranks it above the anchor.
    """
    native_cost = np.asarray(native_cost, dtype=np.float64)
    auxiliary_cost = np.asarray(auxiliary_cost, dtype=np.float64)
    if native_cost.shape != auxiliary_cost.shape or native_cost.ndim != 1:
        raise ValueError("native and auxiliary costs must be same-length vectors")
    native_rank_all = rank_fraction(np.r_[anchor_native_cost, native_cost])
    auxiliary_rank_all = rank_fraction(np.r_[anchor_auxiliary_cost, auxiliary_cost])
    native_gap = np.maximum(native_rank_all[1:] - native_rank_all[0], 0.0)
    auxiliary_gap = np.maximum(auxiliary_rank_all[0] - auxiliary_rank_all[1:], 0.0)
    score = native_gap * auxiliary_gap
    return score, native_rank_all[1:], auxiliary_rank_all[1:]


def farthest_point_indices(
    actions: np.ndarray, anchor: np.ndarray, count: int
) -> np.ndarray:
    """Deterministic greedy action-space coverage."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim < 2 or not 0 < count <= len(actions):
        raise ValueError("invalid action population or requested count")
    flat = actions.reshape(len(actions), -1)
    anchor_flat = np.asarray(anchor, dtype=np.float64).reshape(1, -1)
    distance_to_anchor = np.square(flat - anchor_flat).sum(axis=1)
    chosen = [int(np.argmax(distance_to_anchor))]
    min_distance = np.square(flat - flat[chosen[0]]).sum(axis=1)
    while len(chosen) < count:
        min_distance[chosen] = -np.inf
        chosen.append(int(np.argmax(min_distance)))
        min_distance = np.minimum(
            min_distance,
            np.square(flat - flat[chosen[-1]]).sum(axis=1),
        )
    return np.asarray(chosen, dtype=np.int64)


def top_score_diverse(
    actions: np.ndarray,
    anchor: np.ndarray,
    eligible: np.ndarray,
    score: np.ndarray,
    count: int,
    pool_fraction: float = 0.25,
) -> np.ndarray:
    """Coverage within the top-scoring eligible support."""
    eligible = np.asarray(eligible, dtype=np.int64)
    if len(eligible) < count or not 0 < pool_fraction <= 1:
        raise ValueError("insufficient eligible support or invalid pool fraction")
    pool_size = max(count, int(np.ceil(pool_fraction * len(eligible))))
    ranked = eligible[np.argsort(score[eligible], kind="stable")[::-1]]
    pool = ranked[:pool_size]
    return pool[farthest_point_indices(actions[pool], anchor, count)]


def select_arms(
    actions: np.ndarray,
    anchor: np.ndarray,
    native_cost: np.ndarray,
    auxiliary_cost: np.ndarray,
    anchor_native_cost: float,
    crod_score: np.ndarray,
    native_uncertainty: np.ndarray,
    count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Lock all H0 choices without observing physical outcomes."""
    actions = np.asarray(actions)
    native_cost = np.asarray(native_cost)
    auxiliary_cost = np.asarray(auxiliary_cost)
    crod_score = np.asarray(crod_score)
    native_uncertainty = np.asarray(native_uncertainty)
    n = len(actions)
    if any(len(x) != n for x in (native_cost, auxiliary_cost, crod_score, native_uncertainty)):
        raise ValueError("candidate array lengths differ")
    if not 0 < count < n:
        raise ValueError("invalid acquisition count")
    rejected = np.flatnonzero(native_cost > anchor_native_cost)
    if len(rejected) < count:
        raise RuntimeError("too few native-rejected candidates for matched support")

    rng = np.random.default_rng(seed)
    crod = rejected[np.argsort(crod_score[rejected], kind="stable")[::-1][:count]]
    dino = rejected[np.argsort(auxiliary_cost[rejected], kind="stable")[:count]]
    rejected_diverse_local = farthest_point_indices(actions[rejected], anchor, count)
    return {
        "crod_directional": crod,
        "action_diverse": farthest_point_indices(actions, anchor, count),
        "rejected_action_diverse": rejected[rejected_diverse_local],
        "dino_best_rejected": dino,
        "native_uncertainty_diverse": top_score_diverse(
            actions,
            anchor,
            rejected,
            native_uncertainty,
            count,
        ),
        "random_rejected": np.sort(rng.choice(rejected, size=count, replace=False)),
    }


def mean_action_distance(actions: np.ndarray, anchor: np.ndarray) -> float:
    flat = np.asarray(actions, dtype=np.float64).reshape(len(actions), -1)
    anchor_flat = np.asarray(anchor, dtype=np.float64).reshape(1, -1)
    return float(np.sqrt(np.square(flat - anchor_flat).sum(axis=1)).mean())
