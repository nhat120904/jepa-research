"""Pure utilities for the exact same-state MetaWorld intervention benchmark.

The GPU/environment runner lives in ``49_same_state_intervention.py``.  This
module deliberately contains the deterministic, unit-testable parts: nested
local action-fan construction and within-fan fidelity statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FanStats:
    causal_top1: float
    causal_mrr: float
    factual_prediction_error: float
    counterfactual_prediction_error_mean: float
    effect_spearman: float
    pairwise_spearman: float
    n_candidates: int


def make_local_action_fan(
    nominal: np.ndarray,
    *,
    n_candidates: int = 17,
    xyz_delta: float = 0.25,
    noise_std: float = 0.12,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Return a deterministic local action-sequence fan around ``nominal``.

    ``nominal`` is ``(R, 4)`` in raw MetaWorld action units.  Candidate zero is
    exactly the nominal (factual) sequence.  The other candidates include
    coherent +/- Cartesian perturbations, open/close interventions, and seeded
    smooth local noise.  A single maximum-length fan is generated and callers
    use prefixes for every horizon, making H=1/2/4/8 strictly nested.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] != 4:
        raise ValueError(f"nominal must have shape (R, 4), got {nominal.shape}")
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2")

    fan = [np.clip(nominal, -1.0, 1.0)]
    labels = ["factual"]
    for dim, axis in enumerate("xyz"):
        for sign, word in ((-1.0, "minus"), (1.0, "plus")):
            seq = nominal.copy()
            seq[:, dim] += sign * xyz_delta
            fan.append(np.clip(seq, -1.0, 1.0))
            labels.append(f"{axis}_{word}")
    for value, label in ((-1.0, "gripper_open"), (1.0, "gripper_close")):
        seq = nominal.copy()
        seq[:, 3] = value
        fan.append(np.clip(seq, -1.0, 1.0))
        labels.append(label)

    rng = np.random.default_rng(seed)
    while len(fan) < n_candidates:
        # Coherent low-frequency perturbation plus small per-step jitter.  The
        # former creates meaningfully different effects; the latter avoids an
        # unrealistically constant action over a long horizon.
        coherent = rng.normal(0.0, noise_std, size=(1, 4))
        coherent[:, 3] *= 2.0
        jitter = rng.normal(0.0, noise_std * 0.2, size=nominal.shape)
        seq = np.clip(nominal + coherent + jitter, -1.0, 1.0)
        fan.append(seq)
        labels.append(f"local_random_{len(fan) - 1:02d}")
    return np.stack(fan[:n_candidates]), labels[:n_candidates]


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Stable average ranks, including ties, without a scipy dependency."""
    x = np.asarray(x, dtype=float).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    rx, ry = _average_ranks(x[valid]), _average_ranks(y[valid])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    d = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    return d[np.triu_indices(len(x), k=1)]


def summarize_fan(
    predicted_latents: np.ndarray,
    true_latents: np.ndarray,
    true_object_positions: np.ndarray,
) -> FanStats:
    """Summarize model fidelity on one exact-state action fan.

    Candidate zero is factual.  Causal CRA ranks all action-conditioned model
    predictions against the *factual successor produced from the identical
    starting simulator state*.  The remaining correlations test whether model
    separation across actions tracks true object-effect separation.
    """
    pred = np.asarray(predicted_latents, dtype=float).reshape(len(predicted_latents), -1)
    true = np.asarray(true_latents, dtype=float).reshape(len(true_latents), -1)
    obj = np.asarray(true_object_positions, dtype=float).reshape(len(true_object_positions), -1)
    if not (len(pred) == len(true) == len(obj)) or len(pred) < 2:
        raise ValueError("predicted, true, and object outcomes need the same K>=2")

    pred_err = np.linalg.norm(pred - true, axis=1)
    factual_target_dist = np.linalg.norm(pred - true[0], axis=1)
    d0 = factual_target_dist[0]
    n_better = int(np.sum(factual_target_dist[1:] < d0))
    n_tie = int(np.sum(factual_target_dist[1:] == d0))
    top1 = (1.0 / (1 + n_tie)) if n_better == 0 else 0.0
    mrr = 1.0 / (1.0 + n_better + 0.5 * n_tie)

    # Candidate-to-factual effect sensitivity and all-pairs geometry agreement.
    pred_from_factual = np.linalg.norm(pred - pred[0], axis=1)[1:]
    obj_from_factual = np.linalg.norm(obj - obj[0], axis=1)[1:]
    effect_rho = spearman(pred_from_factual, obj_from_factual)
    pair_rho = spearman(_pairwise_distances(pred), _pairwise_distances(obj))
    return FanStats(
        causal_top1=float(top1),
        causal_mrr=float(mrr),
        factual_prediction_error=float(pred_err[0]),
        counterfactual_prediction_error_mean=float(pred_err[1:].mean()),
        effect_spearman=effect_rho,
        pairwise_spearman=pair_rho,
        n_candidates=len(pred),
    )
