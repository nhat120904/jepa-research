"""Small, testable primitives for the rollout-repair experiment."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch


ARMS = ("one_step_expert", "multistep_expert", "multistep_offpolicy")


def split_for_order(order: int) -> str:
    """Reuse the immutable 80/16/32 PERD episode split."""

    residue = int(order) % 8
    if residue in (0, 4):
        return "test"
    if residue == 2:
        return "val"
    return "train"


def normalized_to_raw(
    normalized: np.ndarray,
    scaler: Any,
    horizon: int,
    action_block: int,
    raw_dim: int,
) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float32)
    expected = (horizon, raw_dim * action_block)
    if normalized.shape[-2:] != expected:
        raise ValueError(
            f"expected trailing normalized action shape {expected}, got "
            f"{normalized.shape[-2:]}"
        )
    return scaler.inverse_transform(normalized.reshape(-1, raw_dim)).reshape(
        *normalized.shape[:-2], horizon, action_block, raw_dim
    )


def autoregressive_rollout(
    model: torch.nn.Module,
    initial: torch.Tensor,
    actions: torch.Tensor,
    history_size: int = 3,
) -> torch.Tensor:
    """Mirror ``LeWM.rollout`` without its inference-time detach.

    Args:
        initial: ``(B, D)`` encoded initial observations.
        actions: ``(B, H, A)`` normalized action blocks.
    Returns:
        ``(B, H, D)`` predicted future embeddings.
    """

    if initial.ndim != 2 or actions.ndim != 3:
        raise ValueError("initial/actions must have shapes (B,D)/(B,H,A)")
    if len(initial) != len(actions):
        raise ValueError("batch mismatch")
    act_emb = model.action_encoder(actions)
    embeddings = [initial]
    predictions = []
    for step in range(actions.shape[1]):
        # At step t there are t+1 state tokens (z0 plus t predictions) and
        # exactly t+1 action tokens (a0..at), matching upstream LeWM.rollout.
        lo = max(0, len(embeddings) - history_size)
        state_window = torch.stack(embeddings[lo:], dim=1)
        action_window = act_emb[:, lo : step + 1]
        prediction = model.predict(state_window, action_window)[:, -1]
        embeddings.append(prediction)
        predictions.append(prediction)
    return torch.stack(predictions, dim=1)


def teacher_forced_rollout(
    model: torch.nn.Module,
    true_embeddings: torch.Tensor,
    actions: torch.Tensor,
    history_size: int = 3,
) -> torch.Tensor:
    """Predict every next latent from true rather than imagined history."""

    if true_embeddings.ndim != 3 or actions.ndim != 3:
        raise ValueError("true_embeddings/actions must have shapes (B,H+1,D)/(B,H,A)")
    if true_embeddings.shape[1] != actions.shape[1] + 1:
        raise ValueError("true embedding sequence must include z0 and H targets")
    act_emb = model.action_encoder(actions)
    predictions = []
    for step in range(actions.shape[1]):
        lo = max(0, step + 1 - history_size)
        prediction = model.predict(
            true_embeddings[:, lo : step + 1], act_emb[:, lo : step + 1]
        )[:, -1]
        predictions.append(prediction)
    return torch.stack(predictions, dim=1)


def masked_prediction_mse(
    predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Mean per-token latent MSE with post-termination horizons excluded."""

    if predicted.shape != target.shape:
        raise ValueError(f"prediction/target mismatch: {predicted.shape} vs {target.shape}")
    if valid.shape != predicted.shape[:2]:
        raise ValueError("valid mask must have shape (B,H)")
    per_token = (predicted.float() - target.float()).square().mean(dim=-1)
    weights = valid.to(per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


def squared_goal_cost(embeddings: np.ndarray, goal: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return np.square(embeddings - goal).sum(axis=-1)


def spearman_no_tie_assumption(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman with average ranks; returns NaN for constant inputs."""

    from scipy.stats import spearmanr

    value = spearmanr(np.asarray(x), np.asarray(y)).statistic
    return float(value) if np.isfinite(value) else float("nan")


def dynamics_state_dict(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {key: value.detach().cpu() for key, value in getattr(model, name).state_dict().items()}
        for name in ("predictor", "action_encoder", "pred_proj")
    }


def load_dynamics_state(model: torch.nn.Module, state: dict[str, Any]) -> None:
    for name in ("predictor", "action_encoder", "pred_proj"):
        getattr(model, name).load_state_dict(state[name], strict=True)


def set_dynamics_trainable(model: torch.nn.Module) -> int:
    model.requires_grad_(False)
    count = 0
    for name in ("predictor", "action_encoder", "pred_proj"):
        module = getattr(model, name)
        module.requires_grad_(True)
        module.train()
        count += sum(parameter.numel() for parameter in module.parameters())
    return int(count)


def list_shards(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise RuntimeError(f"no files matching {pattern} below {root}")
    return paths


def concatenate_sequences(
    arrays: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings, actions, masks = zip(*arrays)
    return (
        np.concatenate(embeddings).astype(np.float32),
        np.concatenate(actions).astype(np.float32),
        np.concatenate(masks).astype(bool),
    )

