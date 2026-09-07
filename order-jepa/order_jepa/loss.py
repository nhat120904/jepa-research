"""PyTorch implementation of the ORDER auxiliary objective."""

from __future__ import annotations

import torch


def order_effect_loss(
    predicted_ij: torch.Tensor,
    predicted_ji: torch.Tensor,
    target_ij: torch.Tensor,
    target_ji: torch.Tensor,
    *,
    duration: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Match the signed endpoint difference of two swapped action orders.

    Targets are detached here so callers cannot accidentally train a frozen or
    target encoder through this loss.  With one fixed duration, ``duration`` is
    only a loss-scale convention and can be left at one.
    """

    if duration <= 0:
        raise ValueError("duration must be positive")
    shapes = {predicted_ij.shape, predicted_ji.shape, target_ij.shape, target_ji.shape}
    if len(shapes) != 1:
        raise ValueError("all four endpoint tensors must have identical shapes")
    true_difference = target_ij.detach() - target_ji.detach()
    predicted_difference = predicted_ij - predicted_ji
    squared = ((predicted_difference - true_difference) / (duration**2)).square()
    if reduction == "none":
        return squared
    if reduction == "sum":
        return squared.sum()
    if reduction == "mean":
        return squared.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def paired_error_decomposition(
    predicted_ij: torch.Tensor,
    predicted_ji: torch.Tensor,
    target_ij: torch.Tensor,
    target_ji: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return branch SSE, mean component, and antisymmetric component.

    The returned tensors satisfy ``branch = mean_component + diff_component``
    up to floating-point error, where the latter two already include the
    factors 2 and 1/2 from the identity in the proposal.
    """

    e1 = predicted_ij - target_ij
    e2 = predicted_ji - target_ji
    branch = e1.square().sum() + e2.square().sum()
    mean_component = 2.0 * ((e1 + e2) / 2.0).square().sum()
    diff_component = 0.5 * (e1 - e2).square().sum()
    return branch, mean_component, diff_component
