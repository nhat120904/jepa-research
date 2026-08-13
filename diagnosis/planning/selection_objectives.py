"""Group-aware objectives for planner candidate selection.

Every ``group_id`` denotes candidates scored at the same MPC snapshot and CEM
iteration.  Comparisons across groups are invalid because their simulator costs
have different offsets and ranges.  The losses below therefore reduce within a
group first and only then average across groups.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _groups(group_id: torch.Tensor):
    if group_id.ndim != 1:
        raise ValueError("group_id must be one-dimensional")
    for gid in torch.unique(group_id, sorted=True):
        idx = torch.nonzero(group_id == gid, as_tuple=False).flatten()
        if idx.numel() >= 2:
            yield idx


def grouped_softmin_regret(
    predicted_cost: torch.Tensor,
    true_cost: torch.Tensor,
    group_id: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Differentiable expected selection regret under a soft argmin.

    ``softmax(-predicted_cost / temperature)`` is the relaxed planner choice.
    The target is simulator-state regret relative to the best registered
    candidate in that same population.  The target is detached deliberately:
    gradients reshape only the learned cost/encoder.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    losses = []
    for idx in _groups(group_id):
        pred = predicted_cost[idx]
        truth = true_cost[idx].detach()
        probability = torch.softmax(-pred / temperature, dim=0)
        losses.append((probability * (truth - truth.min())).sum())
    if not losses:
        raise ValueError("no group contains at least two candidates")
    return torch.stack(losses).mean()


def grouped_pairwise_logistic(
    predicted_cost: torch.Tensor,
    true_cost: torch.Tensor,
    group_id: torch.Tensor,
    *,
    min_true_gap: float = 0.0,
) -> torch.Tensor:
    """Uniform pairwise ranking loss, computed only within populations."""
    losses = []
    for idx in _groups(group_id):
        pred = predicted_cost[idx]
        truth = true_cost[idx].detach()
        i, j = torch.triu_indices(len(idx), len(idx), offset=1, device=pred.device)
        true_delta = truth[j] - truth[i]
        keep = true_delta.abs() > min_true_gap
        if not keep.any():
            continue
        # If truth[j] > truth[i], a correct cost has pred[j] > pred[i].
        signed_pred_delta = true_delta[keep].sign() * (pred[j[keep]] - pred[i[keep]])
        losses.append(F.softplus(-signed_pred_delta).mean())
    if not losses:
        raise ValueError("no non-tied within-group candidate pairs")
    return torch.stack(losses).mean()


def grouped_regression_huber(
    predicted_cost: torch.Tensor,
    true_cost: torch.Tensor,
    group_id: torch.Tensor,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    """Direct ``c*`` regression with equal weight per population.

    This is the required capacity baseline: it gets the same data, encoder
    update and cost head as the ranking arms, but no selection-tail weighting.
    """
    losses = []
    for idx in _groups(group_id):
        losses.append(F.smooth_l1_loss(
            predicted_cost[idx], true_cost[idx].detach(), beta=beta))
    if not losses:
        raise ValueError("no group contains at least two candidates")
    return torch.stack(losses).mean()


def grouped_regret_weighted_pairwise(
    predicted_cost: torch.Tensor,
    true_cost: torch.Tensor,
    group_id: torch.Tensor,
    *,
    kappa: float = 3.0,
    min_true_gap: float = 0.0,
) -> torch.Tensor:
    """Cost-sensitive pairwise ranking, weighted by the truly-worse candidate.

    ``grouped_pairwise_logistic`` treats every inversion alike, and
    ``grouped_softmin_regret`` weights by the *predicted* soft-argmin mass.
    Neither penalises the specific error ``argmin`` consumes: ranking a
    genuinely terrible candidate cheap.  Here each pair is weighted by the
    normalised true regret of its worse member, so an inversion that puts a
    high-regret candidate in the cheap tail costs ``1 + kappa`` times a swap
    between two near-equal candidates.  ``kappa = 0`` recovers the uniform
    pairwise loss, which makes this a strict generalisation and lets the
    ablation isolate the asymmetry rather than the parameterisation.
    """
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    losses = []
    for idx in _groups(group_id):
        pred = predicted_cost[idx]
        truth = true_cost[idx].detach()
        spread = truth.mean() - truth.min()
        if spread <= 0:
            continue
        regret = (truth - truth.min()) / spread
        i, j = torch.triu_indices(len(idx), len(idx), offset=1, device=pred.device)
        true_delta = truth[j] - truth[i]
        keep = true_delta.abs() > min_true_gap
        if not keep.any():
            continue
        i, j, true_delta = i[keep], j[keep], true_delta[keep]
        signed_pred_delta = true_delta.sign() * (pred[j] - pred[i])
        worse = torch.where(true_delta > 0, j, i)
        weight = 1.0 + kappa * regret[worse]
        loss = (weight * F.softplus(-signed_pred_delta)).sum() / weight.sum()
        losses.append(loss)
    if not losses:
        raise ValueError("no non-tied within-group candidate pairs")
    return torch.stack(losses).mean()


@torch.no_grad()
def grouped_hard_selection_regret(
    predicted_cost: torch.Tensor,
    true_cost: torch.Tensor,
    group_id: torch.Tensor,
) -> torch.Tensor:
    """Mean true regret of the hard argmin used by the deployed planner."""
    regrets = []
    for idx in _groups(group_id):
        chosen = torch.argmin(predicted_cost[idx])
        truth = true_cost[idx]
        regrets.append(truth[chosen] - truth.min())
    if not regrets:
        raise ValueError("no group contains at least two candidates")
    return torch.stack(regrets).mean()
