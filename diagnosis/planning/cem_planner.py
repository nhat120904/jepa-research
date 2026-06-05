"""CEM planner — faithful port of the upstream ``CEMPlanner`` (the ``L2_cem`` the
DROID/Metaworld planning configs use), driven through a ``WorldModelAdapter``.

Reference: ``external/jepa-wms/evals/simu_env_planning/planning/planning/planner.py::CEMPlanner``
and the DROID dino-wm config (iterations 15, num_samples 300, num_elites 10,
horizon 3, var_scale 0.1, momentum 0, max_norms [0.1, 0.75],
max_norm_dims [[0,1,2,3,4,5],[6]]).

Objective: ``ReprTargetDistMPCObjective`` with ``sum_all_diffs=False`` — the
planning cost is the **MSE between the last unrolled latent and the goal latent**
(mean over feature dims), no proprio term (``alpha=0``). This mirrors
``adapter.distance_for_planning`` but as a *mean-squared* cost, exactly as upstream.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from models.adapters import WorldModelAdapter


def _clip_actions(actions: torch.Tensor, max_norms, max_norm_dims) -> torch.Tensor:
    """Box-clip per upstream: for each (dims, maxnorm) group, clamp to [-maxnorm, maxnorm]."""
    if max_norms is None:
        return actions
    for dims, maxnorm in zip(max_norm_dims, max_norms):
        actions[..., dims] = torch.clamp(actions[..., dims], min=-maxnorm, max=maxnorm)
    return actions


def _goal_mse(pred_last: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """MSE(last latent, goal) averaged over feature dims → (B,). Matches L2 objective."""
    B = pred_last.shape[0]
    diff = (pred_last.reshape(B, -1) - z_goal.reshape(1, -1)) ** 2
    return diff.mean(dim=-1)


@torch.no_grad()
def cem_plan(
    adapter: WorldModelAdapter,
    z_init: torch.Tensor,
    z_goal: torch.Tensor,
    *,
    horizon: int,
    action_dim: int,
    num_samples: int = 300,
    iterations: int = 15,
    num_elites: int = 10,
    var_scale: float = 0.1,
    max_norms: Optional[List[float]] = (0.1, 0.75),
    max_norm_dims: Optional[List[List[int]]] = ([0, 1, 2, 3, 4, 5], [6]),
    num_act_stepped: Optional[int] = None,
    proprio_t: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Plan an action sequence to reach ``z_goal`` from ``z_init`` via CEM.

    Args:
        z_init: ``(*frame)`` or ``(1, *frame)`` single-frame latent to plan from.
        z_goal: ``(*frame)`` or ``(1, *frame)`` target latent.
        horizon: number of model steps to plan.
        action_dim: per-step action dimension (the adapter's model_action_dim).
        max_norms / max_norm_dims: box-clip groups (``None`` disables clipping).
        num_act_stepped: how many leading actions to return (default: all ``horizon``).

    Returns: ``(num_act_stepped, action_dim)`` planned actions (raw, un-normalized).
    """
    if max_norms is not None:
        max_norms = list(max_norms)
        max_norm_dims = [list(g) for g in max_norm_dims]
    # Caller passes z_init / z_goal as a single frame (no batch dim); we add it.
    device = z_init.device
    frame_shape = tuple(z_init.shape)
    z_goal = z_goal.reshape(1, *frame_shape).to(device, torch.float32)
    z_batch = z_init.reshape(1, *frame_shape).expand(num_samples, *frame_shape).to(
        device, torch.float32).contiguous()

    if proprio_t is not None:
        proprio_t = proprio_t.reshape(1, -1).expand(num_samples, -1).to(device, torch.float32)

    mean = torch.zeros(horizon, action_dim, device=device)
    std = var_scale * torch.ones(horizon, action_dim, device=device)

    for _ in range(iterations):
        noise = torch.randn(num_samples, horizon, action_dim, device=device, generator=generator)
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise   # (num_samples, H, A)
        actions[0] = mean                                        # mean-inclusion trick
        actions = _clip_actions(actions, max_norms, max_norm_dims)

        pred = adapter.predict_rollout(z_batch, actions, proprio_t=proprio_t)  # (num_samples, H+1, *frame)
        cost = _goal_mse(pred[:, -1], z_goal)                    # (num_samples,)

        elite_idx = torch.topk(-cost, num_elites, dim=0).indices
        elites = actions[elite_idx]                              # (num_elites, H, A)
        mean = elites.mean(dim=0)                                # momentum 0 → straight replace
        std = elites.std(dim=0)

    k = horizon if num_act_stepped is None else num_act_stepped
    return mean[:k].detach()
