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

from typing import Any, Dict, List, Optional

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


def _clip_fraction(actions: torch.Tensor, max_norms, max_norm_dims) -> torch.Tensor:
    """Fraction of sampled action entries sitting on a configured clip boundary."""
    if max_norms is None:
        return torch.tensor(0.0, device=actions.device)
    clipped = torch.zeros_like(actions, dtype=torch.bool)
    eps = 1e-6
    for dims, maxnorm in zip(max_norm_dims, max_norms):
        vals = actions[..., dims].abs()
        clipped[..., dims] = vals >= (float(maxnorm) - eps)
    return clipped.to(torch.float32).mean()


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
    cost_fn=None,
    traj_cost_fn=None,
    init_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Plan an action sequence to reach ``z_goal`` from ``z_init`` via CEM.

    Args:
        z_init: ``(*frame)`` or ``(1, *frame)`` single-frame latent to plan from.
        z_goal: ``(*frame)`` or ``(1, *frame)`` target latent.
        horizon: number of model steps to plan.
        action_dim: per-step action dimension (the adapter's model_action_dim).
        max_norms / max_norm_dims: box-clip groups (``None`` disables clipping).
        num_act_stepped: how many leading actions to return (default: all ``horizon``).
        cost_fn: optional ``(pred_last (B,*frame), z_goal (1,*frame)) -> (B,)``
            planning cost. Default: the upstream L2 objective (``_goal_mse``).
        traj_cost_fn: optional ``(pred (B,H+1,*frame), actions (B,H,A), z_goal)
            -> (B,)`` — takes the whole rollout; overrides ``cost_fn`` when set
            (needed for per-step grounded-dynamics costs).
        init_mean: optional ``(H, action_dim)`` (or ``(action_dim,)`` broadcast
            over the horizon) initial CEM mean — the contact-aware *seed* from the
            inverse action proposal (default: zeros, the upstream behaviour). CEM
            then refines from this proposal instead of from a zero-mean Gaussian.

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

    if init_mean is None:
        mean = torch.zeros(horizon, action_dim, device=device)
    else:
        m = init_mean.to(device, torch.float32)
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(horizon, action_dim)
        mean = _clip_actions(m[:horizon].contiguous(), max_norms, max_norm_dims)
    std = var_scale * torch.ones(horizon, action_dim, device=device)

    for _ in range(iterations):
        noise = torch.randn(num_samples, horizon, action_dim, device=device, generator=generator)
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise   # (num_samples, H, A)
        actions[0] = mean                                        # mean-inclusion trick
        actions = _clip_actions(actions, max_norms, max_norm_dims)

        pred = adapter.predict_rollout(z_batch, actions, proprio_t=proprio_t)  # (num_samples, H+1, *frame)
        if traj_cost_fn is not None:
            cost = traj_cost_fn(pred, actions, z_goal)           # (num_samples,)
        else:
            cost = (cost_fn or _goal_mse)(pred[:, -1], z_goal)   # (num_samples,)

        elite_idx = torch.topk(-cost, num_elites, dim=0).indices
        elites = actions[elite_idx]                              # (num_elites, H, A)
        mean = elites.mean(dim=0)                                # momentum 0 → straight replace
        std = elites.std(dim=0)

    k = horizon if num_act_stepped is None else num_act_stepped
    return mean[:k].detach()


@torch.no_grad()
def cem_plan_trace(
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
    cost_fn=None,
    traj_cost_fn=None,
    init_mean: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Run CEM and return the same final plan plus per-iteration diagnostics.

    This is an additive notebook/debugging helper. Its sampling and update order
    intentionally mirrors ``cem_plan`` so the final plan is identical under the
    same fixed ``generator``.
    """
    if max_norms is not None:
        max_norms = list(max_norms)
        max_norm_dims = [list(g) for g in max_norm_dims]
    device = z_init.device
    frame_shape = tuple(z_init.shape)
    z_goal = z_goal.reshape(1, *frame_shape).to(device, torch.float32)
    z_batch = z_init.reshape(1, *frame_shape).expand(num_samples, *frame_shape).to(
        device, torch.float32).contiguous()

    if proprio_t is not None:
        proprio_t = proprio_t.reshape(1, -1).expand(num_samples, -1).to(device, torch.float32)

    if init_mean is None:
        mean = torch.zeros(horizon, action_dim, device=device)
    else:
        m = init_mean.to(device, torch.float32)
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(horizon, action_dim)
        mean = _clip_actions(m[:horizon].contiguous(), max_norms, max_norm_dims)
    std = var_scale * torch.ones(horizon, action_dim, device=device)

    trace = []
    for it in range(iterations):
        mean_before = mean.clone()
        std_before = std.clone()
        noise = torch.randn(num_samples, horizon, action_dim, device=device, generator=generator)
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        actions[0] = mean
        actions = _clip_actions(actions, max_norms, max_norm_dims)

        pred = adapter.predict_rollout(z_batch, actions, proprio_t=proprio_t)
        if traj_cost_fn is not None:
            cost = traj_cost_fn(pred, actions, z_goal)
        else:
            cost = (cost_fn or _goal_mse)(pred[:, -1], z_goal)

        elite_idx = torch.topk(-cost, num_elites, dim=0).indices
        elites = actions[elite_idx]
        elite_cost = cost[elite_idx]
        best_idx = torch.argmin(cost)
        mean = elites.mean(dim=0)
        std = elites.std(dim=0)

        action_norm = actions.reshape(num_samples, -1).norm(dim=-1)
        elite_action_norm = elites.reshape(num_elites, -1).norm(dim=-1)
        trace.append({
            "iteration": it,
            "cost": cost.detach().cpu(),
            "elite_cost": elite_cost.detach().cpu(),
            "best_cost": cost[best_idx].detach().cpu(),
            "mean_cost": cost.mean().detach().cpu(),
            "median_cost": cost.median().detach().cpu(),
            "elite_min_cost": elite_cost.min().detach().cpu(),
            "elite_max_cost": elite_cost.max().detach().cpu(),
            "clip_fraction": _clip_fraction(actions, max_norms, max_norm_dims).detach().cpu(),
            "action_norm_mean": action_norm.mean().detach().cpu(),
            "action_norm_median": action_norm.median().detach().cpu(),
            "elite_action_norm_mean": elite_action_norm.mean().detach().cpu(),
            "sample_action_mean": actions.mean(dim=0).detach().cpu(),
            "sample_action_std": actions.std(dim=0).detach().cpu(),
            "mean_before": mean_before.detach().cpu(),
            "std_before": std_before.detach().cpu(),
            "mean_after": mean.detach().cpu(),
            "std_after": std.detach().cpu(),
            "best_action": actions[best_idx].detach().cpu(),
        })

    k = horizon if num_act_stepped is None else num_act_stepped
    plan = mean[:k].detach()
    return {
        "plan": plan,
        "trace": trace,
        "final_mean": mean.detach().cpu(),
        "final_std": std.detach().cpu(),
        "config": {
            "horizon": horizon,
            "action_dim": action_dim,
            "num_samples": num_samples,
            "iterations": iterations,
            "num_elites": num_elites,
            "var_scale": var_scale,
            "max_norms": max_norms,
            "max_norm_dims": max_norm_dims,
            "num_act_stepped": num_act_stepped,
        },
    }
