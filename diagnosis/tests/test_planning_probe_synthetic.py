"""End-to-end sign check for the planning probe on synthetic models.

This is the offline proof that the production path (cem_plan + action_error)
separates a well-grounded model from an action-ignoring one — exactly the
separation the per-regime correlation is meant to find on real DROID data:

    grounded model      → planner recovers ~expert action → LOW action error
    action-ignoring WM  → cost flat over actions → planner can't recover → HIGH error
"""

import torch

from models.adapters.synthetic import PerfectModel, ActionIgnoringModel
from planning.cem_planner import cem_plan
from metrics.action_score import action_error


def _grounded_linear(W, action_dim):
    def lookup(z_t, a_t):
        return z_t + a_t.to(torch.float32) @ W
    return PerfectModel(lookup_fn=lookup, action_dim=action_dim)


def test_grounded_model_plans_lower_action_error_than_ignoring_model():
    torch.manual_seed(0)
    A, D = 7, 16
    W = torch.randn(A, D) * 0.5
    # A realistic small DROID-like expert action (pose deltas + gripper).
    expert = torch.tensor([[0.04, -0.03, 0.02, 0.0, 0.01, -0.01, 0.3]])
    z0 = torch.zeros(D)
    z_goal = z0 + expert.sum(0) @ W   # goal reachable by the grounded model

    common = dict(horizon=1, action_dim=A, num_samples=300, iterations=15,
                  num_elites=10, var_scale=0.1,
                  max_norms=[0.1, 0.75], max_norm_dims=[[0, 1, 2, 3, 4, 5], [6]])

    grounded = _grounded_linear(W, A)
    plan_g = cem_plan(grounded, z0, z_goal, generator=torch.Generator().manual_seed(1), **common)
    err_g = action_error(plan_g, expert)["total"]

    ignoring = ActionIgnoringModel(action_dim=A)
    plan_i = cem_plan(ignoring, z0, z_goal, generator=torch.Generator().manual_seed(1), **common)
    err_i = action_error(plan_i, expert)["total"]

    # Grounded model recovers the action well; ignoring model cannot.
    assert err_g < 0.1
    assert err_i > err_g * 3
