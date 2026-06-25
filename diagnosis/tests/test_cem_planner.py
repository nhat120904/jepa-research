"""Tests for the CEM planner port (faithful to upstream ``CEMPlanner``).

We use a linear toy world model ``z' = z + a @ W`` exposed through the synthetic
``PerfectModel`` adapter, so a unique optimal action exists and CEM must recover it.
"""

import torch

from models.adapters.synthetic import PerfectModel
from planning.cem_planner import cem_plan, cem_plan_trace


def _linear_adapter(W: torch.Tensor, action_dim: int) -> PerfectModel:
    """PerfectModel whose one-step prediction is z_t + a_t @ W."""
    def lookup(z_t, a_t):
        return z_t + a_t.to(torch.float32) @ W
    return PerfectModel(lookup_fn=lookup, action_dim=action_dim)


def test_recovers_known_action_one_step():
    torch.manual_seed(0)
    A, D = 2, 4
    W = torch.randn(A, D)
    adapter = _linear_adapter(W, action_dim=A)

    z0 = torch.zeros(D)
    a_star = torch.tensor([0.05, -0.04])
    z_goal = z0 + a_star @ W

    gen = torch.Generator().manual_seed(123)
    planned = cem_plan(
        adapter, z0, z_goal, horizon=1, action_dim=A,
        num_samples=512, iterations=12, num_elites=32, var_scale=0.1,
        max_norms=None, generator=gen,
    )
    assert planned.shape == (1, A)
    assert torch.allclose(planned[0], a_star, atol=2e-2)


def test_respects_box_clipping():
    torch.manual_seed(1)
    A, D = 2, 4
    W = torch.randn(A, D)
    adapter = _linear_adapter(W, action_dim=A)

    z0 = torch.zeros(D)
    # Goal needs a large action, but clipping must keep |a| <= 0.1.
    a_big = torch.tensor([0.9, -0.8])
    z_goal = z0 + a_big @ W

    gen = torch.Generator().manual_seed(7)
    planned = cem_plan(
        adapter, z0, z_goal, horizon=1, action_dim=A,
        num_samples=256, iterations=8, num_elites=16, var_scale=0.5,
        max_norms=[0.1], max_norm_dims=[[0, 1]], generator=gen,
    )
    assert planned.abs().max().item() <= 0.1 + 1e-6


def test_deterministic_under_fixed_generator():
    torch.manual_seed(2)
    A, D = 3, 5
    W = torch.randn(A, D)
    adapter = _linear_adapter(W, action_dim=A)
    z0 = torch.zeros(D)
    z_goal = z0 + torch.tensor([0.02, 0.0, -0.03]) @ W

    p1 = cem_plan(adapter, z0, z_goal, horizon=1, action_dim=A,
                  num_samples=128, iterations=5, num_elites=16, var_scale=0.1,
                  max_norms=None, generator=torch.Generator().manual_seed(99))
    p2 = cem_plan(adapter, z0, z_goal, horizon=1, action_dim=A,
                  num_samples=128, iterations=5, num_elites=16, var_scale=0.1,
                  max_norms=None, generator=torch.Generator().manual_seed(99))
    assert torch.equal(p1, p2)


def test_multistep_horizon_returns_h_actions():
    torch.manual_seed(3)
    A, D = 2, 4
    W = torch.randn(A, D)
    adapter = _linear_adapter(W, action_dim=A)
    z0 = torch.zeros(D)
    # net 2-step delta from two small actions
    a_seq = torch.tensor([[0.03, -0.02], [0.01, 0.02]])
    z_goal = z0 + a_seq.sum(0) @ W  # linear: only the sum matters

    planned = cem_plan(adapter, z0, z_goal, horizon=2, action_dim=A,
                       num_samples=512, iterations=12, num_elites=32, var_scale=0.1,
                       max_norms=None, generator=torch.Generator().manual_seed(5))
    assert planned.shape == (2, A)
    # The summed planned delta should match the summed target delta.
    assert torch.allclose(planned.sum(0), a_seq.sum(0), atol=3e-2)


def test_trace_matches_cem_plan_under_fixed_generator():
    torch.manual_seed(4)
    A, D = 3, 5
    W = torch.randn(A, D)
    adapter = _linear_adapter(W, action_dim=A)
    z0 = torch.zeros(D)
    z_goal = z0 + torch.tensor([0.02, -0.01, 0.03]) @ W
    common = dict(
        horizon=2,
        action_dim=A,
        num_samples=128,
        iterations=6,
        num_elites=16,
        var_scale=0.1,
        max_norms=None,
    )

    planned = cem_plan(
        adapter, z0, z_goal, generator=torch.Generator().manual_seed(11), **common)
    traced = cem_plan_trace(
        adapter, z0, z_goal, generator=torch.Generator().manual_seed(11), **common)

    assert torch.equal(traced["plan"], planned)


def test_trace_records_iteration_costs_and_action_stats():
    torch.manual_seed(5)
    A, D = 2, 4
    W = torch.randn(A, D)
    adapter = _linear_adapter(W, action_dim=A)
    z0 = torch.zeros(D)
    z_goal = z0 + torch.tensor([0.04, -0.02]) @ W

    traced = cem_plan_trace(
        adapter, z0, z_goal, horizon=3, action_dim=A,
        num_samples=64, iterations=4, num_elites=8, var_scale=0.1,
        max_norms=[0.1], max_norm_dims=[[0, 1]],
        generator=torch.Generator().manual_seed(17),
    )

    assert len(traced["trace"]) == 4
    assert traced["plan"].shape == (3, A)
    for row in traced["trace"]:
        assert row["cost"].shape == (64,)
        assert row["elite_cost"].shape == (8,)
        assert row["sample_action_mean"].shape == (3, A)
        assert row["sample_action_std"].shape == (3, A)
        assert row["mean_before"].shape == (3, A)
        assert row["mean_after"].shape == (3, A)
        assert torch.isfinite(row["cost"]).all()
        assert torch.isfinite(row["elite_cost"]).all()
        assert torch.isfinite(row["best_cost"])
        assert torch.isfinite(row["mean_cost"])
        assert torch.isfinite(row["median_cost"])
        assert 0.0 <= float(row["clip_fraction"]) <= 1.0
