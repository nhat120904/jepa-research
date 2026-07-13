"""Lightweight tensor tests for the isolated ACID baseline primitives."""

from __future__ import annotations

import torch

from models.heads.acid_idm import (
    ACIDInverseDynamics,
    action_consistency_cost,
    acid_cost,
    transition_features,
)


def test_transition_features_preserve_ordered_pair():
    z0 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    z1 = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    feat01 = transition_features(z0, z1, latent_dim=2)
    feat10 = transition_features(z1, z0, latent_dim=2)
    assert feat01.shape == (1, 8)
    assert not torch.equal(feat01, feat10)


def test_action_consistency_is_mean_time_squared_l2():
    idm = ACIDInverseDynamics(latent_dim=2, action_dim=2, hidden=8)
    for parameter in idm.parameters():
        parameter.data.zero_()
    # G predicts [1, -1] for every transition.
    idm.net[-1].bias.data.copy_(torch.tensor([1.0, -1.0]))
    trajectory = torch.zeros(2, 3, 1, 2)
    actions = torch.tensor(
        [
            [[1.0, -1.0], [3.0, -1.0]],  # residual L2^2: 0, 4 -> mean 2
            [[0.0, 0.0], [0.0, 0.0]],    # residual L2^2: 2, 2 -> mean 2
        ]
    )
    got = action_consistency_cost(trajectory, actions, idm)
    torch.testing.assert_close(got, torch.tensor([2.0, 2.0]))


def test_adaptive_weight_matches_acid_equation():
    goal = torch.tensor([0.0, 1.0, 2.0])
    consistency = torch.tensor([0.0, 2.0, 4.0])
    total, diag = acid_cost(goal, consistency, lambda_acid=0.1)
    # std(goal)/std(consistency) = 1/2 for either population or sample std.
    torch.testing.assert_close(diag["acid_weight"], torch.tensor(0.05))
    torch.testing.assert_close(total, goal + 0.05 * consistency)


def test_zero_lambda_is_exact_terminal_cost_and_constant_residual_is_noop():
    goal = torch.tensor([3.0, 1.0, 2.0])
    consistency = torch.ones(3)
    null, diag0 = acid_cost(goal, consistency, lambda_acid=0.0)
    constant, diag1 = acid_cost(goal, consistency, lambda_acid=0.1)
    assert torch.equal(null, goal)
    assert torch.equal(constant, goal)
    assert diag0["acid_weight"].item() == 0.0
    assert diag1["acid_weight"].item() == 0.0
