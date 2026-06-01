import torch

from metrics.negative_samplers import (
    random_negative, opposite_negative, hard_nn_negative, sample_negatives,
)

B, A, K = 8, 4, 16


def test_random_negative_shape_and_bounds():
    a = torch.zeros(B, A)
    neg = random_negative(a, action_bounds=(-1.0, 1.0), K=K)
    assert neg.shape == (B, K, A)
    assert neg.min() >= -1.0 - 1e-5 and neg.max() <= 1.0 + 1e-5


def test_random_negative_l1_ball():
    a = torch.zeros(B, A)
    neg = random_negative(a, action_bounds=(-1.0, 1.0), K=K, l1_radius=0.075)
    assert neg.abs().sum(-1).max() <= 0.075 + 1e-5


def test_opposite_negative_flips_gripper():
    a = torch.zeros(B, A)
    a[:, 3] = 1.0  # gripper closed
    neg = opposite_negative(a, sigma=0.0, K=K, gripper_dim=3, gripper_range=(0.0, 1.0))
    assert neg.shape == (B, K, A)
    # gripper should flip toward 0 (1 - 1 = 0)
    assert neg[..., 3].abs().mean() < 0.2


def test_hard_nn_returns_actions_from_pool():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(B, 12, generator=g)
    a = torch.randn(B, A, generator=g)
    pool_z = torch.randn(64, 12, generator=g)
    pool_a = torch.randn(64, A, generator=g)
    neg = hard_nn_negative(z, a, pool_z, pool_a, K=K, similarity_radius=10.0)
    assert neg.shape == (B, K, A)
    # every returned action must exist in the pool
    for b in range(B):
        for k in range(K):
            assert ((pool_a - neg[b, k]).abs().sum(-1) < 1e-5).any()


def test_dispatch():
    a = torch.zeros(B, A)
    assert sample_negatives("random", a_t=a, K=K, action_bounds=(-1.0, 1.0)).shape == (B, K, A)
    assert sample_negatives("opposite", a_t=a, K=K, gripper_dim=3).shape == (B, K, A)
