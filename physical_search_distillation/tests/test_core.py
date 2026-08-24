import numpy as np
import torch

from physical_search_distillation.core import hard_elite_refit, split_for_order, straight_through_refit


def test_locked_split_counts():
    values = [split_for_order(i) for i in range(128)]
    assert values.count("train") == 80
    assert values.count("val") == 16
    assert values.count("test") == 32


def test_hard_refit_matches_numpy_forward():
    rng = np.random.default_rng(3)
    actions = rng.normal(size=(20, 3, 4)).astype(np.float32)
    costs = rng.normal(size=20).astype(np.float32)
    elite, mean, std = hard_elite_refit(actions, costs, 5)
    x = torch.tensor(actions, requires_grad=True)
    score = torch.tensor(costs, requires_grad=True)
    got_mean, got_std, got_elite = straight_through_refit(x, score, 5, 0.5)
    np.testing.assert_array_equal(np.sort(got_elite.detach().numpy()), np.sort(elite))
    np.testing.assert_allclose(got_mean.detach().numpy(), mean, atol=1e-6)
    np.testing.assert_allclose(got_std.detach().numpy(), std, atol=1e-6)
    (got_mean.square().mean() + got_std.mean()).backward()
    assert score.grad is not None and torch.isfinite(score.grad).all()
