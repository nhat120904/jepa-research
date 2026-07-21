import pytest
import torch

from planning.selection_objectives import (
    grouped_hard_selection_regret,
    grouped_pairwise_logistic,
    grouped_regression_huber,
    grouped_softmin_regret,
)


def test_softmin_regret_prefers_true_best_and_has_gradient():
    truth = torch.tensor([0.0, 1.0, 3.0, 10.0, 11.0])
    groups = torch.tensor([0, 0, 0, 1, 1])
    good = torch.tensor([0.0, 1.0, 3.0, 10.0, 11.0], requires_grad=True)
    bad = torch.tensor([3.0, 1.0, 0.0, 11.0, 10.0], requires_grad=True)
    good_loss = grouped_softmin_regret(good, truth, groups, temperature=0.2)
    bad_loss = grouped_softmin_regret(bad, truth, groups, temperature=0.2)
    assert good_loss < bad_loss
    bad_loss.backward()
    assert bad.grad is not None and torch.isfinite(bad.grad).all()


def test_pairwise_and_hard_regret_are_group_local():
    truth = torch.tensor([0.0, 2.0, 100.0, 101.0])
    groups = torch.tensor([0, 0, 1, 1])
    ordered = torch.tensor([-2.0, 2.0, -3.0, 3.0])
    reversed_ = -ordered
    assert grouped_pairwise_logistic(ordered, truth, groups) < grouped_pairwise_logistic(
        reversed_, truth, groups)
    assert grouped_hard_selection_regret(ordered, truth, groups).item() == 0.0
    assert grouped_hard_selection_regret(reversed_, truth, groups).item() == 1.5


def test_regression_equal_weights_groups():
    truth = torch.tensor([0.0, 1.0, 10.0, 11.0])
    groups = torch.tensor([0, 0, 1, 1])
    assert grouped_regression_huber(truth, truth, groups).item() == 0.0


def test_rejects_singletons_only():
    with pytest.raises(ValueError):
        grouped_softmin_regret(torch.ones(2), torch.ones(2), torch.arange(2))
