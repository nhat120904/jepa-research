from __future__ import annotations

import numpy as np
import torch

from order_jepa.core import (
    angular_distance,
    order_vector_metrics,
    reacher_physical_cost,
    reacher_success,
    selection_regret,
)
from order_jepa.loss import order_effect_loss, paired_error_decomposition


def test_angular_distance_wraps() -> None:
    assert np.isclose(angular_distance(0.01, 2 * np.pi - 0.01), 0.02)


def test_selection_regret_uses_same_population() -> None:
    selected, regret = selection_regret(np.array([2.0, 0.0, 1.0]), np.array([0.0, 3.0, 1.0]))
    assert selected == 1
    assert regret == 3.0


def test_order_loss_is_zero_for_correct_signed_difference() -> None:
    target_ij = torch.tensor([[2.0, 3.0]])
    target_ji = torch.tensor([[1.0, 1.0]])
    pred_ij = torch.tensor([[5.0, 7.0]])
    pred_ji = torch.tensor([[4.0, 5.0]])
    assert order_effect_loss(pred_ij, pred_ji, target_ij, target_ji).item() == 0.0


def test_paired_decomposition_identity() -> None:
    gen = torch.Generator().manual_seed(7)
    tensors = [torch.randn(4, 8, generator=gen) for _ in range(4)]
    branch, mean, difference = paired_error_decomposition(*tensors)
    assert torch.allclose(branch, mean + difference, atol=1e-5)


def test_order_metrics_detect_commuting_predictor() -> None:
    metrics = order_vector_metrics(
        np.array([1.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    assert metrics.true_norm == 1.0
    assert metrics.predicted_norm == 0.0
    assert metrics.normalized_error == 1.0


def test_reacher_cost_matches_success_threshold_and_wraps_angles() -> None:
    goal = np.array([np.pi - 0.01, -np.pi + 0.01])
    states = np.array(
        [
            [-np.pi + 0.02, np.pi - 0.02],
            [np.pi - 0.07, -np.pi + 0.01],
        ]
    )
    np.testing.assert_allclose(
        reacher_physical_cost(states, goal), [0.6, 1.2], atol=1e-12
    )
    np.testing.assert_array_equal(reacher_success(states, goal), [True, False])


def main() -> None:
    tests = [
        test_angular_distance_wraps,
        test_selection_regret_uses_same_population,
        test_order_loss_is_zero_for_correct_signed_difference,
        test_paired_decomposition_identity,
        test_order_metrics_detect_commuting_predictor,
        test_reacher_cost_matches_success_threshold_and_wraps_angles,
    ]
    for test in tests:
        test()
    print(f"PASS {len(tests)} ORDER-JEPA core tests")


if __name__ == "__main__":
    main()
