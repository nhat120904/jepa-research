from __future__ import annotations

import pytest
import torch

from planning.factorized_state_cost import factorized_state_cost


def _inputs():
    return {
        "decoded_object": torch.tensor([[2.0, 0.0, 0.0]]),
        "decoded_hand": torch.tensor([[5.0, 0.0, 0.0]]),
        "true_object": torch.tensor([[1.0, 0.0, 0.0]]),
        "true_hand": torch.tensor([[3.0, 0.0, 0.0]]),
        "decoded_goal_object": torch.tensor([0.5, 0.0, 0.0]),
        "true_goal_object": torch.tensor([0.0, 0.0, 0.0]),
        "w_hand": 0.5,
    }


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("decoded_both", 3.0),       # |2-.5| + .5|5-2|
        ("true_object", 3.0),        # |1-0| + .5|5-1|
        ("true_hand", 2.0),          # |2-.5| + .5|3-2|
        ("true_both", 2.0),          # |1-0| + .5|3-1|
    ],
)
def test_registered_arms_replace_the_intended_channels(arm, expected):
    cost = factorized_state_cost(arm, **_inputs())
    assert torch.allclose(cost, torch.tensor([expected]))


def test_unknown_arm_and_negative_weight_rejected():
    with pytest.raises(ValueError, match="unknown"):
        factorized_state_cost("bad", **_inputs())
    values = _inputs()
    values["w_hand"] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        factorized_state_cost("decoded_both", **values)
