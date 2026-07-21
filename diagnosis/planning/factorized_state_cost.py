"""Factorized object/hand interventions for the oracle-dynamics cost ladder.

The deployed stateprobe cost is ``||object-goal|| + w_hand*||hand-object||``.
These arms replace the decoded object or decoded hand position with simulator
truth while leaving the other channel untouched.  ``hand`` means end-effector
xyz, matching the existing oracle cost; it does not mean gripper aperture.
"""

from __future__ import annotations

import torch


ARMS = ("decoded_both", "true_object", "true_hand", "true_both")


def factorized_state_cost(
    arm: str,
    *,
    decoded_object: torch.Tensor,
    decoded_hand: torch.Tensor,
    true_object: torch.Tensor,
    true_hand: torch.Tensor,
    decoded_goal_object: torch.Tensor,
    true_goal_object: torch.Tensor,
    w_hand: float = 0.5,
) -> torch.Tensor:
    """Return one cost per candidate for a registered factorization arm.

    Correcting the object channel also corrects the goal-object coordinate;
    otherwise the endpoint and goal remain in the decoded coordinate system.
    This makes each intervention replace one complete information channel.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown factorized cost arm {arm!r}; expected one of {ARMS}")
    if w_hand < 0:
        raise ValueError("w_hand must be non-negative")

    use_true_object = arm in {"true_object", "true_both"}
    use_true_hand = arm in {"true_hand", "true_both"}
    obj = true_object if use_true_object else decoded_object
    hand = true_hand if use_true_hand else decoded_hand
    goal = true_goal_object if use_true_object else decoded_goal_object
    return torch.linalg.vector_norm(obj - goal, dim=-1) + w_hand * torch.linalg.vector_norm(
        hand - obj, dim=-1
    )
