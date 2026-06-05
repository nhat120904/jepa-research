"""Tests for the DROID planning Action Error / Action Score metric.

The Action Error replicates the upstream DROID metric exactly
(`evals/.../planning/plan_evaluator.py`, droid branch):

    d = |Σ_t planned[:,:3]  - Σ_t expert[:,:3]|.sum()      # xyz
      + |Σ_t planned[:,3:6] - Σ_t expert[:,3:6]|.sum()     # orientation
      + |Σ_t planned[:,6:]  - Σ_t expert[:,6:]|.sum()       # gripper closure

i.e. sum each action stream over the executed horizon (pose deltas are additive),
then L1 between the planned and expert net deltas, grouped xyz / orient / grip.
"""

import numpy as np
import torch

from metrics.action_score import action_error, rescale_action_score


def test_identical_plan_has_zero_error():
    a = torch.tensor([[0.1, -0.2, 0.05, 0.0, 0.1, -0.1, 0.7],
                      [0.0, 0.1, -0.05, 0.2, 0.0, 0.1, -0.7]])
    out = action_error(a, a)
    assert out["total"] == 0.0
    assert out["xyz"] == 0.0
    assert out["orient"] == 0.0
    assert out["grip"] == 0.0


def test_grouped_summed_delta_matches_hand_computation():
    planned = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]])
    expert = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # Σ_t planned = [0.2,0,0, 0,0,0, 1.0]; expert net = 0.
    out = action_error(planned, expert)
    assert abs(out["xyz"] - 0.2) < 1e-6   # |0.2| over xyz, only dim 0 nonzero
    assert out["orient"] == 0.0
    assert abs(out["grip"] - 1.0) < 1e-6
    assert abs(out["total"] - 1.2) < 1e-6


def test_error_uses_net_delta_not_per_step():
    # Two paths with the SAME net xyz delta must have zero xyz error,
    # because pose deltas are summed over time before comparison.
    planned = torch.tensor([[0.2, 0.0, 0.0, 0, 0, 0, 0.0],
                            [-0.1, 0.0, 0.0, 0, 0, 0, 0.0]])  # net x = 0.1
    expert = torch.tensor([[0.05, 0.0, 0.0, 0, 0, 0, 0.0],
                           [0.05, 0.0, 0.0, 0, 0, 0, 0.0]])   # net x = 0.1
    out = action_error(planned, expert)
    assert abs(out["xyz"]) < 1e-6


def test_rescale_is_one_at_zero_error_and_monotonic():
    errors = np.array([0.0, 1.0, 2.0, 4.0])
    d_ref = 4.0
    scores = rescale_action_score(errors, d_ref)
    assert abs(scores[0] - 1.0) < 1e-9          # zero error -> perfect score
    assert scores[-1] <= scores[0]              # higher error -> lower score
    assert np.all(np.diff(scores) <= 0)         # monotonic non-increasing
