"""DROID planning Action Error / Action Score.

Replicates the upstream DROID metric exactly (the ``"droid" in task`` branch of
``evals/simu_env_planning/planning/plan_evaluator.py``):

    total_delta = | Σ_t planned − Σ_t expert |        # per action dim
    error_xyz    = total_delta[:3].sum()
    error_orient = total_delta[3:6].sum()
    error_grip   = total_delta[6:].sum()
    error_total  = error_xyz + error_orient + error_grip

DROID actions are pose deltas, so summing over the executed horizon gives the
net displacement; the metric compares the planner's net delta to the expert's.
``Action Score`` rescales the *opposite* of this error to a maximize-able number
(the paper's Table-1 convention), using a dataset-wide reference so regimes are
comparable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

# Action-dim groups for the 7-dim DROID action (xyz / orientation / gripper).
XYZ = slice(0, 3)
ORIENT = slice(3, 6)
GRIP = slice(6, None)


def action_error(planned: torch.Tensor, expert: torch.Tensor) -> Dict[str, float]:
    """Grouped summed-delta L1 error between a planned and expert action sequence.

    Args:
        planned: ``(T, A)`` planned actions over the executed horizon (raw, un-normalized).
        expert:  ``(T, A)`` ground-truth actions over the same horizon.

    Returns: dict with ``xyz``, ``orient``, ``grip``, ``total`` (floats).
    """
    planned = torch.as_tensor(planned, dtype=torch.float32)
    expert = torch.as_tensor(expert, dtype=torch.float32)
    total_delta = torch.abs(planned.sum(0) - expert.sum(0))   # (A,)
    xyz = float(total_delta[XYZ].sum().item())
    orient = float(total_delta[ORIENT].sum().item())
    grip = float(total_delta[GRIP].sum().item())
    return {"xyz": xyz, "orient": orient, "grip": grip, "total": xyz + orient + grip}


def rescale_action_score(errors: np.ndarray, d_ref: float) -> np.ndarray:
    """Rescale Action Error to an Action Score (higher is better): ``1 − d / d_ref``.

    ``d_ref`` is a dataset-wide reference (e.g. the p95 of the error) so the score
    is comparable across regimes. Scores are clipped at 0 below (errors past the
    reference saturate rather than going negative-unbounded).
    """
    errors = np.asarray(errors, dtype=np.float64)
    if d_ref <= 0:
        return np.ones_like(errors)
    return np.clip(1.0 - errors / d_ref, a_min=0.0, a_max=1.0)
