"""Planning objectives: latent-L2, the progress cost, and the mixture of the two.

These implement the upstream ``Objective`` protocol -- score a rolled-out ``info_dict``,
return ``(B, S)`` -- so swapping one for another changes nothing else in the loop. The
world model, the solver, the episodes and the seeds are identical across arms; only the
scalar the planner descends differs. That is what makes the comparison in this program a
controlled one.

``weight = 1.0`` reproduces ``GoalMSE`` exactly and is the reproduction anchor: an arm at
that weight must match the baseline run row for row.
"""

from __future__ import annotations

import torch
from torch import nn

from scene_progress_wm.progress_head import ProgressHead, ProgressTracker

PROTOCOL = "scene_progress_wm_objective_v1"


class ProgressCost(nn.Module):
    """Predicted remaining time after executing each candidate action sequence.

    Reads the rolled-out ``predicted_emb`` ``(B, S, H + T, D)``, whose first ``H``
    entries are the encoded context frames and whose remainder are the model's
    predictions. The head is rolled forward from the *executed* history state held by
    the tracker, over the predicted future latents paired with the candidate blocks that
    produce them.
    """

    def __init__(
        self,
        head: ProgressHead,
        tracker: ProgressTracker,
        history_size: int,
        pred_key: str = "predicted_emb",
        goal_key: str = "goal_emb",
        action_key: str = "action_candidates",
    ) -> None:
        super().__init__()
        self.head = head
        self.tracker = tracker
        self.history_size = history_size
        self.pred_key = pred_key
        self.goal_key = goal_key
        self.action_key = action_key

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred = info_dict[self.pred_key]  # (B, S, H + T, D)
        actions = info_dict[self.action_key]  # (B, S, T, action_block * action_dim)
        goal = info_dict[self.goal_key]  # (B, T_goal, D)

        batch, samples = pred.shape[0], pred.shape[1]
        horizon = actions.shape[2]
        future = pred[:, :, self.history_size : self.history_size + horizon]
        if future.shape[2] != horizon:
            raise RuntimeError(
                f"expected {horizon} predicted future frames, got {future.shape[2]}"
            )

        flat_latents = future.reshape(batch * samples, horizon, -1)
        flat_actions = actions.reshape(batch * samples, horizon, -1)

        # the executed-history state is shared by every candidate and constant for the
        # whole CEM solve; expanding it here is what keeps imagined rollouts out of the
        # head's memory of what actually happened
        hidden, progress = self.tracker.frozen()
        state = (
            hidden.to(flat_latents.dtype).expand(batch * samples, -1).contiguous(),
            progress.to(flat_latents.dtype).expand(batch * samples, -1).contiguous(),
        )

        _hiddens, _progress, final = self.head.scan(flat_latents, flat_actions, state)

        goal_latent = goal[:, -1]  # (B, D)
        goal_latent = (
            goal_latent.unsqueeze(1)
            .expand(batch, samples, goal_latent.shape[-1])
            .reshape(batch * samples, -1)
            .to(flat_latents.dtype)
        )
        return self.head.value(final, goal_latent).view(batch, samples)


class ScaledGoalMSE(nn.Module):
    """``GoalMSE`` divided by a fixed constant, so a mixture weight means something.

    Latent-L2 and predicted remaining time live on different scales; combining them
    without normalisation would make the mixture weight a meaningless dial. The divisor
    is measured once on the baseline arm and then frozen, never refit per arm.
    """

    def __init__(self, scale: float, pred_key: str = "predicted_emb", goal_key: str = "goal_emb"):
        super().__init__()
        from stable_worldmodel.planning import GoalMSE

        if scale <= 0:
            raise ValueError("scale must be positive")
        self.inner = GoalMSE(pred_key=pred_key, goal_key=goal_key)
        self.scale = float(scale)

    def forward(self, info_dict: dict) -> torch.Tensor:
        return self.inner(info_dict) / self.scale


def build_mixture(goal_term: nn.Module, progress_term: nn.Module, weight: float) -> nn.Module:
    """``weight * goal_term + (1 - weight) * progress_term``.

    At ``weight == 1`` the progress term is dropped entirely rather than multiplied by
    zero, so the baseline arm runs the identical computation to a plain ``GoalMSE`` run
    and reproduces it exactly.
    """
    from stable_worldmodel.planning import WeightedSum

    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"mixture weight must lie in [0, 1], got {weight}")
    if weight == 1.0:
        return goal_term
    if weight == 0.0:
        return progress_term
    return WeightedSum([(weight, goal_term), (1.0 - weight, progress_term)])


__all__ = ["PROTOCOL", "ProgressCost", "ScaledGoalMSE", "build_mixture"]
