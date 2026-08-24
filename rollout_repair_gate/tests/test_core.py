from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rollout_repair_gate.core import (
    autoregressive_rollout,
    masked_prediction_mse,
    split_for_order,
    teacher_forced_rollout,
)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_encoder = nn.Identity()

    def predict(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return embeddings + actions


def test_split_counts() -> None:
    values = [split_for_order(i) for i in range(128)]
    assert values.count("train") == 80
    assert values.count("val") == 16
    assert values.count("test") == 32


def test_teacher_forcing_and_autoregression_diverge() -> None:
    model = ToyModel()
    z0 = torch.tensor([[0.0]])
    actions = torch.tensor([[[1.0], [1.0], [1.0]]])
    true = torch.tensor([[[0.0], [10.0], [20.0], [30.0]]])
    autoregressive = autoregressive_rollout(model, z0, actions, history_size=1)
    teacher = teacher_forced_rollout(model, true, actions, history_size=1)
    assert torch.equal(autoregressive[..., 0], torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(teacher[..., 0], torch.tensor([[1.0, 11.0, 21.0]]))


def test_masked_mse_ignores_invalid_horizons() -> None:
    predicted = torch.tensor([[[0.0], [100.0]]])
    target = torch.zeros_like(predicted)
    valid = torch.tensor([[True, False]])
    assert masked_prediction_mse(predicted, target, valid).item() == 0.0

