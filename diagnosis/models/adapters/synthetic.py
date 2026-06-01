"""Synthetic adapters used to validate metric implementations before real eval.

The plan (Task 2.4) requires that we verify the metric code with three
controllable synthetic predictors:

- PerfectModel        → CRA ≈ 1.0, AUG > 0
- ActionIgnoringModel → CRA ≈ 1/(K+1), AUG ≈ 0
- RandomModel         → CRA ≈ 1/(K+1), AUG ≈ 0

If any of these fail, the metric implementation is buggy.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .base import AdapterSpec, WorldModelAdapter


class _NoopAdapterMixin:
    """Shared no-op preprocessing/device handling for synthetic adapters."""

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def preprocess_image(self, img: torch.Tensor) -> torch.Tensor:
        return img

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self


class PerfectModel(_NoopAdapterMixin, WorldModelAdapter):
    """Oracle: returns the true z_{t+1} for the queried (z_t, a_t)."""

    def __init__(self, lookup_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 action_dim: int = 4):
        self.spec = AdapterSpec(
            name="synthetic_perfect", hub_repo="", hub_id="", dataset="synthetic",
            image_size=224, action_dim=action_dim,
        )
        self._lookup_fn = lookup_fn
        self.device = torch.device("cpu")

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def predict(self, z_t, a_t, proprio_t=None) -> torch.Tensor:
        return self._lookup_fn(z_t, a_t)

    def action_dim(self) -> int:
        return self.spec.action_dim


class ActionIgnoringModel(_NoopAdapterMixin, WorldModelAdapter):
    """Returns z_t regardless of a_t — the pathology we are trying to detect."""

    def __init__(self, action_dim: int = 4):
        self.spec = AdapterSpec(
            name="synthetic_action_ignoring", hub_repo="", hub_id="", dataset="synthetic",
            image_size=224, action_dim=action_dim,
        )
        self.device = torch.device("cpu")

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def predict(self, z_t, a_t, proprio_t=None) -> torch.Tensor:
        return z_t

    def action_dim(self) -> int:
        return self.spec.action_dim


class RandomModel(_NoopAdapterMixin, WorldModelAdapter):
    """Returns z_t + Gaussian noise — also action-blind, gives ~chance CRA."""

    def __init__(self, action_dim: int = 4, sigma: float = 0.1, seed: Optional[int] = None):
        self.spec = AdapterSpec(
            name="synthetic_random", hub_repo="", hub_id="", dataset="synthetic",
            image_size=224, action_dim=action_dim,
        )
        self.sigma = sigma
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)
        self.device = torch.device("cpu")

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def predict(self, z_t, a_t, proprio_t=None) -> torch.Tensor:
        noise = torch.randn(z_t.shape, generator=self.gen).to(z_t.device) * self.sigma
        return z_t + noise

    def action_dim(self) -> int:
        return self.spec.action_dim
