"""Unified adapter ABC for DINO-WM, V-JEPA-2-AC, and JEPA-WM checkpoints.

All metrics talk to this interface, so the concrete adapter absorbs every
model-specific quirk (preprocessing, action normalization, proprio
conditioning, predictor forward signature, planning distance).

Latent contract
---------------
A *single-frame* latent is whatever ``encode`` returns per frame, with a
leading batch axis: shape ``(B, *frame_dims)``.

* Real models (``EncPredWMAdapter``): ``frame_dims = (V, H, W, D)`` patch-token
  grid with ``V = 1`` view (e.g. ``(B, 1, 16, 16, 384)`` for a ViT-S/14 at
  224px). Every metric flattens the tail dims for distances/MSE, so the rank
  of ``frame_dims`` does not matter to the metric code.
* Synthetic models: ``frame_dims = (d,)``.

``predict`` maps a single-frame latent (+ action, + optional proprio) to the
predicted *next* single-frame latent of the same shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AdapterSpec:
    name: str
    hub_repo: str
    hub_id: str
    dataset: str            # "metaworld" | "droid" | "robocasa" | "pusht" | ...
    image_size: int
    action_dim: int
    proprio_dim: int = 0
    latent_dim: Optional[int] = None   # filled after load if unknown
    uses_proprio: bool = False         # filled after load from model.use_proprio
    pred_type: Optional[str] = None    # "dino_wm" | "vjepa2_ac" | "AdaLN"
    planning_distance: str = "l2"      # distance the model's CEM planner uses


class WorldModelAdapter(ABC):
    """Single interface every diagnostic metric talks to."""

    spec: AdapterSpec

    # ----- encode --------------------------------------------------------------
    @abstractmethod
    def encode(self, visual: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode raw observations to single-frame latents.

        Args:
            visual: ``(B, T, C, H, W)`` RGB frames in **[0, 255]** (the model's
                own preprocessing — resize + normalize — is applied internally).
            proprio: optional ``(B, T, P)`` raw proprioception. Used only if the
                checkpoint was trained with proprio conditioning.

        Returns:
            Latent ``(B, T, *frame_dims)``.
        """

    # ----- predict -------------------------------------------------------------
    @abstractmethod
    def predict(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        proprio_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One-step prediction from a single-frame latent.

        Args:
            z_t: ``(B, *frame_dims)`` single-frame latent (tau = 1 context).
            a_t: ``(B, A)`` **raw** action (the adapter normalizes it the same
                way the model was trained — this is the #1 source of bugs).
            proprio_t: optional ``(B, P)`` raw proprioception at time t.

        Returns:
            Predicted next single-frame latent ``(B, *frame_dims)``.
        """

    def predict_rollout(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
        proprio_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Default H-step autoregressive rollout. ``actions``: ``(B, H, A)``.

        Returns ``(B, H + 1, *frame_dims)`` with ``z_t`` at index 0. Subclasses
        that maintain a multi-frame context window should override this.
        """
        rollout = [z_t]
        z = z_t
        H = actions.shape[1]
        for h in range(H):
            z = self.predict(z, actions[:, h], proprio_t=proprio_t)
            rollout.append(z)
        return torch.stack(rollout, dim=1)

    # ----- metadata ------------------------------------------------------------
    @abstractmethod
    def action_dim(self) -> int: ...

    def uses_proprio(self) -> bool:
        return bool(getattr(self.spec, "uses_proprio", False))

    @abstractmethod
    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        """Apply the same normalization the model was trained with.

        Action normalization is the #1 source of bugs in this diagnostic.
        Validate with ``scripts/sanity_check.py::check_action_normalization``.
        """

    # ----- planning distance ---------------------------------------------------
    def distance_for_planning(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """Distance the model's CEM planner uses. L2 by default.

        Returns ``(B,)``. Kept consistent with ``spec.planning_distance`` so CRA
        rankings predict planning behaviour (plan Note 2).
        """
        diff = (z_pred - z_target).reshape(z_pred.shape[0], -1)
        return diff.norm(dim=-1)

    # ----- device --------------------------------------------------------------
    @abstractmethod
    def to(self, device: torch.device | str) -> "WorldModelAdapter": ...

    @abstractmethod
    def eval(self) -> "WorldModelAdapter": ...
