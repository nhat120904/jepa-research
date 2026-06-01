"""Unified adapter for the three baselines, all of which load as ``EncPredWM``.

DINO-WM, V-JEPA-2-AC and JEPA-WM are all published in the single
``facebookresearch/jepa-wms`` repo and load through the same ``torch.hub``
entrypoint, returning ``(model, preprocessor)`` where ``model`` is an
``EncPredWM`` wrapper around a ``VideoWM``. They differ only in:

* the encoder backbone (DINOv2/DINOv3/V-JEPA-2) — irrelevant to us, frozen;
* the predictor ``pred_type`` ("dino_wm" / "vjepa2_ac" / "AdaLN") — handled
  *inside* ``VideoWM.forward_pred``, which ``EncPredWM.unroll`` calls;
* env dims + action/proprio normalization stats — carried by the preprocessor.

So one adapter class covers all three. We never reach into ``.encoder`` /
``.predictor`` directly (the old code's #1 mistake); we drive the model through
its own ``encode`` / ``unroll`` methods, exactly like the reference eval
(``evals/unroll_decode/eval.py`` and ``evals/.../planning/planner.py``).

Verified against upstream source; the live numerical path (real checkpoints on
a GPU) is gated on the server by ``scripts/smoke_test.py`` and
``scripts/sanity_check.py::check_action_normalization``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import AdapterSpec, WorldModelAdapter
from ._torchhub import load_from_hub


# hub_id -> (env_name_for_data_stats, nominal_image_size, raw_action_dim, raw_proprio_dim)
# Dims mirror app/plan_common/datasets/__init__.py::DATA_STATS. image_size is
# nominal only — the model's own preprocessor.transform does the real resize.
_HUB_ID_INFO = {
    "jepa_wm_metaworld":  ("metaworld", 256, 4, 4),
    "dino_wm_metaworld":  ("metaworld", 224, 4, 4),
    "jepa_wm_droid":      ("droid",     256, 7, 7),
    "dino_wm_droid":      ("droid",     224, 7, 7),
    "vjepa2_ac_droid":    ("droid",     256, 7, 7),
    "vjepa2_ac_oss":      ("droid",     256, 7, 7),
    "jepa_wm_pusht":      ("pusht",     224, 2, 4),
    "dino_wm_pusht":      ("pusht",     224, 2, 4),
    "jepa_wm_pointmaze":  ("pointmaze", 224, 2, 4),
    "dino_wm_pointmaze":  ("pointmaze", 224, 2, 4),
    "jepa_wm_wall":       ("wall",      224, 2, 2),
    "dino_wm_wall":       ("wall",      224, 2, 2),
    # NOTE: jepa_wm_robocasa has a hub entrypoint but NO checkpoint/config in
    # the upstream registry, so it cannot be loaded. Robocasa is run via the
    # droid-trained checkpoints (shared 7-dim action format) per the plan.
}

# All published planning configs use "L2_cem" (see hubconf _MODEL_CONFIGS),
# so latent L2 is the planner's distance for every baseline.
_PLANNING_DISTANCE = "l2"


class EncPredWMAdapter(WorldModelAdapter):
    """Drives an upstream ``EncPredWM`` checkpoint through its own API."""

    def __init__(self, hub_id: str, device: str | torch.device = "cpu", lazy: bool = False):
        if hub_id not in _HUB_ID_INFO:
            raise ValueError(f"Unknown hub id: {hub_id}. Known: {sorted(_HUB_ID_INFO)}")
        env, image_size, action_dim, proprio_dim = _HUB_ID_INFO[hub_id]
        self.env = env
        self.spec = AdapterSpec(
            name=hub_id,
            hub_repo="facebookresearch/jepa-wms",
            hub_id=hub_id,
            dataset=env,
            image_size=image_size,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            planning_distance=_PLANNING_DISTANCE,
        )
        self.device = torch.device(device)
        self.encpred = None       # EncPredWM
        self.wm = None            # VideoWM (encpred.model)
        self.preprocessor = None
        if not lazy:
            self.load()

    # ----- loading -------------------------------------------------------------
    def load(self) -> "EncPredWMAdapter":
        """Download + build the checkpoint via torch.hub (server-side)."""
        model, preprocessor = load_from_hub(self.spec.hub_id)
        self.encpred = model
        self.preprocessor = preprocessor
        self.wm = getattr(model, "model", model)  # VideoWM
        # Read the genuine config off the loaded model rather than guessing.
        self.spec.uses_proprio = bool(getattr(self.wm, "use_proprio", False))
        self.spec.pred_type = getattr(self.wm, "pred_type", None)
        self.spec.latent_dim = None
        self._model_action_dim = getattr(model, "action_dim", self.spec.action_dim)
        self.eval()
        return self

    def _require_loaded(self):
        if self.encpred is None:
            raise RuntimeError(
                f"{self.spec.hub_id} not loaded. Call .load() (needs the upstream "
                "env: clone external/jepa-wms + uv sync, GPU recommended)."
            )

    # ----- device tensors of normalization stats -------------------------------
    def _stats_to_device(self):
        # Preprocessor stats are CPU tensors; mirror them onto the model device.
        p = self.preprocessor
        for attr in ("action_mean", "action_std", "proprio_mean", "proprio_std",
                     "state_mean", "state_std"):
            v = getattr(p, attr, None)
            if isinstance(v, torch.Tensor):
                setattr(p, attr, v.to(self.device))

    # ----- encode --------------------------------------------------------------
    @torch.no_grad()
    def encode(self, visual: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """visual: (B, T, C, H, W) in [0, 255]. Returns visual latent (B, T, V, H, W, D).

        Uses ``EncPredWM.encode`` which applies /255 + the model's transform +
        the frozen encoder internally — i.e. the exact training-time pipeline.
        """
        self._require_loaded()
        from tensordict.tensordict import TensorDict

        visual = visual.to(self.device, dtype=torch.float32)
        if self.spec.uses_proprio and proprio is not None:
            obs = TensorDict(
                {"visual": visual, "proprio": proprio.to(self.device, dtype=torch.float32)},
                batch_size=[],
            )
            z = self.encpred.encode(obs)
            return z["visual"]
        # rgb-only path (e.g. all *_noprop DROID checkpoints).
        z = self.encpred.encode(visual)
        return z["visual"] if isinstance(z, dict) or hasattr(z, "keys") else z

    @torch.no_grad()
    def encode_proprio_features(self, proprio: torch.Tensor) -> torch.Tensor:
        """Encode raw proprio (B, T, P) into the predictor's proprio features."""
        self._require_loaded()
        p = self.preprocessor.normalize_proprios(proprio.to(self.device, dtype=torch.float32))
        return self.wm.encode_proprio(p)

    # ----- predict -------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        proprio_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One-step prediction via ``EncPredWM.unroll`` (the planner's primitive).

        z_t: (B, V, H, W, D) single-frame visual latent.
        a_t: (B, A) raw action; normalized here the same way the model trained.
        Returns next-frame visual latent (B, V, H, W, D).
        """
        self._require_loaded()
        from einops import rearrange
        from tensordict.tensordict import TensorDict

        B = z_t.shape[0]
        z_t = z_t.to(self.device, dtype=torch.float32)

        # Normalize + shape actions exactly like evals/unroll_decode/eval.py.
        a = a_t.to(self.device, dtype=torch.float32).reshape(B, 1, -1)
        a = self.normalize_action(a)                       # (B, 1, A_raw)
        a = a.reshape(B, -1, self._model_action_dim)       # tubelet/frameskip stack
        act_suffix = rearrange(a, "b t a -> t b a")        # (T=1, B, A)

        z_ctxt_visual = z_t.unsqueeze(1)                   # (B, tau=1, V, H, W, D)
        if self.spec.uses_proprio and proprio_t is not None:
            prop_feat = self.encode_proprio_features(proprio_t.reshape(B, 1, -1))  # (B,1,tok,D)
            z_ctxt = TensorDict({"visual": z_ctxt_visual, "proprio": prop_feat}, batch_size=[])
            pred = self.encpred.unroll(z_ctxt, act_suffix=act_suffix)
            pred_visual = pred["visual"]
        else:
            pred = self.encpred.unroll(z_ctxt_visual, act_suffix=act_suffix)
            pred_visual = pred
        # unroll returns time-first (tau+T, B, V, H, W, D); take the last step.
        return pred_visual[-1]

    @torch.no_grad()
    def predict_rollout(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
        proprio_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """H-step rollout via a single ``unroll`` call (keeps the sliding context).

        actions: (B, H, A) raw. Returns (B, H+1, V, H, W, D) with z_t at index 0.
        """
        self._require_loaded()
        from einops import rearrange
        from tensordict.tensordict import TensorDict

        B, H, _ = actions.shape
        z_t = z_t.to(self.device, dtype=torch.float32)
        a = self.normalize_action(actions.to(self.device, dtype=torch.float32))
        a = a.reshape(B, -1, self._model_action_dim)
        act_suffix = rearrange(a, "b t a -> t b a")        # (H, B, A)

        z_ctxt_visual = z_t.unsqueeze(1)
        if self.spec.uses_proprio and proprio_t is not None:
            prop_feat = self.encode_proprio_features(proprio_t.reshape(B, 1, -1))
            z_ctxt = TensorDict({"visual": z_ctxt_visual, "proprio": prop_feat}, batch_size=[])
            pred = self.encpred.unroll(z_ctxt, act_suffix=act_suffix)["visual"]
        else:
            pred = self.encpred.unroll(z_ctxt_visual, act_suffix=act_suffix)
        # (tau+H, B, ...) time-first -> (B, tau+H, ...)
        return rearrange(pred, "t b ... -> b t ...")

    # ----- metadata ------------------------------------------------------------
    def action_dim(self) -> int:
        return self.spec.action_dim

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        """Apply the model's action normalization: (a - mean) / std (DATA_STATS).

        For DROID/RoboCasa this is identity (mean=0, std=1); for Metaworld it is
        a real shift+scale. This is the #1 bug source — validate on the server
        with ``check_action_normalization``.
        """
        self._require_loaded()
        a = a.to(self.device, dtype=torch.float32)
        return self.preprocessor.normalize_actions(a)

    def distance_for_planning(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        if self.spec.planning_distance == "cosine":
            from metrics.distances import cosine_distance
            return cosine_distance(z_pred, z_target)
        diff = (z_pred - z_target).reshape(z_pred.shape[0], -1)
        return diff.norm(dim=-1)

    # ----- device --------------------------------------------------------------
    def to(self, device):
        self.device = torch.device(device)
        if self.encpred is not None:
            self.encpred = self.encpred.to(self.device)
            self.wm = getattr(self.encpred, "model", self.encpred)
            self._stats_to_device()
        return self

    def eval(self):
        if self.encpred is not None:
            self.encpred.eval()
        return self
