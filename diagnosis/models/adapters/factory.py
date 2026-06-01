"""Build an adapter from a hub identifier string."""

from __future__ import annotations

from .base import WorldModelAdapter
from .enc_pred_adapter import EncPredWMAdapter, _HUB_ID_INFO


def build_adapter(model_name: str, device: str = "cpu", lazy: bool = False) -> WorldModelAdapter:
    """Dispatch to the right concrete adapter class.

    All three baselines (jepa_wm_*, dino_wm_*, vjepa2_ac_*) load as the same
    ``EncPredWM`` wrapper, so they share one adapter. ``lazy=True`` builds the
    adapter spec without downloading the checkpoint (useful for tests / dry
    runs on a machine without the upstream env).
    """
    if model_name in _HUB_ID_INFO:
        return EncPredWMAdapter(model_name, device=device, lazy=lazy)
    raise ValueError(
        f"Unknown model identifier: {model_name}. "
        f"Known checkpoints: {sorted(_HUB_ID_INFO)}"
    )
