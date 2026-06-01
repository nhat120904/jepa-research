"""Shared torch.hub loading helpers — used by every concrete adapter.

The unified `facebookresearch/jepa-wms` repo exposes all three baselines
(DINO-WM, V-JEPA-2-AC, JEPA-WM/Terver) through `torch.hub.load`. The hub
entry returns ``(model, preprocessor)``; the preprocessor encapsulates
per-model image preprocessing (resize, mean/std, channel order).
"""

from __future__ import annotations

from typing import Any, Tuple

import torch


HUB_REPO = "facebookresearch/jepa-wms"


def load_from_hub(hub_id: str, repo: str = HUB_REPO) -> Tuple[Any, Any]:
    """Load (model, preprocessor) from the unified jepa-wms hub.

    If torch.hub returns 503 errors, the plan says: delete
    `external/jepa-wms/uv.lock` and re-run `uv sync` inside that repo.
    """
    return torch.hub.load(repo, hub_id, trust_repo=True)
