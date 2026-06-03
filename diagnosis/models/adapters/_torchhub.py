"""Shared torch.hub loading helpers — used by every concrete adapter.

The unified `facebookresearch/jepa-wms` repo exposes all three baselines
(DINO-WM, V-JEPA-2-AC, JEPA-WM/Terver) through `torch.hub.load`. The hub
entry returns ``(model, preprocessor)``; the preprocessor encapsulates
per-model image preprocessing (resize, mean/std, channel order).
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, Tuple

import torch


HUB_REPO = "facebookresearch/jepa-wms"

# Module the upstream hubconf imports purely to obtain ``init_module``.
_EVAL_MODULE = "evals.simu_env_planning.eval"


def _install_lightweight_eval_stub() -> None:
    """Avoid importing the upstream env/planning stack just to load weights.

    ``hubconf._load_model_with_config`` does
    ``from evals.simu_env_planning.eval import init_module`` — but that module
    pulls in the entire simulation/planning stack at import time (gym, pygame,
    pymunk, nevergrad, …) which the diagnostic never uses and which is painful
    to install. The upstream hubconf docstring itself states datasets/envs are
    *not* needed to load pretrained models.

    ``init_module`` there is just a dispatcher that imports ``module_name`` and
    calls *its* ``init_module`` (the real model builder lives in
    ``app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds``). We pre-register
    a stub module exposing an equivalent dispatcher in ``sys.modules`` so the
    heavy real module is never executed. ``evals`` /
    ``evals.simu_env_planning`` are PEP-420 namespace packages (no __init__), so
    nothing else runs. Idempotent; only installs the stub if absent.
    """
    if _EVAL_MODULE in sys.modules:
        return

    stub = types.ModuleType(_EVAL_MODULE)

    def init_module(folder, checkpoint, module_name, model_kwargs, device,
                    cfgs_data=None, wrapper_kwargs=None, action_dim=None,
                    proprio_dim=None, preprocessor=None):
        return importlib.import_module(module_name).init_module(
            folder=folder, checkpoint=checkpoint, model_kwargs=model_kwargs,
            device=device, action_dim=action_dim, proprio_dim=proprio_dim,
            preprocessor=preprocessor, cfgs_data=cfgs_data,
            wrapper_kwargs=wrapper_kwargs,
        )

    stub.init_module = init_module
    sys.modules[_EVAL_MODULE] = stub


def _hub_repo_dir(repo: str = HUB_REPO):
    """Path of the cached upstream repo inside the torch hub dir, if present."""
    import os
    from pathlib import Path
    owner_name = repo.replace("/", "_")
    d = Path(torch.hub.get_dir()) / f"{owner_name}_main"
    return d if d.exists() else None


def _strip_nonpublic_head_checkpoints(repo: str = HUB_REPO) -> None:
    """Drop ``pretrain_dec_path`` entries that point at non-public local paths.

    ``dino_wm_metaworld``'s config asks for a ``state_head`` checkpoint at
    ``${JEPAWM_LOGS}/.../jepa-latest.pth.tar`` — a training artifact that is not
    published, so loading it hard-fails with FileNotFoundError. That head is a
    *decoder* the diagnostic never uses (we only drive ``encode``/``unroll`` on
    the visual latent). ``init_module`` skips loading a head whose
    ``pretrain_dec_path`` entry is absent (the module is still built, just left
    randomly initialized and unused), so we simply remove any head checkpoint
    whose source is an unexpanded ``${...}`` env-var path. Idempotent; operates
    on the cached config files in the hub repo.
    """
    import glob
    import os

    import yaml

    repo_dir = _hub_repo_dir(repo)
    if repo_dir is None:
        return
    for cfg_path in glob.glob(os.path.join(str(repo_dir), "configs", "**", "*.yaml"),
                              recursive=True):
        try:
            with open(cfg_path) as f:
                text = f.read()
            if "pretrain_dec_path" not in text or "${" not in text:
                continue
            cfg = yaml.safe_load(text)
        except Exception:
            continue
        # ``pretrain_dec_path`` can be nested (e.g. model_kwargs/pretrain_kwargs/
        # heads_cfg/pretrain_dec_path), so walk the whole tree.
        changed = False

        def _scrub(node):
            nonlocal changed
            if isinstance(node, dict):
                pdp = node.get("pretrain_dec_path")
                if isinstance(pdp, dict):
                    for name in [n for n, src in pdp.items()
                                 if isinstance(src, str) and "${" in src
                                 and not src.startswith(("http://", "https://"))]:
                        pdp.pop(name, None)
                        changed = True
                for v in node.values():
                    _scrub(v)
            elif isinstance(node, list):
                for v in node:
                    _scrub(v)

        _scrub(cfg)
        if changed:
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)


def load_from_hub(hub_id: str, repo: str = HUB_REPO) -> Tuple[Any, Any]:
    """Load (model, preprocessor) from the unified jepa-wms hub.

    If torch.hub returns 503 errors, the plan says: delete
    `external/jepa-wms/uv.lock` and re-run `uv sync` inside that repo.
    """
    _install_lightweight_eval_stub()
    _strip_nonpublic_head_checkpoints(repo)
    return torch.hub.load(repo, hub_id, trust_repo=True)
