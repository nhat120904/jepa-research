"""LoRA partial-unfreeze of the frozen ViTPredictor — the model-side Rung A fix
(docs/plans/2026-06-24-object-aware-predictor-design.md).

Option D (scripts/25) added a correction OUTSIDE the predictor
(``ẑ = F(z,a) + Δ(z,a)``): the corrected *readout* tracked the counterfactual
object (cf-corr +0.37) but the recursively-rolled latent was not a dynamically
consistent contact rollout, so success stayed 0/32 and the additive head saturated
below the dyn-head teacher (0.68). This module instead puts a small trainable
correction *inside* the predictor's transformer blocks (LoRA adapters: a frozen
``W`` plus a low-rank ``B·A``), so the corrected dynamics propagate through the
multi-step rollout the planner searches over. The encoder, the predictor's base
weights, and everything else stay frozen; only the rank-``r`` adapters train.

LoRA is zero-initialised (``B=0``) so the model starts *exactly* at the frozen
predictor, and is **toggleable** (``set_lora_enabled``) so a single adapter serves
both a frozen ``l2`` arm and a corrected ``lora`` arm in the same paired run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank update.

    ``y = W x + b  (+  scaling · (x Aᵀ) Bᵀ   when enabled)``.  ``A`` is
    kaiming-init, ``B`` is zero-init → the update starts at 0 (output identical to
    the frozen layer). Set ``enabled=False`` to recover the exact frozen layer."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.enabled = True
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B stays zero -> initial delta is exactly 0.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + (x @ self.A.t() @ self.B.t()) * self.scaling
        return out


def _predictor_of(adapter):
    """The upstream ViTPredictor: ``adapter.wm.predictor`` (VideoWM.predictor)."""
    wm = getattr(adapter, "wm", None) or getattr(adapter.encpred, "model", adapter.encpred)
    pred = getattr(wm, "predictor", None)
    if pred is None:
        raise RuntimeError("could not find adapter.wm.predictor (ViTPredictor)")
    return pred


def inject_lora(adapter, *, r: int = 8, alpha: int = 16,
                target_substrs=("layers", "blocks")) -> List[LoRALinear]:
    """Replace every ``nn.Linear`` inside the predictor's transformer blocks with a
    ``LoRALinear``. ``target_substrs`` matches the block ModuleList name across
    predictor variants: ``layers`` (vit.py ViTPredictor — DINO-WM/JEPA-WM) and
    ``predictor_blocks``/``blocks`` (AdaLN_vit). Returns the injected adapters (for
    the optimizer / toggling). Base predictor weights are frozen; only LoRA trains."""
    if isinstance(target_substrs, str):
        target_substrs = (target_substrs,)
    predictor = _predictor_of(adapter)
    targets = []
    for name, module in list(predictor.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if target_substrs and not any(s in name for s in target_substrs):
            continue
        targets.append(name)
    if not targets:
        linears = [n for n, m in predictor.named_modules() if isinstance(m, nn.Linear)]
        raise RuntimeError(
            f"inject_lora matched 0 Linear layers under {target_substrs}. "
            f"Predictor Linear modules are: {linears[:12]}{' ...' if len(linears) > 12 else ''}")
    injected: List[LoRALinear] = []
    for name in targets:
        parent = predictor
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        base = getattr(parent, attr)
        if isinstance(base, LoRALinear):
            injected.append(base)
            continue
        lora = LoRALinear(base, r=r, alpha=alpha).to(next(base.parameters()).device)
        setattr(parent, attr, lora)
        injected.append(lora)
    return injected


def set_lora_enabled(modules: List[LoRALinear], enabled: bool) -> None:
    for m in modules:
        m.enabled = bool(enabled)


def lora_state_dict(injected: List[LoRALinear], adapter) -> dict:
    """{module_name: {'A':..., 'B':...}} keyed by the predictor-relative name, so a
    fresh inject + load round-trips regardless of object identity."""
    predictor = _predictor_of(adapter)
    name_of = {}
    for name, module in predictor.named_modules():
        if isinstance(module, LoRALinear):
            name_of[id(module)] = name
    sd = {}
    for m in injected:
        nm = name_of.get(id(m))
        if nm is not None:
            sd[nm] = {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()}
    return sd


def load_predictor_lora(adapter, ckpt_path: str | Path, device) -> tuple[List[LoRALinear], dict]:
    """Inject LoRA into ``adapter`` and load trained weights — the inference entry
    point for scripts/18/23/24 (`--predictor-lora`)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tgt = ckpt.get("target_substrs", ckpt.get("target_substr", ("layers", "blocks")))
    injected = inject_lora(adapter, r=ckpt["r"], alpha=ckpt["alpha"], target_substrs=tgt)
    predictor = _predictor_of(adapter)
    by_name = {name: m for name, m in predictor.named_modules() if isinstance(m, LoRALinear)}
    sd = ckpt["lora"]
    missing = 0
    for nm, params in sd.items():
        m = by_name.get(nm)
        if m is None:
            missing += 1
            continue
        with torch.no_grad():
            m.A.copy_(params["A"].to(device))
            m.B.copy_(params["B"].to(device))
    if missing:
        print(f"[load_predictor_lora] WARNING: {missing}/{len(sd)} LoRA modules "
              f"in the ckpt did not match the live predictor", flush=True)
    return injected, ckpt
