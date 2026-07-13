"""LoRA partial-unfreeze of the frozen ENCODER — Phase D, the geometry-level fix
(docs/plans/2026-07-02-encoder-lora-action-grounding-design.md).

Phase 4 closed the frozen-encoder program empirically: even an adversarially
hardened, relearned representation φ = A(z) (Phase C v2) gates at push 1/16 with
healthy grounding, because any φ is a function of the frozen ``z`` — latents the
encoder has entangled stay entangled for every possible head. This module moves
the lever inside the encoder itself: zero-init LoRA adapters (the exact
``LoRALinear`` mechanism scripts/26 already trains on the *predictor*) injected
into the encoder's transformer blocks, so the Phase-C losses can separate merged
latents at the source instead of reading around them.

Same contract as ``lora_predictor``: zero-init (``B=0``) → the model starts
exactly at the frozen encoder; toggleable (``set_lora_enabled``) → one injected
model serves both the frozen baseline arm and the LoRA arm (and gives the
preservation loss its frozen reference for free).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from models.heads.lora_predictor import LoRALinear


def _encoder_of(adapter):
    """The upstream encoder module: ``adapter.wm.encoder`` (VideoWM.encoder)."""
    wm = getattr(adapter, "wm", None) or getattr(adapter.encpred, "model", adapter.encpred)
    enc = getattr(wm, "encoder", None)
    if enc is None:
        raise RuntimeError("could not find adapter.wm.encoder")
    return enc


def inject_encoder_lora(adapter, *, r: int = 16, alpha: int = 32,
                        target_substrs=("blocks", "layers")) -> List[LoRALinear]:
    """Replace every ``nn.Linear`` inside the encoder's transformer blocks with a
    ``LoRALinear``. ``target_substrs`` matches the block ModuleList name across
    backbones: ``blocks`` (DINOv2/v3, V-JEPA-2 ViTs) with ``layers`` as fallback.
    Returns the injected adapters (for the optimizer / toggling). Base encoder
    weights are frozen; only LoRA trains."""
    if isinstance(target_substrs, str):
        target_substrs = (target_substrs,)
    encoder = _encoder_of(adapter)
    targets = []
    for name, module in list(encoder.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if target_substrs and not any(s in name for s in target_substrs):
            continue
        targets.append(name)
    if not targets:
        linears = [n for n, m in encoder.named_modules() if isinstance(m, nn.Linear)]
        raise RuntimeError(
            f"inject_encoder_lora matched 0 Linear layers under {target_substrs}. "
            f"Encoder Linear modules are: {linears[:12]}{' ...' if len(linears) > 12 else ''}")
    injected: List[LoRALinear] = []
    for name in targets:
        parent = encoder
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


def encoder_lora_state_dict(injected: List[LoRALinear], adapter) -> dict:
    """{module_name: {'A':..., 'B':...}} keyed by the encoder-relative name, so a
    fresh inject + load round-trips regardless of object identity."""
    encoder = _encoder_of(adapter)
    name_of = {}
    for name, module in encoder.named_modules():
        if isinstance(module, LoRALinear):
            name_of[id(module)] = name
    sd = {}
    for m in injected:
        nm = name_of.get(id(m))
        if nm is not None:
            sd[nm] = {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()}
    return sd


def load_encoder_lora(adapter, ckpt_path: str | Path, device) -> tuple[List[LoRALinear], dict]:
    """Inject LoRA into ``adapter``'s encoder and load trained weights — the
    inference entry point for ``scripts/30 --encoder-lora``."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tgt = ckpt.get("target_substrs", ("blocks", "layers"))
    injected = inject_encoder_lora(adapter, r=ckpt["r"], alpha=ckpt["alpha"], target_substrs=tgt)
    encoder = _encoder_of(adapter)
    by_name = {name: m for name, m in encoder.named_modules() if isinstance(m, LoRALinear)}
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
        print(f"[load_encoder_lora] WARNING: {missing}/{len(sd)} LoRA modules "
              f"in the ckpt did not match the live encoder", flush=True)
    return injected, ckpt
