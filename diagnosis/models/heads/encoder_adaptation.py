"""Last-block encoder adaptation utilities used by the feasibility sprint."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.heads.lora_encoder import _encoder_of


def _find_block_list(encoder: nn.Module) -> tuple[str, nn.ModuleList]:
    candidates = []
    for name, module in encoder.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            priority = int(any(token in name.lower() for token in ("block", "layer")))
            candidates.append((priority, len(module), name, module))
    if not candidates:
        raise RuntimeError("encoder contains no ModuleList of transformer blocks")
    _, _, name, module = max(candidates, key=lambda item: (item[0], item[1]))
    return name, module


def unfreeze_last_encoder_blocks(adapter, n_blocks: int = 4):
    """Freeze the encoder, then unfreeze its last ``n_blocks`` and final norm."""
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")
    encoder = _encoder_of(adapter)
    encoder.requires_grad_(False)
    block_name, blocks = _find_block_list(encoder)
    start = max(0, len(blocks) - n_blocks)
    for block in blocks[start:]:
        block.requires_grad_(True)
    # DINO/V-JEPA encoders conventionally expose a final norm.  It is part of
    # the adapted tail, but we avoid broad substring matches inside every block.
    for name, module in encoder.named_children():
        if "norm" in name.lower():
            module.requires_grad_(True)
    named = [(name, p) for name, p in encoder.named_parameters() if p.requires_grad]
    if not named:
        raise RuntimeError("last-block adaptation selected zero parameters")
    return named, {"block_list": block_name, "n_total_blocks": len(blocks), "start": start}


def trainable_encoder_state_dict(adapter) -> dict[str, torch.Tensor]:
    encoder = _encoder_of(adapter)
    return {name: p.detach().cpu() for name, p in encoder.named_parameters() if p.requires_grad}


def load_trainable_encoder_state(adapter, state: dict[str, torch.Tensor], n_blocks: int, device):
    named, meta = unfreeze_last_encoder_blocks(adapter, n_blocks=n_blocks)
    live = dict(named)
    missing = sorted(set(state) - set(live))
    if missing:
        raise RuntimeError(f"checkpoint encoder parameters do not match live model: {missing[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            live[name].copy_(value.to(device=device, dtype=live[name].dtype))
    return named, meta
