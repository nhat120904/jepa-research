"""Latent-space distance functions used by every metric."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """L2 over flattened tail dims. Inputs (B, ...). Output (B,)."""
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    return (a_flat - b_flat).norm(dim=-1)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    return 1.0 - F.cosine_similarity(a_flat, b_flat, dim=-1)


_DISTANCES = {
    "l2": l2_distance,
    "cosine": cosine_distance,
}


def get_distance(name: str):
    if name not in _DISTANCES:
        raise ValueError(f"Unknown distance '{name}'. Choices: {list(_DISTANCES)}")
    return _DISTANCES[name]
