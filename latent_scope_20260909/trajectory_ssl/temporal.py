"""Causal windows and partition-consistent memory inputs.

o[r] is observed BEFORE a[r] is executed. A history token at r contains o[r]
and the incoming action a[r-1], never the next proposal a[r]. At reset there
is no incoming action: return zero plus an explicit validity mask.
"""

from __future__ import annotations

import torch


def causal_window(visual, proprio, actions, start: int, length: int,
                  history_tokens: int = 96, stride: int = 4) -> dict:
    n = len(visual)
    if len(proprio) != n or len(actions) not in (n, n - 1):
        raise ValueError("unaligned observations/actions")
    if start < 0 or length < 1 or start + length >= n:
        raise ValueError("window must include both boundary observations")
    if history_tokens < 1 or stride < 1:
        raise ValueError("history_tokens and stride must be positive")
    dev = visual.device
    indices = start - stride * torch.arange(history_tokens - 1, -1, -1, device=dev)
    valid = indices >= 0
    action_valid = indices > 0
    hist_actions = actions[(indices - 1).clamp_min(0)].clone()
    hist_actions[~action_valid] = 0
    return {
        "hist_visual": visual[indices.clamp_min(0)],
        "hist_proprio": proprio[indices.clamp_min(0)],
        "hist_actions": hist_actions,
        "hist_mask": valid,
        "hist_action_mask": action_valid,
        "hist_frame_indices": indices,
        "cur_visual": visual[start],
        "cur_proprio": proprio[start],
        "actions": actions[start:start + length],
        "fut_visual": visual[start + 1:start + length + 1],
        "fut_proprio": proprio[start + 1:start + length + 1],
        # Inclusive boundaries are REQUIRED for path signatures. Adjacent paths
        # share an observation, but their action/increment intervals do not overlap.
        "path_visual": visual[start:start + length + 1],
        "path_proprio": proprio[start:start + length + 1],
    }


def memory_positions(length: int, stride: int = 4, offset: int = 0,
                     device=None) -> torch.Tensor:
    """Indices of future frames on one fixed grid, even for non-aligned splits.

Frames are o[t+1],...,o[t+length]; actions are a[t],...,a[t+length-1].
Index i refers to BOTH frame i and its incoming action i. offset counts
transitions already processed since the grid's origin; do not reset it per chunk.
"""
    if length < 0 or offset < 0 or stride < 1:
        raise ValueError("invalid memory grid")
    indices = torch.arange(length, device=device)
    return indices[(offset + indices + 1) % stride == 0]


def continue_memory(memory, h, future_frames, actions, *, stride: int = 4,
                    offset: int = 0):
    """Adapter for modules exposing tokens(frames, actions) and forward(..., h0).

The caller must carry offset across arbitrary splits. No clamping of a missing
next action is needed, and no future candidate action enters the observed prefix.
"""
    if future_frames.shape[:2] != actions.shape[:2]:
        raise ValueError("one incoming action per future frame is required")
    pos = memory_positions(future_frames.shape[1], stride, offset, h.device)
    if pos.numel() == 0:
        return h
    return memory(memory.tokens(future_frames[:, pos], actions[:, pos]), None, h)
