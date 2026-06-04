"""Regime classification on DROID using proprioception + latent-change heuristics.

DROID has no MuJoCo ground truth for contact, so we approximate:
    contact_manipulation: latent change much larger than baseline AND gripper closed
    gripper_actuation:    significant gripper state delta
    pre_grasp:            gripper open and latent change above baseline
    free_space:           otherwise
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


GRIPPER_DELTA_THRESHOLD = 0.2
# "high latent change" multiplier over the dataset-median baseline. Encoder-
# dependent: DINOv2 ViT-S/14 patch-token L2 over the full grid has a large
# near-constant component, so per-step changes have a *narrow* dynamic range
# (real DROID-wrist: median≈622, max≈842 → a 1.5× gate is unreachable and
# collapses contact_manipulation to 0%). 1.0× (above-median) is the calibrated
# "scene actively moving" threshold for this encoder; a higher-dynamic-range
# encoder (e.g. DINOv3 ViT-L) would warrant a larger ratio. Contact on DROID-
# wrist is thus a proxy: gripper closed (holding) AND above-median visual change.
CONTACT_LATENT_RATIO = 1.0
GRIPPER_CLOSED_THRESHOLD = 0.5
GRIPPER_OPEN_THRESHOLD = 0.3


def droid_baseline_change(z_t: torch.Tensor, z_t1: torch.Tensor) -> float:
    """Median of ||z_{t+1} - z_t|| across the dataset — call once, pass into
    `classify_droid_regime` via `transition_info['global_median_latent_change']`.
    """
    diff = (z_t1 - z_t).reshape(z_t.shape[0], -1).norm(dim=-1)
    return diff.median().item()


def classify_droid_regime(transition_info: Dict, latent_t, latent_t1) -> str:
    g_t = transition_info["gripper_state_t"]
    g_t1 = transition_info["gripper_state_t1"]

    if abs(g_t1 - g_t) > GRIPPER_DELTA_THRESHOLD:
        return "gripper_actuation"

    latent_change = float(np.linalg.norm(np.asarray(latent_t1).flatten()
                                          - np.asarray(latent_t).flatten()))
    baseline = transition_info.get("global_median_latent_change", 1.0)

    if latent_change > CONTACT_LATENT_RATIO * baseline and g_t > GRIPPER_CLOSED_THRESHOLD:
        return "contact_manipulation"

    if g_t < GRIPPER_OPEN_THRESHOLD and latent_change > baseline:
        return "pre_grasp"

    return "free_space"
