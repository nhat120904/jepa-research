"""Regime classification on DROID using proprioception + latent-change heuristics.

DROID-wrist has no MuJoCo ground truth for contact, so we approximate the four
manipulation regimes from the **real gripper sensor** (primary) plus the
latent-change proxy (secondary):

    gripper_actuation:    the gripper is actively opening/closing (large |Δgrip|)
    contact_manipulation: the gripper is held *closed* — i.e. in contact with /
                          holding an object. This is a real sensor signal, so it
                          does NOT depend on visual motion: a closed gripper
                          resting on or holding an object is in contact even when
                          the scene is momentarily static.
    pre_grasp:            gripper open AND the scene is actively changing — the
                          arm is approaching / an object is entering the wrist
                          view (no object pose on DROID-wrist, so above-baseline
                          ‖Δz‖ is the only available "something is happening" cue).
    free_space:           gripper open AND the scene is static — moving in empty
                          space, no object interaction.

Rationale for making contact gripper-primary (changed 2026-06-14): the earlier
rule gated contact on ‖Δz‖>baseline AND gripper-closed, which dumped every
*static holding* step (closed gripper, small per-step latent change) into
free_space. On the cached DROID subset that mislabeled ~54% of all closed-gripper
transitions as free_space — visible in the regime gallery as a gripper clearly
grasping an object yet tagged free_space. Because the decision metric is
*effect-conditioned* CRA (restricted to ‖Δz‖>median, the same baseline), this
relabel leaves the headline effect-conditioned numbers unchanged — the
low-‖Δz‖ holding steps were excluded by effect-conditioning either way — but it
makes the raw per-regime counts and the visualization match physical reality.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


GRIPPER_DELTA_THRESHOLD = 0.2
# Gripper-state split (0 = fully open .. 1 = fully closed). On the DROID-wrist
# cache the gripper signal is bimodal (a large open cluster near 0 and a closed
# cluster near 0.8-1.0), so the exact cut is not sensitive; 0.5 cleanly separates
# "holding/contacting" from "open".
GRIPPER_CLOSED_THRESHOLD = 0.5

# "Scene actively changing" multiplier over the dataset-median ‖Δz‖ baseline,
# used (gripper-open only) to split pre_grasp (approaching) from free_space
# (empty-space motion). Encoder-dependent: DINOv2 ViT-S/14 patch-token L2 over
# the full grid has a large near-constant component, so per-step changes have a
# *narrow* dynamic range (real DROID-wrist: median≈619, max≈840). 1.0× (above
# the median) is the calibrated "scene is moving" threshold for this encoder; a
# higher-dynamic-range encoder (e.g. DINOv3 ViT-L) would warrant a larger ratio.
SCENE_CHANGE_RATIO = 1.0

# Backward-compatible alias for the prior name (was used as the contact gate;
# now gates the pre_grasp/free_space split). Kept so external references resolve.
CONTACT_LATENT_RATIO = SCENE_CHANGE_RATIO


def droid_baseline_change(z_t: torch.Tensor, z_t1: torch.Tensor) -> float:
    """Median of ||z_{t+1} - z_t|| across the dataset — call once, pass into
    `classify_droid_regime` via `transition_info['global_median_latent_change']`.
    """
    diff = (z_t1 - z_t).reshape(z_t.shape[0], -1).norm(dim=-1)
    return diff.median().item()


def classify_droid_regime(transition_info: Dict, latent_t, latent_t1) -> str:
    g_t = transition_info["gripper_state_t"]
    g_t1 = transition_info["gripper_state_t1"]

    # 1. The gripper is actively opening/closing — the grasp/release event itself.
    if abs(g_t1 - g_t) > GRIPPER_DELTA_THRESHOLD:
        return "gripper_actuation"

    # 2. Gripper held closed → in contact with / holding an object, regardless of
    #    motion (real sensor; see module docstring for why this is not Δz-gated).
    if g_t > GRIPPER_CLOSED_THRESHOLD:
        return "contact_manipulation"

    # 3. Gripper open: split "approaching an object" from "moving in free space"
    #    by scene change (object entering/growing in the wrist view drives ‖Δz‖
    #    above the dataset median; panning empty space stays below it).
    latent_change = float(np.linalg.norm(np.asarray(latent_t1).flatten()
                                          - np.asarray(latent_t).flatten()))
    baseline = transition_info.get("global_median_latent_change", 1.0)
    if latent_change > SCENE_CHANGE_RATIO * baseline:
        return "pre_grasp"

    return "free_space"
