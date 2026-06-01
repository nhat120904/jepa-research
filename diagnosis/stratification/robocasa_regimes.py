"""RoboCasa regime classification.

RoboCasa exposes object-grasp flags and proprioception per stage; we map
those to the same four-regime taxonomy as Metaworld/DROID.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


GRIPPER_DELTA_THRESHOLD = 0.2
PRE_GRASP_DISTANCE = 0.12  # slightly looser than Metaworld (DROID-scale scenes)


def classify_robocasa_regime(transition_info: Dict) -> str:
    obs_t = transition_info["obs_t"]
    obs_t1 = transition_info["obs_t1"]

    if abs(obs_t1["gripper_state"] - obs_t["gripper_state"]) > GRIPPER_DELTA_THRESHOLD:
        return "gripper_actuation"

    if obs_t.get("object_in_hand", False) or obs_t1.get("object_in_hand", False):
        return "contact_manipulation"

    if "ee_position" in obs_t and "active_object_position" in obs_t:
        ee = np.asarray(obs_t["ee_position"])
        obj = np.asarray(obs_t["active_object_position"])
        if np.linalg.norm(ee - obj) < PRE_GRASP_DISTANCE:
            return "pre_grasp"

    return "free_space"
