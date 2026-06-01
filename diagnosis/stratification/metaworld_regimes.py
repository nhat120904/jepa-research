"""Regime classification on Metaworld from the 39-dim state vector.

IMPORTANT — this is a **proxy**, not MuJoCo contact ground truth.
The HuggingFace Metaworld dataset (``MetaworldHFDataset``) does NOT ship
per-frame contact flags. It does ship the full 39-dim Metaworld v2 observation
as ``state``, whose layout is:

    state[0:3]    end-effector (hand) xyz
    state[3]      gripper state
    state[4:7]    primary object xyz
    state[7:11]   primary object quaternion
    state[11:14]  secondary object xyz
    state[14:18]  secondary object quaternion
    state[18:36]  previous-timestep copy of the above 18
    state[36:39]  goal xyz

So we recover ee/object/gripper exactly, and use **object displacement** as a
contact proxy (the object only moves when something pushes it). This keeps
Metaworld as the primary diagnostic while being honest that "contact" here means
"the object measurably moved", not a MuJoCo contact sensor.

Returns one of: 'free_space' | 'pre_grasp' | 'gripper_actuation' | 'contact_manipulation'.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Proxy thresholds (tunable; documented in the plan).
GRIPPER_DELTA_THRESHOLD = 0.10     # |Δ gripper| over a step
OBJECT_MOVE_THRESHOLD = 0.005      # 5 mm object displacement → "contact"
PRE_GRASP_DISTANCE = 0.10          # 10 cm ee↔object proximity

# State-vector slices (Metaworld v2 39-dim layout).
EE_SLICE = slice(0, 3)
GRIPPER_IDX = 3
OBJECT_SLICE = slice(4, 7)


def classify_metaworld_regime(
    state_t: np.ndarray,
    state_t1: np.ndarray,
    gripper_delta_threshold: float = GRIPPER_DELTA_THRESHOLD,
    object_move_threshold: float = OBJECT_MOVE_THRESHOLD,
    pre_grasp_distance: float = PRE_GRASP_DISTANCE,
) -> str:
    """Classify a single (s_t -> s_{t+1}) transition from raw 39-dim states."""
    state_t = np.asarray(state_t, dtype=np.float32)
    state_t1 = np.asarray(state_t1, dtype=np.float32)

    grip_t = float(state_t[GRIPPER_IDX])
    grip_t1 = float(state_t1[GRIPPER_IDX])
    if abs(grip_t1 - grip_t) > gripper_delta_threshold:
        return "gripper_actuation"

    obj_t = state_t[OBJECT_SLICE]
    obj_t1 = state_t1[OBJECT_SLICE]
    if float(np.linalg.norm(obj_t1 - obj_t)) > object_move_threshold:
        return "contact_manipulation"

    ee_t = state_t[EE_SLICE]
    if float(np.linalg.norm(ee_t - obj_t)) < pre_grasp_distance:
        return "pre_grasp"

    return "free_space"
