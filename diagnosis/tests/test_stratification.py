import numpy as np

from stratification.metaworld_regimes import classify_metaworld_regime


def _state(ee=(0, 0, 0), gripper=0.0, obj=(1.0, 0.0, 0.0)):
    s = np.zeros(39, dtype=np.float32)
    s[0:3] = ee
    s[3] = gripper
    s[4:7] = obj
    return s


def test_gripper_actuation():
    s0 = _state(gripper=0.0)
    s1 = _state(gripper=0.5)            # |Δgrip| = 0.5 > 0.10
    assert classify_metaworld_regime(s0, s1) == "gripper_actuation"


def test_contact_manipulation_from_object_motion():
    s0 = _state(gripper=0.0, obj=(0.0, 0.0, 0.0))
    s1 = _state(gripper=0.0, obj=(0.1, 0.0, 0.0))   # object moved 10cm >> 5mm
    assert classify_metaworld_regime(s0, s1) == "contact_manipulation"


def test_pre_grasp_when_ee_near_static_object():
    s0 = _state(ee=(0, 0, 0), gripper=0.0, obj=(0.05, 0.0, 0.0))   # 5cm < 10cm
    s1 = _state(ee=(0, 0, 0), gripper=0.0, obj=(0.05, 0.0, 0.0))   # static object
    assert classify_metaworld_regime(s0, s1) == "pre_grasp"


def test_free_space_when_ee_far_and_object_static():
    s0 = _state(ee=(0, 0, 0), gripper=0.0, obj=(1.0, 0.0, 0.0))
    s1 = _state(ee=(0, 0, 0), gripper=0.0, obj=(1.0, 0.0, 0.0))
    assert classify_metaworld_regime(s0, s1) == "free_space"
