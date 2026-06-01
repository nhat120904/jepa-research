from .metaworld_regimes import classify_metaworld_regime
from .droid_regimes import classify_droid_regime, droid_baseline_change
from .robocasa_regimes import classify_robocasa_regime

REGIMES = ["free_space", "pre_grasp", "gripper_actuation", "contact_manipulation"]

__all__ = [
    "REGIMES",
    "classify_metaworld_regime",
    "classify_droid_regime",
    "droid_baseline_change",
    "classify_robocasa_regime",
]
