"""Pure input/output contracts; does not import models or simulators."""

import math
from numbers import Integral, Real


def validate_train_arguments(args):
    for key in ("steps", "decisions", "eval_every", "warmup", "log_every"):
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("selection_limit", "train_limit"):
        value = getattr(args, key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value <= 0):
            raise ValueError(f"{key} must be None or a positive integer")
    for key in ("lr", "hit_weight", "standard_mass", "mixed_frac"):
        value = getattr(args, key)
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if (key == "lr" and value <= 0) or (key == "hit_weight" and value < 0):
            raise ValueError(f"Invalid {key}: {value}")
        if key in ("standard_mass", "mixed_frac") and not 0 <= value <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if isinstance(args.seed, bool) or not isinstance(args.seed, Integral) or args.seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def validate_bank_shapes(name, geometry_shape, hit_shape, action_shape, candidates=8, chunk=15, feature_shapes=None):
    geometry_shape, hit_shape, action_shape = map(tuple, (geometry_shape, hit_shape, action_shape))
    if len(geometry_shape) != 2 or geometry_shape[0] <= 0 or geometry_shape[1] != candidates:
        raise ValueError(f"{name}: expected nonempty K={candidates} geometry bank, got {geometry_shape}")
    if hit_shape != geometry_shape:
        raise ValueError(f"{name}: native-hit and geometry shapes differ")
    expected = (geometry_shape[0], candidates, chunk, 2)
    if action_shape != expected:
        raise ValueError(f"{name}: expected action shape {expected}, got {action_shape}")
    if feature_shapes is not None:
        cur = tuple(feature_shapes.get("cur", ()))
        if len(cur) != 3 or cur[:2] != (geometry_shape[0], 256) or cur[2] <= 0:
            raise ValueError(f"{name}: invalid current feature shape {cur}")
        n, _, dimension = cur
        expected_features = {"cur": (n, 256, dimension), "prev": (n, 64, dimension),
                             "end": (n, candidates, 256, dimension), "seg": (n, candidates, 3, 64, dimension)}
        for key, expected in expected_features.items():
            actual = tuple(feature_shapes.get(key, ()))
            if actual != expected:
                raise ValueError(f"{name}: expected {key} feature shape {expected}, got {actual}")


def validate_decision_domains(choices, hit_values, candidates=8):
    for value in choices:
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < candidates:
            raise ValueError(f"chosen candidate must be an integer in [0,{candidates - 1}], got {value!r}")
    for value in hit_values:
        if value not in (False, True):
            raise ValueError(f"native_hit must contain binary labels, got {value!r}")


def json_ready(value):
    """Undefined ratios become null; preserve finite values and discrete counts."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    return value


def format_stat(value):
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.3f}"
