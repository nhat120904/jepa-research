"""Dense goal progress from BDDL goal predicates (docs/LIBERO_QUALIFICATION_PROTOCOL.md, Amendment 2).

Pure functions, no LIBERO import (unit-tested in cta_tests/). The LIBERO side that reads joints and positions is
ti_wm.libero_runtime.goal_progress. Per predicate:
- On / In: negative distance (m) from the object to its target (site or object);
- Open / Close / TurnOn / TurnOff: joint progress from its value at episode start toward the nearest endpoint of the
  predicate's range, clipped to [0, 1];
- a satisfied predicate scores 1, above every unsatisfied value of either kind.
A conjunction scores the mean over its predicates (every LIBERO-Goal task has exactly one).
"""

import numpy as np

JOINT_KINDS = {"open": "default_open_ranges", "close": "default_close_ranges",
               "turnon": "default_turnon_ranges", "turnoff": "default_turnoff_ranges"}
DISTANCE_KINDS = ("on", "in")
SATISFIED = 1.0


def nearest_endpoint(ranges, q0):
    """The boundary the joint must cross: the endpoint of the predicate's range nearest to its start value."""
    return float(min(ranges, key=lambda r: abs(float(r) - q0)))


def joint_progress(q, q0, target):
    span = target - q0
    if abs(span) < 1e-9:
        return 1.0
    return float(np.clip((q - q0) / span, 0.0, 1.0))


def distance_progress(pos, target_pos):
    return -float(np.linalg.norm(np.asarray(pos, float) - np.asarray(target_pos, float)))


def combine(values, satisfied):
    return float(np.mean([SATISFIED if s else v for v, s in zip(values, satisfied)]))
