"""Planner-facing scalar feedbacks over the Scene event automaton.

`scene_core.feedback_reward` balances the two task-5 branches against each
other: each branch is normalised to one and the two are averaged.  A hint^2
style automaton potential instead treats every milestone as one step towards
the accepting state, so the longer drawer branch carries proportionally more
weight.  Both are capped at 0.90 below acceptance and return 1.0 on stable
success, so they differ only in shaping, not in scale or in the success bonus.

On task 4 the two functions are identical by construction (0.90 * cube / 4);
the feedback contrast therefore exists only on task 5, and task-4 rows are a
sanity check that must match exactly.
"""

from __future__ import annotations

from event_smdp_h0.core import ARM_EVENT, ARM_TERMINAL
from event_smdp_h0.scene_core import MilestoneState, feedback_reward


FEEDBACKS = ("event_progress", "automaton_potential")

# The two feedbacks above are not two families: they are one branch-weighted
# family at w = 0.500 and w = 0.625 respectively, verified exactly on every
# task-5 event state.  The robustness sweep therefore walks w instead of
# inventing unrelated scalars, and adds two designs outside the family.
SWEEP_FEEDBACKS = (
    "branch_w030",
    "branch_w040",
    "branch_w050",
    "branch_w056",
    "branch_w062",
    "branch_w070",
    "anti_livelock",
    "shaped_gamma09",
)
ANTI_LIVELOCK_PENALTY = 0.25
SHAPED_GAMMA = 0.9
_SWEEP_WEIGHTS = {
    "branch_w030": 0.300,
    "branch_w040": 0.400,
    "branch_w050": 0.500,
    "branch_w056": 0.5625,
    "branch_w062": 0.625,
    "branch_w070": 0.700,
}

# Milestones that must still fire before the accepting state, per branch.
_BRANCH_TOTALS = {4: (4, 0), 5: (5, 3)}


def automaton_potential(state: MilestoneState) -> float:
    """hint^2 style: uniform credit per remaining automaton milestone."""

    if state.stable_success:
        return 1.0
    cube_total, window_total = _BRANCH_TOTALS[state.task_id]
    total = cube_total + window_total
    remaining = (cube_total - min(state.cube_stage, cube_total)) + (
        window_total - min(state.window_stage, window_total)
    )
    return 0.90 * (1.0 - remaining / total)


def scalar(state: MilestoneState, feedback: str) -> float:
    if feedback == "event_progress":
        return feedback_reward(state, ARM_EVENT)
    if feedback == "automaton_potential":
        return automaton_potential(state)
    if feedback == "terminal":
        return feedback_reward(state, ARM_TERMINAL)
    raise ValueError(f"unknown feedback: {feedback}")


def branch_weighted(state: MilestoneState, weight: float) -> float:
    """One-parameter family; `weight` splits credit between the two branches.

    `weight=0.500` reproduces `scene_core.feedback_reward` and `weight=0.625`
    reproduces :func:`automaton_potential`, both exactly.
    """

    if state.stable_success:
        return 1.0
    cube_total, window_total = _BRANCH_TOTALS[state.task_id]
    if window_total == 0:
        return 0.90 * min(state.cube_stage, cube_total) / cube_total
    return 0.90 * (
        weight * min(state.cube_stage, cube_total) / cube_total
        + (1.0 - weight) * min(state.window_stage, window_total) / window_total
    )


def sweep_weight(feedback: str) -> float:
    return _SWEEP_WEIGHTS[feedback]


def is_branch_family(feedback: str) -> bool:
    return feedback in _SWEEP_WEIGHTS
