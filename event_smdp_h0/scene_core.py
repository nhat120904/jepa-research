"""Pure event automata and matched Skill-UCT for OGBench-Scene H0.

The simulator runner supplies exact skill-conditioned transitions.  This file
contains no MuJoCo or model imports, which keeps the treatment distinction
auditable: both arms use the same tree search and differ only in the scalar
feedback backed up through that tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Callable

from event_smdp_h0.core import ARM_EVENT, ARM_TERMINAL, proposal_order


SKILLS = (
    "toggle_button_0",
    "toggle_button_1",
    "drawer_open",
    "drawer_close",
    "window_open",
    "window_close",
    "place_cube_in_drawer",
)


@dataclass(frozen=True)
class ScenePredicates:
    button_0: int
    button_1: int
    drawer_open: bool
    drawer_closed: bool
    window_open: bool
    window_closed: bool
    cube_in_drawer: bool
    cube_at_goal: bool
    native_success: bool


@dataclass(frozen=True)
class MilestoneState:
    """Task-conditioned, history-bearing event state.

    Task 4 uses ``cube_stage`` 0..4.  Task 5 uses the same drawer/cube
    branch at 0..5 and an order-independent window branch at 0..3.
    ``stable_count`` is deliberately not latched: terminal success must still
    hold in the current physical state.
    """

    task_id: int
    cube_stage: int = 0
    window_stage: int = 0
    stable_count: int = 0
    transitions: tuple[str, ...] = ()

    @property
    def stable_success(self) -> bool:
        return self.stable_count >= 3


def initial_milestones(task_id: int) -> MilestoneState:
    if task_id not in (4, 5):
        raise ValueError("Scene H0 is locked to tasks 4 and 5")
    return MilestoneState(task_id=task_id)


def advance_milestones(
    state: MilestoneState, predicates: ScenePredicates
) -> MilestoneState:
    """Advance only through causally ordered milestones.

    The two branches in task 5 are independent, but ordering *within* each
    branch is required.  Final success is current-state and three-step stable;
    it is never inferred from milestone history alone.
    """

    cube_stage = state.cube_stage
    window_stage = state.window_stage
    events: list[str] = []

    if cube_stage == 0 and predicates.button_0 == 1:
        cube_stage = 1
        events.append("drawer_unlocked")
    if cube_stage == 1 and predicates.button_0 == 1 and predicates.drawer_open:
        cube_stage = 2
        events.append("drawer_opened_after_unlock")
    if cube_stage == 2 and predicates.drawer_open and predicates.cube_in_drawer:
        cube_stage = 3
        events.append("cube_placed_in_open_drawer")
    if cube_stage == 3 and predicates.drawer_closed and predicates.cube_at_goal:
        cube_stage = 4
        events.append("drawer_closed_with_cube")
    if (
        state.task_id == 5
        and cube_stage == 4
        and predicates.button_0 == 0
        and predicates.drawer_closed
        and predicates.cube_at_goal
    ):
        cube_stage = 5
        events.append("drawer_relocked")

    if state.task_id == 5:
        if window_stage == 0 and predicates.button_1 == 1:
            window_stage = 1
            events.append("window_unlocked")
        if window_stage == 1 and predicates.button_1 == 1 and predicates.window_open:
            window_stage = 2
            events.append("window_opened_after_unlock")
        if window_stage == 2 and predicates.button_1 == 0 and predicates.window_open:
            window_stage = 3
            events.append("window_relocked")

    stable_count = state.stable_count + 1 if predicates.native_success else 0
    return replace(
        state,
        cube_stage=cube_stage,
        window_stage=window_stage,
        stable_count=stable_count,
        transitions=state.transitions + tuple(events),
    )


def feedback_reward(state: MilestoneState, arm: str) -> float:
    if arm == ARM_TERMINAL:
        return float(state.stable_success)
    if arm != ARM_EVENT:
        raise ValueError(f"unknown arm: {arm}")
    if state.stable_success:
        return 1.0
    if state.task_id == 4:
        return 0.90 * state.cube_stage / 4.0
    cube_fraction = state.cube_stage / 5.0
    window_fraction = state.window_stage / 3.0
    return 0.90 * (cube_fraction + window_fraction) / 2.0


@dataclass
class SearchNode:
    path: tuple[int, ...] = ()
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class PlanSearchResult:
    selected_action: int
    simulations: int
    unique_sequences: int
    max_reward: float
    best_sequence: tuple[int, ...]
    root_stats: tuple[dict[str, float | int | str], ...]
    evaluations: tuple[dict[str, object], ...] = ()


def _select(
    node: SearchNode,
    *,
    search_seed: int,
    depth: int,
    exploration: float,
) -> tuple[int, SearchNode]:
    branching = len(SKILLS)
    order = proposal_order(search_seed, node.path, depth, branching)
    for action in order:
        if action not in node.children:
            child = SearchNode(path=node.path + (action,))
            node.children[action] = child
            return action, child

    rank = {action: index for index, action in enumerate(order)}
    log_parent = math.log(max(node.visits, 1))

    def key(item: tuple[int, SearchNode]) -> tuple[float, float, int]:
        action, child = item
        bonus = exploration * math.sqrt(log_parent / max(child.visits, 1))
        return child.mean + bonus, child.mean, -rank[action]

    return max(node.children.items(), key=key)


def uct_plan_search(
    *,
    horizon: int,
    simulations: int,
    search_seed: int,
    exploration: float,
    evaluate: Callable[[tuple[int, ...]], float],
    record_evaluations: bool = False,
) -> PlanSearchResult:
    """Matched finite-budget UCT returning the robust root action."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if simulations < len(SKILLS):
        raise ValueError("simulations must expand every root skill")

    root = SearchNode()
    seen: set[tuple[int, ...]] = set()
    best_reward = -float("inf")
    best_sequence: tuple[int, ...] = ()
    evaluation_trace: list[dict[str, object]] = []
    for _ in range(simulations):
        node = root
        visited = [root]
        actions: list[int] = []
        for depth in range(horizon):
            action, node = _select(
                node,
                search_seed=search_seed,
                depth=depth,
                exploration=exploration,
            )
            actions.append(action)
            visited.append(node)
        sequence = tuple(actions)
        reward = float(evaluate(sequence))
        if not math.isfinite(reward):
            raise RuntimeError("non-finite search reward")
        seen.add(sequence)
        if record_evaluations:
            evaluation_trace.append(
                {"sequence": list(sequence), "predicted_reward": reward}
            )
        if reward > best_reward:
            best_reward = reward
            best_sequence = sequence
        for item in visited:
            item.visits += 1
            item.value_sum += reward

    order = proposal_order(search_seed, (), 0, len(SKILLS))
    rank = {action: index for index, action in enumerate(order)}

    def robust_key(item: tuple[int, SearchNode]) -> tuple[int, float, int]:
        action, child = item
        return child.visits, child.mean, -rank[action]

    selected, _ = max(root.children.items(), key=robust_key)
    stats = tuple(
        {
            "action": action,
            "skill": SKILLS[action],
            "visits": root.children[action].visits,
            "mean_value": root.children[action].mean,
        }
        for action in range(len(SKILLS))
    )
    return PlanSearchResult(
        selected_action=selected,
        simulations=simulations,
        unique_sequences=len(seen),
        max_reward=best_reward,
        best_sequence=best_sequence,
        root_stats=stats,
        evaluations=tuple(evaluation_trace),
    )


def self_check() -> dict[str, bool]:
    p0 = ScenePredicates(0, 0, False, True, False, True, False, False, False)
    state = advance_milestones(initial_milestones(5), p0)
    locked_open_does_not_advance = state.cube_stage == 0
    p1 = replace(p0, button_0=1)
    state = advance_milestones(state, p1)
    p2 = replace(p1, drawer_open=True, drawer_closed=False)
    state = advance_milestones(state, p2)
    ordered_advance = state.cube_stage == 2
    sparse_is_zero = feedback_reward(state, ARM_TERMINAL) == 0.0
    event_is_dense = feedback_reward(state, ARM_EVENT) > 0.0
    checks = {
        "locked_open_does_not_advance": locked_open_does_not_advance,
        "ordered_advance": ordered_advance,
        "terminal_sparse": sparse_is_zero,
        "event_progress": event_is_dense,
        "seven_shared_skills": len(SKILLS) == 7,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Scene core self-check failed: {checks}")
    return checks
