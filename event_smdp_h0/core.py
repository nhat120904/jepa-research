"""Pure search and event-scoring utilities for the Event-SMDP H0 gate.

This module deliberately has no MuJoCo, torch, or stable-worldmodel imports so
that its invariants can be checked before the expensive simulator job starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Callable, Iterable

import numpy as np


ARM_TERMINAL = "terminal_only"
ARM_EVENT = "event_state"


@dataclass(frozen=True)
class EventSummary:
    """Ordinal multi-state summary of one action-conditioned rollout."""

    stable_success: bool
    final_stage: int
    max_stage: int
    first_max_step: int
    goal_suffix: int
    drop: bool
    final_distance_m: float
    min_distance_m: float
    final_cube_position: tuple[float, float, float]
    stage_trace: tuple[int, ...]

    def reward(self, arm: str) -> float:
        if arm == ARM_TERMINAL:
            return float(self.stable_success)
        if arm != ARM_EVENT:
            raise ValueError(f"unknown arm: {arm}")
        if self.stable_success:
            return 1.0

        # An ordinal encoding, not a physical-distance surrogate.  A one-state
        # increase is larger than every possible timing/dwell tie-breaker.
        final_term = self.final_stage / 6.0
        max_term = self.max_stage / 60.0
        time_term = 0.0
        if self.max_stage > 0 and self.stage_trace:
            time_term = 0.02 * (
                1.0 - self.first_max_step / max(len(self.stage_trace) - 1, 1)
            )
        dwell_term = 0.04 * min(self.goal_suffix, 3) / 3.0
        drop_term = 0.05 if self.drop else 0.0
        return float(max(0.0, final_term + max_term + time_term + dwell_term - drop_term))


def suffix_count(mask: Iterable[bool]) -> int:
    total = 0
    for value in reversed(tuple(bool(x) for x in mask)):
        if not value:
            break
        total += 1
    return total


def summarize_events(
    distances_m: np.ndarray,
    cube_positions: np.ndarray,
    robot_contacts: np.ndarray,
    *,
    start_cube_position: np.ndarray,
    start_distance_m: float,
    success_tolerance_m: float = 0.04,
    near_tolerance_m: float = 0.08,
    moved_tolerance_m: float = 0.025,
    toward_goal_tolerance_m: float = 0.015,
    lift_tolerance_m: float = 0.012,
    stable_dwell: int = 3,
) -> EventSummary:
    """Map a true trajectory to a cyclic task-conditioned event state trace.

    States are evaluated at every step and may move backwards.  They are not
    treated as globally competing terminal events.  Within a state, success
    and drop are possible next events, matching the multi-state/semi-Markov
    interpretation used by the proposed learned model.
    """

    distances = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    cubes = np.asarray(cube_positions, dtype=np.float64)
    contacts = np.asarray(robot_contacts, dtype=bool).reshape(-1)
    start_cube = np.asarray(start_cube_position, dtype=np.float64).reshape(3)
    if not len(distances) or cubes.shape != (len(distances), 3):
        raise ValueError("event trace arrays have inconsistent shapes")
    if contacts.shape != distances.shape:
        raise ValueError("contact trace has inconsistent shape")
    if stable_dwell <= 0 or near_tolerance_m <= success_tolerance_m:
        raise ValueError("invalid event thresholds")

    goal_mask = distances <= success_tolerance_m
    goal_suffix = suffix_count(goal_mask)
    stable = goal_suffix >= stable_dwell
    lifted = cubes[:, 2] >= start_cube[2] + lift_tolerance_m
    displacement = np.linalg.norm(cubes - start_cube[None], axis=1)
    toward_goal = distances <= float(start_distance_m) - toward_goal_tolerance_m
    manipulated = lifted | ((displacement >= moved_tolerance_m) & toward_goal)

    stages = np.zeros(len(distances), dtype=np.int64)
    stages[contacts] = 1
    stages[manipulated] = 2
    stages[distances <= near_tolerance_m] = 3
    stages[goal_mask] = 4
    if stable:
        stages[-stable_dwell:] = 5

    # A drop is a failure branch only after a real lift and only when the
    # object has returned to the table away from the target without contact.
    lifted_seen = np.maximum.accumulate(lifted)
    dropped = lifted_seen & (cubes[:, 2] <= start_cube[2] + 0.005) & ~contacts & ~goal_mask
    max_stage = int(stages.max())
    first_max = int(np.flatnonzero(stages == max_stage)[0])
    return EventSummary(
        stable_success=bool(stable),
        final_stage=int(stages[-1]),
        max_stage=max_stage,
        first_max_step=first_max,
        goal_suffix=int(goal_suffix),
        drop=bool(dropped.any()),
        final_distance_m=float(distances[-1]),
        min_distance_m=float(distances.min()),
        final_cube_position=tuple(float(x) for x in cubes[-1]),
        stage_trace=tuple(int(x) for x in stages),
    )


def proposal_order(seed: int, path: tuple[int, ...], depth: int, branching: int) -> tuple[int, ...]:
    """Node-local, arm-independent expansion order."""

    payload = f"{seed}|{depth}|" + ",".join(str(x) for x in path)
    digest = hashlib.sha256(payload.encode()).digest()
    node_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(node_seed)
    return tuple(int(x) for x in rng.permutation(branching))


@dataclass
class SearchNode:
    path: tuple[int, ...] = ()
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    selected_action: int
    selected_visits: int
    selected_mean_value: float
    simulations: int
    unique_sequences: int
    max_observed_reward: float
    root_stats: tuple[dict[str, float | int], ...]


def _select_child(
    node: SearchNode,
    *,
    seed: int,
    absolute_depth: int,
    branching: int,
    exploration: float,
) -> tuple[int, SearchNode]:
    order = proposal_order(seed, node.path, absolute_depth, branching)
    for action in order:
        if action not in node.children:
            child = SearchNode(path=node.path + (action,))
            node.children[action] = child
            return action, child

    log_parent = math.log(max(node.visits, 1))
    order_rank = {action: rank for rank, action in enumerate(order)}

    def key(item: tuple[int, SearchNode]) -> tuple[float, float, int]:
        action, child = item
        bonus = exploration * math.sqrt(log_parent / max(child.visits, 1))
        return (child.mean_value + bonus, child.mean_value, -order_rank[action])

    return max(node.children.items(), key=key)


def uct_search(
    *,
    start_depth: int,
    total_depth: int,
    branching: int,
    simulations: int,
    seed: int,
    exploration: float,
    evaluate: Callable[[tuple[int, ...]], float],
) -> SearchResult:
    """Run deterministic UCT over fixed, depth-indexed skill proposals."""

    if not 0 <= start_depth < total_depth:
        raise ValueError("invalid search depth")
    if branching < 2 or simulations < branching:
        raise ValueError("simulations must cover every root proposal")
    root = SearchNode()
    seen: set[tuple[int, ...]] = set()
    max_reward = -float("inf")

    for _ in range(simulations):
        node = root
        path_nodes = [root]
        choices: list[int] = []
        for depth in range(start_depth, total_depth):
            action, node = _select_child(
                node,
                seed=seed,
                absolute_depth=depth,
                branching=branching,
                exploration=exploration,
            )
            choices.append(action)
            path_nodes.append(node)
        sequence = tuple(choices)
        reward = float(evaluate(sequence))
        if not np.isfinite(reward):
            raise RuntimeError("search evaluator returned a non-finite reward")
        seen.add(sequence)
        max_reward = max(max_reward, reward)
        for visited in path_nodes:
            visited.visits += 1
            visited.value_sum += reward

    order = proposal_order(seed, (), start_depth, branching)
    order_rank = {action: rank for rank, action in enumerate(order)}

    def robust_key(item: tuple[int, SearchNode]) -> tuple[int, float, int]:
        action, child = item
        return (child.visits, child.mean_value, -order_rank[action])

    selected, child = max(root.children.items(), key=robust_key)
    stats = tuple(
        {
            "action": int(action),
            "visits": int(root.children[action].visits),
            "mean_value": float(root.children[action].mean_value),
        }
        for action in sorted(root.children)
    )
    return SearchResult(
        selected_action=int(selected),
        selected_visits=int(child.visits),
        selected_mean_value=float(child.mean_value),
        simulations=int(simulations),
        unique_sequences=len(seen),
        max_observed_reward=float(max_reward),
        root_stats=stats,
    )


def make_skill_lattice(
    nominal_actions: np.ndarray,
    *,
    action_block: int,
    branching: int,
    noise_scale: np.ndarray,
    noise_rho: float,
    seed: int,
) -> np.ndarray:
    """Create one support-matched nominal-plus-noise skill lattice.

    Output shape is ``(depth, branching, action_block, action_dim)``.  Variant
    zero is exactly the logged nominal chunk at every depth; all other variants
    use temporally correlated normalized-action noise.  This is a privileged H0
    proposal support shared by both arms, not the proposed learned skill prior.
    """

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    if nominal.ndim != 2 or len(nominal) % action_block:
        raise ValueError("nominal action horizon must be divisible by action_block")
    if branching < 2 or not 0 <= noise_rho < 1:
        raise ValueError("invalid proposal configuration")
    scale = np.asarray(noise_scale, dtype=np.float64).reshape(-1)
    if scale.shape != (nominal.shape[1],) or np.any(scale < 0):
        raise ValueError("noise scale does not match action dimension")

    depth = len(nominal) // action_block
    lattice = np.repeat(
        nominal.reshape(depth, 1, action_block, nominal.shape[1]), branching, axis=1
    )
    rng = np.random.default_rng(seed)
    innovation_scale = math.sqrt(1.0 - noise_rho**2)
    for d in range(depth):
        for variant in range(1, branching):
            noise = np.zeros((action_block, nominal.shape[1]), dtype=np.float64)
            noise[0] = rng.normal(size=nominal.shape[1]) * scale
            for t in range(1, action_block):
                innovation = rng.normal(size=nominal.shape[1]) * scale
                noise[t] = noise_rho * noise[t - 1] + innovation_scale * innovation
            lattice[d, variant] = np.clip(lattice[d, variant] + noise, -1.0, 1.0)
    return lattice.astype(np.float32)


def self_check() -> dict[str, bool]:
    """Cheap invariants executed inside every Slurm smoke job."""

    stages = summarize_events(
        np.array([0.12, 0.07, 0.03, 0.03, 0.03]),
        np.array([[0.3, 0.0, 0.02], [0.33, 0.0, 0.02], [0.38, 0.0, 0.02],
                  [0.39, 0.0, 0.02], [0.4, 0.0, 0.02]]),
        np.array([False, True, True, False, False]),
        start_cube_position=np.array([0.3, 0.0, 0.02]),
        start_distance_m=0.12,
    )
    if not stages.stable_success or stages.final_stage != 5:
        raise AssertionError("stable-success event invariant failed")

    def reward(path: tuple[int, ...]) -> float:
        return float(sum(x == 0 for x in path))

    search = uct_search(
        start_depth=0,
        total_depth=3,
        branching=3,
        simulations=18,
        seed=7,
        exploration=0.5,
        evaluate=reward,
    )
    if search.simulations != 18 or sum(int(x["visits"]) for x in search.root_stats) != 18:
        raise AssertionError("UCT accounting invariant failed")

    nominal = np.zeros((10, 5), dtype=np.float32)
    lattice = make_skill_lattice(
        nominal,
        action_block=5,
        branching=3,
        noise_scale=np.ones(5) * 0.1,
        noise_rho=0.5,
        seed=9,
    )
    if lattice.shape != (2, 3, 5, 5) or not np.array_equal(lattice[:, 0], nominal.reshape(2, 5, 5)):
        raise AssertionError("skill lattice invariant failed")
    return {"event_scoring": True, "uct_accounting": True, "skill_lattice": True}
