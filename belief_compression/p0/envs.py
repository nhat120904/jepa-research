"""POPGym Arcade substrate wrapper.

This is the ONLY module in `p0` that needs the substrate installed.  jax and
popgym_arcade are imported lazily inside the functions, so the rest of `p0`
(and its unit tests) run on numpy alone in `diagnosis/.venv`, which has torch
but no jax.

Every fact in `TASKS` below was verified by running the package on
2026-07-30 (see `docs/gateP0_design.md` §1), not read off documentation:

  * `popgym-arcade==0.0.7` installs from PyPI with no build step.
  * `popgym_arcade.make(env_id, partial_obs=...)` returns `(env, params)`;
    `env.reset` / `env.step` are jax-jittable and vmappable.
  * Observations are `(128, 128, 3) uint8` pixels.
  * The oracle hidden state is exposed on the returned `EnvState`:
    `board` (BattleShip, Navigator) and `mine_grid` (MineSweeper).
  * 128 parallel envs x 200 steps ran at ~2.8e5 steps/s on CPU alone, with
    rendering, in `jax.jit(jax.vmap(scan))`.

The `hidden_cardinality` numbers are exact combinatorics, and the MineSweeper
Easy one was confirmed empirically: 2000 independent resets produced exactly
120 distinct `mine_grid`s, matching C(16, 2) = 120.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    hidden_field: str        # attribute on EnvState holding the hidden grid
    grid: Tuple[int, int]
    n_positive: int          # mines / ship cells
    hidden_cardinality: int  # exact size of the hidden-state space
    enumerable: bool         # small enough for ExactEnumerationBelief
    target_bit: int          # what the decision layer asks the agent to find

    @property
    def n_cells(self) -> int:
        return self.grid[0] * self.grid[1]


# 0 = SAFE (name a non-mine cell), 1 = TARGET (name a ship cell).
TASKS = {
    # PRIMARY: the hidden space is exactly enumerable, so the EXACT Bayes
    # posterior is computable with zero GPU-hours and every learned number can
    # be read against a ground truth.
    "MineSweeperEasy": TaskSpec("MineSweeperEasy", "mine_grid", (4, 4), 2,
                                comb(16, 2), True, 0),
    "MineSweeperMedium": TaskSpec("MineSweeperMedium", "mine_grid", (6, 6), 6,
                                  comb(36, 6), False, 0),
    "MineSweeperHard": TaskSpec("MineSweeperHard", "mine_grid", (8, 8), 10,
                                comb(64, 10), False, 0),
    # SECONDARY: hidden space astronomically large, so K is purely a knob of
    # the FILTER -- the second reading Gate C1 §1.2 requires us to test.
    # Verified: 2000 resets gave 2000 distinct boards.
    "BattleShipEasy": TaskSpec("BattleShipEasy", "board", (8, 8), 12,
                               comb(64, 12), False, 1),
    "BattleShipMedium": TaskSpec("BattleShipMedium", "board", (10, 10), 12,
                                 comb(100, 12), False, 1),
    "BattleShipHard": TaskSpec("BattleShipHard", "board", (12, 12), 12,
                               comb(144, 12), False, 1),
}


def available() -> bool:
    """True if the substrate is importable in the current interpreter."""
    try:
        import popgym_arcade  # noqa: F401
        return True
    except Exception:
        return False


def make(task: str, partial_obs: bool = True):
    """`(env, params)` for one POPGym Arcade task.

    `partial_obs=False` gives the paired fully-observable variant, which Gate
    C1 §2.1 uses as a free upper bound.  NOTE (verified, and recorded honestly
    in the design doc): on BattleShip the two renderings demonstrably differ
    from step 3 onwards, but on MineSweeper a short random-action probe found
    no pixel difference.  Confirm MineSweeper's MDP/POMDP pairing on day 0
    before relying on it; the hidden MINE POSITIONS are unobservable in both
    variants either way, so P0's belief measurement does not depend on it.
    """
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    import popgym_arcade
    return popgym_arcade.make(TASKS[task].env_id, partial_obs=partial_obs)


def oracle_params(state, task: str) -> np.ndarray:
    """The TRUE hidden parameter as a flat binary vector.

    MineSweeper folds "revealed" into the same grid (0 = empty, 1 = mine,
    2 = viewed), so the hidden BIT is `grid == 1` only for still-hidden cells;
    a viewed cell is no longer hidden.  We therefore return the *mine
    indicator* as `grid == 1`, and the caller gets the revealed set from
    `revealed_cells`.
    """
    spec = TASKS[task]
    grid = np.asarray(getattr(state, spec.hidden_field))
    if spec.hidden_field == "mine_grid":
        return (grid == 1).astype(np.int8).reshape(-1)
    return (grid > 0).astype(np.int8).reshape(-1)


def revealed_cells(state, task: str) -> dict:
    """Cells whose value the agent has already uncovered -> their bit.

    Used to condition `belief.ExactEnumerationBelief` on the same evidence the
    learned model saw, so the two are compared on identical information.
    """
    spec = TASKS[task]
    grid = np.asarray(getattr(state, spec.hidden_field)).reshape(-1)
    if spec.hidden_field == "mine_grid":
        # value 2 == viewed-and-safe (a viewed mine terminates the episode)
        return {int(i): 0 for i in np.flatnonzero(grid == 2)}
    guesses = getattr(state, "guesses", None)
    if guesses is None:
        return {}
    g = np.asarray(guesses).reshape(-1)
    board = np.asarray(getattr(state, spec.hidden_field)).reshape(-1)
    return {int(i): int(board[i] > 0) for i in np.flatnonzero(g > 0)}


def collect(task: str, n_episodes: int = 64, ep_len: int = 64,
            seed: int = 0, partial_obs: bool = True):
    """Random-policy rollouts; returns (obs, oracle_params, states).

    Deliberately simple and un-jitted: P0's data collection is not the
    bottleneck (the substrate does ~2.8e5 steps/s on CPU even unbatched-ish),
    and a readable collector is worth more here than a fast one.  The
    production trainer should use the vmapped `jax.lax.scan` form measured in
    the design doc.

    Returns
        obs      (n_episodes, ep_len, H, W, 3) uint8
        params   (n_episodes, ep_len, n_cells) int8   -- oracle hidden state
        states   list of the final EnvState per episode
    """
    import jax
    env, ps = make(task, partial_obs=partial_obs)
    n_actions = env.action_space(ps).n
    obs_all, par_all, finals = [], [], []
    key = jax.random.PRNGKey(seed)
    for e in range(n_episodes):
        key, k0 = jax.random.split(key)
        o, st = env.reset(k0, ps)
        os_, ps_ = [], []
        for t in range(ep_len):
            key, ka, ks = jax.random.split(key, 3)
            a = int(jax.random.randint(ka, (), 0, n_actions))
            o, st, r, d, _ = env.step(ks, st, a, ps)
            os_.append(np.asarray(o))
            ps_.append(oracle_params(st, task))
            if bool(d):
                break
        obs_all.append(np.stack(os_))
        par_all.append(np.stack(ps_))
        finals.append(st)
    n = min(len(x) for x in obs_all)
    return (np.stack([x[:n] for x in obs_all]),
            np.stack([x[:n] for x in par_all]),
            finals)
