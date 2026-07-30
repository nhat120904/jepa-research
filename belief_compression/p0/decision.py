"""The decision layer P0 puts on top of a pixel POMDP.

POPGym Arcade's BattleShip / MineSweeper give us property (P) of Gate C1 §1.2 --
a genuine probing action that reveals hidden state -- but not property (G), a
family of goals sharing the SAME hidden state.  Gate C1 §1.2 records that no
released benchmark has (P) and (G) together.

P0 adds (G) on the DECISION side, without touching the environment: the hidden
parameter (the mine grid / the ship board) is exactly the environment's own
hidden state, and a *goal* is a region of the board over which the agent must
name one cell.  Nothing about the simulator, its dynamics, its rewards or its
observations is modified -- only the read-out question we ask of the belief.
That is the same move Gate B made on the numpy tasks, and it is what makes the
multi-goal-invariance pillar testable on an established suite.

    hidden parameter  z   : a length-n binary vector (1 = mine / ship cell)
    goal              g   : a REGION, i.e. an ordered tuple of cell indices
    terminal action   a   : "I name cell a" (a in g)
    reward(z, a, g)       : +1 if z[a] == target_bit else -1, plus a small
                            fixed per-cell utility u_g(a) that breaks ties

The tie-breaking utility matters and is not cosmetic.  Without it, every
hypothesis whose region contains *any* matching cell ties at +1 and
`Task.preferred_action`'s first-wins tie-break makes the signature depend on
cell ORDER rather than on the hidden bits.  With it, the preferred action is
"the highest-utility cell in region g that has the target property" -- a
well-posed task ("pick the best safe cell to open next"), and a signature that
genuinely reads the hidden bits inside the region.

ANALYTIC BOUND.  For one goal g, the preferred action can only be one of the
|g| cells in the region, so there are at most |g| distinct per-goal signatures;
across a family G the full signature is a tuple, so

    M  <=  min( K,  prod_{g in G} |g| )        [`RegionCommit.bound`]

This is the P0 analogue of Gate C0's `min(K, |G|(|A|-1)+1)`.  The product form
(rather than C0's sum form) is a consequence of the hidden space being a bit
vector with no 1-D geometry, and it is the HARDER test: the bound blows past K
quickly as |G| grows, so the goal-richness sweep has a pre-computable point at
which compression must stop working.  Measuring M against this bound over
(K, |G|, |A|) is exactly Gate P0's deliverable (c).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# Which bit the agent is asked to find.
SAFE = 0      # MineSweeper: name a cell that is NOT a mine
TARGET = 1    # BattleShip: name a cell that IS a ship


@dataclass(frozen=True)
class RegionCommit:
    """Decision structure over a length-`n_cells` binary hidden parameter.

    `utility` is a fixed (n_cells,) array of per-cell utilities, seeded once and
    shared by every goal; it is part of the task definition, not a nuisance
    parameter, and is never re-drawn between conditions.
    """

    n_cells: int
    target_bit: int
    utility: Tuple[float, ...]

    # ---------------------------------------------------------------- build --
    @classmethod
    def build(cls, n_cells: int, target_bit: int = SAFE, seed: int = 0) -> "RegionCommit":
        rng = np.random.default_rng(seed)
        # Utilities are strictly ordered and bounded well inside (0, 1) so the
        # match/no-match gap (2.0) always dominates: the preferred action is
        # ALWAYS a matching cell when one exists, and utility only ranks among
        # the matching cells.  This keeps the decision semantics unambiguous.
        u = np.sort(rng.uniform(0.05, 0.45, size=n_cells))
        rng.shuffle(u)
        return cls(n_cells=int(n_cells), target_bit=int(target_bit),
                   utility=tuple(float(x) for x in u))

    # ---------------------------------------------------------------- goals --
    def goal_family(self, n_goals: int, region_size: int,
                    seed: int = 0, disjoint: bool = True) -> List[Tuple[int, ...]]:
        """`n_goals` regions of `region_size` cells each.

        `disjoint=True` partitions distinct cells across goals (requires
        n_goals * region_size <= n_cells); this is the honest default because
        overlapping regions inflate signature agreement for free.  The regions
        are drawn once from a fixed seed and are part of the pre-registered
        configuration -- they are NOT re-drawn per condition or per seed.
        """
        if n_goals < 1 or region_size < 2:
            raise ValueError("need n_goals >= 1 and region_size >= 2")
        rng = np.random.default_rng(1000 + seed)
        if disjoint:
            if n_goals * region_size > self.n_cells:
                raise ValueError(
                    f"cannot fit {n_goals} disjoint regions of size {region_size} "
                    f"into {self.n_cells} cells"
                )
            perm = rng.permutation(self.n_cells)
            return [
                tuple(int(c) for c in sorted(perm[j * region_size:(j + 1) * region_size]))
                for j in range(n_goals)
            ]
        return [
            tuple(int(c) for c in sorted(rng.choice(self.n_cells, region_size, replace=False)))
            for j in range(n_goals)
        ]

    def bound(self, goal_family: Sequence[Sequence[int]], K: int) -> int:
        """min(K, prod_g |g|) -- the analytic ceiling on M (see module docstring)."""
        prod = 1
        for g in goal_family:
            prod *= len(g)
            if prod >= K:
                return int(K)
        return int(min(K, prod))

    # ------------------------------------------------------------- rewards --
    def terminal_actions(self, goal: Sequence[int]) -> List[int]:
        return list(goal)

    def reward(self, z, a: int, goal) -> float:
        """+/-1 for matching the target bit, plus the cell's ranking utility."""
        match = 1.0 if int(z[a]) == self.target_bit else -1.0
        return match + self.utility[a]

    def preferred_action(self, z, goal: Sequence[int]) -> int:
        """argmax_a reward(z, a, goal) -- the highest-utility MATCHING cell.

        Equivalent to `core.Task.preferred_action` but computed directly, so it
        can be used to label the ORACLE hidden state without building a Task.
        """
        best_a, best_r = None, -np.inf
        for a in goal:
            r = self.reward(z, a, goal)
            if r > best_r + 1e-12:
                best_r, best_a = r, a
        return best_a

    def signature(self, z, goal_family) -> Tuple[int, ...]:
        """The decision signature of ONE hidden parameter over the goal family.

        Identical in meaning to `compression._preferred_signature`; provided
        here so oracle hidden states can be labelled without instantiating a
        Task (Gate C1 STOP S5 asks for the number of distinct signatures under
        the ORACLE hidden state).
        """
        return tuple(self.preferred_action(z, g) for g in goal_family)
