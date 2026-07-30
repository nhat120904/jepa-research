"""Two tiny, fully-controllable POMDPs.

Both share the same abstract shape so the planners are task-agnostic:

  hidden parameter  ->  probe(s) that inform it  ->  terminal commit action
  goal family that controls how much of the hidden parameter is
  DECISION-RELEVANT (and therefore how much can be compressed away).

mass_sort
    N objects, each with a hidden mass class in {LIGHT, HEAVY}.  A TAP probe
    on object i returns a noisy reading of its class.  The terminal action is
    a force assignment (one force per object); each ACTIVE object (per the
    goal) rewards +1 if force matches true mass else -1.  Objects outside the
    goal's active set are don't-cares (reward 0) -> the source of compression.

occluder_push
    A hidden target LOCATION in {0..L-1} behind an occluder.  bin = loc // fine
    is the coarse (decision-relevant) part; loc % fine is a nuisance part.  A
    LOOK_COARSE probe reveals the bin (noisy); a LOOK_FINE probe reveals the
    nuisance index (noisy, high info, decision-irrelevant under a coarse goal).
    The terminal action commits a PUSH toward a bin; +1 if it matches the
    goal's binning of the true location else -1.
"""

from __future__ import annotations

import itertools
from typing import List

import numpy as np

from .core import ComputeCounter, Task

LIGHT, HEAVY = 0, 1


# --------------------------------------------------------------------------- #
# mass_sort
# --------------------------------------------------------------------------- #
class MassSort(Task):
    name = "mass_sort"

    def __init__(
        self,
        n_objects: int = 4,
        p_heavy=0.5,
        probe_noise=0.15,
        probe_cost: float = 0.05,
        max_budget: int = 2,
        noise_overrides: dict | None = None,
    ):
        self.n_objects = n_objects
        self.probe_cost = probe_cost
        self.max_budget = max_budget

        # per-object flip probability for the TAP probe (lower = more informative)
        self.noise = np.full(n_objects, float(probe_noise))
        if noise_overrides:
            for i, v in noise_overrides.items():
                self.noise[i] = v

        # enumerate hidden states: every mass assignment over N objects
        self.hidden_states = [
            tuple(bits) for bits in itertools.product((LIGHT, HEAVY), repeat=n_objects)
        ]
        # prior
        if np.isscalar(p_heavy):
            p_heavy = np.full(n_objects, float(p_heavy))
        self._p_heavy = np.asarray(p_heavy, dtype=float)
        pri = np.array(
            [
                np.prod(
                    [
                        self._p_heavy[i] if z[i] == HEAVY else 1 - self._p_heavy[i]
                        for i in range(n_objects)
                    ]
                )
                for z in self.hidden_states
            ]
        )
        self.prior = pri / pri.sum()

        self.probes = [f"tap{i}" for i in range(n_objects)]

    # goals ----------------------------------------------------------------
    def goal_family(self, richness: int) -> List[frozenset]:
        """richness = number of decision-relevant objects.  Each goal activates
        one distinct object; the family's union covers `richness` objects."""
        richness = max(0, min(richness, self.n_objects))
        return [frozenset({i}) for i in range(richness)]

    def all_objects_goal(self) -> frozenset:
        return frozenset(range(self.n_objects))

    def active(self, goal) -> tuple:
        return tuple(sorted(goal))

    # terminal decision -----------------------------------------------------
    def terminal_actions(self, goal) -> List[tuple]:
        """A terminal action = force pattern over the ACTIVE objects only.
        (Forces on don't-care objects are irrelevant, so we omit them.)"""
        act = self.active(goal)
        if not act:
            return [()]
        return [tuple(bits) for bits in itertools.product((LIGHT, HEAVY), repeat=len(act))]

    def reward(self, z, a, goal, counter: ComputeCounter | None = None) -> float:
        if counter is not None:
            counter.rewards()
        act = self.active(goal)
        r = 0.0
        for j, i in enumerate(act):
            r += 1.0 if a[j] == z[i] else -1.0
        return r

    # probes ----------------------------------------------------------------
    def obs_space(self, probe) -> List[int]:
        return [LIGHT, HEAVY]

    def obs_prob(self, z, probe, o) -> float:
        i = int(probe[3:])  # "tap{i}"
        true = z[i]
        noise = self.noise[i]
        return (1.0 - noise) if o == true else noise


# --------------------------------------------------------------------------- #
# occluder_push
# --------------------------------------------------------------------------- #
class OccluderPush(Task):
    name = "occluder_push"

    def __init__(
        self,
        n_locations: int = 8,
        fine: int = 4,
        coarse_noise: float = 0.1,
        fine_noise: float = 0.05,
        probe_cost: float = 0.1,
        max_budget: int = 1,
        prior=None,
    ):
        assert n_locations % fine == 0, "n_locations must be divisible by fine"
        self.n_locations = n_locations
        self.fine = fine
        self.n_bins = n_locations // fine
        self.coarse_noise = coarse_noise
        self.fine_noise = fine_noise
        self.probe_cost = probe_cost
        self.max_budget = max_budget

        self.hidden_states = list(range(n_locations))  # location id
        if prior is None:
            pri = np.ones(n_locations)
        else:
            pri = np.asarray(prior, dtype=float)
        self.prior = pri / pri.sum()

        self.probes = ["look_coarse", "look_fine"]

    def param_embedding(self):
        # locations are an ordered 1-D axis
        return np.arange(self.n_locations, dtype=float)

    def bin_of(self, loc: int) -> int:
        return loc // self.fine

    def fine_of(self, loc: int) -> int:
        return loc % self.fine

    # goals ----------------------------------------------------------------
    def goal_family(self, richness: int):
        """A goal is a binning of locations (tuple loc->bin-id).  richness =
        finest granularity present in the family.  Coarser binnings merge many
        locations into one push target (decision-equivalent)."""
        richness = max(1, min(richness, self.n_locations))
        grans = self._granularities_up_to(richness)
        return [self._binning(g) for g in grans]

    def coarse_goal(self):
        """The natural coarse goal: push toward the location's bin."""
        return self._binning(self.n_bins)

    def _granularities_up_to(self, richness: int) -> List[int]:
        # granularities that evenly partition the locations, up to `richness`
        grans = [g for g in range(1, richness + 1) if self.n_locations % g == 0]
        if richness not in grans and self.n_locations % richness == 0:
            grans.append(richness)
        return sorted(set(grans))

    def _binning(self, granularity: int) -> tuple:
        size = self.n_locations / granularity
        return tuple(int(loc // size) for loc in range(self.n_locations))

    # terminal decision -----------------------------------------------------
    def terminal_actions(self, goal) -> List[int]:
        return sorted(set(goal))  # the set of push targets (bin ids) in this goal

    def reward(self, z, a, goal, counter: ComputeCounter | None = None) -> float:
        if counter is not None:
            counter.rewards()
        return 1.0 if a == goal[z] else -1.0

    # probes ----------------------------------------------------------------
    def obs_space(self, probe) -> List[int]:
        if probe == "look_coarse":
            return list(range(self.n_bins))
        return list(range(self.fine))

    def obs_prob(self, z, probe, o) -> float:
        if probe == "look_coarse":
            true = self.bin_of(z)
            k = self.n_bins
            noise = self.coarse_noise
        else:  # look_fine
            true = self.fine_of(z)
            k = self.fine
            noise = self.fine_noise
        if k == 1:
            return 1.0
        return (1.0 - noise) if o == true else noise / (k - 1)


# --------------------------------------------------------------------------- #
# grid_param  (Gate C0: belief resolution is a first-class knob)
# --------------------------------------------------------------------------- #
class GridParam(Task):
    """A hidden CONTINUOUS parameter theta in [0,1) discretized on a uniform
    grid of `resolution` cells, so K = resolution is a free knob.

    The point of this task is that **belief resolution and decision structure
    are decoupled by construction**:

      * The hidden parameter is continuous (think: object mass, friction
        coefficient, contact-point offset).  `resolution` only controls how
        finely the belief discretizes it -> K = resolution.
      * The DECISION structure lives in continuous theta-space and never sees
        the grid: a goal is a fixed tuple of `n_controllers` controller
        centres c_a, reward(theta, a) = 1 - 2|theta - c_a|, so the optimal
        controller is the nearest centre.  Refining the grid cannot create new
        optimal controllers.

    Consequence (the Gate-C0 hypothesis, provable here): for a goal family of
    G goals with A controllers each, the preferred-action signature is constant
    on every cell of the arrangement of all goals' decision boundaries.  There
    are at most G*(A-1) distinct boundaries, hence

        M  <=  min(K,  G*(A-1) + 1)                      [`decision_bound()`]

    which is INDEPENDENT of K.  If the empirical M tracks this bound while K
    grows, M/K -> 0 and the compression win grows without bound.

    Probes are quantized sensors with a FIXED number of readout bins (the
    sensor does not get better when the belief grid gets finer), so the
    observation space is held constant as K grows.
    """

    name = "grid_param"

    def __init__(
        self,
        resolution: int = 16,
        n_controllers: int = 3,
        n_goals: int = 1,
        sensor_bins: tuple = (3, 8),
        sensor_noise: tuple = (0.15, 0.05),
        probe_cost: float = 0.05,
        max_budget: int = 1,
    ):
        assert resolution >= 1 and n_controllers >= 2 and n_goals >= 1
        self.resolution = int(resolution)
        self.n_controllers = int(n_controllers)
        self.n_goals = int(n_goals)
        self.probe_cost = probe_cost
        self.max_budget = max_budget

        # hidden states are grid-cell indices; theta[i] is the cell centre
        self.hidden_states = list(range(self.resolution))
        self.theta = (np.arange(self.resolution) + 0.5) / self.resolution
        self.prior = np.full(self.resolution, 1.0 / self.resolution)

        # quantized sensors: bin count fixed, independent of the belief grid
        self.sensor_bins = tuple(int(b) for b in sensor_bins)
        self.sensor_noise = tuple(float(v) for v in sensor_noise)
        assert len(self.sensor_bins) == len(self.sensor_noise)
        self.probes = [f"sense{b}" for b in self.sensor_bins]
        self._noise_of = dict(zip(self.probes, self.sensor_noise))
        self._bins_of = dict(zip(self.probes, self.sensor_bins))

    def param_embedding(self):
        return self.theta

    # goals ----------------------------------------------------------------- #
    def _goal(self, j: int) -> tuple:
        """Goal j = a tuple of controller centres, offset so that different
        goals put their decision boundaries in different places (this is what
        makes a richer goal family carve theta into more cells)."""
        A = self.n_controllers
        off = j / (self.n_goals * A) if self.n_goals > 1 else 0.0
        return tuple(float((a + 0.5) / A + off) for a in range(A))

    def goal_family(self, richness: int) -> List[tuple]:
        """richness = number of goals in the family (each adds A-1 boundaries)."""
        richness = max(1, min(int(richness), self.n_goals))
        return [self._goal(j) for j in range(richness)]

    def decision_bound(self, richness: int | None = None) -> int:
        """Analytic upper bound on M: cells in the boundary arrangement."""
        G = self.n_goals if richness is None else max(1, min(int(richness), self.n_goals))
        return min(self.resolution, G * (self.n_controllers - 1) + 1)

    # terminal decision ------------------------------------------------------ #
    def terminal_actions(self, goal) -> List[int]:
        return list(range(len(goal)))

    def reward(self, z, a, goal, counter: ComputeCounter | None = None) -> float:
        if counter is not None:
            counter.rewards()
        return 1.0 - 2.0 * abs(self.theta[z] - goal[a])

    # probes ----------------------------------------------------------------- #
    def obs_space(self, probe) -> List[int]:
        return list(range(self._bins_of[probe]))

    def obs_prob(self, z, probe, o) -> float:
        b = self._bins_of[probe]
        if b == 1:
            return 1.0
        noise = self._noise_of[probe]
        true = min(int(self.theta[z] * b), b - 1)
        return (1.0 - noise) if o == true else noise / (b - 1)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_task(name: str, **kw) -> Task:
    if name == "mass_sort":
        return MassSort(**kw)
    if name == "occluder_push":
        return OccluderPush(**kw)
    if name == "grid_param":
        return GridParam(**kw)
    raise ValueError(f"unknown task {name!r}")
