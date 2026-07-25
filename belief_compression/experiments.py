"""The two required Gate-B measurements.

(i)  COMPRESSION vs GOAL-RICHNESS
     As the goal family grows (more of the hidden parameter becomes
     decision-relevant), how many decision modes M survive and what is the
     planning regret vs the full-belief planner?

(ii) PROBE POLICY COMPARISON
     On both tasks, compare return / decision-regret / planning-compute across
     certainty-equivalent, entropy-seeking, decision-regret VOI, amortized-VOI,
     full-belief oracle, and oracle-fully-observed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .compression import compress
from .core import Belief
from .evaluate import exact_return, fully_observed_return
from .planners import (
    AmortizedVOI,
    CertaintyEquivalent,
    CompressionPlanner,
    DecisionRegretVOI,
    EntropySeeking,
    FullBeliefPlanner,
)
from .tasks import MassSort, OccluderPush


# --------------------------------------------------------------------------- #
# Measurement (i): compression vs goal-richness
# --------------------------------------------------------------------------- #
@dataclass
class CompressionRow:
    richness: int
    n_goals: int
    K: int
    M: int
    ratio: float
    regret_in_family: float      # executing goals INSIDE the compressed family
    regret_out_of_family: float  # executing a goal that needs the full parameter


def _mean_regret(task, family, exec_goals, tol, budget):
    full = FullBeliefPlanner()
    comp = CompressionPlanner(goal_family=family, tol=tol)
    regs = []
    for g in exec_goals:
        rf = exact_return(task, full, g, budget=budget).ret
        rc = exact_return(task, comp, g, budget=budget).ret
        regs.append(rf - rc)
    return float(np.mean(regs)) if regs else 0.0


def measurement1_richness(task, richness_values, full_goal, budget=0) -> List[CompressionRow]:
    """Exact decision-equivalence compression across a growing goal family."""
    rows = []
    prior = Belief.prior(task)
    for r in richness_values:
        family = task.goal_family(r)
        comp = compress(prior, family, tol=0.0)
        reg_in = _mean_regret(task, family, family, tol=0.0, budget=budget)
        reg_out = _mean_regret(task, family, [full_goal], tol=0.0, budget=budget)
        rows.append(
            CompressionRow(
                richness=r,
                n_goals=len(family),
                K=comp.K,
                M=comp.M,
                ratio=comp.ratio,
                regret_in_family=reg_in,
                regret_out_of_family=reg_out,
            )
        )
    return rows


@dataclass
class ToleranceRow:
    tol: float
    K: int
    M: int
    ratio: float
    regret_in_family: float


def measurement1_tolerance(task, tol_values, richness, budget=0) -> List[ToleranceRow]:
    """Lossy compression frontier at fixed (maximal) goal-richness: trade
    regret for smaller M via Q-signature tolerance."""
    rows = []
    prior = Belief.prior(task)
    family = task.goal_family(richness)
    for tol in tol_values:
        comp = compress(prior, family, tol=tol)
        reg_in = _mean_regret(task, family, family, tol=tol, budget=budget)
        rows.append(
            ToleranceRow(tol=tol, K=comp.K, M=comp.M, ratio=comp.ratio, regret_in_family=reg_in)
        )
    return rows


# --------------------------------------------------------------------------- #
# Measurement (ii): probe policy comparison
# --------------------------------------------------------------------------- #
@dataclass
class ProbeRow:
    policy: str
    ret: float
    regret_vs_full: float
    regret_vs_observed: float
    exp_probes: float
    reward_evals: int
    voi_inner: int
    total_compute: int


def measurement2_probes(task, goal) -> List[ProbeRow]:
    full = FullBeliefPlanner()
    policies = [
        ("certainty_equiv", CertaintyEquivalent()),
        ("entropy_seek", EntropySeeking()),
        ("voi", DecisionRegretVOI()),
        ("amortized_voi", AmortizedVOI()),
        ("full_belief", full),
    ]
    full_res = exact_return(task, full, goal)
    obs_ret = fully_observed_return(task, goal)

    rows = []
    for name, pol in policies:
        res = exact_return(task, pol, goal)
        rows.append(
            ProbeRow(
                policy=name,
                ret=res.ret,
                regret_vs_full=full_res.ret - res.ret,
                regret_vs_observed=obs_ret - res.ret,
                exp_probes=res.n_probes_expected,
                reward_evals=res.compute["reward_evals"],
                voi_inner=res.compute["voi_inner"],
                total_compute=res.compute["total"],
            )
        )
    # oracle fully-observed (upper bound; no planning, no probes)
    rows.append(
        ProbeRow(
            policy="fully_observed",
            ret=obs_ret,
            regret_vs_full=full_res.ret - obs_ret,
            regret_vs_observed=0.0,
            exp_probes=0.0,
            reward_evals=0,
            voi_inner=0,
            total_compute=0,
        )
    )
    return rows


# --------------------------------------------------------------------------- #
# Default configurations used by the Gate-B run
# --------------------------------------------------------------------------- #
def default_mass_sort_measure1():
    task = MassSort(n_objects=4, p_heavy=0.5, probe_noise=0.15, probe_cost=0.05, max_budget=0)
    richness = [1, 2, 3, 4]
    full_goal = task.all_objects_goal()
    return task, richness, full_goal


def default_occluder_measure1():
    task = OccluderPush(n_locations=8, fine=4, probe_cost=0.1, max_budget=0)
    # goal-richness = finest granularity present; granularities dividing 8
    richness = [2, 4, 8]
    full_goal = task._binning(task.n_locations)  # every location its own target
    return task, richness, full_goal


def default_mass_sort_measure2():
    # object 2 is decision-IRRELEVANT for the goal, yet its probe is very
    # informative -> an entropy trap that decision-regret VOI must avoid.
    task = MassSort(
        n_objects=3,
        p_heavy=0.5,
        probe_noise=0.15,
        noise_overrides={2: 0.02},
        probe_cost=0.05,
        max_budget=2,
    )
    goal = frozenset({0, 1})
    return task, goal


def default_occluder_measure2():
    # coarse goal: only the bin matters; look_fine is high-info but irrelevant.
    task = OccluderPush(
        n_locations=8,
        fine=4,
        coarse_noise=0.1,
        fine_noise=0.05,
        probe_cost=0.1,
        max_budget=1,
    )
    goal = task.coarse_goal()
    return task, goal
