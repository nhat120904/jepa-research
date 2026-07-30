"""The Gate P0 measurement protocol and its PRE-REGISTERED verdict.

Every threshold in `PREREG` is fixed here, in code, before any training run.
`verdict()` is a pure function of the measured summary and the frozen
thresholds, so the decision cannot drift after seeing the numbers.  This is the
same discipline as `scaling.make_verdict` (Gate C0) and `evaluate` (Gate B).

What is measured, per (task, belief model, K, goal family, seed):

  M, M/K      via the EXISTING `compression.compress` on a `HypothesisTask`.
              No parallel implementation.
  bound       min(K, prod_g |g|) -- the analytic ceiling (see `decision`).
  ESS         effective sample size of the K hypothesis weights.
  n_distinct  how many of the K hypotheses are different parameter values.
  ops         `ComputeCounter` total charged by `compress()` -- the O(K|G||A|)
              signature build, which is the honest `C_comp` register Gate C0
              §A6 requires as the headline (never the cached column).

and, once per (task, model, K):

  belief_share  C_belief / (C_belief + C_search).  Gate C1 STOP S3.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..compression import compress
from ..core import Belief, ComputeCounter
from .belief import BeliefModel
from .decision import RegionCommit
from .hypotheses import HypothesisSet, HypothesisTask

# --------------------------------------------------------------------------- #
# PRE-REGISTERED THRESHOLDS -- frozen before any run.
# --------------------------------------------------------------------------- #
PREREG: Dict[str, float] = {
    # Reference operating point.  K_REF is the smallest K at which Gate C1's G1
    # is stated ("M/K <= 0.25 at K >= 128"), so P0 measures at exactly that K.
    "K_REF": 128,

    # --- P0-A: the hypotheses are decision-diverse (not VACUOUS) ------------ #
    "MIN_MODES": 3,          # median M at K_REF must be >= this
    "MIN_ESS": 8.0,          # median ESS of the K weights must be >= this
    "MIN_ORACLE_SIGS": 3,    # distinct signatures of the TRUE hidden state
                             # across eval episodes must be >= this (Gate C1 S5)

    # --- P0-B: something is actually compressed ----------------------------- #
    "MAX_RATIO": 0.25,       # median M/K at K_REF must be <= this (Gate C1 G1)

    # --- P0-C: the K >> bound regime that Gate C0 requires exists ----------- #
    "MIN_K_OVER_BOUND": 4.0,   # K / min(K, prod_g|g|) must reach this
    "BOUND_SLACK": 1.25,       # and measured M must be <= BOUND_SLACK * bound,
                               # i.e. M tracks the decision structure, not K

    # --- P0-D: the filter does not eat the win (Gate C1 STOP S3) ------------ #
    "MAX_BELIEF_SHARE": 0.80,

    # --- STOP conditions (fire independently of the PASS gates) ------------- #
    "STOP_VACUOUS_MODES": 2,     # M <= this on every task  -> VACUOUS
    "STOP_VACUOUS_ESS": 2.0,     # ESS < this on every task -> VACUOUS
    "STOP_NOCOMPRESS_RATIO": 0.5,  # M/K > this everywhere  -> NO-COMPRESSION
}

VERDICT_PASS = "PASS"
VERDICT_AMBER = "AMBER"
VERDICT_STOP_VACUOUS = "STOP-VACUOUS"
VERDICT_STOP_NOCOMPRESS = "STOP-NO-COMPRESSION"
VERDICT_STOP_S3 = "STOP-S3-FILTER-DOMINATES"


# --------------------------------------------------------------------------- #
# One measurement cell
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    task: str
    model: str
    K: int
    n_goals: int
    region_size: int
    seed: int
    M: int
    ratio: float
    bound: int
    K_over_bound: float
    ess: float
    n_distinct: int
    ops: int
    tol: float = 0.0
    # Fraction of the episode elapsed when the belief was read.  The belief's
    # confidence changes enormously within an episode (early = diffuse, late =
    # nearly resolved), and M with it, so a single-point measurement would be
    # an arbitrary operating point.  Phase is part of the cell identity.
    phase: float = 0.0

    @property
    def key(self) -> tuple:
        return (self.n_goals, self.region_size, self.phase)

    def as_dict(self) -> dict:
        return asdict(self)


def measure_cell(hyps: HypothesisSet, decision: RegionCommit, goal_family,
                 task: str, seed: int = 0, tol: float = 0.0,
                 phase: float = 0.0) -> Cell:
    """Compress one hypothesis set and record everything P0 needs.

    The compression is `belief_compression.compression.compress` verbatim, run
    on a `HypothesisTask` whose hidden states are the K hypotheses.  The
    `ComputeCounter` therefore charges exactly the signature-build cost Gate C0
    calls `C_comp`.
    """
    hyp_task = HypothesisTask(hyps, decision)
    belief = Belief.prior(hyp_task)
    counter = ComputeCounter()
    comp = compress(belief, list(goal_family), tol=tol, counter=counter)
    bound = decision.bound(goal_family, hyps.K)
    return Cell(
        task=task,
        model=hyps.source,
        K=hyps.K,
        n_goals=len(goal_family),
        region_size=len(goal_family[0]),
        seed=seed,
        M=comp.M,
        ratio=comp.ratio,
        bound=bound,
        K_over_bound=hyps.K / max(1, bound),
        ess=hyps.ess,
        n_distinct=hyps.n_distinct,
        ops=counter.total(),
        tol=tol,
        phase=float(phase),
    )


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def sweep(model_factory: Callable[..., BeliefModel],
          decision: RegionCommit,
          K_values: Sequence[int],
          goal_specs: Sequence[tuple],
          task: str,
          n_seeds: int = 5,
          tol: float = 0.0,
          family_seed: int = 0,
          phases: Sequence[float] = (0.0,)) -> List[Cell]:
    """Sweep K x goal-family richness x episode phase x seed.

    `model_factory(seed, phase, rng)` returns the belief model for one
    evaluation draw -- in production, the belief the trained network holds at
    `phase` of the way through a logged episode; offline, a synthetic stand-in.
    The goal REGIONS are built once from `family_seed` and reused across every
    cell, so richness is the only thing that varies between goal conditions.
    """
    rows: List[Cell] = []
    for n_goals, region_size in goal_specs:
        family = decision.goal_family(n_goals, region_size, seed=family_seed)
        for phase in phases:
            for K in K_values:
                for s in range(n_seeds):
                    rng = np.random.default_rng(10_000 * s + 97 * K
                                                + int(1000 * phase))
                    model = model_factory(s, phase, rng)
                    hyps = model.hypotheses(K, rng)
                    rows.append(measure_cell(hyps, decision, family, task=task,
                                             seed=s, tol=tol, phase=phase))
    return rows


def oracle_signature_count(true_params: np.ndarray, decision: RegionCommit,
                           goal_family) -> int:
    """Distinct decision signatures of the TRUE hidden state across episodes.

    Gate C1 STOP S5 is stated partly on this number: if the ORACLE hidden state
    only ever produces one or two signatures across the evaluation set, the task
    itself carries no decision diversity and no filter could represent any.  It
    is a TASK-side control, and it needs no belief model at all -- so it is the
    first thing P0 runs.
    """
    sigs = {decision.signature(z, goal_family) for z in np.asarray(true_params)}
    return len(sigs)


# --------------------------------------------------------------------------- #
# Cost split (Gate C1 STOP S3)
# --------------------------------------------------------------------------- #
@dataclass
class CostSplit:
    c_belief: float
    c_search: float
    unit: str = "ms"

    @property
    def belief_share(self) -> float:
        tot = self.c_belief + self.c_search
        return float(self.c_belief / tot) if tot > 0 else 1.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["belief_share"] = self.belief_share
        return d


def compress_ops(hyps: HypothesisSet, decision: RegionCommit, goal_family,
                 tol: float = 0.0) -> int:
    """`C_comp`: primitive ops the compression itself charges, via ComputeCounter."""
    return measure_cell(hyps, decision, goal_family, task="_ops", tol=tol).ops


def expectimax_nodes(H: int, obs_branch: int, n_probes: int = 1) -> int:
    """Nodes in a depth-H expectimax tree with `obs_branch` outcomes per probe.

    Matches the recursion in `planners.expectimax`: the root plus, at every
    level, one child per (probe, observation).
    """
    if H < 0:
        raise ValueError("H must be >= 0")
    b = int(n_probes) * int(obs_branch)
    total, level = 1, 1
    for _ in range(H):
        level *= b
        total += level
    return total


def search_flops(K: int, M: int, n_goals: int, n_actions: int, H: int,
                 obs_branch: int, q_flops: float, n_probes: int = 1,
                 compressed: bool = True) -> float:
    """FLOPs the SEARCH side spends per decision.

    Two terms, matching Gate C0's accounting:
      * the signature / Q build, K * |G| * |A| head forwards -- Gate C0 §A6's
        `C_comp`, the honest headline (never the cached column);
      * the expectimax tree, which costs |A| head forwards per node per
        hypothesis carried: M for the compressed planner, K for full belief.

    `q_flops` is the cost of ONE `Q(z, a, g)` evaluation.  It is the parameter
    that decides Gate C1 STOP S3, and it is measurable in P0 without training
    anything: a forward pass costs the same whatever the weights are, so an
    untrained head of the specified size gives the exact number.
    """
    carried = M if compressed else K
    build = K * n_goals * n_actions * q_flops if compressed else 0.0
    tree = expectimax_nodes(H, obs_branch, n_probes) * carried * n_actions * q_flops
    return float(build + tree)


def belief_flops(K: int, encoder_flops: float, per_hypothesis_flops: float,
                 amortized: bool = True) -> float:
    """FLOPs the BELIEF side spends per decision -- the S3 numerator.

    The two architectures land on opposite sides of S3, which is why P0
    measures both:

      amortized=True   one encoder forward produces the posterior parameters,
                       then K cheap decodes.  C_belief is nearly FLAT in K, so
                       the belief share FALLS as K grows and S3 is unlikely to
                       fire at large K -- but at small K the encoder dominates
                       everything and S3 fires trivially.
      amortized=False  a particle filter re-scores every hypothesis with a
                       learned likelihood: K encoder-scale forwards.  C_belief
                       is LINEAR in K, exactly like the search it is competing
                       with, and S3 becomes a genuine coin-flip.
    """
    if amortized:
        return float(encoder_flops + K * per_hypothesis_flops)
    return float(K * (encoder_flops + per_hypothesis_flops))


def time_callable(fn: Callable[[], object], repeats: int = 5) -> float:
    """Median wall-clock milliseconds of `fn`, after one warm-up call."""
    fn()
    ts = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(ts))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
@dataclass
class TaskSummary:
    """Everything the verdict needs about one task, aggregated over seeds.

    Every quantity is keyed by the goal family `(n_goals, region_size)`, because
    the PASS gates must all hold on the SAME family -- see `verdict`.
    """

    task: str
    K_ref: int
    oracle_sigs: Dict[tuple, int] = field(default_factory=dict)
    median_M: Dict[tuple, float] = field(default_factory=dict)
    median_ratio: Dict[tuple, float] = field(default_factory=dict)
    median_ess: Dict[tuple, float] = field(default_factory=dict)
    median_bound: Dict[tuple, float] = field(default_factory=dict)

    def keys(self) -> List[tuple]:
        return sorted(self.median_M)

    def best_ratio(self) -> Optional[float]:
        return min(self.median_ratio.values()) if self.median_ratio else None

    def max_M(self) -> Optional[float]:
        return max(self.median_M.values()) if self.median_M else None

    def max_ess(self) -> Optional[float]:
        return max(self.median_ess.values()) if self.median_ess else None


def summarise(rows: Sequence[Cell], task: str, oracle_sigs,
              K_ref: int = int(PREREG["K_REF"])) -> TaskSummary:
    """Aggregate `rows` over seeds, per (goal family, phase), at K = K_ref.

    `oracle_sigs` is either a dict keyed by `Cell.key` -- (n_goals,
    region_size, phase) -- or by (n_goals, region_size), or a single int
    applied to every cell.  The oracle signature count does not depend on the
    belief model, so the two-element keying is the usual case.
    """
    out = TaskSummary(task=task, K_ref=K_ref)
    sel = [r for r in rows if r.task == task and r.K == K_ref]
    keys = sorted({r.key for r in sel})
    for key in keys:
        grp = [r for r in sel if r.key == key]
        out.median_M[key] = statistics.median(r.M for r in grp)
        out.median_ratio[key] = statistics.median(r.ratio for r in grp)
        out.median_ess[key] = statistics.median(r.ess for r in grp)
        out.median_bound[key] = statistics.median(r.bound for r in grp)
        if isinstance(oracle_sigs, dict):
            out.oracle_sigs[key] = int(
                oracle_sigs.get(key, oracle_sigs.get(key[:2], 0))
            )
        else:
            out.oracle_sigs[key] = int(oracle_sigs)
    return out


# --------------------------------------------------------------------------- #
# The pre-registered verdict
# --------------------------------------------------------------------------- #
def verdict(summaries: Sequence[TaskSummary],
            rows: Sequence[Cell],
            cost: CostSplit,
            cost_deep: Optional[CostSplit] = None,
            prereg: Optional[Dict[str, float]] = None):
    """PASS / AMBER / STOP-*, decided purely from measured numbers.

    PASS requires a single WITNESS CELL -- one (task, goal family) at K_REF on
    which A, B and C all hold simultaneously -- plus D globally.  The witness
    must be a single cell rather than "A somewhere, B somewhere else": a filter
    whose hypotheses are diverse only under a rich goal family and which
    compresses only under a poor one has not demonstrated the regime Gate C0
    requires.  It is satisfiable by construction -- a family with
    `MIN_MODES <= prod_g|g| <= MAX_RATIO * K` meets both -- so this is a real
    constraint, not an impossible one.

    Returns (verdict_string, list_of_reason_strings).
    """
    P = dict(PREREG)
    if prereg:
        P.update(prereg)
    reasons: List[str] = []

    # ---- STOP conditions first (they can fire even if a PASS gate is met) -- #
    all_M = [s.max_M() for s in summaries if s.max_M() is not None]
    all_ess = [s.max_ess() for s in summaries if s.max_ess() is not None]
    all_best_ratio = [s.best_ratio() for s in summaries if s.best_ratio() is not None]

    if all_M and max(all_M) <= P["STOP_VACUOUS_MODES"]:
        reasons.append(
            f"VACUOUS: max median M over all tasks/goal-families = {max(all_M):g} "
            f"<= {P['STOP_VACUOUS_MODES']:g} at K={P['K_REF']:g}. The learned "
            "hypotheses are not decision-diverse; compression is trivially "
            "lossless and meaningless (Gate C1 STOP S5)."
        )
        return VERDICT_STOP_VACUOUS, reasons
    if all_ess and max(all_ess) < P["STOP_VACUOUS_ESS"]:
        reasons.append(
            f"VACUOUS: max median ESS = {max(all_ess):.2f} < {P['STOP_VACUOUS_ESS']:g}. "
            "The filter carries one hypothesis with a decorative tail "
            "(Gate C1 STOP S5)."
        )
        return VERDICT_STOP_VACUOUS, reasons
    if all_best_ratio and min(all_best_ratio) > P["STOP_NOCOMPRESS_RATIO"]:
        reasons.append(
            f"NO-COMPRESSION: best median M/K over all tasks/goal-families = "
            f"{min(all_best_ratio):.3f} > {P['STOP_NOCOMPRESS_RATIO']:g} at "
            f"K={P['K_REF']:g}. The decision structure is rich relative to the "
            "filter; M ~ K and the method buys nothing (Gate C1 STOP S1)."
        )
        return VERDICT_STOP_NOCOMPRESS, reasons
    if cost.belief_share >= P["MAX_BELIEF_SHARE"] and (
        cost_deep is None or cost_deep.belief_share >= P["MAX_BELIEF_SHARE"]
    ):
        deep = "" if cost_deep is None else f" (and {cost_deep.belief_share:.3f} at max H)"
        reasons.append(
            f"FILTER DOMINATES: C_belief share = {cost.belief_share:.3f}{deep} "
            f">= {P['MAX_BELIEF_SHARE']:g}. Saving planning compute is "
            "irrelevant end-to-end (Gate C1 STOP S3)."
        )
        return VERDICT_STOP_S3, reasons

    # ---- PASS gates, all on a single witness (task, goal family) cell ------ #
    witness = None
    for s in summaries:
        for key in s.keys():
            M, ratio = s.median_M[key], s.median_ratio[key]
            ess, osig = s.median_ess[key], s.oracle_sigs.get(key, 0)
            a_ok = (M >= P["MIN_MODES"] and ess >= P["MIN_ESS"]
                    and osig >= P["MIN_ORACLE_SIGS"])
            b_ok = ratio <= P["MAX_RATIO"]
            c_rows = [
                r for r in rows
                if r.task == s.task
                and r.key == key
                and r.K_over_bound >= P["MIN_K_OVER_BOUND"]
                and r.M <= P["BOUND_SLACK"] * r.bound
            ]
            c_ok = len(c_rows) > 0
            label = f"{s.task} |G|={key[0]}x{key[1]} phase={key[2]:g}"
            reasons.append(
                f"[{label}] A(diverse)={a_ok} "
                f"(M={M:g}, ESS={ess:.1f}, oracle_sigs={osig})  "
                f"B(compresses)={b_ok} (M/K={ratio:.4f})  "
                f"C(regime K>>bound)={c_ok} ({len(c_rows)} cells)"
            )
            if a_ok and b_ok and c_ok and witness is None:
                witness = label

    reasons.append(
        f"[cost] C_belief share = {cost.belief_share:.3f} "
        f"(threshold {P['MAX_BELIEF_SHARE']:g}) -> D(not S3)="
        f"{cost.belief_share < P['MAX_BELIEF_SHARE']}"
    )

    if witness is not None and cost.belief_share < P["MAX_BELIEF_SHARE"]:
        reasons.insert(
            0, f"PASS: witness cell = {witness}; A, B and C all hold there, "
               "D holds globally.")
        return VERDICT_PASS, reasons

    reasons.insert(
        0,
        "AMBER: no single task satisfies A, B and C together (or D failed), but "
        "no STOP condition fired either. Per the P0 design this buys ONE "
        "targeted one-week extension, not the 10-12 week Gate C1 commitment.",
    )
    return VERDICT_AMBER, reasons
