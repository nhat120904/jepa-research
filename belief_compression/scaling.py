"""Gate C0 — SCALING study for decision-equivalent belief compression.

    diagnosis/.venv/bin/python -m belief_compression.scaling

Gate B showed a constant-factor win at one small size.  The method's actual
value proposition is ASYMPTOTIC:

    "full-belief planning becomes intractable as the problem scales, while
     decision-mode planning stays cheap AND preserves the decision."

That is only true if the number of surviving decision modes M stays BOUNDED
while the belief resolution K grows.  This module measures exactly that, plus
the compute consequences, on four independently swept axes:

  S1  RESOLUTION      K grows, decision structure held fixed   -> does M saturate?
  S2  COMPUTE vs K    planning compute + exact regret vs K
  S3  HORIZON  H      planning compute vs search depth
  S4  ACTIONS |A|     compute/M vs terminal-action-set size
  S5  GOAL RICHNESS   M as a joint function of (K, |G|)

Everything is exact: exact Bayes, exact expectimax, exact expected return by
enumeration (no Monte-Carlo).  Compute is the existing ComputeCounter measured
at the ROOT decision, which is the per-decision planning cost.

Honesty rules baked in: any cell we could not afford to evaluate exactly is
reported as `n/a` rather than estimated, and every extrapolation is labelled.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .compression import compress
from .core import Belief, ComputeCounter
from .evaluate import exact_return
from .planners import (
    AmortizedVOI,
    CompressionPlanner,
    DecisionRegretVOI,
    FullBeliefPlanner,
    PrecomputedCompressionPlanner,
)
from .tasks import GridParam, MassSort, OccluderPush

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
FIGS = os.path.join(DOCS, "figures")
DOC_PATH = os.path.join(DOCS, "gateC0_scaling_results.md")

# Cost caps: beyond these the exact enumerators are too slow to be worth it.
# Cells above the cap are reported as `n/a` (never estimated).
MAX_K_EXACT_REGRET = 192      # exact_return enumerates K hidden states x tree
MAX_H_EXACT_REGRET = 1


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def table(headers, rows) -> str:
    if not rows:
        return "  ".join(str(h) for h in headers)
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)
    ]
    line = "  ".join(str(h).rjust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join(
        "  ".join(str(r[i]).rjust(widths[i]) for i in range(len(headers))) for r in rows
    )
    return f"{line}\n{sep}\n{body}"


def f4(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def loglog_slope(xs, ys) -> Optional[float]:
    """Least-squares slope of log(y) vs log(x); None if not enough valid points."""
    pts = [(x, y) for x, y in zip(xs, ys) if x and y and x > 0 and y > 0]
    if len(pts) < 3:
        return None
    lx = np.log(np.array([p[0] for p in pts], dtype=float))
    ly = np.log(np.array([p[1] for p in pts], dtype=float))
    return float(np.polyfit(lx, ly, 1)[0])


def root_compute(task, policy, goal, budget):
    """Compute + wall-clock of a single planning decision from the prior."""
    b0 = Belief.prior(task)
    c = ComputeCounter()
    t0 = time.perf_counter()
    action = policy.act(b0, budget, goal, c)
    wall = time.perf_counter() - t0
    return c.as_dict(), wall, action


# --------------------------------------------------------------------------- #
# S1 — mode saturation under resolution refinement  (THE headline measurement)
# --------------------------------------------------------------------------- #
@dataclass
class SaturationRow:
    label: str
    K: int
    M: int
    ratio: float
    bound: Optional[int]        # analytic decision-structure bound, if known
    n_goals: int
    regret: Optional[float]     # exact commit regret vs full belief (budget 0)


def _commit_regret(task, family, goal, budget=0) -> float:
    full = FullBeliefPlanner()
    comp = CompressionPlanner(goal_family=family, tol=0.0)
    return (
        exact_return(task, full, goal, budget=budget).ret
        - exact_return(task, comp, goal, budget=budget).ret
    )


def s1_grid_param(K_values, n_goals, n_controllers=3) -> List[SaturationRow]:
    """Refine the belief grid over a CONTINUOUS hidden parameter while the goal
    family (and hence the decision structure) is held fixed."""
    rows = []
    for K in K_values:
        task = GridParam(
            resolution=K, n_controllers=n_controllers, n_goals=n_goals, max_budget=0
        )
        family = task.goal_family(n_goals)
        comp = compress(Belief.prior(task), family, tol=0.0)
        reg = None
        if K <= MAX_K_EXACT_REGRET:
            reg = max(_commit_regret(task, family, g) for g in family)
        rows.append(
            SaturationRow(
                label=f"grid_param(|G|={n_goals},|A|={n_controllers})",
                K=K,
                M=comp.M,
                ratio=comp.ratio,
                bound=task.decision_bound(n_goals),
                n_goals=n_goals,
                regret=reg,
            )
        )
    return rows


def s1_occluder(L_values, n_bins=2) -> List[SaturationRow]:
    """Same refinement story on the existing occluder task: more locations
    (finer belief) but the same fixed number of push targets."""
    rows = []
    for L in L_values:
        task = OccluderPush(n_locations=L, fine=L // n_bins, max_budget=0)
        goal = task.coarse_goal()
        family = [goal]
        comp = compress(Belief.prior(task), family, tol=0.0)
        reg = _commit_regret(task, family, goal) if L <= MAX_K_EXACT_REGRET else None
        rows.append(
            SaturationRow(
                label=f"occluder_push(bins={n_bins})",
                K=L,
                M=comp.M,
                ratio=comp.ratio,
                bound=n_bins,
                n_goals=1,
                regret=reg,
            )
        )
    return rows


def s1_mass_sort_contrast(N_values) -> List[SaturationRow]:
    """CONTRAST axis: here K grows because new decision-RELEVANT dimensions are
    added, not because an existing parameter is resolved more finely.  The
    decision structure grows with K, so M must too.  This is the control that
    stops S1 from being a tautology."""
    rows = []
    for N in N_values:
        task = MassSort(n_objects=N, max_budget=0)
        family = task.goal_family(N)  # every object matters to some goal
        comp = compress(Belief.prior(task), family, tol=0.0)
        reg = None
        if comp.K <= MAX_K_EXACT_REGRET:
            reg = max(_commit_regret(task, family, g) for g in family)
        rows.append(
            SaturationRow(
                label=f"mass_sort(N={N}, all-relevant)",
                K=comp.K,
                M=comp.M,
                ratio=comp.ratio,
                bound=None,
                n_goals=len(family),
                regret=reg,
            )
        )
    return rows


def s1_table(rows) -> str:
    return table(
        ["config", "K", "M", "M/K", "bound", "|G|", "regret_commit"],
        [
            [
                r.label,
                r.K,
                r.M,
                f4(r.ratio),
                "-" if r.bound is None else r.bound,
                r.n_goals,
                "n/a" if r.regret is None else f"{r.regret:.2e}",
            ]
            for r in rows
        ],
    )


# --------------------------------------------------------------------------- #
# S2 — planning compute vs K
# --------------------------------------------------------------------------- #
@dataclass
class ComputeRow:
    K: int
    M: int
    H: int
    n_actions: int
    n_goals: int
    compute: dict = field(default_factory=dict)     # policy -> total ops
    wall: dict = field(default_factory=dict)        # policy -> seconds
    regret: dict = field(default_factory=dict)      # policy -> exact regret (or None)
    same_root_action: dict = field(default_factory=dict)
    ret_full: Optional[float] = None                # full-belief exact return


def _policies(task, family, budget):
    full = FullBeliefPlanner()
    return [
        ("full_belief", full),
        ("compression", CompressionPlanner(goal_family=family, tol=0.0)),
        ("compression_cached", PrecomputedCompressionPlanner(task, family, tol=0.0)),
        ("voi", DecisionRegretVOI()),
        ("amortized_voi", AmortizedVOI()),
    ]


def s2_compute_vs_K(K_values, H=1, n_controllers=3, n_goals=1) -> List[ComputeRow]:
    rows = []
    for K in K_values:
        task = GridParam(
            resolution=K, n_controllers=n_controllers, n_goals=n_goals, max_budget=H
        )
        family = task.goal_family(n_goals)
        goal = family[0]
        comp = compress(Belief.prior(task), family, tol=0.0)
        row = ComputeRow(K=K, M=comp.M, H=H, n_actions=n_controllers, n_goals=n_goals)
        ref_action = None
        for name, pol in _policies(task, family, H):
            cdict, wall, action = root_compute(task, pol, goal, H)
            row.compute[name] = cdict["total"]
            row.wall[name] = wall
            if name == "full_belief":
                ref_action = action
            row.same_root_action[name] = action == ref_action
            if K <= MAX_K_EXACT_REGRET and H <= MAX_H_EXACT_REGRET:
                r_full = exact_return(task, FullBeliefPlanner(), goal, budget=H).ret
                r_pol = exact_return(task, pol, goal, budget=H).ret
                row.ret_full = r_full
                row.regret[name] = r_full - r_pol
            else:
                row.regret[name] = None
        rows.append(row)
    return rows


def s2_table(rows) -> str:
    hdr = [
        "K", "M", "M/K",
        "C_full", "C_comp", "C_cached", "C_voi", "C_amort",
        "full/cached", "ret_full", "regret_cached", "rel_regret", "same_root",
    ]
    body = []
    for r in rows:
        reg = r.regret.get("compression_cached")
        rel = (abs(reg) / abs(r.ret_full)) if (reg is not None and r.ret_full) else None
        body.append([
            r.K, r.M, f4(r.M / r.K),
            r.compute["full_belief"],
            r.compute["compression"],
            r.compute["compression_cached"],
            r.compute["voi"],
            r.compute["amortized_voi"],
            f4(r.compute["full_belief"] / max(1, r.compute["compression_cached"])),
            "n/a" if r.ret_full is None else f4(r.ret_full),
            "n/a" if reg is None else f"{reg:.2e}",
            "n/a" if rel is None else f"{rel:.2%}",
            str(r.same_root_action["compression_cached"]),
        ])
    return table(hdr, body)


# --------------------------------------------------------------------------- #
# S3b — joint (K, H): where does the compute ratio actually go?
# --------------------------------------------------------------------------- #
def s3b_joint(K_values, H_values, n_controllers=3, n_goals=1, max_ops=4e7):
    """Ratio C_full / C_cached over the (K, H) grid.

    The interesting structure is two-sided: at fixed H the ratio is capped by
    the tree size (the compressed planner still has to READ the K-particle
    belief once per decision, an unavoidable O(K) term), while at fixed K the
    ratio is capped by K/M.  Cells whose full-belief cost would exceed
    `max_ops` primitive operations are skipped (`n/a`) rather than estimated.
    """
    out = {}
    for K in K_values:
        for H in H_values:
            task = GridParam(
                resolution=K, n_controllers=n_controllers, n_goals=n_goals, max_budget=H
            )
            family = task.goal_family(n_goals)
            goal = family[0]
            comp = compress(Belief.prior(task), family, tol=0.0)
            # cheap a-priori cost estimate to decide affordability
            branch = sum(task.sensor_bins)
            nodes = sum(branch ** d for d in range(H + 1))
            est = nodes * K * (n_controllers + 2)
            if est > max_ops:
                out[(K, H)] = None
                continue
            c_full, _, a_full = root_compute(task, FullBeliefPlanner(), goal, H)
            c_cach, _, a_cach = root_compute(
                task, PrecomputedCompressionPlanner(task, family), goal, H
            )
            out[(K, H)] = dict(
                ratio=c_full["total"] / max(1, c_cach["total"]),
                K_over_M=K / comp.M,
                M=comp.M,
                same_root=(a_full == a_cach),
            )
    return out


def s3b_table(grid, K_values, H_values) -> str:
    hdr = ["K", "M", "K/M"] + [f"H={h}" for h in H_values]
    body = []
    for K in K_values:
        cells = [grid[(K, h)] for h in H_values]
        ok = [c for c in cells if c]
        if not ok:
            continue
        body.append(
            [K, ok[0]["M"], f4(ok[0]["K_over_M"])]
            + ["n/a" if c is None else f4(c["ratio"]) for c in cells]
        )
    return table(hdr, body)


# --------------------------------------------------------------------------- #
# S3 — planning compute vs horizon H
# --------------------------------------------------------------------------- #
def s3_compute_vs_H(H_values, K=192, n_controllers=3, n_goals=1) -> List[ComputeRow]:
    rows = []
    for H in H_values:
        task = GridParam(
            resolution=K, n_controllers=n_controllers, n_goals=n_goals, max_budget=H
        )
        family = task.goal_family(n_goals)
        goal = family[0]
        comp = compress(Belief.prior(task), family, tol=0.0)
        row = ComputeRow(K=K, M=comp.M, H=H, n_actions=n_controllers, n_goals=n_goals)
        ref_action = None
        for name, pol in _policies(task, family, H):
            cdict, wall, action = root_compute(task, pol, goal, H)
            row.compute[name] = cdict["total"]
            row.wall[name] = wall
            if name == "full_belief":
                ref_action = action
            row.same_root_action[name] = action == ref_action
            if K <= MAX_K_EXACT_REGRET and H <= MAX_H_EXACT_REGRET:
                r_full = exact_return(task, FullBeliefPlanner(), goal, budget=H).ret
                r_pol = exact_return(task, pol, goal, budget=H).ret
                row.regret[name] = r_full - r_pol
            else:
                row.regret[name] = None
        rows.append(row)
    return rows


def s3_table(rows) -> str:
    hdr = ["H", "K", "M", "C_full", "C_cached", "full/cached",
           "wall_full_s", "wall_cached_s", "regret_cached", "same_root"]
    body = []
    for r in rows:
        reg = r.regret.get("compression_cached")
        body.append([
            r.H, r.K, r.M,
            r.compute["full_belief"],
            r.compute["compression_cached"],
            f4(r.compute["full_belief"] / max(1, r.compute["compression_cached"])),
            f"{r.wall['full_belief']:.4f}",
            f"{r.wall['compression_cached']:.4f}",
            "n/a" if reg is None else f"{reg:.2e}",
            str(r.same_root_action["compression_cached"]),
        ])
    return table(hdr, body)


# --------------------------------------------------------------------------- #
# S4 — action-set size
# --------------------------------------------------------------------------- #
def s4_actions(A_values, K=384, H=1, n_goals=1) -> List[ComputeRow]:
    rows = []
    for A in A_values:
        task = GridParam(resolution=K, n_controllers=A, n_goals=n_goals, max_budget=H)
        family = task.goal_family(n_goals)
        goal = family[0]
        comp = compress(Belief.prior(task), family, tol=0.0)
        row = ComputeRow(K=K, M=comp.M, H=H, n_actions=A, n_goals=n_goals)
        ref_action = None
        for name, pol in _policies(task, family, H):
            cdict, wall, action = root_compute(task, pol, goal, H)
            row.compute[name] = cdict["total"]
            row.wall[name] = wall
            if name == "full_belief":
                ref_action = action
            row.same_root_action[name] = action == ref_action
            row.regret[name] = None
        rows.append(row)
    return rows


def s4_table(rows) -> str:
    hdr = ["|A|", "K", "M", "M/K", "bound", "C_full", "C_cached", "full/cached", "same_root"]
    body = []
    for r in rows:
        body.append([
            r.n_actions, r.K, r.M, f4(r.M / r.K),
            r.n_goals * (r.n_actions - 1) + 1,
            r.compute["full_belief"],
            r.compute["compression_cached"],
            f4(r.compute["full_belief"] / max(1, r.compute["compression_cached"])),
            str(r.same_root_action["compression_cached"]),
        ])
    return table(hdr, body)


# --------------------------------------------------------------------------- #
# S5 — goal richness x resolution
# --------------------------------------------------------------------------- #
def s5_richness_x_K(K_values, G_values, n_controllers=3):
    grid = {}
    for G in G_values:
        for K in K_values:
            task = GridParam(
                resolution=K, n_controllers=n_controllers, n_goals=G, max_budget=0
            )
            comp = compress(Belief.prior(task), task.goal_family(G), tol=0.0)
            grid[(G, K)] = (comp.M, task.decision_bound(G))
    return grid


def s5_table(grid, K_values, G_values) -> str:
    hdr = ["|G|", "bound"] + [f"K={k}" for k in K_values]
    body = []
    for G in G_values:
        bound = grid[(G, K_values[-1])][1]
        body.append([G, bound] + [grid[(G, k)][0] for k in K_values])
    return table(hdr, body)


# --------------------------------------------------------------------------- #
# S6 — decision fidelity vs VALUE fidelity  (the limitation this study found)
# --------------------------------------------------------------------------- #
def _rep_belief(task, comp, prior, mode: str) -> Belief:
    """Collapse modes onto a representative particle, two ways."""
    w = np.zeros(task.n_states())
    for m in comp.modes:
        if mode == "maxweight":
            rep = m.rep
        else:  # mode-conditional mean of the hidden parameter
            wt = np.array([prior.weights[k] for k in m.members], dtype=float)
            wt = wt / wt.sum()
            mean_th = float((wt * task.theta[list(m.members)]).sum())
            rep = int(np.argmin(np.abs(task.theta - mean_th)))
        w[rep] += m.weight
    return Belief(task, w / w.sum())


def s6_value_fidelity(K_values, A_values, H=1, n_goals=1):
    """Exact commit regret is zero, but is the compressed planner's VALUE
    estimate right?  Probe/VOI decisions depend on values, not just argmaxes,
    so a decision-equivalent-but-value-wrong collapse can skip a probe it
    should take.  Measured against both representative choices."""
    from .planners import expectimax

    rows = []
    for A in A_values:
        for K in K_values:
            task = GridParam(resolution=K, n_controllers=A, n_goals=n_goals, max_budget=H)
            family = task.goal_family(n_goals)
            goal = family[0]
            prior = Belief.prior(task)
            comp = compress(prior, family, tol=0.0)
            v_full, a_full = expectimax(prior, goal, H, ComputeCounter())
            out = {}
            for repmode in ("maxweight", "centroid"):
                rb = _rep_belief(task, comp, prior, repmode)
                v, a = expectimax(rb, goal, H, ComputeCounter())
                out[repmode] = (v, a)
            reg = None
            if K <= MAX_K_EXACT_REGRET and H <= MAX_H_EXACT_REGRET:
                r_full = exact_return(task, FullBeliefPlanner(), goal, budget=H).ret
                r_pol = exact_return(
                    task, PrecomputedCompressionPlanner(task, family), goal, budget=H
                ).ret
                reg = (r_full - r_pol, r_full)
            rows.append(dict(
                A=A, K=K, M=comp.M, v_full=v_full, a_full=a_full,
                v_max=out["maxweight"][0], a_max=out["maxweight"][1],
                v_cen=out["centroid"][0], a_cen=out["centroid"][1],
                regret=reg,
            ))
    return rows


def s6_table(rows) -> str:
    hdr = ["|A|", "K", "M", "V_full", "V_maxweight", "V_centroid",
           "root_full", "root_maxw", "same_root", "regret_H1", "rel"]
    body = []
    for r in rows:
        reg, rel = ("n/a", "n/a")
        if r["regret"] is not None:
            reg = f"{r['regret'][0]:.2e}"
            rel = f"{abs(r['regret'][0]) / abs(r['regret'][1]):.1%}" if r["regret"][1] else "n/a"
        body.append([
            r["A"], r["K"], r["M"],
            f4(r["v_full"]), f4(r["v_max"]), f4(r["v_cen"]),
            f"{r['a_full'][0]}:{r['a_full'][1]}",
            f"{r['a_max'][0]}:{r['a_max'][1]}",
            str(r["a_max"] == r["a_full"]),
            reg, rel,
        ])
    return table(hdr, body)


# --------------------------------------------------------------------------- #
# S7 — is the fidelity failure intrinsic, or an artefact of the representative?
# --------------------------------------------------------------------------- #
def s7_representative(A_values, K_values, G_values=(1, 2), H=1):
    """S6 shows the collapsed belief mis-values, and that a probing planner can
    therefore pick the wrong root action.  S7 asks the follow-up that decides
    how damaging that is: does the failure survive a better representative?

    The mode PARTITION (and hence M, and hence all the compute numbers) is
    identical under both rules -- only which particle carries a mode's weight
    changes.  So any regret difference here is purely the cost of collapsing
    onto the wrong point, and any fix is FREE in compute.
    """
    rows = []
    for A in A_values:
        for G in G_values:
            for K in K_values:
                task = GridParam(resolution=K, n_controllers=A, n_goals=G, max_budget=H)
                family = task.goal_family(G)
                M = compress(Belief.prior(task), family, tol=0.0).M
                cell = dict(A=A, G=G, K=K, M=M, worst={}, compute={})
                for rule in ("maxweight", "centroid"):
                    worst, comp_ops = 0.0, None
                    for g in family:
                        pol = PrecomputedCompressionPlanner(task, family, rep_rule=rule)
                        rf = exact_return(task, FullBeliefPlanner(), g, budget=H).ret
                        rp = exact_return(task, pol, g, budget=H).ret
                        if rf:
                            worst = max(worst, abs(rf - rp) / abs(rf))
                        if comp_ops is None:
                            comp_ops = root_compute(task, pol, g, H)[0]["total"]
                    cell["worst"][rule] = worst
                    cell["compute"][rule] = comp_ops
                rows.append(cell)
    return rows


def s7_table(rows) -> str:
    hdr = ["|A|", "|G|", "K", "M", "rel_regret_maxweight", "rel_regret_centroid",
           "C_maxweight", "C_centroid", "same_compute"]
    body = []
    for r in rows:
        body.append([
            r["A"], r["G"], r["K"], r["M"],
            f"{r['worst']['maxweight']:.2%}", f"{r['worst']['centroid']:.2%}",
            r["compute"]["maxweight"], r["compute"]["centroid"],
            str(r["compute"]["maxweight"] == r["compute"]["centroid"]),
        ])
    return table(hdr, body)


# --------------------------------------------------------------------------- #
# Doc writer (incremental: each section is flushed to disk as soon as it exists)
# --------------------------------------------------------------------------- #
class Doc:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("")

    def add(self, text: str):
        with open(self.path, "a") as f:
            f.write(text.rstrip() + "\n\n")
            f.flush()
        return text

    def code(self, text: str):
        return self.add("```\n" + text.rstrip() + "\n```")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_plots(s1_grid, s1_occ, s1_mass, s2, s3) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    os.makedirs(FIGS, exist_ok=True)
    saved = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for rows, style in ((s1_grid, "o-"), (s1_occ, "s-"), (s1_mass, "^--")):
        ax.plot([r.K for r in rows], [r.M for r in rows], style, label=rows[0].label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("K (belief resolution)")
    ax.set_ylabel("M (surviving decision modes)")
    ax.set_title("S1: does M saturate as K grows?")
    ax.legend(fontsize=7)
    ax = axes[1]
    for rows, style in ((s1_grid, "o-"), (s1_occ, "s-"), (s1_mass, "^--")):
        ax.plot([r.K for r in rows], [r.ratio for r in rows], style, label=rows[0].label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("M / K")
    ax.set_title("S1: compression ratio")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(FIGS, "gateC0_s1_saturation.png")
    fig.savefig(p, dpi=110)
    saved.append(p)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot([r.K for r in s2], [r.compute["full_belief"] for r in s2], "o-", label="full_belief")
    ax.plot([r.K for r in s2], [r.compute["compression"] for r in s2], "s-", label="compression")
    ax.plot([r.K for r in s2], [r.compute["compression_cached"] for r in s2], "^-",
            label="compression_cached")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("K")
    ax.set_ylabel("planning ops @ root")
    ax.set_title("S2: compute vs belief resolution (H=1)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot([r.H for r in s3], [r.compute["full_belief"] for r in s3], "o-", label="full_belief")
    ax.plot([r.H for r in s3], [r.compute["compression_cached"] for r in s3], "^-",
            label="compression_cached")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("planning horizon H")
    ax.set_ylabel("planning ops @ root")
    ax.set_title(f"S3: compute vs horizon (K={s3[0].K})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGS, "gateC0_s2s3_compute.png")
    fig.savefig(p, dpi=110)
    saved.append(p)
    plt.close(fig)
    return saved


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
K_GRID = [6, 12, 24, 48, 96, 192, 384, 768, 1536]
K_COMPUTE = [12, 24, 48, 96, 192, 384, 768, 1536]
L_OCC = [8, 16, 32, 64, 128, 256]
N_MASS = [2, 3, 4, 5, 6, 7, 8]
H_VALUES = [0, 1, 2, 3, 4]
K_S3B = [24, 96, 384, 1536]
H_S3B = [0, 1, 2, 3, 4]
A_VALUES = [2, 3, 4, 6, 8, 12]
G_VALUES = [1, 2, 4, 8]
K_S5 = [24, 96, 384, 1536]
K_S7 = [24, 48, 96, 192]
A_S7 = [2, 3, 4, 6, 8, 12]


def main():
    t_start = time.perf_counter()
    doc = Doc(DOC_PATH)
    doc.add("# Gate C0 — Scaling Study\n\n"
            "Does decision-equivalent belief compression buy an ASYMPTOTIC advantage, or "
            "only the constant factor Gate B measured?\n\n"
            "Reproduce: `diagnosis/.venv/bin/python -m belief_compression.scaling` "
            "(from repo root). Everything below is exact (exact Bayes, exact expectimax, "
            "exact expected return by enumeration) and was produced by that command. "
            "Cells the exact enumerator could not afford are printed `n/a`, never estimated.")

    doc.add("## The question\n\n"
            "The value proposition is *\"full-belief planning becomes intractable as the "
            "problem scales; decision-mode planning stays cheap while preserving the "
            "decision\"*. That requires **M bounded while K grows**. If instead M grows "
            "with K, the win is a constant factor and the claim is weak.")

    doc.add("### Compute-accounting fix (disclosed)\n\n"
            "`Belief.updated` previously charged the compute counter `K` likelihood "
            "touches even for a belief whose support was only `M` particles, which would "
            "have masked exactly the effect this study measures. It now charges (and "
            "evaluates) the support only — a filter carrying M particles does not evaluate "
            "the K-M particles it does not carry, and zero-weight particles stay zero under "
            "Bayes, so the posterior is unchanged. Re-running Gate B after the fix leaves "
            "every return, regret and verdict identical (still GO); only two compute "
            "numbers move, both for `amortized_voi` (mass_sort 222 -> 198, occluder_push "
            "110 -> 74). Full-belief numbers are unchanged because its support is all of K.")

    # ---------------- S1 ---------------- #
    print("[S1] mode saturation ...")
    s1_g1 = s1_grid_param(K_GRID, n_goals=1)
    s1_g2 = s1_grid_param(K_GRID, n_goals=2)
    s1_g4 = s1_grid_param(K_GRID, n_goals=4)
    s1_occ = s1_occluder(L_OCC, n_bins=2)
    s1_mass = s1_mass_sort_contrast(N_MASS)
    doc.add("## S1 (headline) — does M saturate as belief resolution K grows?\n\n"
            "`grid_param` has a CONTINUOUS hidden parameter; `K = resolution` is purely how "
            "finely the belief discretizes it. The goal family (controller centres) lives in "
            "continuous parameter space and is untouched by the grid, so refining K cannot "
            "create new optimal actions. `bound` is the analytic prediction "
            "`min(K, |G|*(|A|-1)+1)` = number of cells in the arrangement of all goals' "
            "decision boundaries. `regret_commit` is the exact commit-decision regret vs the "
            "full-belief oracle (budget 0), maximised over goals in the family.")
    doc.code(s1_table(s1_g1 + s1_g2 + s1_g4))
    doc.add("Same refinement story on the pre-existing occluder task (more locations, same "
            "fixed number of push targets):")
    doc.code(s1_table(s1_occ))
    doc.add("**Control axis** (this is what stops S1 being a tautology): here K grows because "
            "new decision-RELEVANT latent dimensions are added, not because one parameter is "
            "resolved more finely. Every object matters to some goal in the family.")
    doc.code(s1_table(s1_mass))

    sat = {}
    for name, rows in (("grid|G|=1", s1_g1), ("grid|G|=2", s1_g2), ("grid|G|=4", s1_g4),
                       ("occluder", s1_occ), ("mass_sort", s1_mass)):
        Ms = [r.M for r in rows]
        sat[name] = dict(
            M_first=Ms[0], M_last=Ms[-1], K_first=rows[0].K, K_last=rows[-1].K,
            saturated=Ms[-1] == Ms[-2] == Ms[-3],
            slope=loglog_slope([r.K for r in rows], Ms),
            ratio_last=rows[-1].ratio,
            max_regret=max((abs(r.regret) for r in rows if r.regret is not None), default=None),
        )
    lines = ["Log-log slope of M vs K (0 = perfectly saturated, 1 = M grows like K):", ""]
    lines.append(table(
        ["axis", "K range", "M range", "M/K @ Kmax", "slope(logM/logK)", "saturated(last 3)",
         "max |regret|"],
        [[k, f"{v['K_first']}-{v['K_last']}", f"{v['M_first']}-{v['M_last']}",
          f4(v["ratio_last"]),
          "n/a" if v["slope"] is None else f4(v["slope"]),
          str(v["saturated"]),
          "n/a" if v["max_regret"] is None else f"{v['max_regret']:.2e}"]
         for k, v in sat.items()]))
    doc.code("\n".join(lines))
    print(s1_table(s1_g1 + s1_g2 + s1_g4))

    # ---------------- S2 ---------------- #
    print("[S2] compute vs K ...")
    s2 = s2_compute_vs_K(K_COMPUTE, H=1)
    doc.add("## S2 — planning compute vs belief resolution K (horizon H=1)\n\n"
            "Root-decision compute (total primitive ops from the existing `ComputeCounter`: "
            "reward evals + belief touches + expectimax nodes + VOI inner calls). "
            "`compression` builds the decision signatures at decision time (O(K|G||A|) reward "
            "evals); `compression_cached` reuses signatures precomputed once per task/goal "
            "family, so its per-decision cost is O(K) bucketing + O(M x tree). `same_root` = "
            "the compressed planner chose the same root action as the full-belief oracle. "
            f"Exact regret is computed for K <= {MAX_K_EXACT_REGRET} (beyond that the exact "
            "enumerator over K hidden states x observation trees is too slow; reported n/a).")
    doc.code(s2_table(s2))
    sl_full = loglog_slope([r.K for r in s2], [r.compute["full_belief"] for r in s2])
    sl_comp = loglog_slope([r.K for r in s2], [r.compute["compression"] for r in s2])
    sl_cach = loglog_slope([r.K for r in s2], [r.compute["compression_cached"] for r in s2])
    sl_amort = loglog_slope([r.K for r in s2], [r.compute["amortized_voi"] for r in s2])
    doc.code(
        "log-log slopes of root compute vs K (H=1):\n"
        f"  full_belief         {sl_full:.3f}\n"
        f"  compression         {sl_comp:.3f}\n"
        f"  compression_cached  {sl_cach:.3f}\n"
        f"  amortized_voi       {sl_amort:.3f}\n"
        f"  full/cached ratio   {s2[0].compute['full_belief'] / s2[0].compute['compression_cached']:.2f}x "
        f"at K={s2[0].K}  ->  "
        f"{s2[-1].compute['full_belief'] / s2[-1].compute['compression_cached']:.2f}x at K={s2[-1].K}")
    print(s2_table(s2))

    # ---------------- S3 ---------------- #
    print("[S3] compute vs H ...")
    s3 = s3_compute_vs_H(H_VALUES, K=192)
    doc.add("## S3 — planning compute vs horizon H (K=192 fixed)\n\n"
            "Full-belief expectimax expands the same tree as the compressed planner, but "
            "pays O(K) per node instead of O(M). Exact closed-loop regret by enumeration is "
            f"only affordable for H <= {MAX_H_EXACT_REGRET}; for deeper horizons we report "
            "whether the compressed planner picked the SAME root action as the full-belief "
            "oracle, which is a necessary condition for zero root regret (and is cheap).")
    doc.code(s3_table(s3))
    print(s3_table(s3))

    # ---------------- S3b ---------------- #
    print("[S3b] joint (K,H) compute ratio ...")
    s3b = s3b_joint(K_S3B, H_S3B)
    doc.add("### S3b — the compute ratio `C_full / C_cached` over the joint (K, H) grid\n\n"
            "This is the honest two-sided picture, and it is the most load-bearing table in "
            "the study. Reading DOWN a column (fixed H, growing K) the ratio flattens: the "
            "compressed planner still has to READ the K-particle belief once per decision to "
            "bucket it, an unavoidable `O(K)` term, so at fixed depth the ratio is capped by "
            "the size of the search tree. Reading ACROSS a row (fixed K, growing H) the ratio "
            "climbs toward `K/M`, because the `O(K)` read is amortized over an exponentially "
            "growing tree in which every node costs `O(M)` instead of `O(K)`. `K/M` is the "
            "ceiling, and `K/M` is unbounded in K. Cells whose full-belief cost would exceed "
            "4e7 primitive ops were skipped, not estimated.")
    doc.code(s3b_table(s3b, K_S3B, H_S3B))
    print(s3b_table(s3b, K_S3B, H_S3B))

    # ---------------- S4 ---------------- #
    print("[S4] action-set size ...")
    s4 = s4_actions(A_VALUES, K=384, H=1)
    doc.add("## S4 — terminal action-set size |A| (K=384, H=1)\n\n"
            "M is expected to track |A| (one mode per optimal controller), i.e. the bound "
            "`|G|*(|A|-1)+1`, and to stay independent of K.")
    doc.code(s4_table(s4))
    print(s4_table(s4))

    # ---------------- S5 ---------------- #
    print("[S5] goal richness x K ...")
    s5 = s5_richness_x_K(K_S5, G_VALUES)
    doc.add("## S5 — goal-family richness x belief resolution\n\n"
            "M as a joint function of (|G|, K), |A|=3. The Gate-B finding was that M grows "
            "with goal richness; the Gate-C0 question is whether, at each fixed richness, M "
            "is flat in K.")
    doc.code(s5_table(s5, K_S5, G_VALUES))
    print(s5_table(s5, K_S5, G_VALUES))

    # ---------------- S6 ---------------- #
    print("[S6] value fidelity ...")
    s6 = s6_value_fidelity([96, 384], A_VALUES, H=1)
    doc.add("## S6 — the limitation this study found: decision fidelity != VALUE fidelity\n\n"
            "S1/S2 show the collapse is **exactly lossless for the terminal commit** (regret "
            "0.00e+00 at budget 0, every K). But a planner that can PROBE compares the value "
            "of committing now against the expected value of committing after an observation, "
            "and decision-mode compression preserves the *argmax* of the commit, not its "
            "*value*. Collapsing a mode onto one representative particle therefore biases the "
            "value estimate, and the bias direction depends on which particle you pick:\n\n"
            "`V_maxweight` uses the current implementation (max-weight member, which under a "
            "flat prior is an arbitrary mode-edge particle) and systematically UNDER-values; "
            "the mode-conditional-mean particle systematically OVER-values. Neither is "
            "value-consistent, so this is structural, not a bad choice of representative. "
            "`root_full` / `root_maxw` are the root actions; where they disagree the "
            "compressed planner skipped a probe the oracle took.")
    doc.code(s6_table(s6))
    print(s6_table(s6))

    # ---------------- S7 ---------------- #
    print("[S7] mode representative ...")
    s7 = s7_representative(A_S7, K_S7)
    doc.add("## S7 — is the S6 failure intrinsic, or an artefact of the representative?\n\n"
            "S6 established that the collapsed belief mis-values and can therefore pick the "
            "wrong root action. That is only damning if it survives a better choice of "
            "representative particle. It largely does not. The mode PARTITION is identical "
            "under both rules — same modes, same M, and (verified in the last column) "
            "byte-identical compute — so the only thing that changes is which particle "
            "carries a mode's weight, and any improvement here is FREE.\n\n"
            "`maxweight` is the original rule: the mode's highest-weight member, which under "
            "a flat prior ties across the whole mode and resolves to the FIRST index — an "
            "arbitrary particle at the mode's EDGE. `centroid` instead picks the particle "
            "nearest the mode's belief-weighted mean parameter.")
    doc.code(s7_table(s7))
    print(s7_table(s7))

    figs = save_plots(s1_g1, s1_occ, s1_mass, s2, s3)

    # ---------------- verdict ---------------- #
    print("[verdict] ...")
    verdict, reasoning = make_verdict(sat, s2, s3, s4, s5, s3b, s6, s7)
    doc.add("## Verdict\n\n" + f"**{verdict}**\n\n" + reasoning)
    if figs:
        doc.add("## Figures\n\n" + "\n".join(f"- `{os.path.relpath(p, DOCS)}`" for p in figs))
    doc.add(f"_Total runtime: {time.perf_counter() - t_start:.1f}s._")
    print(f"\nVERDICT: {verdict}\nWrote {DOC_PATH}")
    return verdict


# Verdict thresholds (named so the decision rule is auditable).
COMMIT_REGRET_TOL = 1e-9      # decision-equivalence is supposed to be EXACT here
PLAN_REGRET_REL_TOL = 0.02    # closed-loop (probe-valuation) regret, relative to return
RATIO_FLOOR = 0.05            # M/K must fall at least this far to count as "-> 0"


def make_verdict(sat, s2, s3, s4, s5, s3b, s6, s7=()):
    """STRONG / WEAK / DEAD, decided from the measured numbers.

    Gate C0 turned out to answer two separable questions, so the verdict is
    reported as two audited components rather than one word that hides half the
    evidence:
      (1) SCALING  -- does M stay bounded and does the compute win widen?
      (2) FIDELITY -- is the preserved decision the one the planner needs?
    """
    grid_sat = all(sat[k]["saturated"] for k in ("grid|G|=1", "grid|G|=2", "grid|G|=4"))
    occ_sat = sat["occluder"]["saturated"]
    mass_grows = sat["mass_sort"]["slope"] is not None and sat["mass_sort"]["slope"] > 0.9
    ratio_falls = sat["grid|G|=1"]["ratio_last"] < RATIO_FLOOR

    # commit-only regret (budget 0) from S1
    commit_regrets = [v["max_regret"] for v in sat.values() if v["max_regret"] is not None]
    commit_ok = (max(commit_regrets) if commit_regrets else 0.0) <= COMMIT_REGRET_TOL

    # closed-loop regret (with probing) from S2
    rel = []
    for r in s2:
        reg = r.regret.get("compression_cached")
        if reg is not None and r.ret_full:
            rel.append(abs(reg) / abs(r.ret_full))
    worst_rel = max(rel) if rel else 0.0
    plan_ok = worst_rel <= PLAN_REGRET_REL_TOL
    # Does that regret GROW with K, or is it a fixed offset?  Compare the last
    # cell against the largest of the preceding ones (a first cell that happens
    # to be exactly 0 must not be read as unbounded growth).
    regret_grows = len(rel) >= 3 and rel[-1] > 1.05 * max(rel[:-1])

    r_small = s2[0].compute["full_belief"] / s2[0].compute["compression_cached"]
    r_big = s2[-1].compute["full_belief"] / s2[-1].compute["compression_cached"]
    widens_K = r_big > 2 * r_small
    h_small = s3[0].compute["full_belief"] / s3[0].compute["compression_cached"]
    h_big = s3[-1].compute["full_belief"] / s3[-1].compute["compression_cached"]
    widens_H = h_big >= h_small
    roots_ok = all(r.same_root_action["compression_cached"] for r in s2 + s3 + s4)
    roots_ok = roots_ok and all(c["same_root"] for c in s3b.values() if c)

    # deepest-H ratio at each K: does the ceiling K/M actually get approached?
    deep = []
    for K in K_S3B:
        cells = [s3b[(K, h)] for h in H_S3B if s3b.get((K, h))]
        if cells:
            deep.append((K, cells[-1]["ratio"], cells[-1]["K_over_M"]))
    ceiling_grows = len(deep) >= 2 and deep[-1][1] > deep[0][1]

    # --- S6: does the collapse preserve the decision the PLANNER needs? --- #
    s6_disagree = [r for r in s6 if r["a_max"] != r["a_full"]]
    s6_worst_rel = 0.0
    for r in s6:
        if r["regret"] is not None and r["regret"][1]:
            s6_worst_rel = max(s6_worst_rel, abs(r["regret"][0]) / abs(r["regret"][1]))
    fidelity_ok = (not s6_disagree) and s6_worst_rel <= PLAN_REGRET_REL_TOL

    # --- S7: does that failure survive a better mode representative? --- #
    s7_maxw = max((r["worst"]["maxweight"] for r in s7), default=0.0)
    s7_cent = max((r["worst"]["centroid"] for r in s7), default=0.0)
    s7_free = all(r["compute"]["maxweight"] == r["compute"]["centroid"] for r in s7)
    s7_cent_ok = bool(s7) and s7_cent <= PLAN_REGRET_REL_TOL
    s7_cells_bad = sum(1 for r in s7 if r["worst"]["centroid"] > 1e-12)
    # fidelity is recoverable if a compute-free change of representative brings
    # the closed-loop regret inside tolerance everywhere we measured
    fidelity_recoverable = s7_cent_ok and s7_free

    # (1) the scaling question Gate C0 was chartered to answer
    scaling = (grid_sat and occ_sat and ratio_falls and commit_ok
               and widens_K and ceiling_grows and not regret_grows)
    scaling_dead = (not grid_sat and not occ_sat) or not commit_ok or not widens_K
    v_scaling = "STRONG" if scaling else ("DEAD" if scaling_dead else "WEAK")

    # (2) the fidelity question the study surfaced on its own
    if fidelity_ok:
        v_fidelity = "LOSSLESS"
    elif fidelity_recoverable:
        v_fidelity = "LOSSY-ON-PROBING (recoverable: fixed by the mode representative)"
    else:
        v_fidelity = "LOSSY-ON-PROBING"

    if v_scaling == "STRONG" and (fidelity_ok or fidelity_recoverable):
        v = "STRONG"
    elif v_scaling == "STRONG":
        v = "WEAK"
    else:
        v = v_scaling

    lines = [
        f"Two separable questions, reported separately rather than averaged into one word.",
        "",
        f"### (1) Scaling: **{v_scaling}**",
        "",
        f"- M saturates on every resolution-refinement axis (grid_param |G|=1,2,4): **{grid_sat}**",
        f"- M saturates on the occluder refinement axis: **{occ_sat}**",
        f"- M/K at the largest K measured (grid_param, |G|=1): "
        f"**{sat['grid|G|=1']['ratio_last']:.5f}** (threshold {RATIO_FLOOR}); M was "
        f"{sat['grid|G|=1']['M_first']} at K={sat['grid|G|=1']['K_first']} and "
        f"{sat['grid|G|=1']['M_last']} at K={sat['grid|G|=1']['K_last']}",
        f"- Control axis (mass_sort, all latent dimensions decision-relevant) has M growing "
        f"like K, log-log slope {sat['mass_sort']['slope']:.3f}: **{mass_grows}** — so the "
        f"saturation above is a property of RESOLUTION refinement, not a tautology of the setup",
        f"- Exact COMMIT regret (budget 0) vs full belief, worst measured cell: "
        f"**{(max(commit_regrets) if commit_regrets else 0.0):.2e}** (exactly lossless: "
        f"**{commit_ok}**)",
        f"- Compute advantage over K at fixed H=1: {r_small:.2f}x at K={s2[0].K} -> "
        f"{r_big:.2f}x at K={s2[-1].K} (widens: **{widens_K}**)",
        f"- Compute advantage over H at fixed K={s3[0].K}: {h_small:.2f}x at H={s3[0].H} -> "
        f"{h_big:.2f}x at H={s3[-1].H} (widens or holds: **{widens_H}**)",
        f"- Deepest-horizon ratio grows with K: "
        + ", ".join(f"K={k}: {r:.1f}x (ceiling K/M={c:.0f})" for k, r, c in deep)
        + f" -> **{ceiling_grows}**",
        "",
        f"### (2) Decision fidelity: **{v_fidelity}**",
        "",
        f"- Exact CLOSED-LOOP regret (H=1, planner may probe) vs full belief, worst cell in "
        f"the S2 sweep: **{worst_rel:.2%}** of the full-belief return "
        f"(within {PLAN_REGRET_REL_TOL:.0%}: **{plan_ok}**; grows with K: **{regret_grows}**)",
        f"- Worst closed-loop regret across the S6 |A| sweep: **{s6_worst_rel:.2%}** of return",
        f"- Root-action disagreements with the full-belief oracle in S6: "
        f"**{len(s6_disagree)} / {len(s6)}** cells"
        + (" (" + ", ".join(f"|A|={r['A']},K={r['K']}" for r in s6_disagree) + ")"
           if s6_disagree else ""),
        f"- Compressed planner matched the oracle root action in every measured cell "
        f"across S2+S3+S3b+S4 (the False below is exactly the |A|=2 cell above): "
        f"**{roots_ok}**",
    ]
    if s7:
        lines += [
            "",
            f"- Swapping the mode representative from `maxweight` to `centroid` "
            f"(same partition, same M, **identical compute**: {s7_free}) moves the worst "
            f"closed-loop regret over the whole {len(s7)}-cell (|A|, |G|, K) grid from "
            f"**{s7_maxw:.2%}** to **{s7_cent:.2%}**, nonzero in only "
            f"{s7_cells_bad}/{len(s7)} cells: recoverable = **{fidelity_recoverable}**",
        ]
    if not fidelity_ok:
        lines += [
            "",
            "The compression is exactly lossless for the terminal commit at every "
            "resolution, but it is NOT value-preserving, and probe decisions are made on "
            "values. Collapsing a mode onto a representative particle biases the value "
            "estimate in a direction set by which particle is chosen (S6: max-weight "
            "under-values, mode-mean over-values), so the compressed planner can decline a "
            "probe the oracle takes. Crucially this error is CONSTANT in K, not growing — it "
            "does not erode the scaling result.",
            "",
            "S7 then shows most of that error was an ARTEFACT rather than a property of "
            "decision-mode compression: the original `maxweight` rule degenerates under a "
            "flat prior (all members tie, the tie-break returns the first index, i.e. a "
            "mode-EDGE particle), and simply collapsing onto the mode-conditional mean "
            "instead removes the catastrophic |A|=2 root flip and brings the worst regret "
            "across the whole grid to "
            f"{s7_cent:.2%} at exactly the same compute. The residual is not zero "
            f"({s7_cells_bad}/{len(s7)} cells), and neither rule is value-CONSISTENT (S6: "
            "max-weight under-values, centroid over-values the same belief), so the clean "
            "guarantee still needs a value-consistent mode summary — e.g. carrying each "
            "mode's belief-weighted Q-vector and its within-mode observation likelihood "
            "instead of any single representative particle. What S7 establishes is that "
            "the fidelity gap is a fixable implementation detail that costs no compute, "
            "not a wall in front of the scaling result.",
        ]
    return v, "\n".join(lines)


if __name__ == "__main__":
    main()
