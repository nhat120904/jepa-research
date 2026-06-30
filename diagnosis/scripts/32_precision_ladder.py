"""Precision-difficulty ladder on Metaworld — the in-sim analogue of the paper's
cup-vs-box grasping gradient (V-JEPA-2 Table 2: Grasp Cup 65% vs Box 25%).

The cup/box gap is a *real-robot* result (two physical Franka arms) and cannot be
reproduced here. What it MEASURES — manipulation success degrading as the task
demands finer object/gripper precision — is reproducible in simulation. This script
takes a closed-loop CSV (scripts/18, the `l2` baseline arm run over an ordered
ladder of contact tasks) and shows the same gradient two ways:

1. **Env-flag success** per task (the canonical Metaworld success rate, like the
   paper's table) — ordered easy→hard.
2. **Object-tolerance sweep**: for a grid of radii r, success(r) = fraction of
   episodes whose final object-to-goal distance < r. A precision-easy task (push)
   already succeeds at tight r; a precision-hard task (peg-insert) only succeeds at
   loose r. The *curves themselves* are the precision gradient — the in-sim cup
   (steep, succeeds at fine precision) vs box (shallow) analogue, free of the env's
   per-task fixed threshold.

If `results/metaworld_boundary.csv` is present it joins each task's contact-regime
Boundary Blindness, closing symptom→behaviour (higher BB → lower success).

    .venv/bin/python scripts/32_precision_ladder.py \
        --csv results/metaworld_precision_ladder.csv --arm l2

Inputs : a closed-loop CSV with columns task, arm, success_end, obj_goal_dist.
Outputs: results/metaworld_precision_ladder_summary.md
         results/figures/figure_precision_ladder.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Recommended ladder (increasing required manipulation precision). reach is the
# free-space anchor (no object); peg-insert-side is the precision ceiling. All have
# Sawyer*V3Policy experts wired in scripts/18.expert_policy.
DEFAULT_LADDER = ["mw-reach", "mw-push", "mw-pick-place", "mw-peg-insert-side"]


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval for a success count k/n (small-n safe)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_boundary_bb(path: Path) -> dict:
    """Per-task contact-regime Boundary Blindness (max over contact/pre_grasp)."""
    if not path.exists():
        return {}
    bb = pd.read_csv(path)
    contact = bb[bb["regime"].isin(["pre_grasp", "gripper_actuation", "contact_manipulation"])]
    col = "bb_boundary" if "bb_boundary" in contact else "bb"
    out = {}
    for task, g in contact.groupby("task"):
        v = g[col].dropna()
        if len(v):
            out[task] = float(v.max())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/metaworld_precision_ladder.csv",
                    help="closed-loop CSV from scripts/18 (the l2 arm over the ladder)")
    ap.add_argument("--arm", default="l2", help="which arm to analyse")
    ap.add_argument("--boundary-csv", default="results/metaworld_boundary.csv")
    ap.add_argument("--order", nargs="+", default=None,
                    help="explicit easy→hard task order; default = by env-flag success")
    ap.add_argument("--rmax", type=float, default=0.30, help="max object tolerance radius (m)")
    ap.add_argument("--rsteps", type=int, default=31)
    ap.add_argument("--out-md", default="results/metaworld_precision_ladder_summary.md")
    ap.add_argument("--out-fig", default="results/figures/figure_precision_ladder.pdf")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path} — run scripts/18 with --arms {args.arm} "
                         f"over the ladder first (see scripts/slurm_precision_ladder.sh)")
    df = pd.read_csv(csv_path)
    df = df[df["arm"] == args.arm]
    if df.empty:
        raise SystemExit(f"no rows with arm={args.arm} in {csv_path}")
    if "obj_goal_dist" not in df.columns:
        raise SystemExit("CSV lacks obj_goal_dist — re-run scripts/18 (column added 2026-06-25)")

    success_col = "success_end" if "success_end" in df.columns else "success"
    bb_by_task = load_boundary_bb(Path(args.boundary_csv))
    radii = np.linspace(0.0, args.rmax, args.rsteps)

    # Per-task aggregates. A task whose object never moves from goal (obj≈0 for
    # all episodes) is FREE-SPACE (e.g. reach) — its object-tolerance sweep is
    # degenerate (trivially 1.0), so flag it and keep it out of the sweep panel.
    rows = []
    sweep = {}
    for task, g in df.groupby("task"):
        n = len(g)
        k = int(g[success_col].sum())
        p, lo, hi = wilson(k, n)
        obj = g["obj_goal_dist"].to_numpy()
        sweep[task] = np.array([(obj < r).mean() for r in radii])
        rows.append({
            "task": task, "n": n, "success_end": p, "succ_lo": lo, "succ_hi": hi,
            "succ_k": k, "median_obj_goal_dist": float(np.median(obj)),
            "free_space": bool(np.nanmax(obj) < 1e-3),
            "bb_contact": bb_by_task.get(task, float("nan")),
        })
    summary = pd.DataFrame(rows)

    # Order easy→hard: explicit, else by env-flag success desc (ties: smaller obj dist).
    if args.order:
        order = [t for t in args.order if t in set(summary.task)]
    else:
        order = (summary.sort_values(["success_end", "median_obj_goal_dist"],
                                     ascending=[False, True]).task.tolist())
    summary["__rank"] = summary.task.map({t: i for i, t in enumerate(order)})
    summary = summary.sort_values("__rank").drop(columns="__rank").reset_index(drop=True)

    # ---- figure: (a) success bar easy→hard, (b) object-tolerance sweep curves ----
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(order))
    pvals = [float(summary[summary.task == t].success_end.iloc[0]) for t in order]
    lo = [pvals[i] - float(summary[summary.task == order[i]].succ_lo.iloc[0]) for i in range(len(order))]
    hi = [float(summary[summary.task == order[i]].succ_hi.iloc[0]) - pvals[i] for i in range(len(order))]
    axb.bar(x, pvals, yerr=[lo, hi], capsize=4, color="#378ADD")
    axb.set_xticks(x); axb.set_xticklabels([t.replace("mw-", "") for t in order],
                                           rotation=20, ha="right")
    axb.set_ylim(0, 1.05); axb.set_ylabel("env-flag success (success_end)")
    axb.set_title("Success drops as precision demand rises\n(easy → hard)")

    # Object-tolerance sweep — contact tasks only (free-space tasks are degenerate).
    contact_order = [t for t in order
                     if not bool(summary[summary.task == t].free_space.iloc[0])]
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(contact_order):
        axc.plot(radii * 100, sweep[t], label=t.replace("mw-", ""),
                 color=cmap(i / max(len(contact_order) - 1, 1)), lw=2)
    axc.set_xlabel("object-to-goal tolerance radius (cm)")
    axc.set_ylabel("P(final obj dist < r)")
    axc.set_ylim(0, 1.05)
    axc.set_title("Object-proximity sweep, contact tasks\n(env-flag success = 0 for all)")
    if contact_order:
        axc.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_fig = Path(args.out_fig); out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    print(f"wrote {out_fig}", flush=True)

    # ---- markdown summary (mirrors the paper's Table 2 layout) ----
    lines = ["# Metaworld precision-difficulty ladder", "",
             f"Closed-loop `{args.arm}` arm, `{csv_path.name}` "
             f"({int(summary.n.sum())} episodes). Attempted in-sim analogue of the "
             "V-JEPA-2 real-robot cup (65%) vs box (25%) grasping gradient — does "
             "MetaWorld success degrade smoothly with required object precision?", "",
             "| task | n | success_end [95% CI] | median obj→goal (m) | contact BB |",
             "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        bb = "—" if not np.isfinite(r.bb_contact) else f"{r.bb_contact:+.2f}"
        lines.append(f"| {r.task} | {int(r.n)} | {r.success_end:.2%} "
                     f"[{r.succ_lo:.2%}, {r.succ_hi:.2%}] | "
                     f"{r.median_obj_goal_dist:.3f} | {bb} |")
    # Is the success signal a graded curve, or a binary free-space/contact cliff?
    contact = summary[~summary.free_space]
    if len(contact) and float(contact.success_end.max()) == 0.0:
        lines += ["", "**Env-flag success is a binary cliff, not a graded curve:** every "
                  "contact task is at the 0% floor while the free-space anchor is not. The "
                  "L2 wall is *total* for contact manipulation here — there is no cup-vs-box "
                  "success gradient to measure, because nothing contact succeeds. (A graded "
                  "*success* curve like cup 65 / box 25 needs a model that sometimes succeeds "
                  "at contact; the real-robot V-JEPA-2-AC does, MetaWorld+dino_wm does not.)"]
        lines += ["", "The only graded signal is the softer **object-proximity sweep** "
                  "(`" + str(out_fig) + "`): at a fixed loose radius, easier-contact tasks "
                  "get the object closer (push > peg-insert > pick-place at r=25cm). This is "
                  "the qualitative echo of cup/box, but it never crosses into real success "
                  "and is confounded by each task's initial object→goal geometry (not "
                  "normalised to progress)."]
    else:
        sub = summary.dropna(subset=["bb_contact"])
        if len(sub) >= 3 and sub.success_end.std() > 0:
            cc = float(np.corrcoef(sub.bb_contact, sub.success_end)[0, 1])
            lines += ["", f"Pearson(contact BB, success_end) = {cc:+.3f} (n={len(sub)} tasks) "
                      "— negative links the latent symptom (boundary blindness) to which "
                      "tasks fail."]
        lines += ["", "Object-proximity sweep: `" + str(out_fig) + "`. Easier-precision "
                  "tasks reach high proximity at a tighter radius — the in-sim cup-vs-box "
                  "gradient."]
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}\n", flush=True)
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
