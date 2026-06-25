"""Figure 3 (C2): BB -> planning outcome -> contact failure signature.

The planning claim is regime-level, not a dense pointwise correlation. The
figure therefore shows the full inference chain:

(a) the diagnostic signal: BB is low in free space and high at the pre-grasp
    boundary on both Metaworld model families;
(b) the task outcome: reach, the low-BB/free-space task, reproduces the
    published baseline while contact-boundary tasks fail;
(c) the failure signature and partial repair: on contact tasks the end-effector
    arrives but the object/state distance remains large; the grounded cost
    improves final state distance without flipping success.

All numbers are read live from the result CSVs. No GPU/data needed.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXCLUDE = ("mw-door-close",)
REGIMES = ["free_space", "pre_grasp", "contact_manipulation"]
REG_LBL = ["free\nspace", "pre-grasp\nboundary", "contact\nmanip."]
TASKS = ["mw-reach", "mw-push", "mw-pick-place"]
TASK_LBL = ["reach", "push", "pick-\nplace"]
ARMS = [("l2", "L2", "tab:blue"), ("hdyn", "grounded", "tab:red")]


def pool_regime(bb, model_key, regime):
    g = bb[bb.model.str.contains(model_key) & (bb.regime == regime)
           & ~bb.task.isin(EXCLUDE)]
    g = g[np.isfinite(g.bb_boundary) & (g.n_boundary > 0)]
    return float(np.average(g.bb_boundary, weights=g.n_boundary)) if len(g) else np.nan


bb = pd.read_csv("results/metaworld_boundary.csv")
cl = pd.read_csv("results/metaworld_closed_loop.csv")
# Reach: strict episode-end re-score (D.2). success_end is the upstream
# final-state convention; the any-step `success` latch in closed_loop.csv is
# retracted as the headline (it gave the inflated 94%/100%).
rs = pd.read_csv("results/metaworld_reach_strict.csv")


def succ_pct(task, arm):
    """Closed-loop success % for (task, arm): strict episode-end for reach,
    any-step (== strict, both 0) for the contact tasks."""
    if task == "mw-reach":
        g = rs[(rs.task == task) & (rs.arm == arm)]
        return float(g.success_end.mean() * 100) if len(g) else np.nan
    g = cl[(cl.task == task) & (cl.arm == arm)]
    return float(g.success.mean() * 100) if len(g) else np.nan


def contact_delta_ci():
    rng = np.random.default_rng(0)
    deltas = []
    for task in ["mw-push", "mw-pick-place"]:
        p = cl[cl.task == task].pivot_table(
            index="seed", columns="arm", values="final_state_dist"
        )
        deltas.extend((p.l2 - p.hdyn).dropna().values)
    deltas = np.asarray(deltas)
    boot = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def contact_means(arm):
    g = cl[(cl.task.isin(["mw-push", "mw-pick-place"])) & (cl.arm == arm)]
    return float(g.final_state_dist.mean()), float(g.ee_dist.mean())


fig, axes = plt.subplots(
    1, 3, figsize=(12.5, 3.9), width_ratios=[1.1, 1.0, 1.05]
)
fig.patch.set_facecolor("white")

# (a) regime-pooled BB, both model families
ax = axes[0]
x = np.arange(len(REGIMES))
ax.axvspan(-0.5, 0.5, color="tab:green", alpha=0.08, zorder=0)
ax.axvspan(0.5, 1.5, color="tab:red", alpha=0.07, zorder=0)
for i, (m, lbl, color) in enumerate(
    [("dino", "DINO-WM", "#4C78A8"), ("jepa", "JEPA-WM", "#F58518")]
):
    y = [pool_regime(bb, m, r) for r in REGIMES]
    bars = ax.bar(x + (i - 0.5) * 0.38, y, 0.36, label=lbl, color=color)
    for b, val in zip(bars, y):
        ax.text(
            b.get_x() + b.get_width() / 2,
            val + 0.04,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
ax.set_xticks(x, REG_LBL)
ax.set_ylabel("Boundary Blindness (pooled)")
ax.set_ylim(0, 1.55)
ax.set_title("(a) Diagnostic: BB by regime")
ax.legend(frameon=False, fontsize=9)
ax.text(0, 1.43, "low-BB\nfree space", ha="center", fontsize=7.5, color="tab:green")
ax.text(1, 1.43, "high-BB\nboundary", ha="center", fontsize=7.5, color="tab:red")

# (b) closed-loop success by task, grouped by the regime they must traverse
ax = axes[1]
xt = np.arange(len(TASKS))
ax.axvspan(-0.5, 0.5, color="tab:green", alpha=0.08, zorder=0)
ax.axvspan(0.5, 2.5, color="tab:red", alpha=0.06, zorder=0)
for i, (arm, lbl, color) in enumerate(ARMS):
    s = [succ_pct(t, arm) for t in TASKS]
    offset = (i - 0.5) * 0.32
    bars = ax.bar(xt + offset, s, 0.30, color=color, label=lbl)
    for b, val in zip(bars, s):
        ax.text(
            b.get_x() + b.get_width() / 2,
            val + 3,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
ax.hlines(
    [35.9, 53.7],
    xmin=-0.47,
    xmax=0.47,
    colors="0.35",
    linestyles=":",
    linewidth=1,
)
ax.text(0, 58, "published\nReach CI", ha="center", fontsize=6.5, color="0.3")
ax.text(1.5, 88, "must cross\ncontact boundary", ha="center", fontsize=7.5,
        color="tab:red")
ax.set_xticks(xt, TASK_LBL)
ax.set_ylim(-5, 108)
ax.set_ylabel("closed-loop success (%)")
ax.set_title("(b) Outcome: success by task")
ax.legend(frameon=False, fontsize=9, loc="upper right")

# (c) contact failure signature plus the grounded partial repair
ax = axes[2]
state_means = []
ee_means = []
for arm, _, _ in ARMS:
    state, ee = contact_means(arm)
    state_means.append(state)
    ee_means.append(ee)

groups = np.arange(2)
w = 0.32
state_bars = ax.bar(
    groups - w / 2,
    state_means,
    w,
    label="state/object dist.",
    color="#B279A2",
)
ee_bars = ax.bar(
    groups + w / 2,
    ee_means,
    w,
    label="ee dist.",
    color="#59A14F",
)
for bars in [state_bars, ee_bars]:
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.018,
            f"{b.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

delta, lo, hi = contact_delta_ci()
ax.annotate(
    "",
    xy=(1 - w / 2, state_means[1]),
    xytext=(0 - w / 2, state_means[0]),
    arrowprops=dict(arrowstyle="->", lw=1.2, color="tab:red"),
)
ax.text(
    0.5,
    max(state_means) + 0.06,
    f"grounded state gain\n+{delta:.2f} [{lo:.2f},{hi:.2f}]",
    ha="center",
    va="bottom",
    fontsize=7.2,
    color="tab:red",
)
ax.text(
    0.5,
    0.12,
    "arm arrives\nobject does not",
    ha="center",
    fontsize=7.5,
    color="0.25",
)
ax.set_xticks(groups, ["L2\n0/32", "grounded\n0/32"])
ax.set_ylabel("final distance (contact tasks)")
ax.set_ylim(0, max(state_means) + 0.16)
ax.set_title("(c) Signature: score improves, no success")
ax.legend(frameon=False, fontsize=8, loc="upper right")

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/figures/figure_bb_vs_planning.pdf")
print("wrote results/figures/figure_bb_vs_planning.pdf")
print("regime BB dino:", {r: round(pool_regime(bb, 'dino', r), 3) for r in REGIMES})
print("regime BB jepa:", {r: round(pool_regime(bb, 'jepa', r), 3) for r in REGIMES})
print("reach success (strict): l2 %.1f%%  hdyn %.1f%%"
      % (succ_pct("mw-reach", "l2"), succ_pct("mw-reach", "hdyn")))
print(f"contact paired delta +{delta:.3f} [{lo:.3f}, {hi:.3f}]")
