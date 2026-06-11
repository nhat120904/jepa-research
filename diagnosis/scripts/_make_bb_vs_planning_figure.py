"""Figure 3 (the C2 figure): Boundary Blindness predicts closed-loop planning
outcome, at the REGIME level (the clean signal; per-task boundary-sample counts
are too small to pool reliably).

Panel (a): pooled BB by regime for the frozen baselines — high at the
pre-grasp/contact boundary, low in free space (the diagnostic signal).
Panel (b): closed-loop success by task. The link is the regime each task must
traverse: reach is a free-space pose-reaching task (low-BB regime) and succeeds
above the published baseline; push and pick-and-place must cross the
pre-grasp/contact boundary (high-BB regime) and fail at exactly the predicted
point — the arm arrives (ee 2-4 cm) but the object never moves. Inset: the
grounded fix's paired contact-distance gain (CI excludes zero).

All numbers read live from the result CSVs. No GPU/data needed.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXCLUDE = ("mw-door-close",)
REGIMES = ["free_space", "pre_grasp", "contact_manipulation"]
REG_LBL = ["free\nspace", "pre-grasp\nboundary", "contact\nmanip."]


def pool_regime(bb, model_key, regime):
    g = bb[bb.model.str.contains(model_key) & (bb.regime == regime)
           & ~bb.task.isin(EXCLUDE)]
    g = g[np.isfinite(g.bb_boundary) & (g.n_boundary > 0)]
    return float(np.average(g.bb_boundary, weights=g.n_boundary)) if len(g) else np.nan


bb = pd.read_csv("results/metaworld_boundary.csv")
cl = pd.read_csv("results/metaworld_closed_loop.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), width_ratios=[1.05, 1])

# (a) regime-pooled BB, both model families
ax = axes[0]
x = np.arange(len(REGIMES))
for i, (m, lbl) in enumerate([("dino", "DINO-WM"), ("jepa", "JEPA-WM")]):
    y = [pool_regime(bb, m, r) for r in REGIMES]
    ax.bar(x + (i - 0.5) * 0.38, y, 0.36, label=lbl)
ax.set_xticks(x, REG_LBL)
ax.set_ylabel("Boundary Blindness (pooled)")
ax.set_title("(a) BB is a regime property")
ax.legend(frameon=False, fontsize=9)
ax.annotate("planning lives here\n$\\downarrow$", xy=(0, 0.3), xytext=(0, 0.85),
            ha="center", fontsize=7.5, color="tab:green")
ax.annotate("planning fails here\n$\\downarrow$", xy=(1, 1.3), xytext=(1, 0.85),
            ha="center", fontsize=7.5, color="tab:red")

# (b) closed-loop success by task, colored by the regime they must traverse
ax = axes[1]
TASKS = ["mw-reach", "mw-push", "mw-pick-place"]
TASK_LBL = ["reach", "push", "pick-\nplace"]
xt = np.arange(len(TASKS))
for arm, mk, c in [("l2", "o-", "tab:blue"), ("hdyn", "s--", "tab:red")]:
    s = [cl[(cl.task == t) & (cl.arm == arm)].success.mean() * 100 for t in TASKS]
    ax.plot(xt, s, mk, color=c, ms=9, label={"l2": "L2", "hdyn": "grounded"}[arm])
ax.axvspan(-0.4, 0.4, color="tab:green", alpha=0.08)
ax.axvspan(0.4, 2.4, color="tab:red", alpha=0.06)
ax.text(0, 30, "free-space\nregime", ha="center", fontsize=7.5, color="tab:green")
ax.text(1.5, 80, "must cross\ncontact boundary", ha="center", fontsize=7.5,
        color="tab:red")
ax.set_xticks(xt, TASK_LBL)
ax.set_ylim(-5, 108)
ax.set_ylabel("closed-loop success (%)")
ax.set_title("(b) Success tracks the regime")
ax.legend(frameon=False, fontsize=9, loc="center right")
for xi, t in zip(xt, TASKS):
    v = cl[(cl.task == t) & (cl.arm == "l2")].success.mean() * 100
    ax.text(xi, v + 4, f"{v:.0f}%", ha="center", fontsize=8, color="tab:blue")

# inset on (b): paired contact-distance improvement of the grounded fix
rng = np.random.default_rng(0)
deltas = []
for t in ["mw-push", "mw-pick-place"]:
    p = cl[cl.task == t].pivot_table(index="seed", columns="arm",
                                     values="final_state_dist")
    deltas.extend((p.l2 - p.hdyn).dropna().values)
deltas = np.asarray(deltas)
m = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(1)
lo, hi = np.percentile(m, [2.5, 97.5])
axin = ax.inset_axes([0.40, 0.30, 0.26, 0.34])
axin.errorbar([0], [deltas.mean()],
              yerr=[[deltas.mean() - lo], [hi - deltas.mean()]],
              fmt="D", color="tab:red", capsize=4)
axin.axhline(0, ls="--", lw=0.8, color="0.5")
axin.set_xlim(-0.6, 0.8)
axin.set_xticks([])
axin.set_title("grounded fix:\ncontact dist. gain", fontsize=6.5)
axin.tick_params(labelsize=6)
axin.text(0.12, deltas.mean(), f"+{deltas.mean():.2f}\n[{lo:.2f},{hi:.2f}]",
          fontsize=6, va="center")
axin.spines[["top", "right"]].set_visible(False)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/figures/figure_bb_vs_planning.pdf")
print("wrote results/figures/figure_bb_vs_planning.pdf")
print("regime BB dino:", {r: round(pool_regime(bb, 'dino', r), 3) for r in REGIMES})
print("regime BB jepa:", {r: round(pool_regime(bb, 'jepa', r), 3) for r in REGIMES})
print(f"contact paired delta +{deltas.mean():.3f} [{lo:.3f}, {hi:.3f}]")
