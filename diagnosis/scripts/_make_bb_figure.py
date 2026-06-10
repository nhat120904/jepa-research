"""Figure: Boundary Blindness per regime (the headline figure for the
Boundary-Blind framing). Bars = n_boundary-weighted pooled bb_boundary per
(model, regime); whiskers = n_boundary-weighted pooled bootstrap CI bounds.
Metaworld pooled over tasks excluding mw-door-close (articulated-object proxy
anomaly — shown hatched as a sensitivity variant); DROID is the single flat pool.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIMES = ["free_space", "pre_grasp", "gripper_actuation", "contact_manipulation"]
LABELS = ["free\nspace", "pre\ngrasp", "gripper\nactuation", "contact\nmanip."]


def pooled(df, model, regime, exclude=()):
    g = df[(df.model == model) & (df.regime == regime) & ~df.task.isin(exclude)]
    g = g[np.isfinite(g.bb_boundary) & (g.n_boundary > 0)]
    if not len(g):
        return np.nan, np.nan, np.nan
    w = g.n_boundary
    return (float(np.average(g.bb_boundary, weights=w)),
            float(np.average(g.bb_boundary_lo, weights=w)),
            float(np.average(g.bb_boundary_hi, weights=w)))


mw = pd.read_csv("results/metaworld_boundary.csv")
dr = pd.read_csv("results/droid_boundary.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), width_ratios=[2, 1])

ax = axes[0]
models = ["dino_wm_metaworld", "jepa_wm_metaworld"]
x = np.arange(len(REGIMES))
for i, m in enumerate(models):
    pts = [pooled(mw, m, r, exclude=("mw-door-close",)) for r in REGIMES]
    y = [p[0] for p in pts]
    lo = [p[0] - p[1] if np.isfinite(p[1]) else 0 for p in pts]
    hi = [p[2] - p[0] if np.isfinite(p[2]) else 0 for p in pts]
    ax.bar(x + (i - 0.5) * 0.38, y, 0.36, yerr=[lo, hi], capsize=3,
           label=m.replace("_metaworld", ""))
ax.set_xticks(x, LABELS)
ax.set_ylabel("BB (boundary subset, pooled)")
ax.set_title("Metaworld (object-Δ outcome, excl. mw-door-close)")
ax.legend(frameon=False)

ax = axes[1]
pts = [pooled(dr, "dino_wm_droid", r) for r in REGIMES]
y = [p[0] for p in pts]
lo = [p[0] - p[1] if np.isfinite(p[1]) else 0 for p in pts]
hi = [p[2] - p[0] if np.isfinite(p[2]) else 0 for p in pts]
ax.bar(x, y, 0.6, yerr=[lo, hi], capsize=3, color="tab:green", label="dino_wm")
ax.set_xticks(x, LABELS)
ax.set_title("DROID (‖Δz‖ proxy outcome, transfer)")
ax.legend(frameon=False)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/figures/figure_bb_per_regime.pdf")
print("wrote results/figures/figure_bb_per_regime.pdf")
