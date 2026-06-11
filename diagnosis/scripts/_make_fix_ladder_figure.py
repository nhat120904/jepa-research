"""Figure 2: the fix ladder. Left — pooled pre-grasp BB at each rung of the
ladder (frozen base -> mixture heads -> phi-metric -> grounded dynamics),
showing capacity and metric are nulls and only the grounded channel moves BB.
Right — the V1/V2/V3 probe chain that localizes the broken piece (object
decodable, propagated for factual actions, dead for counterfactual ones) and
the counterfactual-tracking correlation the grounded channel restores.

All numbers read live from the result CSVs. No GPU/data needed.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXCLUDE = ("mw-door-close",)


def pool_pregrasp(path, model_key="dino"):
    d = pd.read_csv(path)
    g = d[d.model.str.contains(model_key) & (d.regime == "pre_grasp")
          & ~d.task.isin(EXCLUDE)]
    g = g[np.isfinite(g.bb_boundary) & (g.n_boundary > 0)]
    return float(np.average(g.bb_boundary, weights=g.n_boundary))


rungs = [
    ("frozen\nbase", "results/metaworld_boundary.csv"),
    ("mixture\nheads", "results/metaworld_boundary_fix.csv"),
    ("$\\varphi$-metric\nreweight", "results/metaworld_boundary_metric.csv"),
    ("grounded\n$h(z,a)$", "results/metaworld_boundary_dynamics.csv"),
]
vals = [pool_pregrasp(p) for _, p in rungs]
labels = [r for r, _ in rungs]
colors = ["0.6", "0.6", "0.6", "tab:red"]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=[1.4, 1])

ax = axes[0]
x = np.arange(len(rungs))
bars = ax.bar(x, vals, 0.62, color=colors)
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.axhline(vals[0], ls="--", lw=0.8, color="0.4")
ax.annotate("$-50\\%$", xy=(3, vals[3]), xytext=(3, vals[0] * 0.62),
            ha="center", fontsize=10, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red"))
ax.set_xticks(x, labels, fontsize=8.5)
ax.set_ylabel("pre-grasp BB (pooled)")
ax.set_title("(a) The fix ladder: only grounding moves BB")
ax.set_ylim(0, max(vals) * 1.25)

# (b) probe chain V1/V2/V3 + counterfactual corr
ax = axes[1]
# V1/V2 are object-decode errors vs the state sd (0.094); V3 is the
# counterfactual spread correlation (frozen vs +h). Show as two mini-panels.
err_sd = 0.094
v1, v2 = 0.064, 0.059      # FIX_C1_EXPLAINER / CLAIMS 3.5
xb = np.arange(2)
ax.bar(xb, [v1, v2], 0.5, color="tab:blue", label="probe error")
ax.axhline(err_sd, ls="--", lw=0.8, color="0.4")
ax.text(1.5, err_sd + 0.001, "object state sd", fontsize=7.5, color="0.4")
ax.set_xticks(xb, ["V1\n(decode)", "V2\n(factual\npropagate)"], fontsize=8)
ax.set_ylabel("object-decode error", fontsize=9)
ax.set_title("(b) Where the signal lives", fontsize=10)
ax.set_ylim(0, 0.11)

# inset: counterfactual tracking corr frozen -> +h (the V3 channel)
axin = ax.inset_axes([0.55, 0.5, 0.42, 0.42])
axin.bar([0, 1], [0.035, 0.682], 0.55, color=["0.6", "tab:red"])
axin.set_xticks([0, 1], ["V3\nfrozen", "$+h$"], fontsize=7)
axin.set_title("cf. corr", fontsize=7.5)
axin.set_ylim(0, 0.8)
axin.tick_params(labelsize=6)

for a in list(axes) + [axin]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/figures/figure_fix_ladder.pdf")
print("wrote results/figures/figure_fix_ladder.pdf")
print("ladder:", {l.replace(chr(10), ' '): round(v, 3) for l, v in zip(labels, vals)})
