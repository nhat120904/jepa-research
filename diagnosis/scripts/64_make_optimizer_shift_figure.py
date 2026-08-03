"""Create the main-paper optimizer-conditioned ranking figure.

This is an offline analysis of persisted candidate dumps and summary CSVs.  It
does not load a model or simulator.  The two scatter panels use a population
fixed before plotting (DINO-WM push, seed 41000, replan 0); the lower panels
show seed-clustered aggregates over all four checkpoint--task cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "DINO / push": "#0072B2",
    "DINO / pick-place": "#56B4E9",
    "JEPA / push": "#D55E00",
    "JEPA / pick-place": "#E69F00",
}


def scatter_population(
    ax: plt.Axes, frame: pd.DataFrame, *, iteration: int, title: str
) -> None:
    pop = frame[frame["iter"] == iteration].copy()
    if len(pop) != 100:
        raise ValueError(f"expected 100 candidates at iteration {iteration}, got {len(pop)}")

    proxy = 100.0 * pop["proxy_cost"].to_numpy(float)
    truth = 100.0 * pop["true_shaped_cost"].to_numpy(float)
    proxy_top = np.argsort(proxy, kind="mergesort")[:10]
    truth_top = np.argsort(truth, kind="mergesort")[:10]

    ax.scatter(proxy, truth, s=12, color="#999999", alpha=0.55, linewidth=0)
    ax.scatter(
        proxy[truth_top],
        truth[truth_top],
        s=34,
        facecolors="none",
        edgecolors="#0072B2",
        linewidth=1.2,
        label="reference top 10%",
    )
    ax.scatter(
        proxy[proxy_top],
        truth[proxy_top],
        s=25,
        color="#D55E00",
        edgecolors="white",
        linewidth=0.35,
        label="proxy top 10%",
        zorder=3,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("stateprobe cost (cm)")
    ax.set_ylabel("reference cost (cm)")
    ax.grid(alpha=0.18, linewidth=0.5)


def aggregate_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    chance: float | None = None,
) -> None:
    labels = []
    for (model, task), cell in summary.groupby(["model", "task"], sort=True):
        label = f"{model} / {task}"
        labels.append(label)
        cell = cell.set_index("stage").loc[["initial", "final"]]
        x = np.array([0.0, 1.0])
        y = cell[metric].to_numpy(float)
        lo = cell[f"{metric}_ci_lo"].to_numpy(float)
        hi = cell[f"{metric}_ci_hi"].to_numpy(float)
        ax.plot(x, y, marker="o", ms=4, lw=1.5, color=COLORS[label], label=label)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt="none",
            ecolor=COLORS[label],
            elinewidth=0.9,
            capsize=2,
        )
    if chance is not None:
        ax.axhline(
            chance,
            color="#555555",
            ls="--",
            lw=0.9,
            label="random-set expectation",
        )
    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1], ["initial", "final"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--replan", type=int, default=0)
    args = parser.parse_args()

    candidates = pd.read_csv(
        args.candidates,
        usecols=[
            "seed",
            "replan",
            "iter",
            "candidate",
            "proxy_cost",
            "true_shaped_cost",
        ],
    )
    candidates = candidates[
        (candidates["seed"] == args.seed) & (candidates["replan"] == args.replan)
    ]
    summary = pd.read_csv(args.summary)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True)

    scatter_population(
        axes[0, 0], candidates, iteration=0, title="A  Before adaptive refitting"
    )
    scatter_population(
        axes[0, 1], candidates, iteration=5, title="B  After five refits"
    )
    handles, labels = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, loc="upper left", frameon=False)

    aggregate_panel(
        axes[1, 0],
        summary,
        metric="cost_spearman",
        ylabel="proxy/reference Spearman",
    )
    axes[1, 0].set_title("C  Rank agreement across all cells", fontsize=9)
    axes[1, 0].set_ylim(0.0, 0.65)

    aggregate_panel(
        axes[1, 1],
        summary,
        metric="reference_top10_recall",
        ylabel="reference top-10% recall",
        chance=0.10,
    )
    axes[1, 1].set_title("D  Elite-set recall across all cells", fontsize=9)
    axes[1, 1].set_ylim(0.0, 0.34)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc="upper right", frameon=False, ncol=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
