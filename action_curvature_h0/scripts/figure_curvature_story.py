#!/usr/bin/env python3
"""The two-panel figure that carries the whole finding.

Same x-axis on both panels -- per-state action-space curvature, normalised by
probe size so it is a property of the state and not of how hard we pushed.

Left: curvature vs how much the gripper actually moves.  Strong relation.
Right: curvature vs how often the model picks the wrong action.  Flat.

Read together: curvature tells you where motion happens, not where the model
is wrong -- and only the second is what a planner suffers from.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else float("nan")


def main() -> None:
    args = parse_args()
    per = defaultdict(list)
    for path in sorted(args.records.glob("snapshot_*/records.json")):
        snap = int(path.parent.name.split("_")[1])
        for r in json.loads(path.read_text()):
            if not r.get("valid_unclipped"):
                continue
            if not np.isfinite(r.get("model_angular_fraction", np.nan)):
                continue
            # Normalise by probe size: curvature scales linearly with it, so the
            # ratio is the part that belongs to the state.
            r["_ratio"] = (r["k_model_self"]
                           * np.sqrt(max(r["model_angular_fraction"], 0.0))
                           / r["sigma"])
            per[snap].append(r)

    snaps = sorted(per)
    curv = np.array([np.median([r["_ratio"] for r in per[s]]) for s in snaps])
    motion = np.array([1000 * np.median([r["effector_span_m"] for r in per[s]])
                       for s in snaps])
    wrong = np.array([100 * np.mean([not r["ordinal_argmin_agree"] for r in per[s]])
                      for s in snaps])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    style = dict(s=34, c="#1f77b4", alpha=0.75, edgecolors="none")

    axes[0].scatter(curv, motion, **style)
    axes[0].set_ylabel("gripper movement (mm)")
    axes[0].set_title(f"Curvature tracks motion\nrank corr = {spearman(curv, motion):+.2f}",
                      fontsize=11)
    axes[0].set_yscale("log")

    axes[1].scatter(curv, wrong, **{**style, "c": "#d62728"})
    axes[1].axhline(np.mean(wrong), ls="--", lw=1.2, c="#666",
                    label=f"mean {np.mean(wrong):.0f}%")
    axes[1].set_ylabel("model picks the wrong action (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title(f"Curvature says nothing about being wrong\n"
                      f"rank corr = {spearman(curv, wrong):+.2f}", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("action-space curvature of the state\n(per unit action probe)")
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Action-space curvature measures where motion happens, "
                 "not where the world model is wrong",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(snaps)} states)")
    print(f"  curvature vs motion : {spearman(curv, motion):+.3f}")
    print(f"  curvature vs wrong  : {spearman(curv, wrong):+.3f}")
    print(f"  wrong-action rate   : {wrong.min():.0f}% - {wrong.max():.0f}%, "
          f"mean {wrong.mean():.0f}%")


if __name__ == "__main__":
    main()
