#!/usr/bin/env python3
"""Compose the rendered triplets into a side-by-side video and a bend figure.

Video: the low-curvature state above the high-curvature one, each showing its
three nudged rollouts side by side. The point is visible without narration --
in the low state nothing moves whichever way the action is nudged; in the high
state the arm and cube visibly go three different ways.

Figure: the model's three predicted terminal latents, projected onto the plane
spanned by v- and v+. Collinear means a locally straight map; the angle between
the two segments is exactly what "action-space curvature" measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--viz", type=Path, required=True)
    p.add_argument("--low", type=int, required=True)
    p.add_argument("--high", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


def load(viz: Path, snap: int) -> tuple[dict, dict]:
    d = np.load(viz / f"snapshot_{snap:03d}/viz.npz")
    meta = json.loads((viz / f"snapshot_{snap:03d}/meta.json").read_text())
    return {k: d[k] for k in d.files}, meta


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Two-pixel top band tinted per arm, so the three columns stay legible."""
    tint = {"-": (90, 140, 230), "0": (150, 150, 150), "+": (230, 120, 90)}[text]
    out = img.copy()
    out[:6, :] = np.array(tint, dtype=out.dtype)
    return out


def strip(data: dict, n: int) -> np.ndarray:
    """One frame of the three-arm strip, padded to the longest rollout."""
    cols = []
    for key, tag in (("frames_minus", "-"), ("frames_centre", "0"), ("frames_plus", "+")):
        f = data[key]
        cols.append(label(f[min(n, len(f) - 1)], tag))
    return np.concatenate(cols, axis=1)


def main() -> None:
    args = parse_args()
    lo, lo_meta = load(args.viz, args.low)
    hi, hi_meta = load(args.viz, args.high)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = max(max(len(lo[k]) for k in ("frames_minus", "frames_centre", "frames_plus")),
            max(len(hi[k]) for k in ("frames_minus", "frames_centre", "frames_plus")))
    frames = [np.concatenate([strip(lo, i), strip(hi, i)], axis=0) for i in range(n)]
    path = args.out_dir / "curvature_low_vs_high.mp4"
    try:
        imageio.mimsave(path, frames, fps=args.fps, macro_block_size=1)
    except Exception:
        path = args.out_dir / "curvature_low_vs_high.gif"
        imageio.mimsave(path, frames, duration=1.0 / args.fps)
    print(f"wrote {path}  ({len(frames)} frames)")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (d, m, name) in zip(axes, [(lo, lo_meta, "LOW curvature"),
                                       (hi, hi_meta, "HIGH curvature")]):
        c = d["latent_coords"] * 1.0
        ax.plot(c[:, 0], c[:, 1], "-o", color="#1f77b4", lw=2, ms=9, zorder=3)
        ax.plot([c[0, 0], c[2, 0]], [c[0, 1], c[2, 1]], "--", color="#999", lw=1.4,
                label="straight line", zorder=2)
        for pt, tag in zip(c, ["a - d", "a", "a + d"]):
            ax.annotate(tag, pt, textcoords="offset points", xytext=(7, 7), fontsize=9)
        ax.set_title(f"{name}  (snapshot {m['snapshot']})\n"
                     f"bend = {m['latent_angle_deg']:.1f}°   "
                     f"cube travels {max(m['object_travel_mm']):.2f} mm",
                     fontsize=10.5)
        ax.set_xlabel("latent, along v−")
        ax.set_ylabel("latent, perpendicular")
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(frameon=False, fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("What action-space curvature measures: the model's predicted "
                 "outcome for three nudged actions", fontsize=12)
    fig.tight_layout()
    out = args.out_dir / "curvature_bend.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    print(json.dumps({"low": lo_meta, "high": hi_meta}, indent=2))


if __name__ == "__main__":
    main()
