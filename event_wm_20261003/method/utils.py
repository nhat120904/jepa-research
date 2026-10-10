"""Shared helpers of the method (method/README.md). State vector of an entity: pos (2: u = column, v = row, px of the
64 x 64 frame), app (A values in [0, 1]), covered (1); D = A + 3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def layout(D):
    """slices of an entity state vector of length D: position, appearance, covered index."""
    return slice(0, 2), slice(2, D - 1), D - 1


def two_means_threshold(x, iters=50):
    """Same rule as sfa_code.two_means_threshold / frontend.two_means_threshold. -> (threshold, Ashman's D, fraction above)."""
    x = np.asarray(x, np.float64)
    c = np.percentile(x, [10, 90]).astype(np.float64)
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None]).argmin(1)
        c = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(2)])
    thr = c.mean()
    lo, hi = x[x <= thr], x[x > thr]
    sep = abs(c[1] - c[0]) / (np.sqrt(0.5 * (lo.var() + hi.var())) + 1e-9)
    return float(thr), float(sep), float((x > thr).mean())


def bimodal_split(x):
    """2-means split of x; (threshold, D); inf threshold if not bimodal (Ashman's D <= 2)."""
    thr, sep, _ = two_means_threshold(x)
    return (thr if sep > 2.0 else np.inf), sep


def save_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")


class Frames:
    """(N, 64, 64, 3) uint8 frames, memory-mapped or in RAM. A memory map is reopened every `remap` reads, so the pages
    read so far leave the working set (on Windows it otherwise grows to the file size)."""

    def __init__(self, path, n, ram=False, remap=200):
        self.path, self.n, self.remap, self.k, self.ram = path, n, remap, 0, ram
        src = np.load(path, mmap_mode="r")
        self.a = np.array(src[:n]) if ram else src[:n]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if not self.ram:
            self.k += 1
            if self.k % self.remap == 0:
                self.a = np.load(self.path, mmap_mode="r")[:self.n]
        return np.array(self.a[idx])


def episode_bounds(cache: Path, split: str, episodes: int | None = None):
    """-> starts, ends (inclusive) of the first `episodes` episodes of the cached split."""
    term = np.load(Path(cache) / f"{split}_terminals.npy")
    ends = np.flatnonzero(term)
    if episodes is not None:
        ends = ends[:episodes]
    starts = np.r_[0, ends[:-1] + 1]
    return starts, ends
