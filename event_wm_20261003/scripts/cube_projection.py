#!/usr/bin/env python3
"""PRIVILEGED attribution helper: map simulator cube positions (x, y, z) to the label-free pixel state.

Fits a quadratic map (x, y, z) -> (u, v) from val frames where a discovered object is visible to the
colour track and its cube is at rest (the pixel state is the top-face centroid of the colour track, so
the map lands in the same space as the reader, the planner and the skill targets). Also stores the
object -> cube matching from cube_discover.py. Used only by the attribution arms of cube_closed_loop.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SLICES = {"single": [14], "double": [14, 21], "triple": [14, 21, 28], "quadruple": [14, 21, 28, 35]}


def feats(x):
    x = np.atleast_2d(np.asarray(x, np.float64))
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    return np.stack([np.ones(len(x)), a, b, c, a * a, b * b, c * c, a * b, a * c, b * c], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--discover", type=Path, required=True)
    ap.add_argument("--kind", default="triple")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    disc = json.loads((a.discover / "discover.json").read_text())
    o2c = np.array([d["cube"] for d in sorted(disc["privileged_diagnostic"], key=lambda d: d["object"])])
    z = np.load(a.discover / "tracks_val.npz")
    uv, mass = z["uv"], z["mass"]
    q = np.load(a.cache / "val_qpos.npy", mmap_mode="r")
    xyz = np.stack([np.asarray(q[:, s:s + 3]) for s in SLICES[a.kind]], 1)
    speed = np.r_[np.zeros((1, xyz.shape[1])), np.linalg.norm(np.diff(xyz, axis=0), axis=-1)]
    X, Y = [], []
    for k, j in enumerate(o2c):
        m = (mass[:, k] >= 3) & (speed[:, j] < 1e-3)
        X.append(xyz[m, j]); Y.append(uv[m, k])
    X, Y = np.concatenate(X), np.concatenate(Y)
    half = len(X) // 2
    idx = np.random.default_rng(0).permutation(len(X))
    tr, te = idx[:half], idx[half:]
    W = np.linalg.lstsq(feats(X[tr]), Y[tr], rcond=None)[0]
    err = np.linalg.norm(feats(X[te]) @ W - Y[te], axis=-1)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "projection.npz", W=W, o2c=o2c)
    rep = {"frames": int(len(X)), "heldout_err_px_median": float(np.median(err)), "p95": float(np.percentile(err, 95)),
           "o2c": o2c.tolist()}
    (a.out / "projection.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
