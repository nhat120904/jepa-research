#!/usr/bin/env python3
"""PRIVILEGED attribution helper: qpos move events (cube_events.py) re-expressed in the label-free pixel
state, so a skill can be trained on privileged segments with pixel targets.

Moves, times and segments are the qpos ones; the cube index becomes the discovered object index and
every position becomes its projected pixel position (cube_projection.py), plus the true coverage bit.
Comparing a skill trained on these events with one trained on cube_events_px.py events separates the
cost of label-free segmentation from the cost of the pixel target space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cube_projection import feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos-events", type=Path, required=True, help="cube_events.py output dir")
    ap.add_argument("--project", type=Path, required=True, help="projection.npz")
    ap.add_argument("--px-events", type=Path, required=True, help="cube_events_px.py dir (lo/hi/thr_px)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    zp = np.load(a.project)
    W, o2c = zp["W"], zp["o2c"]
    c2o = np.argsort(o2c)
    ref = np.load(a.px_events / "cube_events_train.npz")
    a.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        ev = np.load(a.qpos_events / f"cube_events_{split}.npz")
        N, K, _ = ev["before"].shape

        def to_px(xyz):
            cov = np.zeros((N, K), np.float32)
            for i in range(K):
                for j in range(K):
                    if i != j:
                        cov[:, i] = np.maximum(cov[:, i], ((xyz[:, j, 2] > xyz[:, i, 2] + 0.02)
                                                           & (np.linalg.norm(xyz[:, i, :2] - xyz[:, j, :2], axis=-1) < 0.03)))
            uv = (feats(xyz.reshape(-1, 3)) @ W).reshape(N, K, 2)
            return np.concatenate([uv, cov[..., None]], -1)[:, o2c].astype(np.float32)

        before, after = to_px(ev["before"]), to_px(ev["after"])
        k = c2o[ev["k"]]
        np.savez_compressed(a.out / f"cube_events_{split}.npz", t_start=ev["t_start"], t=ev["t"], k=k, before=before, after=after,
                            target_xy=after[np.arange(N), k, :2], seg_start=ev["seg_start"], knock=ev["knock"],
                            episode=ev["episode"], lo=ref["lo"], hi=ref["hi"], thr_px=ref["thr_px"], n_bin=1)
        print(split, {"moves": int(N)}, flush=True)


if __name__ == "__main__":
    main()
