#!/usr/bin/env python3
"""Cube play-data inventory (privileged qpos; describes the data, not used by the method).

Moves = maximal intervals in which a cube's position changes (it is being carried or falls).
Reports moves per episode, move durations, steps between moves, final heights (stacking) and how
often a single move relocates more than one cube.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SLICES = {"single": [14], "double": [14, 21], "triple": [14, 21, 28], "quadruple": [14, 21, 28, 35]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--kind", default="triple")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    q = np.load(a.cache / "train_qpos.npy", mmap_mode="r")
    term = np.load(a.cache / "train_terminals.npy")
    n = len(term)
    ep = np.concatenate([[0], np.cumsum(term[:-1])])
    xyz = np.stack([np.asarray(q[:, s:s + 3]) for s in SLICES[a.kind]], 1)          # (N, K, 3)
    K = xyz.shape[1]
    same = ep[1:] == ep[:-1]
    speed = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)                               # (N-1, K)
    moving = (speed > 2e-3) & same[:, None]
    rep = {"frames": int(n), "episodes": int(ep[-1] + 1), "cubes": K}
    moves = []
    for k in range(K):
        m = moving[:, k].astype(np.int8)
        d = np.diff(np.r_[0, m, 0])
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        # merge interruptions shorter than 5 frames (the cube pauses while lifted)
        keep_s, keep_e = [], []
        for s0, e0 in zip(st, en):
            if keep_e and s0 - keep_e[-1] < 5 and ep[s0] == ep[keep_e[-1]]:
                keep_e[-1] = e0
            else:
                keep_s.append(s0)
                keep_e.append(e0)
        for s0, e0 in zip(keep_s, keep_e):
            disp = np.linalg.norm(xyz[min(e0, n - 1), k] - xyz[s0, k])
            if disp > 0.02:
                moves.append((s0, e0, k, disp, xyz[min(e0, n - 1), k, 2]))
    moves.sort()
    mv = np.array([(m[0], m[1], m[2], m[3], m[4]) for m in moves])
    rep["moves"] = int(len(mv))
    rep["moves_per_episode"] = float(len(mv) / (ep[-1] + 1))
    dur = mv[:, 1] - mv[:, 0]
    rep["move_duration_pct"] = {p: float(np.percentile(dur, p)) for p in (5, 50, 95)}
    starts = mv[:, 0].astype(int)
    gaps = np.diff(starts)[ep[starts[1:]] == ep[starts[:-1]]]
    rep["steps_between_move_starts_pct"] = {p: float(np.percentile(gaps, p)) for p in (5, 50, 95)}
    z = mv[:, 4]
    rep["final_height_hist"] = {"ground(<0.04)": float((z < 0.04).mean()), "level2(0.04-0.08)": float(((z >= 0.04) & (z < 0.08)).mean()),
                                "level3(>=0.08)": float((z >= 0.08).mean())}
    # overlapping moves of different cubes (carrying a stack or knocking another cube)
    ov = 0
    for i in range(1, len(mv)):
        if mv[i, 0] < mv[i - 1, 1] and mv[i, 2] != mv[i - 1, 2]:
            ov += 1
    rep["overlapping_moves_frac"] = ov / max(1, len(mv))
    rep["xy_range"] = {"x": [float(xyz[..., 0].min()), float(xyz[..., 0].max())], "y": [float(xyz[..., 1].min()), float(xyz[..., 1].max())]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"cube_inventory_{a.kind}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
