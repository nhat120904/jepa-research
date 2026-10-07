#!/usr/bin/env python3
"""Cube move events from the PRIVILEGED cube positions (qpos) -- direction D of the cube extension.

Direction D tests whether the planning and skill parts transfer to cubes when the object state is
given (privileged), before replacing it with label-free object discovery (direction A). A move is a
maximal interval in which one cube's position changes by more than 2 cm (pauses < 5 frames merged).
Output per split: t_start, t_end, cube k, state before/after (K x 3 positions), target xy = the cube's
final xy, the start of the segment leading to the move (previous move end + 1), and whether another
cube moved during the same interval (knocks / carried stacks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SLICES = {"single": [14], "double": [14, 21], "triple": [14, 21, 28], "quadruple": [14, 21, 28, 35]}


def moves_of(xyz, ep):
    n, K, _ = xyz.shape
    same = ep[1:] == ep[:-1]
    speed = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    moving = (speed > 2e-3) & same[:, None]
    out = []
    for k in range(K):
        d = np.diff(np.r_[0, moving[:, k].astype(np.int8), 0])
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        ks, ke = [], []
        for s0, e0 in zip(st, en):
            if ke and s0 - ke[-1] < 5 and ep[s0] == ep[ke[-1]]:
                ke[-1] = e0
            else:
                ks.append(s0)
                ke.append(e0)
        for s0, e0 in zip(ks, ke):
            e0 = min(e0, n - 1)
            if np.linalg.norm(xyz[e0, k] - xyz[s0, k]) > 0.02:
                out.append((s0, e0, k))
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--kind", default="triple")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for split in ("train", "val"):
        q = np.load(a.cache / f"{split}_qpos.npy", mmap_mode="r")
        term = np.load(a.cache / f"{split}_terminals.npy")
        ep = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)
        xyz = np.stack([np.asarray(q[:, s:s + 3]) for s in SLICES[a.kind]], 1).astype(np.float32)
        mv = moves_of(xyz, ep)
        ts = np.array([m[0] for m in mv]); te = np.array([m[1] for m in mv]); k = np.array([m[2] for m in mv])
        before, after = xyz[ts], xyz[np.minimum(te + 1, len(xyz) - 1)]
        others = np.ones((len(mv), xyz.shape[1]), bool)
        others[np.arange(len(mv)), k] = False
        knock = (np.linalg.norm(after - before, axis=-1) * others).max(1) > 0.02
        first = np.r_[0, np.nonzero(term)[0] + 1]
        seg = first[ep[ts]].copy()
        same_ep = np.r_[False, ep[ts][1:] == ep[ts][:-1]]
        seg[same_ep] = te[:-1][same_ep[1:]] + 1
        seg = np.minimum(seg, ts)
        np.savez_compressed(a.out / f"cube_events_{split}.npz", t_start=ts, t=te, k=k, before=before, after=after,
                            target_xy=after[np.arange(len(mv)), k, :2], seg_start=seg, knock=knock, episode=ep[ts])
        rep[split] = {"moves": int(len(mv)), "per_episode": float(len(mv) / (ep[-1] + 1)), "knock_frac": float(knock.mean()),
                      "stack_frac": float((after[np.arange(len(mv)), k, 2] > 0.04).mean())}
        print(split, rep[split], flush=True)
    (a.out / "cube_events_report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
