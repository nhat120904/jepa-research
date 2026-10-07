#!/usr/bin/env python3
"""PRIVILEGED diagnostic (visual-scene): event recall per object group.

Reference events per group on processed episodes (val): cube = contiguous qpos[14:17] motion with net
displacement >= 2 cm; drawer (qpos 23) / window (qpos 24) = contiguous joint motion with net change >= 25%
of the joint's observed range; button b = a toggle of button_states[:, b]. Detected = some u_events event
overlaps the reference interval within 15 frames. Also: which identity is acted in the overlapping events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def intervals(moving):
    d = np.diff(np.r_[0, moving.astype(np.int8), 0]); return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    ev = np.load(a.run / "events" / f"events_{a.split}.npz"); ent = np.load(a.run / "front" / f"entities_{a.split}.npz")
    q = np.load(a.cache / f"{a.split}_qpos.npy", mmap_mode="r"); bs = np.load(a.cache / f"{a.split}_button_states.npy", mmap_mode="r")
    term = np.load(a.cache / f"{a.split}_terminals.npy"); n = len(term)
    starts = np.r_[0, np.nonzero(term)[0] + 1]; ends = np.r_[np.nonzero(term)[0] + 1, n]
    starts, ends = starts[starts < ends], ends[starts < ends]
    eps = [e for e in range(len(starts)) if ent["processed"][starts[e]]]
    ts, te, ee = ev["t_start"], ev["t"], ev["e"]
    rng = {j: float(np.ptp(np.asarray(q[:, j]))) for j in (23, 24)}
    groups = {"cube": [], "drawer": [], "window": [], "button0": [], "button1": []}
    for e in eps:
        s0, s1 = starts[e], ends[e]
        Q = np.asarray(q[s0:s1]); Bs = np.asarray(bs[s0:s1]).astype(int)
        mv = np.r_[False, np.abs(np.diff(Q[:, 14:17], axis=0)).max(1) > 2e-3]
        for x, y in intervals(mv):
            if np.linalg.norm(Q[min(y + 1, len(Q) - 1), 14:16] - Q[max(x - 1, 0), 14:16]) >= 0.02:
                groups["cube"].append((s0 + x, s0 + y))
        for j, g in ((23, "drawer"), (24, "window")):
            mv = np.r_[False, np.abs(np.diff(Q[:, j])) > 1e-3]
            for x, y in intervals(mv):
                if abs(Q[min(y + 1, len(Q) - 1), j] - Q[max(x - 1, 0), j]) >= 0.25 * rng[j]:
                    groups[g].append((s0 + x, s0 + y))
        for b in range(Bs.shape[1]):
            for t in np.nonzero(Bs[1:, b] != Bs[:-1, b])[0]:
                groups[f"button{b}"].append((s0 + t, s0 + t + 1))
    out = {}
    for g, iv in groups.items():
        hits, acted = [], []
        for x, y in iv:
            ov = np.nonzero((ts <= y + 15) & (te >= x - 15))[0]
            hits.append(len(ov) > 0); acted += [int(ee[i]) for i in ov]
        out[g] = {"ref": len(iv), "recall": round(float(np.mean(hits)), 3) if iv else None,
                  "acted_identities": np.bincount(acted, minlength=ent["pos"].shape[1]).tolist() if acted else []}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
