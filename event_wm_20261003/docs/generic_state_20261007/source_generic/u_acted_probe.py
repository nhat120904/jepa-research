#!/usr/bin/env python3
"""PRIVILEGED diagnostic (puzzle): which acted-entity rule picks the pressed button?

Pressed button (privileged): at a button_states change, the toggled set S; the pressed button is the member
of S whose cross (itself + 4-neighbours on the rows x cols grid) contains the most of S. Identity -> button
map: best single-identity reader per button (u_button_probe logic, simplified: max |corr| of app with state).
For every u_events event whose interval overlaps exactly one press (within 15 frames), each rule's choice
among the changed identities of that event (identities whose rest state differs before / after):
  near    min distance from agent pixels to the identity (current rule), frames [t0 - 5, t0 + 10);
  cover   max fraction of the identity's place disc covered by agent pixels over those frames;
  tip     min distance to the agent tip = agent pixel farthest from the agent base (the pixel most often
          covered by agent segments over the split);
  central min sum of distances to the other changed identities (ties -> near).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    d = json.loads((a.run / "front" / "discover.json").read_text())
    E = np.load(a.run / "front" / f"entities_{a.split}.npz"); ev = np.load(a.run / "events" / f"events_{a.split}.npz")
    bs = np.load(a.cache / f"{a.split}_button_states.npy", mmap_mode="r")
    proc = E["processed"]; pf = np.nonzero(proc)[0]
    K = E["pos"].shape[1]
    # identity -> button by correlation on processed visible frames
    Bp = np.asarray(bs[pf]).astype(float); app = E["app"][pf]; vis = E["area"][pf] >= 1
    corr = np.zeros((K, Bp.shape[1]))
    for k in range(K):
        if vis[:, k].sum() < 50:
            continue
        x = app[vis[:, k], k]; x = x - x.mean(0)
        for j in range(Bp.shape[1]):
            y = Bp[vis[:, k], j] - Bp[vis[:, k], j].mean()
            if y.std() == 0:
                continue
            corr[k, j] = max(abs(np.corrcoef(x[:, c], y)[0, 1]) for c in range(3) if x[:, c].std() > 0) if (x.std(0) > 0).any() else 0
    but_of = {k: int(corr[k].argmax()) for k in range(K) if corr[k].max() > 0.8}
    # agent base: pixel most often covered by agent segments
    A = np.unpackbits(E["agent"][pf], axis=-1)[:, :, :64].astype(bool)
    freq = A.mean(0); base = np.array(np.unravel_index(freq.argmax(), freq.shape))[::-1].astype(float)   # (u, v)
    v_, u_ = np.mgrid[0:64, 0:64]
    thr = d["thr_pos"]
    centre = {k: np.array(i["centre"]) for k, i in enumerate(d["identities"]) if i["centre"] is not None}
    discs = {k: np.hypot(u_ - c[0], v_ - c[1]) <= thr / 2 for k, c in centre.items()}
    # presses
    B = np.asarray(bs).astype(int)
    tog = np.nonzero((B[1:] != B[:-1]).any(1))[0] + 1
    presses = []
    for t in tog:
        if not proc[max(t - 300, 0):t].any():
            continue
        S = set(np.nonzero(B[t] != B[t - 1])[0].tolist())
        def cross(i):
            r, c = divmod(i, a.cols)
            return {i} | {rr * a.cols + cc for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= rr < a.rows and 0 <= cc < a.cols}
        presses.append((t, max(S, key=lambda i: len(cross(i) & S))))
    pt = np.array([p[0] for p in presses])
    res = {r: [] for r in ("near", "cover", "tip", "central")}
    for t0, t1, e, b4, af in zip(ev["t_start"], ev["t"], ev["e"], ev["before"], ev["after"]):
        ov = np.nonzero((pt >= t0 - 15) & (pt <= t1 + 15))[0]
        if len(ov) != 1:
            continue
        pressed = presses[ov[0]][1]
        ch = [k for k in range(K) if k in centre and np.abs(af[k, 2:5] - b4[k, 2:5]).max() > 1e-6]
        if not ch or not any(but_of.get(k) == pressed for k in ch):
            continue
        fr = [t for t in range(max(t0 - 5, 0), t0 + 10) if proc[t]]
        Am = [np.unpackbits(E["agent"][t], axis=-1)[:, :64].astype(bool) for t in fr]
        def near(k):
            return min((np.min(np.hypot(u_[m] - centre[k][0], v_[m] - centre[k][1])) for m in Am if m.any()), default=np.inf)
        def cover(k):
            return max(((m & discs[k]).sum() / discs[k].sum() for m in Am), default=0)
        def tipd(k):
            ds = []
            for m in Am:
                if m.any():
                    far = np.argmax(np.hypot(u_[m] - base[0], v_[m] - base[1]))
                    ds.append(np.hypot(u_[m][far] - centre[k][0], v_[m][far] - centre[k][1]))
            return min(ds, default=np.inf)
        def central(k):
            return sum(np.hypot(*(centre[k] - centre[j])) for j in ch if j != k)
        picks = {"near": min(ch, key=near), "cover": max(ch, key=cover), "tip": min(ch, key=tipd),
                 "central": min(ch, key=lambda k: (central(k), near(k)))}
        for r, k in picks.items():
            res[r].append(but_of.get(k) == pressed)
    out = {"events_scored": len(res["near"]), "base_uv": base.tolist(), "mapped_identities": len(but_of),
           "accuracy": {r: round(float(np.mean(v)), 3) if v else None for r, v in res.items()}}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
