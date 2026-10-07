#!/usr/bin/env python3
"""Diagnostic: do mask-IoU tracklets of the cached SAM 2 segments separate agent, objects and scenery?

Tracklets: consecutive sampled frames of an episode, Hungarian matching on 1 - IoU of the 64x64 masks, a
match accepted at IoU >= 0.5 (the usual detection-match criterion). Per tracklet: length, fraction of
steps whose centroid moved by more than tol (2 x the median-based noise split from u_events: passed in),
mean colour, median area. Reported per merged colour type of an identity-stage run (for reading only):
share of the type's segments in tracklets of length >= L, and the length-weighted moving fraction.
PRIVILEGED (cube): per cube, the fraction of its visible frames covered by tracklets of length >= L.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--segments", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--tol", type=float, required=True)
    ap.add_argument("--L", type=int, default=5)
    a = ap.parse_args()
    d = json.loads((a.front / "discover.json").read_text())
    z = np.load(a.segments / "segments_train.npz")
    raw = {r["cluster"]: np.array(r["rgb"]) / 255.0 for r in d["raw_clusters"]}
    mg = {r["cluster"]: r["merged_into"] for r in d["raw_clusters"]}
    ks = np.array(sorted(raw)); cent = np.stack([raw[k] for k in ks])
    dd = np.linalg.norm(z["col"][:, None] - cent[None], axis=-1)
    lab = np.array([mg[k] for k in ks[dd.argmin(1)]]); lab[dd.min(1) >= d["colour_merge"]] = -1
    fr = z["frames"]; fi = {t: i for i, t in enumerate(fr)}; f = np.array([fi[t] for t in z["t"]])
    term = np.load(a.cache / "train_terminals.npy"); ep = np.concatenate([[0], np.cumsum(term[:-1])])
    P = z["pos"]
    byf = {}
    for i in range(len(f)):
        byf.setdefault(f[i], []).append(i)
    tid = -np.ones(len(f), int); nt = 0
    for x in sorted(byf):
        cur = byf[x]
        prev = byf.get(x - 1) if (x - 1 in byf and ep[fr[x - 1]] == ep[fr[x]]) else None
        if prev:
            Mc = np.unpackbits(z["mask"][cur], axis=-1)[:, :, :64].reshape(len(cur), -1).astype(np.float32)
            Mp = np.unpackbits(z["mask"][prev], axis=-1)[:, :, :64].reshape(len(prev), -1).astype(np.float32)
            inter = Mc @ Mp.T; iou = inter / (Mc.sum(1)[:, None] + Mp.sum(1)[None] - inter + 1e-9)
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= 0.5:
                    tid[cur[i]] = tid[prev[j]]
        for i in cur:
            if tid[i] < 0:
                tid[i] = nt; nt += 1
    # tracklet stats
    order = np.lexsort((f, tid))
    T = {}
    for i in order:
        T.setdefault(tid[i], []).append(i)
    length = np.zeros(nt, int); movef = np.zeros(nt)
    for t_, ii in T.items():
        length[t_] = len(ii)
        if len(ii) > 1:
            movef[t_] = float((np.hypot(*(P[ii[1:]] - P[ii[:-1]]).T) > a.tol).mean())
    seg_len, seg_mv = length[tid], movef[tid]
    out = {"tracklets": int(nt), "segments": int(len(f)), "len_pct_50_90_99": np.percentile(length, [50, 90, 99]).tolist(),
           "segments_in_len_ge_L": round(float((seg_len >= a.L).mean()), 3), "types": []}
    for k in sorted(set(lab[lab >= 0].tolist())):
        s = lab == k
        if (np.bincount(f[s], minlength=len(fr)) > 0).mean() < 0.3:
            continue
        lg = s & (seg_len >= a.L)
        out["types"].append({"type": int(k), "share_len_ge_L": round(float(lg.sum() / s.sum()), 3),
                             "median_len": float(np.median(seg_len[s])),
                             "moving_frac_long": round(float(seg_mv[lg].mean()), 3) if lg.any() else None,
                             "frac_long_moving_gt_half": round(float((seg_mv[lg] > 0.5).mean()), 3) if lg.any() else None,
                             "moving_frac_all_tracklets_len_gt1": round(float(seg_mv[s & (seg_len > 1)].mean()), 3) if (s & (seg_len > 1)).any() else None})
    print(json.dumps(out, indent=None))


if __name__ == "__main__":
    main()
