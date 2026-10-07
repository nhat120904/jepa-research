#!/usr/bin/env python3
"""Diagnostic: candidate motion / structure statistics per merged colour type (no privileged input).

For a front-end run (discover.json with raw_clusters / merged_into) and its cached segments:
  count       median segments per sampled frame;      present   fraction of frames with the type;
  cont_move   continuity track moved > thr_pos;        big_move  largest segment moved > thr_pos;
  any_move    fraction of frame pairs in which some segment has no same-type segment within thr_pos in
              the previous frame;                       seg_move  the same per segment;
  at_places   fraction of segments at fixed places (density peeling, 30% occupancy);
  elong       median mask elongation (sqrt of the eigenvalue ratio of the mask covariance);
  area        median area (px).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, required=True, help="identity-stage front/ dir (discover.json)")
    ap.add_argument("--segments", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
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
    thr = d["thr_pos"]; P = z["pos"]; A = z["area"]
    v_, u_ = np.mgrid[0:64, 0:64]
    out = []
    for k in sorted(set(lab[lab >= 0].tolist())):
        sel = np.nonzero(lab == k)[0]
        cnt = np.bincount(f[sel], minlength=len(fr))
        if (cnt > 0).mean() < 0.3:
            continue
        byf = {}
        for i in sel:
            byf.setdefault(f[i], []).append(i)
        fs = sorted(byf)
        # continuity / largest tracks
        cont, big, prev, pe = [], [], None, None
        for x in fs:
            if ep[fr[x]] != pe:
                prev, pe = None, ep[fr[x]]
            ii = byf[x]
            j = max(ii, key=lambda i: A[i])
            big.append((x, P[j]))
            if prev is not None:
                dd_ = [np.hypot(*(P[i] - prev)) for i in ii]
                if min(dd_) <= thr:
                    j = ii[int(np.argmin(dd_))]
            cont.append((x, P[j])); prev = P[j]

        def frac_moved(track):
            m = n = 0
            for (x0, p0), (x1, p1) in zip(track[:-1], track[1:]):
                if x1 == x0 + 1 and ep[fr[x0]] == ep[fr[x1]]:
                    n += 1; m += np.hypot(*(p1 - p0)) > thr
            return m / max(1, n)

        anym = segm = npair = nseg = 0
        for x in fs:
            if x - 1 in byf and ep[fr[x - 1]] == ep[fr[x]]:
                Q = P[byf[x - 1]]
                mv = [np.min(np.hypot(*(Q - P[i]).T)) > thr for i in byf[x]]
                npair += 1; anym += any(mv); nseg += len(mv); segm += sum(mv)
        # fixed places (same peeling as the front end)
        from scipy.ndimage import convolve
        res, r = 4, thr / 2
        g = 64 * res; rr = int(np.ceil(r * res)); yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        disc = (np.hypot(xx, yy) <= r * res).astype(np.float64)
        Ps, Fs = P[sel], f[sel]
        ij = np.clip(np.round(Ps * res).astype(int), 0, g - 1)
        left = np.ones(len(Ps), bool); at = 0
        while left.any():
            H = np.zeros((g, g)); np.add.at(H, (ij[left, 1], ij[left, 0]), 1.0)
            D = convolve(H, disc, mode="constant"); y0, x0 = np.unravel_index(D.argmax(), D.shape)
            inside = left & (np.hypot(*(Ps - np.array([x0, y0]) / res).T) <= r)
            if not inside.any() or len(np.unique(Fs[inside])) < 0.3 * len(fr):
                break
            at += inside.sum(); left &= ~inside
        # elongation
        ss = sel[:: max(1, len(sel) // 3000)]
        el = []
        for i in ss:
            w = np.unpackbits(z["mask"][i], axis=-1)[:, :64].astype(np.float64); s = w.sum()
            cu, cv = (w * u_).sum() / s, (w * v_).sum() / s
            C = np.array([[(w * (u_ - cu) ** 2).sum(), (w * (u_ - cu) * (v_ - cv)).sum()],
                          [(w * (u_ - cu) * (v_ - cv)).sum(), (w * (v_ - cv) ** 2).sum()]]) / s
            ev = np.linalg.eigvalsh(C)
            el.append(np.sqrt(ev[1] / max(ev[0], 1e-3)))
        out.append({"type": int(k), "clusters": [int(c) for c in ks if mg[c] == k], "rgb": (np.mean([raw[c] for c in ks if mg[c] == k], 0) * 255).round().astype(int).tolist(),
                    "count": int(np.median(cnt[cnt > 0])), "present": round(float((cnt > 0).mean()), 3),
                    "cont_move": round(frac_moved(cont), 3), "big_move": round(frac_moved(big), 3),
                    "any_move": round(anym / max(1, npair), 3), "seg_move": round(segm / max(1, nseg), 3),
                    "at_places": round(at / max(1, len(Ps)), 3), "elong": round(float(np.median(el)), 2), "area": int(np.median(A[sel]))})
    for o in out:
        print(json.dumps(o))


if __name__ == "__main__":
    main()
