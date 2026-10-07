#!/usr/bin/env python3
"""Diagnostic: mask solidity (area / convex-hull area) per colour type and its bimodality.

For each non-agent colour type of an identity-stage run: solidity of up to 2000 train segments, 2-means
split with Ashman's D, the share in the upper (convex) group, and for each group the fraction of sampled
frames with >= 1 / >= 2 / >= 3 instances (segments of the group grouped within thr_pos).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from sfa_code import two_means_threshold


def solidity(mask):
    v, u = np.nonzero(mask)
    if len(u) < 3:
        return 1.0
    pts = np.c_[u, v].astype(float)
    c = np.concatenate([pts + o for o in ([0, 0], [1, 0], [0, 1], [1, 1])])
    try:
        return float(mask.sum() / ConvexHull(c).volume)
    except Exception:
        return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--segments", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads((a.front / "discover.json").read_text())
    z = np.load(a.segments / "segments_train.npz")
    raw = {r["cluster"]: np.array(r["rgb"]) / 255.0 for r in d["raw_clusters"]}
    mg = {r["cluster"]: r["merged_into"] for r in d["raw_clusters"]}
    ks = np.array(sorted(raw)); cent = np.stack([raw[k] for k in ks])
    dd = np.linalg.norm(z["col"][:, None] - cent[None], axis=-1)
    lab = np.array([mg[k] for k in ks[dd.argmin(1)]]); lab[dd.min(1) >= d["colour_merge"]] = -1
    fr = z["frames"]; fi = {t: i for i, t in enumerate(fr)}; f = np.array([fi[t] for t in z["t"]])
    thr = d["thr_pos"]
    for t in d["types"]:
        if t["agent"]:
            continue
        idx = np.nonzero(lab == t["type"])[0]
        sub = idx[:: max(1, len(idx) // 2000)]
        sol = np.array([solidity(np.unpackbits(z["mask"][i], axis=-1)[:, :64].astype(bool)) for i in sub])
        th, D, up = two_means_threshold(sol)
        out = {"type": t["type"], "anchor": t.get("anchor"), "n": int(len(sub)), "solidity_p10_50_90": np.round(np.percentile(sol, [10, 50, 90]), 2).tolist(),
               "split": round(th, 3), "D": round(D, 2), "convex_share": round(up, 3)}
        for name, g in (("convex", sol > th), ("concave", sol <= th)):
            gi = sub[g]
            byf = {}
            for i in gi:
                byf.setdefault(f[i], []).append(z["pos"][i])
            cnt = []
            for x in range(len(fr)):
                P = byf.get(x, [])
                groups = []
                for p in P:
                    for G in groups:
                        if min(np.hypot(*(np.array(G) - p).T)) <= thr:
                            G.append(p); break
                    else:
                        groups.append([p])
                cnt.append(len(groups))
            cnt = np.array(cnt)
            out[name] = {"segments": int(g.sum()), "P_inst_ge": [round(float((cnt >= k).mean()), 3) for k in (1, 2, 3)],
                         "mean_pos": np.round(z["pos"][gi].mean(0), 1).tolist() if len(gi) else None}
        print(json.dumps(out))


if __name__ == "__main__":
    main()
