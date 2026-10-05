#!/usr/bin/env python3
"""Diagnostic: per merged colour type, how often do its pixels change between consecutive sampled frames?

For segment s at sampled frame t (with t + stride in the same episode): fraction of its mask pixels whose
RGB changes by more than `tau` (max over channels, /255) between t and the next sampled frame. Per type:
mean of that fraction, and the fraction of segments with more than half of their pixels changed.
tau = 2-means split of log |dI| over all pixels of the sampled pairs (Ashman's D reported).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sfa_code import two_means_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--segments", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=1500)
    a = ap.parse_args()
    d = json.loads((a.front / "discover.json").read_text())
    z = np.load(a.segments / "segments_train.npz")
    raw = {r["cluster"]: np.array(r["rgb"]) / 255.0 for r in d["raw_clusters"]}
    mg = {r["cluster"]: r["merged_into"] for r in d["raw_clusters"]}
    ks = np.array(sorted(raw)); cent = np.stack([raw[k] for k in ks])
    dd = np.linalg.norm(z["col"][:, None] - cent[None], axis=-1)
    lab = np.array([mg[k] for k in ks[dd.argmin(1)]]); lab[dd.min(1) >= d["colour_merge"]] = -1
    fr = z["frames"][: a.max_frames]
    term = np.load(a.cache / "train_terminals.npy"); ep = np.concatenate([[0], np.cumsum(term[:-1])])
    obs = np.load(a.cache / "train_observations.npy", mmap_mode="r")
    stride = int(np.median(np.diff(z["frames"])))
    ok = np.array([t + stride < len(term) and ep[t + stride] == ep[t] for t in fr])
    fr = fr[ok]
    I0 = np.asarray(obs[fr]).astype(np.int16); I1 = np.asarray(obs[fr + stride]).astype(np.int16)
    dI = np.abs(I1 - I0).max(-1) / 255.0                                   # (F, 64, 64)
    x = np.log(dI[dI > 0] + 1e-6)
    thr, D, _ = two_means_threshold(x)
    tau = float(np.exp(thr))
    ch = dI > tau
    fidx = {t: i for i, t in enumerate(fr)}
    rows = [i for i in range(len(z["t"])) if z["t"][i] in fidx and lab[i] >= 0]
    frac = np.zeros(len(rows)); lb = lab[rows]
    for j, i in enumerate(rows):
        m = np.unpackbits(z["mask"][i], axis=-1)[:, :64].astype(bool)
        frac[j] = ch[fidx[z["t"][i]]][m].mean()
    out = {"tau": tau, "D": D, "changed_pixels_all": round(float(ch.mean()), 4), "types": []}
    for k in sorted(set(lb.tolist())):
        s = lb == k
        out["types"].append({"type": int(k), "segments": int(s.sum()), "mean_changed_frac": round(float(frac[s].mean()), 3),
                             "segs_majority_changed": round(float((frac[s] > 0.5).mean()), 3)})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
