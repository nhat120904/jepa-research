#!/usr/bin/env python3
"""PRIVILEGED diagnostic (cube envs): position noise of colour-anchored identities at rest.

For each identity -> cube (PRIVILEGED map from discover.json) and consecutive sampled frames in which that
cube did not move (qpos), the displacement of the identity's position under two measurements:
  sam-union       the front end's position (union of SAM segments near the continuity pick);
  colour-support  centroid of the pixels within thr_pos of it whose colour is within the merge radius of
                  any of the identity's colour clusters (all faces and shades).
Also the depth sensitivity: px per cm of the measured v coordinate vs the cube's y, at rest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="identity-stage dir with front/")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=2000)
    a = ap.parse_args()
    d = json.loads((a.run / "front" / "discover.json").read_text())
    E = np.load(a.run / "front" / "entities_train.npz")
    obs = np.load(a.cache / "train_observations.npy", mmap_mode="r")
    term = np.load(a.cache / "train_terminals.npy"); q = np.load(a.cache / "train_qpos.npy", mmap_mode="r")
    raw = {r["cluster"]: np.array(r["rgb"]) / 255.0 for r in d["raw_clusters"]}
    merge, thr = d["colour_merge"], d["thr_pos"]
    ep = np.concatenate([[0], np.cumsum(term[:-1])])
    pf = np.nonzero(E["processed"])[0][: a.max_frames]
    imgs = np.asarray(obs[pf]).astype(np.float32) / 255.0
    v_, u_ = np.mgrid[0:64, 0:64]
    out = []
    for j, ident in enumerate(d["identities"]):
        if ident["anchor"] != "colour":
            continue
        diag = d["privileged_diagnostic"][j] if j < len(d["privileged_diagnostic"]) else None
        if not diag or diag["median_err_cm"] > 5:
            continue
        cube = diag["cube"]
        cl = [c for t in d["types"] if t["type"] in ident["types"] for c in t["clusters"]]
        C = np.stack([raw[c] for c in cl if c in raw])
        P_sam = E["pos"][pf, j].astype(np.float64)
        vis = E["area"][pf, j] >= 1
        dc = np.linalg.norm(imgs[:, :, :, None, :] - C[None, None, None], axis=-1).min(-1)          # (F, 64, 64)
        win = np.hypot(u_[None] - P_sam[:, 0, None, None], v_[None] - P_sam[:, 1, None, None]) <= thr
        m = win & (dc < merge) & vis[:, None, None]
        cnt = m.sum((1, 2))
        P_col = np.stack([(m * u_[None]).sum((1, 2)), (m * v_[None]).sum((1, 2))], -1) / np.maximum(cnt, 1)[:, None]
        okc = cnt >= 3
        xyz = np.asarray(q[pf, 14 + 7 * cube:17 + 7 * cube])
        i0, i1 = np.arange(len(pf) - 1), np.arange(1, len(pf))
        same = ep[pf[i0]] == ep[pf[i1]]
        still = np.linalg.norm(xyz[i1] - xyz[i0], axis=-1) < 1e-3
        r = {"identity": j, "cube": cube}
        for name, P, ok in (("sam_union", P_sam, vis), ("colour_support", P_col, okc)):
            sel = same & still & ok[i0] & ok[i1]
            dp = np.linalg.norm(P[i1] - P[i0], axis=-1)[sel]
            r[name] = {"static_jitter_px_p50_p90_p99": np.round(np.percentile(dp, [50, 90, 99]), 2).tolist(), "pairs": int(sel.sum())}
            # depth sensitivity at rest: least squares v ~ a + b x + c y over resting frames
            rest = ok & np.r_[still, False]
            if rest.sum() > 30:
                A = np.c_[np.ones(rest.sum()), xyz[rest, 0] * 100, xyz[rest, 1] * 100]
                for ax, nm in ((0, "u"), (1, "v")):
                    coef = np.linalg.lstsq(A, P[rest, ax], rcond=None)[0]
                    r[name][f"{nm}_px_per_cm_x_y"] = np.round(coef[1:], 3).tolist()
        out.append(r)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
