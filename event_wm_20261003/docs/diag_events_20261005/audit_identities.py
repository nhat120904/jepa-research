#!/usr/bin/env python3
"""PRIVILEGED visual audit (2026-10-05, offline, local): which pixels the cube-double front end assigns to each colour
type, so the two extra identities (type 9 colour identity, type 4 place identity) can be checked by eye.

Inputs (copied from run 57420 and diag job 57586): front/discover.json, front/segments_val.npz, obs_cubedouble_val_ep*.npy.
Segments are labelled with the nearest raw-cluster colour of discover.json (display only; the front end used its
own unrounded centres). Output: audit_cubedouble_identities.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(sys.argv[1]); OUT = Path(sys.argv[2])
front = D / "family_visual-cube-double-play-v0_57420/front"
disc = json.loads((front / "discover.json").read_text())
z = np.load(front / "segments_val.npz")
obs = [np.load(D / "frames" / f"obs_cubedouble_val_ep{k}.npy") for k in (0, 1)]
raw = {r["cluster"]: np.array(r["rgb"]) / 255.0 for r in disc["raw_clusters"]}
rep = {r["cluster"]: r["merged_into"] for r in disc["raw_clusters"]}
cl_ids = np.array(sorted(raw)); cl_rgb = np.stack([raw[c] for c in cl_ids])
lab = cl_ids[np.linalg.norm(z["col"][:, None] - cl_rgb[None], axis=-1).argmin(1)]
typ = np.array([rep[c] for c in lab])
show = {0: ("type 0 = id 0 (blue cube)", (0.2, 0.6, 1.0)), 2: ("type 2 = id 1 (red cube)", (1.0, 0.2, 0.2)),
        9: ("type 9 = id 2 (extra)", (1.0, 0.0, 1.0)), 4: ("type 4 = id 3 (extra place)", (1.0, 1.0, 0.0)),
        6: ("type 6 = agent", (0.3, 1.0, 0.3)), 8: ("type 8 = agent", (0.0, 0.8, 0.8))}
place = next(i for i in disc["identities"] if i["anchor"] == "location")
t_all = z["t"]
# frames: where type 9 is present, and a few where it is absent, from val episodes 0 and 1
t9 = np.unique(t_all[(typ == 9) & (t_all < 2002)])
t_no9 = np.setdiff1d(np.unique(t_all[t_all < 2002]), t9)
rng = np.random.default_rng(0)
pick = list(rng.choice(t9, min(6, len(t9)), replace=False)) + list(rng.choice(t_no9, 2, replace=False))
pick = sorted(int(x) for x in pick)
fig, axes = plt.subplots(len(pick), 1 + len(show), figsize=(2.0 * (1 + len(show)), 2.1 * len(pick)))
vg, ug = np.mgrid[0:64, 0:64]
for r, t in enumerate(pick):
    fr = obs[t // 1001][t % 1001]
    axes[r, 0].imshow(fr, interpolation="nearest"); axes[r, 0].set_title(f"val frame {t}", fontsize=7)
    for c, (ty, (name, colr)) in enumerate(show.items(), start=1):
        img = fr.astype(np.float32) / 255.0 * 0.35
        for i in np.nonzero((t_all == t) & (typ == ty))[0]:
            m = np.unpackbits(z["mask"][i], axis=-1)[:, :64].astype(bool)
            img[m] = colr
        if ty == 4:
            d = np.hypot(ug - place["centre"][0], vg - place["centre"][1])
            img[np.abs(d - place["radius"]) < 0.5] = (1.0, 1.0, 1.0)
        axes[r, c].imshow(img, interpolation="nearest")
        if r == 0:
            axes[r, c].set_title(name, fontsize=7)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("cube-double front end (57420): pixels per colour type, val episodes 0-1 (white circle = place disc of id 3)", fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("saved", OUT, "frames", pick, "type9 frames", len(t9))
