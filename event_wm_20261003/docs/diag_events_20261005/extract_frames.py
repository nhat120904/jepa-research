#!/usr/bin/env python3
"""PRIVILEGED diagnostic export (2026-10-05): raw val frames of the first episodes of a family, for the visual
audit of front-end identities. Read-only on the cache; never used by the method."""
import sys
from pathlib import Path
import numpy as np

C = Path("/mnt/data/nhatnc129/jepa/event_wm/cache")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
for env, tag, n_ep in (("visual-cube-double-play-v0", "cubedouble", 2), ("visual-puzzle-4x6-play-v0", "puzzle46", 1),
                       ("visual-puzzle-4x5-play-v0", "puzzle45", 1)):
    o = np.load(C / env / "val_observations.npy", mmap_mode="r"); term = np.load(C / env / "val_terminals.npy")
    s = np.r_[0, np.nonzero(term)[0] + 1]; e = np.r_[np.nonzero(term)[0] + 1, len(term)]
    for k in range(n_ep):
        np.save(OUT / f"obs_{tag}_val_ep{k}.npy", np.asarray(o[s[k]:e[k]]))
        print(tag, k, s[k], e[k], flush=True)
print("done", flush=True)
