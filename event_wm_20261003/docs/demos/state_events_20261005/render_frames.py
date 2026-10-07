#!/usr/bin/env python3
"""Demo export (2026-10-05): render val episodes of the OGBench STATE datasets from their saved simulator states
(qpos, qvel; button states for puzzle) with the env's front pixel camera, for the event demo of the state track.
Read-only on the datasets; no method code and no policy involved."""
import sys
from pathlib import Path

import numpy as np
import gymnasium
import ogbench  # noqa: F401

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/mnt/data/nhatnc129/jepa/ogbench/data")
W = H = 160
for env_id, ds in (("cube-triple-v0", "cube-triple-play-v0"), ("puzzle-4x5-v0", "puzzle-4x5-play-v0")):
    z = np.load(DATA / f"{ds}-val.npz")
    term, qpos, qvel = z["terminals"], z["qpos"], z["qvel"]
    bs = z["button_states"] if "button_states" in z.files else None
    ends = np.nonzero(term)[0]; starts = np.r_[0, ends[:-1] + 1]
    env = gymnasium.make(env_id, ob_type="pixels", width=W, height=H, visualize_info=False)
    env.reset(seed=0)
    u = env.unwrapped
    for ep in (0, 1):
        a, b = int(starts[ep]), int(ends[ep]) + 1
        frames = []
        for t in range(a, b):
            if bs is not None:
                u.set_state(qpos[t], qvel[t], bs[t])
            else:
                u.set_state(qpos[t], qvel[t])
            frames.append(np.asarray(u.render(), np.uint8))
        np.savez_compressed(OUT / f"{ds}_val_ep{ep}.npz", frames=np.stack(frames), t0=a)
        print(ds, ep, a, b, frames[0].shape, flush=True)
print("done", flush=True)
