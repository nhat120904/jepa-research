#!/usr/bin/env python3
"""Render the official evaluation tasks with the current MUJOCO_GL backend and compare with dataset frames.

Writes tasks_<size>.npz (same format as inventory.py) and a PNG strip, and prints pixel statistics
next to those of dataset frames. Used to pick a rendering backend that matches the training data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4x5")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import gymnasium
    import ogbench  # noqa: F401
    from PIL import Image

    env = gymnasium.make(f"visual-puzzle-{a.size}-v0")
    store = []
    for task_id in range(1, 6):
        ob, info = env.reset(seed=task_id, options=dict(task_id=task_id, render_goal=False))
        u = env.unwrapped
        store.append(dict(task_id=task_id, init_ob=ob, goal_ob=info["goal"], init=u._cur_button_states.copy(),
                          goal=u._target_button_states.copy()))
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / f"tasks_{a.size}.npz", **{k: np.stack([s[k] for s in store]) for k in store[0]})
    data = np.asarray(np.load(a.cache / "val_observations.npy", mmap_mode="r")[:2000])
    env_frames = np.stack([s["init_ob"] for s in store] + [s["goal_ob"] for s in store])
    rep = {"backend": os.environ.get("MUJOCO_GL"), "env_mean": float(env_frames.mean()),
           "env_channel_mean": env_frames.reshape(-1, 3).mean(0).round(2).tolist(),
           "data_mean": float(data.mean()), "data_channel_mean": data.reshape(-1, 3).mean(0).round(2).tolist()}
    strip = np.concatenate([store[0]["init_ob"], store[0]["goal_ob"], data[0], data[1000]], 1)
    Image.fromarray(strip).resize((strip.shape[1] * 4, strip.shape[0] * 4), Image.NEAREST).save(a.out / "strip.png")
    (a.out / "render_check.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
