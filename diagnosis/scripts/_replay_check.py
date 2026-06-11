"""Decisive world-equality test: replay a dataset trajectory's raw actions in
the live env and compare (a) the proprio trajectory — physics/world equality —
and (b) pixel frames at matched timesteps — camera equality with identical arm
poses (markers/object still differ via rand_vec, but the arm dominates).

python scripts/_replay_check.py --config configs/diagnostic_metaworld.yaml --task mw-reach
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import decord  # noqa: F401  — ffmpeg DLLs for torchcodec (HF video decode)
import yaml
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders import iterate_metaworld_trajectories  # noqa: E402

FRAMESKIP = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="mw-reach")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tb = next(iter(iterate_metaworld_trajectories(
        cfg["dataset"]["root"], [args.task], max_trajectories_per_task=1)))
    actions = tb.action.numpy()            # (T-1, 4) raw, per raw step
    ds_prop = tb.proprio.numpy()           # (T, 4)  [ee xyz, gripper]
    ds_frames = tb.obs_visual.numpy()      # (T, 3, H, W) [0,255]
    print(f"dataset traj: {actions.shape[0]} actions, proprio[0]={ds_prop[0].round(3)}")

    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module
    cl = import_module("18_closed_loop_eval")
    env, obs = cl.make_env(args.task, seed=10000)
    print(f"env init proprio  ={obs[:4].round(3)}")

    env_prop = [obs[:4].copy()]
    env_frames = [cl.render(env)]
    for a in actions:
        obs, _, _, _, _ = env.step(np.clip(a, -1, 1))
        env_prop.append(obs[:4].copy())
        env_frames.append(cl.render(env))
    env.close()
    env_prop = np.stack(env_prop)

    n = min(len(env_prop), len(ds_prop))
    err = np.linalg.norm(env_prop[:n, :3] - ds_prop[:n, :3], axis=1)
    print(f"ee-pos replay error: t0={err[0]:.4f} median={np.median(err):.4f} "
          f"max={err.max():.4f}  (same world+physics => ~1e-3)")
    print(f"gripper err median: {np.median(np.abs(env_prop[:n,3]-ds_prop[:n,3])):.4f}")

    out = ROOT / "results" / "logs"
    pairs = []
    for t in (0, 25, 50, 75):
        if t < n:
            dsf = np.transpose(ds_frames[t], (1, 2, 0)).astype(np.uint8)
            evf = env_frames[t].astype(np.uint8)
            pairs.append(np.concatenate([dsf, evf], axis=1))
    imageio.imwrite(out / "replay_pairs.png", np.concatenate(pairs, axis=0))
    print("wrote results/logs/replay_pairs.png (rows: t=0,25,50,75; left=dataset right=env)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
