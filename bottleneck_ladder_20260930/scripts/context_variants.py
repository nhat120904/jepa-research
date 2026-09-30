#!/usr/bin/env python3
"""Closed-loop context controls on Reacher (paired with ladder h1/h3 arms).

Variants (same CEM seeds as consensus_planning single30 / ladder h3):
  static3 : three copies of the current frame, zero (dataset-mean) action history
            -> same number of context frames as h3 but no motion information;
  h2      : frames t-5, t and one executed block;
  h1_rh1  : history 1, replan after every block (10 plans per episode);
  h3_rh1  : history 3, replan after every block.
Replanning variants use fewer seeds (they cost 5x more planning).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder as L  # noqa: E402


def plan(task, frames, goal_image, blocks, seed):
    import torch

    task.solver.callbacks = []
    task.solver.torch_gen.manual_seed(int(seed))
    with torch.inference_mode():
        return task.solver.solve(dict(L.prepared_hist(task, frames, goal_image, blocks)))["actions"][0].numpy()


def episode_rh5(task, root, init, goal, goal_image, ctx0, s, mode):
    """Standard receding horizon (whole 5-block plan), context by mode."""
    frames0, blocks0 = ctx0
    task.restore(root, init, goal)
    p1 = plan(task, frames0, goal_image, blocks0, L.seed_of(root.root, s, 0))
    raw = task.to_raw(p1)
    shots, term, steps = {}, False, 0
    for a in raw:
        _, _, term, _, _ = task.raw_env.step(a)
        steps += 1
        if term:
            break
        if steps in (15, 20):
            shots[steps] = task.render()
    if not term:
        f25 = task.render()
        if mode == "static3":
            f, b = [f25] * 3, np.zeros((2, p1.shape[1]), np.float32)
        elif mode == "h2":
            f, b = [shots[20], f25], p1[4:5]
        else:
            raise ValueError(mode)
        p2 = plan(task, f, goal_image, b, L.seed_of(root.root, s, 1))
        term, s2 = task.execute(task.to_raw(p2))
        steps += s2
    d = task.distance(goal)
    return bool(task.success(term, d))


def episode_rh1(task, root, init, goal, goal_image, hist_frames, hist_blocks, s, h):
    """Replan after every block; context = last h block-boundary frames."""
    task.restore(root, init, goal)
    frames = list(hist_frames)      # includes the current frame last
    blocks = list(hist_blocks)
    term = False
    for k in range(10):
        f = frames[-h:]
        b = np.stack(blocks[-(h - 1):]) if h > 1 else np.zeros((0, 1), np.float32)
        p = plan(task, f, goal_image, b, L.seed_of(root.root, s, k))
        raw = task.scaler.inverse_transform(p[:1].reshape(-1, task.raw_dim).astype(np.float32))
        term, _ = task.execute(raw)
        if term:
            break
        frames.append(task.render())
        blocks.append(p[0].astype(np.float32))
    d = task.distance(goal)
    return bool(task.success(term, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--rh1-seeds", type=int, default=3)
    ap.add_argument("--task", default="reacher", choices=("reacher", "cube"))
    ap.add_argument("--modes", default="static3,h2,h1_rh1,h3_rh1")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model and physics work must run under sbatch")
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import stable_worldmodel as swm

    from cemstop.tasks import TASKS, build_roots

    task = TASKS[a.task](swm)
    modes = a.modes.split(",")
    roots = build_roots(task.dataset, L.N_MANIFEST, L.MANIFEST_SEED)
    out = a.out_dir / a.task
    out.mkdir(parents=True, exist_ok=True)
    try:
        for rid in range(a.first + a.offset, a.last, a.stride):
            path = out / f"root_{rid:04d}.json"
            if path.exists():
                continue
            root = roots[rid]
            hist = L.history_rows(task, root)
            if hist is None:
                continue
            t = time.perf_counter()
            init, goal = task.rows(root)
            goal_image = np.asarray(goal["goal"])
            frames = []
            for row in (0, L.BLOCK):
                task.restore(root, {"qpos": hist["qpos"][row], "qvel": hist["qvel"][row]}, goal)
                frames.append(task.render())
            task.restore(root, init, goal)
            frames.append(task.render())
            acts = np.asarray(hist["action"][: L.BLOCK * (L.HIST - 1)], np.float32)
            blocks = task.scaler.transform(acts).astype(np.float32).reshape(L.HIST - 1, -1)
            ctx = {"static3": ([frames[-1]] * 3, np.zeros_like(blocks)),
                   "h2": (frames[-2:], blocks[-1:])}
            res = {"root": rid}
            for mode, c in ctx.items():
                if mode in modes:
                    res[mode] = [episode_rh5(task, root, init, goal, goal_image, c, s, mode)
                                 for s in range(L.M)]
            if "h1_rh1" in modes:
                res["h1_rh1"] = [episode_rh1(task, root, init, goal, goal_image, frames[-1:], [], s, 1)
                                 for s in range(a.rh1_seeds)]
            if "h3_rh1" in modes:
                res["h3_rh1"] = [episode_rh1(task, root, init, goal, goal_image, frames, list(blocks), s, 3)
                                 for s in range(a.rh1_seeds)]
            res["seconds"] = time.perf_counter() - t
            path.write_text(json.dumps(res) + "\n")
            print(json.dumps({k: (sum(v) if isinstance(v, list) else v) for k, v in res.items()}), flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()
