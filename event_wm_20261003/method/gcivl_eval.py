#!/usr/bin/env python3
"""Flat image-goal baseline in our code path (V2_PLAN B2 flat-raw / easy-task path D_i): the gcivl.py actor given the
task's goal image directly, on the official OGBench visual tasks, with the episode seeds of closed_loop_objects.py
(matched comparison: seed * 10000 + task * 100 + episode). Success = info['success'] at any step within the env's horizon.
Outputs (--out): gcivl_eval.json (summary + per-episode success / steps).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--gcivl", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch / local/run_stage.ps1")
    import gymnasium
    import ogbench  # noqa: F401
    import torch

    from gcivl import make_nets

    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.gcivl, map_location="cpu", weights_only=False)
    _, actor = make_nets(ck["act_dim"]); actor.load_state_dict(ck["actor"]); actor = actor.to(a.device).eval()
    env = gymnasium.make(a.env)
    eps = []
    for task in a.tasks:
        for epi in range(a.episodes):
            ob, info = env.reset(seed=a.seed * 10000 + task * 100 + epi, options=dict(task_id=task, render_goal=False))
            goal = info["goal"]
            g = torch.as_tensor(goal, device=a.device).permute(2, 0, 1)[None].float() / 255.0
            done, success, steps = False, False, 0
            while not done:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.device == "cuda")):
                    x = torch.as_tensor(ob, device=a.device).permute(2, 0, 1)[None].float() / 255.0
                    act = np.clip(actor(x, g).float().cpu().numpy()[0], -1, 1)
                ob, _, term, trunc, info = env.step(act)
                steps += 1
                success = success or bool(info["success"])
                done = term or trunc or success
            eps.append({"task": task, "episode": epi, "success": success, "steps": steps})
            print(eps[-1], flush=True)
    by_task = {str(t): float(np.mean([e["success"] for e in eps if e["task"] == t])) for t in a.tasks}
    summ = {"arm": "FLAT BASELINE: gcivl.py actor on the task goal image (no planner)", "env": a.env, "success": float(np.mean([e["success"] for e in eps])),
            "by_task": by_task, "checkpoint": str(a.gcivl), "minutes": round((time.time() - t0) / 60, 1)}
    save_json(a.out / "gcivl_eval.json", {"summary": summ, "episodes": eps})
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
