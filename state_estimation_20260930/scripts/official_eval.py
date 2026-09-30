#!/usr/bin/env python3
"""Upstream LeWM evaluation (le-wm/eval.py logic, World.evaluate) with start-state switches.

Reproduces the released evaluator for reacher / pusht / cube / tworoom: episode
sampling with rng(seed) over valid starts, dataset start image for the first
plan, CEM 300/30/30, horizon 5 = receding 5, frameskip 5, budget 50.
Switches (all else identical):
  --history H   PlanConfig.history_len (released: 1). Upstream fills history
                from executed steps only, so the first plan sees one frame.
  --prefill     seed the history with the dataset frames/actions that preceded
                the start, so the first plan also sees H frames.
  --innov-gamma G, --innov-mode M   add G * (one-step innovation of the newest
                context frame) to the rollout (needs --history 4); see se/models.py.
  --min-start   restrict sampling to starts with at least this many prior steps
                (needed for --prefill; use the same value for every paired arm).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("reacher", "pusht", "cube", "tworoom"))
    ap.add_argument("--history", type=int, required=True)
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("--min-start", type=int, default=0)
    ap.add_argument("--innov-gamma", type=float, default=0.0)
    ap.add_argument("--innov-mode", default="dist", choices=("dist", "post"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model and physics work must run under sbatch")
    import torch

    import stable_worldmodel as swm
    from se import tasks as T
    from se.policy import load_prefill, make_policy_class

    c = T.TASKS[a.task]
    world = swm.World(c["env"], num_envs=a.num_eval, image_shape=(224, 224),
                      max_episode_steps=2 * T.BUDGET, **c["env_kw"])
    tf = T.image_transform()
    dataset = T.load_dataset(swm, a.task)
    process = T.fit_processors(dataset, a.task)
    model = swm.wm.utils.load_pretrained(c["repo"]).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if a.innov_gamma:
        from se.models import enable_innovation

        enable_innovation(model, a.innov_gamma, a.innov_mode)
    solver = swm.planning.CEMSolver(
        cost=swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE()),
        batch_size=1, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device="cuda", seed=a.seed)
    episodes, starts = T.sample_episodes(dataset, a.seed, a.num_eval, a.min_start)
    prefill = None
    if a.prefill:
        if a.history < 2:
            raise ValueError("--prefill needs --history >= 2")
        prefill = load_prefill(dataset, episodes, starts, (a.history - 1) * T.BLOCK)
    Policy = make_policy_class(swm)
    policy = Policy(solver=solver,
                    config=swm.PlanConfig(horizon=T.HORIZON, receding_horizon=T.HORIZON,
                                          action_block=T.BLOCK, history_len=a.history),
                    process=process, transform={"pixels": tf, "goal": tf}, prefill=prefill)
    world.set_policy(policy)
    t = time.time()
    with torch.inference_mode():
        metrics = world.evaluate(dataset=dataset, start_steps=starts, goal_offset=T.GOAL_OFFSET,
                                 eval_budget=T.BUDGET, episodes_idx=episodes, callables=c["callables"])
    out = {"task": a.task, "history": a.history, "prefill": a.prefill, "min_start": a.min_start,
           "innov_gamma": a.innov_gamma, "innov_mode": a.innov_mode,
           "seed": a.seed, "success_rate": float(metrics["success_rate"]),
           "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
           "episodes": episodes, "start_steps": starts, "seconds": time.time() - t}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out) + "\n")
    print(json.dumps({k: out[k] for k in ("task", "history", "prefill", "innov_gamma", "innov_mode", "seed",
                                          "success_rate", "seconds")}),
          flush=True)
    world.close()


if __name__ == "__main__":
    main()
