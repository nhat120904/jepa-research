#!/usr/bin/env python3
"""Upstream LeWM evaluation (le-wm/eval.py logic, World.evaluate) with a history_len switch.

Reproduces the released Reacher/Cube evaluator: episode sampling with rng(seed)
over valid starts, dataset start image for the first plan, rendered frames
afterwards, CEM 300/30/30, horizon 5 = receding 5, budget 50. The only change
is PlanConfig.history_len. Upstream fills history from executed steps only, so
with history_len=3 the first plan still sees one frame (warm-up) and the second
plan sees three.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

CFG = {
    "reacher": {"env": "swm/ReacherDMControl-v0", "kw": {"task": "qpos_match"},
                "repo": "quentinll/lewm-reacher",
                "data": "/mnt/data/nhatnc129/jepa/lewm_stage0/downloads/lewm/lewm-reacher/extracted/reacher.h5",
                "callables": [{"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
                              {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}}]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="reacher", choices=list(CFG))
    ap.add_argument("--history", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model and physics work must run under sbatch")
    import stable_pretraining as spt
    import torch
    from sklearn import preprocessing
    from torchvision.transforms import v2 as transforms

    import stable_worldmodel as swm

    c = CFG[a.task]
    world = swm.World(c["env"], num_envs=a.num_eval, image_shape=(224, 224),
                      max_episode_steps=100, **c["kw"])
    tf = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                             transforms.Normalize(**spt.data.dataset_stats.ImageNet),
                             transforms.Resize(size=224)])
    dataset = swm.data.load_dataset(c["data"], keys_to_cache=["action"])
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    acts = dataset.get_col_data("action")
    scaler = preprocessing.StandardScaler().fit(acts[~np.isnan(acts).any(axis=1)])
    model = swm.wm.utils.load_pretrained(c["repo"]).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver = swm.planning.CEMSolver(
        cost=swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE()),
        batch_size=1, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device="cuda", seed=a.seed)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5,
                                             history_len=a.history),
        process={"action": scaler}, transform={"pixels": tf, "goal": tf})
    # Episode sampling exactly as le-wm/eval.py.
    ep_col = dataset.get_col_data(col)
    step_idx = dataset.get_col_data("step_idx")
    lengths = np.array([np.max(step_idx[ep_col == e]) + 1 for e in ep_indices])
    max_start = dict(zip(ep_indices, lengths - 25 - 1))
    valid = np.nonzero(step_idx <= np.array([max_start[e] for e in ep_col]))[0]
    g = np.random.default_rng(a.seed)
    pick = np.sort(valid[g.choice(len(valid) - 1, size=a.num_eval, replace=False)])
    rows = dataset.get_row_data(pick)
    world.set_policy(policy)
    t = time.time()
    metrics = world.evaluate(dataset=dataset, start_steps=rows["step_idx"].tolist(), goal_offset=25,
                             eval_budget=50, episodes_idx=rows[col].tolist(), callables=c["callables"])
    out = {"task": a.task, "history": a.history, "seed": a.seed,
           "success_rate": float(metrics["success_rate"]),
           "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
           "episodes": rows[col].tolist(), "start_steps": rows["step_idx"].tolist(),
           "seconds": time.time() - t}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out) + "\n")
    print(json.dumps({k: out[k] for k in ("task", "history", "seed", "success_rate", "seconds")}), flush=True)
    world.close()


if __name__ == "__main__":
    main()
