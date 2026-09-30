#!/usr/bin/env python3
"""Can per-block feedback fix rollout error once the objective's arrival time is freed?

Context: two frames (motion state) for every arm. Arms (3 CEM seeds, paired):
  rh5_term     : released structure, execute 5 blocks, terminal cost at block 5;
  rh1_term     : replan every block, terminal cost at block 5 (known to collapse);
  rh1_min      : replan every block, cost = min over predicted blocks of goal distance;
  rh1_shrink   : replan every block, terminal cost at block min(5, blocks left);
  rh1_min_c6   : rh1_min with 6 CEM iterations (same WM evaluations as rh5_term).
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

BUDGET_BLOCKS = 10
SEEDS = 3


class MinOverTime:
    """Goal distance of the closest predicted block (arrival at any step)."""

    def __call__(self, info):
        pred = info["predicted_emb"]
        t = info["action_candidates"].shape[2]
        fut = pred[..., -t:, :]
        goal = info["goal_emb"][:, None, -1:, :]
        return ((fut - goal.detach()) ** 2).mean(-1).min(-1).values


def make_solver(swm, task, horizon, n_steps, objective):
    # ShootingCostEvaluator calls objective(info); a plain callable is enough.
    cost = swm.planning.ShootingCostEvaluator(task.model, objective or swm.planning.GoalMSE())
    s = swm.planning.CEMSolver(cost=cost, batch_size=1, num_samples=300, n_steps=n_steps, topk=30,
                               var_scale=1.0, device=task.device, seed=0)
    cfg = swm.PlanConfig(horizon=horizon, receding_horizon=horizon, action_block=5, history_len=2)
    s.configure(action_space=task.world.envs.action_space, n_envs=1, config=cfg)
    return s


def solve(solver, prepared, seed):
    import torch

    solver.callbacks = []
    solver.torch_gen.manual_seed(int(seed))
    with torch.inference_mode():
        return solver.solve(dict(prepared))["actions"][0].numpy()


def block_raw(task, plan_blocks):
    return task.scaler.inverse_transform(
        np.asarray(plan_blocks, np.float32).reshape(-1, task.raw_dim)).astype(np.float32)


def run_arm(task, solvers, root, init, goal, goal_image, frames0, block0, s, arm):
    task.restore(root, init, goal)
    frames, blocks = list(frames0), [block0]
    term = False
    if arm == "rh5_term":
        k = 0
        while k < BUDGET_BLOCKS and not term:
            p = solve(solvers[("term", 5, 30)],
                      L.prepared_hist(task, frames[-2:], goal_image, np.stack(blocks[-1:])),
                      L.seed_of(root.root, s, k))
            for j in range(5):
                term, _ = task.execute(block_raw(task, p[j:j + 1]))
                if term:
                    break
                frames.append(task.render())
                blocks.append(p[j])
            k += 5
    else:
        for k in range(BUDGET_BLOCKS):
            if arm == "rh1_term":
                key = ("term", 5, 30)
            elif arm == "rh1_min":
                key = ("min", 5, 30)
            elif arm == "rh1_min_c6":
                key = ("min", 5, 6)
            elif arm == "rh1_shrink":
                key = ("term", min(5, BUDGET_BLOCKS - k), 30)
            else:
                raise ValueError(arm)
            p = solve(solvers[key],
                      L.prepared_hist(task, frames[-2:], goal_image, np.stack(blocks[-1:])),
                      L.seed_of(root.root, s, k))
            term, _ = task.execute(block_raw(task, p[:1]))
            if term:
                break
            frames.append(task.render())
            blocks.append(p[0])
    d = task.distance(goal)
    return bool(task.success(term, d)), float(d)


ARMS = ("rh5_term", "rh1_term", "rh1_min", "rh1_shrink", "rh1_min_c6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("cube", "reacher"), required=True)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=50)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
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
    solvers = {("term", h, 30): make_solver(swm, task, h, 30, None) for h in range(1, 6)}
    solvers[("min", 5, 30)] = make_solver(swm, task, 5, 30, MinOverTime())
    solvers[("min", 5, 6)] = make_solver(swm, task, 5, 6, MinOverTime())
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
            task.restore(root, {"qpos": hist["qpos"][L.BLOCK], "qvel": hist["qvel"][L.BLOCK]}, goal)
            f_prev = task.render()
            task.restore(root, init, goal)
            f_now = task.render()
            acts = np.asarray(hist["action"][L.BLOCK: 2 * L.BLOCK], np.float32)
            block0 = task.scaler.transform(acts).astype(np.float32).reshape(-1)
            res = {"root": rid}
            for arm in ARMS:
                out_arm = [run_arm(task, solvers, root, init, goal, goal_image, [f_prev, f_now],
                                   block0, s, arm) for s in range(SEEDS)]
                res[arm] = [o[0] for o in out_arm]
                res[arm + "_dist"] = [o[1] for o in out_arm]
            res["seconds"] = time.perf_counter() - t
            path.write_text(json.dumps(res) + "\n")
            print(json.dumps({k: sum(v) for k, v in res.items() if k in ARMS} | {"root": rid,
                              "sec": round(res["seconds"], 1)}), flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()
