#!/usr/bin/env python3
"""Consensus planning probe: M independent short CEM runs, execute the consensus plan.

Per development root and task, closed-loop episodes (two plans each) for:
- single30[s], single3[s], s = 0..M-1: one CEM run with seed s (30 / 3 iterations);
- medoid3: action-space medoid of the M 3-iteration plans (compute = 1 x 30);
- avg3: mean of the M 3-iteration plans;
- bestcost3: lowest world-model cost among the M 3-iteration plans;
- latmedoid3: medoid of the M predicted endpoint latents;
- medoid30: action medoid of M 30-iteration plans (M x compute, diagnostic).
Plan-1 open-loop: every plan-1 candidate/aggregate simulated for 25 steps.
Reuses the verified harness in ../cem_stopping_20260929 (import only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cem_stopping_20260929"))

MANIFEST_SEED = 20260929   # same roots as cem_stopping dev split
N_MANIFEST = 300
BASE_SEED = 7_000_000
M = 10
AGG3 = ("medoid3", "avg3", "bestcost3", "latmedoid3")


def seed_of(root: int, s: int, plan: int) -> int:
    return BASE_SEED + 1000 * root + 10 * s + plan


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("cube", "reacher"), required=True)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=100)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def make_solver3(swm, task):
    s = swm.planning.CEMSolver(cost=task.solver.cost, batch_size=1, num_samples=300, n_steps=3,
                               topk=30, var_scale=1.0, device=task.device, seed=0)
    s.configure(action_space=task.world.envs.action_space, n_envs=1, config=task.policy.cfg)
    return s


def solve(solver, prepared, seed, recorder=None):
    import torch

    solver.callbacks = [recorder] if recorder is not None else []
    solver.torch_gen.manual_seed(int(seed))
    with torch.inference_mode():
        return solver.solve(dict(prepared))["actions"][0].numpy()


def score_plans(task, prepared, plans: np.ndarray):
    """World-model cost and predicted endpoint latent of each plan (M, H, D)."""
    import torch

    cost = task.solver.cost
    dev = task.device
    m = len(plans)
    dtype = task.solver.dtype
    info = {}
    for k, v in prepared.items():
        if torch.is_tensor(v):
            vb = v[0:1].to(device=dev, dtype=dtype if v.is_floating_point() else None)
            info[k] = vb.unsqueeze(1).expand(1, m, *vb.shape[1:])
        elif isinstance(v, np.ndarray):
            info[k] = np.repeat(v[0:1][:, None, ...], m, axis=1)
    cand = torch.as_tensor(plans, dtype=dtype, device=dev)[None]
    with torch.inference_mode():
        out = cost._rollout(info, cand)
        c = cost.objective(out)[0].float().cpu().numpy()
        end = out["predicted_emb"][0, :, -1].float().reshape(m, -1).cpu().numpy()
    return c, end


def medoid(x: np.ndarray) -> int:
    x = x.reshape(len(x), -1).astype(np.float64)
    d = ((x[:, None] - x[None]) ** 2).sum(-1)
    return int(np.argmin(d.sum(1)))


def aggregate(name: str, plans: np.ndarray, costs, ends) -> np.ndarray:
    if name in ("medoid3", "medoid30"):
        return plans[medoid(plans)]
    if name == "avg3":
        return plans.mean(0).astype(np.float32)
    if name == "bestcost3":
        return plans[int(np.argmin(costs))]
    if name == "latmedoid3":
        return plans[medoid(ends)]
    raise ValueError(name)


def run_root(task, solver3, root) -> tuple[dict, dict]:
    from cemstop.recorder import PlanRecorder

    init, goal = task.rows(root)
    goal_image = np.asarray(goal["goal"])
    task.restore(root, init, goal)
    d0 = float(task.distance(goal))
    frame0 = task.render()
    prep0 = task.prepared(frame0, goal_image)
    t = time.perf_counter()
    means3, means30 = [], []
    for s in range(M):
        rec = PlanRecorder((3, 30))
        m30 = solve(task.solver, prep0, seed_of(root.root, s, 0), rec)
        assert np.array_equal(m30, rec.means[30])
        means3.append(rec.means[3])
        means30.append(m30)
    means3, means30 = np.stack(means3), np.stack(means30)
    plan1_seconds = time.perf_counter() - t
    c3, e3 = score_plans(task, prep0, means3)
    c30, _ = score_plans(task, prep0, means30)

    def multi_plan(prep, plan_idx, which):
        if which == "medoid30":
            ms = np.stack([solve(task.solver, prep, seed_of(root.root, s, plan_idx)) for s in range(M)])
            return aggregate(which, ms, None, None)
        ms = np.stack([solve(solver3, prep, seed_of(root.root, s, plan_idx)) for s in range(M)])
        c, e = score_plans(task, prep, ms) if which in ("bestcost3", "latmedoid3") else (None, None)
        return aggregate(which, ms, c, e)

    def episode(first_plan, second_planner):
        task.restore(root, init, goal)
        term, steps = task.execute(task.to_raw(first_plan))
        plan_s = 0.0
        if not term:
            frame = task.render()
            t0 = time.perf_counter()
            plan2 = second_planner(task.prepared(frame, goal_image))
            plan_s = time.perf_counter() - t0
            term, s2 = task.execute(task.to_raw(plan2))
            steps += s2
        d = task.distance(goal)
        return {"success": bool(task.success(term, d)), "distance": float(d), "steps": int(steps),
                "plan2_seconds": plan_s}

    arms = {}
    for s in range(M):
        arms[f"single30_{s}"] = episode(
            means30[s], lambda prep, s=s: solve(task.solver, prep, seed_of(root.root, s, 1)))
        arms[f"single3_{s}"] = episode(
            means3[s], lambda prep, s=s: solve(solver3, prep, seed_of(root.root, s, 1)))
    first = {"medoid3": aggregate("medoid3", means3, c3, e3),
             "avg3": aggregate("avg3", means3, c3, e3),
             "bestcost3": aggregate("bestcost3", means3, c3, e3),
             "latmedoid3": aggregate("latmedoid3", means3, c3, e3),
             "medoid30": aggregate("medoid30", means30, None, None)}
    for name, plan in first.items():
        arms[name] = episode(plan, lambda prep, name=name: multi_plan(prep, 1, name))

    # Plan-1 open loop: every candidate and aggregate from the root, 25 steps.
    def open_loop(plan):
        task.restore(root, init, goal)
        term, _ = task.execute(task.to_raw(plan))
        d = task.distance(goal)
        return float(d), bool(task.success(term, d))

    ol3 = np.array([open_loop(m) for m in means3])
    ol30 = np.array([open_loop(m) for m in means30])
    ol_agg = {k: open_loop(v) for k, v in first.items()}
    arrays = {"means3": means3, "means30": means30, "cost3": c3, "cost30": c30, "end3": e3,
              "ol3_dist": ol3[:, 0], "ol3_succ": ol3[:, 1].astype(bool),
              "ol30_dist": ol30[:, 0], "ol30_succ": ol30[:, 1].astype(bool)}
    summary = {"root": root.root, "episode": root.episode, "start_step": root.start_step,
               "start_distance": d0,
               "arms": arms, "plan1_open_loop_aggregates": ol_agg,
               "plan1_seconds_10x30": plan1_seconds,
               "idx": {"medoid3": medoid(means3), "latmedoid3": medoid(e3),
                       "bestcost3": int(np.argmin(c3)), "medoid30": medoid(means30)}}
    return arrays, summary


def main():
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model and physics work must run under sbatch")
    if args.last > 100:
        raise RuntimeError("development roots only (0-99)")
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import stable_worldmodel as swm

    from cemstop.tasks import TASKS, build_roots

    task = TASKS[args.task](swm)
    solver3 = make_solver3(swm, task)
    roots = build_roots(task.dataset, N_MANIFEST, MANIFEST_SEED)
    out = args.out_dir / args.task
    out.mkdir(parents=True, exist_ok=True)
    try:
        for rid in range(args.first + args.offset, args.last, args.stride):
            path = out / f"root_{rid:04d}.json"
            if path.exists():
                continue
            t = time.perf_counter()
            arrays, summary = run_root(task, solver3, roots[rid])
            np.savez_compressed(out / f"root_{rid:04d}.npz", **arrays)
            summary["seconds"] = time.perf_counter() - t
            path.write_text(json.dumps(summary, indent=1) + "\n")
            a = summary["arms"]
            print(json.dumps({"root": rid, "s30": sum(a[f"single30_{s}"]["success"] for s in range(M)),
                              "s3": sum(a[f"single3_{s}"]["success"] for s in range(M)),
                              **{k: a[k]["success"] for k in (*AGG3, "medoid30")},
                              "sec": round(summary["seconds"], 1)}), flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()
