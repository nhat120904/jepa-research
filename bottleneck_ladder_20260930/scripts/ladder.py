#!/usr/bin/env python3
"""Oracle ladder for released LeWM planning on Cube/Reacher (development roots).

For the 20 plan-1 candidates of each root recorded by consensus_planning dev_v1
(10 seeds x CEM-3/CEM-30), score each candidate with
  L0  world-model cost, history 1 (released evaluator setting),
  L1  world-model cost, history 3 (context the predictor was trained with:
      frames at t-10, t-5, t and the two executed action blocks),
  L2  latent cost of the TRUE endpoint (execute, render, encode),
  L3  physical distance (already recorded),
and measure ranking (Spearman, top-1 success). Also closed loop with history 3
under the same 10 CEM seeds as the history-1 single30 arms (paired).
History frames are rendered from restored dataset states (same renderer as the
current frame); dataset action alignment is checked by replay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cem_stopping_20260929"))

MANIFEST_SEED = 20260929
N_MANIFEST = 300
CONS = Path("/mnt/data/nhatnc129/jepa/consensus_planning/dev_v1")
BASE_SEED = 7_000_000   # identical to consensus_planning seed_of
M = 10
HIST = 3
BLOCK = 5


def seed_of(root, s, plan):
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


def history_rows(task, root):
    """Dataset rows start-10 .. start (inclusive); None if unavailable."""
    if root.start_step < BLOCK * (HIST - 1):
        return None
    lo = root.start_step - BLOCK * (HIST - 1)
    chunk = task.dataset.load_chunk(np.array([root.episode]), np.array([lo]),
                                    np.array([root.start_step + 1]))[0]
    out = {}
    for k in ("qpos", "qvel", "action"):
        v = chunk[k]
        out[k] = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    return out


def prepared_hist(task, frames, goal_image, blocks_norm):
    import torch

    info = {"pixels": np.stack(frames)[None], "goal": np.asarray(goal_image)[None, None],
            "action": np.full((1, 1, task.raw_dim), np.nan, np.float32)}
    p = task.policy._prepare_info(info)
    if len(frames) > 1:
        p["action_history"] = torch.as_tensor(np.asarray(blocks_norm)[None], dtype=torch.float32)
    return p


def score(task, prepared, plans):
    """WM cost and predicted endpoint for each plan under the given context."""
    import torch

    cost = task.solver.cost
    m, dev, dtype = len(plans), task.device, task.solver.dtype
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
        goal = out["goal_emb"][0, -1].float().reshape(-1).cpu().numpy()
    return c, end, goal


def encode(task, images):
    import torch

    with torch.inference_mode():
        p = task.policy._prepare_info({"pixels": np.asarray(images)[:, None]})["pixels"]
        e = task.model.encode({"pixels": p.to(task.device)})["emb"][:, -1]
    return e.float().reshape(len(images), -1).cpu().numpy()


def spearman(a, b):
    a, b = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_root(task, root, cons_npz, cons_json):
    init, goal = task.rows(root)
    goal_image = np.asarray(goal["goal"])
    plans = np.concatenate([cons_npz["means3"], cons_npz["means30"]]).astype(np.float32)
    phys = np.concatenate([cons_npz["ol3_dist"], cons_npz["ol30_dist"]])
    succ = np.concatenate([cons_npz["ol3_succ"], cons_npz["ol30_succ"]]).astype(bool)

    task.restore(root, init, goal)
    frame_t = task.render()
    hist = history_rows(task, root)
    res = {"root": root.root, "start_step": root.start_step, "has_history": hist is not None}
    p1 = task.prepared(frame_t, goal_image)
    c_h1, e_h1, g_emb = score(task, p1, plans)
    res["h1_recompute_max_rel"] = float(np.max(np.abs(
        c_h1 - np.concatenate([cons_npz["cost3"], cons_npz["cost30"]])) / (np.abs(c_h1) + 1e-9)))
    frames, blocks = None, None
    if hist is not None:
        frames = []
        for row in (0, BLOCK):   # t-10, t-5
            task.restore(root, {"qpos": hist["qpos"][row], "qvel": hist["qvel"][row]}, goal)
            frames.append(task.render())
        frames.append(frame_t)
        acts = np.asarray(hist["action"][: BLOCK * (HIST - 1)], np.float32)
        blocks = task.scaler.transform(acts).astype(np.float32).reshape(HIST - 1, BLOCK * task.raw_dim)
        # Alignment check: replay the dataset actions from t-10 and compare with the state at t.
        task.restore(root, {"qpos": hist["qpos"][0], "qvel": hist["qvel"][0]}, goal)
        task.execute(acts)
        qpos = (task.raw_env._data.qpos if task.name == "cube"
                else task.raw_env.env.physics.data.qpos)
        res["replay_qpos_err"] = float(np.max(np.abs(np.asarray(qpos) - np.asarray(init["qpos"]))))
        c_h3, e_h3, _ = score(task, prepared_hist(task, frames, goal_image, blocks), plans)
    # True endpoints.
    ends = []
    for plan in plans:
        task.restore(root, init, goal)
        task.execute(task.to_raw(plan))
        ends.append(task.render())
    z_true = encode(task, np.stack(ends))
    c_true = np.sum((z_true - g_emb[None]) ** 2, axis=1)
    c_true_mse = c_true / z_true.shape[1]
    arrays = {"phys": phys, "succ": succ, "c_h1": c_h1, "c_true": c_true_mse,
              "err_h1": np.sum((e_h1 - z_true) ** 2, 1)}
    for name, c in (("h1", c_h1), ("true", c_true_mse)):
        res[f"rho_{name}"] = spearman(c, phys)
        res[f"top1_{name}"] = bool(succ[int(np.argmin(c))])
    if hist is not None:
        arrays.update({"c_h3": c_h3, "err_h3": np.sum((e_h3 - z_true) ** 2, 1)})
        res["rho_h3"] = spearman(c_h3, phys)
        res["top1_h3"] = bool(succ[int(np.argmin(c_h3))])
    res["top1_random"] = float(succ.mean())
    res["oracle_any"] = bool(succ.any())
    res["err_h1_median"] = float(np.median(arrays["err_h1"]))
    if hist is not None:
        res["err_h3_median"] = float(np.median(arrays["err_h3"]))
    return res, arrays, frames, blocks


def closed_loop_h3(task, root, frames, blocks, s):
    """CEM-30 episode with history-3 context, same seeds as consensus single30_s."""
    init, goal = task.rows(root)
    goal_image = np.asarray(goal["goal"])
    task.restore(root, init, goal)
    import torch

    task.solver.callbacks = []
    task.solver.torch_gen.manual_seed(seed_of(root.root, s, 0))
    with torch.inference_mode():
        plan1 = task.solver.solve(dict(prepared_hist(task, frames, goal_image, blocks)))["actions"][0].numpy()
    raw = task.to_raw(plan1)
    shots = {}
    term = False
    steps = 0
    for i, a in enumerate(raw):
        _, _, term, trunc, _ = task.raw_env.step(a)
        steps += 1
        if term or trunc:
            break
        if steps in (15, 20):
            shots[steps] = task.render()
    if not term:
        f2 = [shots[15], shots[20], task.render()]
        b2 = np.stack([plan1[3], plan1[4]]).astype(np.float32)
        task.solver.torch_gen.manual_seed(seed_of(root.root, s, 1))
        with torch.inference_mode():
            plan2 = task.solver.solve(dict(prepared_hist(task, f2, goal_image, b2)))["actions"][0].numpy()
        term, s2 = task.execute(task.to_raw(plan2))
        steps += s2
    d = task.distance(goal)
    return {"success": bool(task.success(term, d)), "distance": float(d), "steps": steps}


def main():
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model and physics work must run under sbatch")
    if args.last > 100:
        raise RuntimeError("development roots only")
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import stable_worldmodel as swm

    from cemstop.tasks import TASKS, build_roots

    task = TASKS[args.task](swm)
    roots = build_roots(task.dataset, N_MANIFEST, MANIFEST_SEED)
    out = args.out_dir / args.task
    out.mkdir(parents=True, exist_ok=True)
    try:
        for rid in range(args.first + args.offset, args.last, args.stride):
            path = out / f"root_{rid:04d}.json"
            if path.exists():
                continue
            t = time.perf_counter()
            cons_npz = dict(np.load(CONS / args.task / f"root_{rid:04d}.npz"))
            cons_json = json.loads((CONS / args.task / f"root_{rid:04d}.json").read_text())
            res, arrays, frames, blocks = run_root(task, roots[rid], cons_npz, cons_json)
            res["closed_h1"] = [cons_json["arms"][f"single30_{s}"]["success"] for s in range(M)]
            if frames is not None:
                res["closed_h3"] = [closed_loop_h3(task, roots[rid], frames, blocks, s)["success"]
                                    for s in range(M)]
            res["seconds"] = time.perf_counter() - t
            np.savez_compressed(out / f"root_{rid:04d}.npz", **arrays)
            path.write_text(json.dumps(res, indent=1) + "\n")
            print(json.dumps({k: res[k] for k in res if k.startswith(("rho", "top1", "closed", "replay", "h1_re"))}),
                  flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()
