"""Frozen-model diagnostics. Oracle puzzle dynamics/distances are diagnostic controls only."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from lightsout import solver, min_press_set
from u_wm import Model

if "SLURM_JOB_ID" not in os.environ:
    raise RuntimeError("Use sbatch")
root = Path(os.environ["DIAG_DIR"])
out = root / ("job_" + os.environ["SLURM_JOB_ID"])
out.mkdir()
torch.set_num_threads(2)
torch.manual_seed(0)
np.random.seed(0)
checkpoint = Path("/mnt/data/nhatnc129/jepa/event_wm/state_puzzle-4x5-play-v0_57614/wm_57636/u_model.pt")
M = Model(torch.load(checkpoint, map_location="cpu", weights_only=False), "cuda")
episodes = json.loads((root / "dev.json").read_text())["episodes"]
roots = [ep for ep in episodes if ep["episode"] == 0]
A, reduced, piv, span = solver(4, 5)
orig_step, orig_h = M.step_batch, M.heuristic

def bits(s):
    return (np.asarray(s)[..., 2:5].mean(-1) > 0.5).astype(np.uint8)

def oracle_h(states, goal):
    delta = bits(states) ^ bits(goal)[None]
    transformed = (delta @ reduced[:, 20:].T) % 2
    xs = np.zeros((len(states), 20), np.uint8)
    xs[:, piv] = transformed[:, :len(piv)]
    result = (xs[:, None] ^ span[None]).sum(-1).min(-1).astype(np.float32)
    result[transformed[:, len(piv):].any(-1)] = 30.0
    return result

def oracle_step(states, events):
    nxt = np.array(states, np.float32, copy=True)
    ee = np.array([e for e, _ in events])
    nxt[..., 2:5] = (bits(states) ^ A[:, ee].T)[..., None]
    nxt[..., 5] = 0
    return nxt

def project(states):
    result = np.array(states, copy=True)
    for k, prototypes in enumerate(M.proto):
        ds = ((result[:, k, None, :2] - prototypes[None, :, :2]) / M.sc.thr_pos) ** 2
        da = ((result[:, k, None, 2:5] - prototypes[None, :, 2:5]) / M.app_tol[k]) ** 2
        best = (ds.sum(-1) + da.sum(-1)).argmin(-1)
        result[:, k] = prototypes[best]
    return result

def projected_step(states, events):
    return project(orig_step(states, events))

def measure(s, true, goal, start):
    return dict(h=float(orig_h(s[None], goal)[0]),
                exact_distance_of_thresholded_bits=int(oracle_h(s[None], goal)[0]),
                goal_test=bool(M.at_goal(s, goal)),
                wrong_bits=int((bits(s) != bits(true)).sum()),
                max_position_drift=float(np.linalg.norm(s[:, :2] - start[:, :2], axis=-1).max()),
                continuous_min=float(s[:, 2:5].min()), continuous_max=float(s[:, 2:5].max()),
                ambiguous_entities=int(((s[:, 2:5].mean(-1) > .2) & (s[:, 2:5].mean(-1) < .8)).sum()),
                blocked_entities=int((s[:, 5] > .5).sum()),
                max_state_error=float(np.abs(s-true).max()))

report = dict(job_id=os.environ["SLURM_JOB_ID"], checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              note="Saved development roots rounded to3 decimals. Oracle controls use simulator puzzle rule for attribution, not method results. Projection is nearest training rest-state prototype, no oracle transitions.",
              trajectories=[], searches=[])
for ep in roots:
    start = np.array(ep["read_start"], np.float32); goal = np.array(ep["read_goal"], np.float32)
    assert np.array_equal(bits(start), ep["sim_start"]["buttons"])
    assert np.array_equal(bits(goal), ep["sim_goal"]["buttons"])
    chosen = min_press_set(bits(start), bits(goal), 4, 5)
    assert chosen is not None
    order = np.flatnonzero(chosen).tolist()
    for mode in ("raw", "prototype_projected"):
        cur, true = start.copy(), start.copy()
        rows = [dict(depth=0, **measure(cur, true, goal, start))]
        for e in order:
            target = true[e].copy(); target[2:5] = 1 - target[2:5]
            teacher = orig_step(true[None], [(e, target)])[0]
            true = oracle_step(true[None], [(e, target)])[0]
            cur = orig_step(cur[None], [(e, target)])[0]
            if mode == "prototype_projected":
                cur = project(cur[None])[0]
            rows.append(dict(depth=len(rows), acted=e,
                             teacher_forced_wrong_bits=int((bits(teacher) != bits(true)).sum()),
                             **measure(cur, true, goal, start)))
        report["trajectories"].append(dict(task=ep["task"], mode=mode, reference_order=order, rows=rows))
        print("TRAJECTORY", ep["task"], mode, rows[-1], flush=True)

ep = next(ep for ep in roots if ep["task"] == 2)
start = np.array(ep["read_start"], np.float32); goal = np.array(ep["read_goal"], np.float32)
for name, step_fn, heuristic in (
    ("raw_WM_learned_h", orig_step, orig_h),
    ("oracle_WM_learned_h", oracle_step, orig_h),
    ("raw_WM_oracle_h", orig_step, oracle_h),
    ("oracle_WM_oracle_h", oracle_step, oracle_h),
    ("projected_WM_learned_h", projected_step, orig_h),
):
    M.step_batch, M.heuristic = step_fn, heuristic
    t0 = time.time(); plan, info = M.plan(start, goal, max_expansions=20000)
    selected = plan if plan is not None else info.get("best_plan", [])
    pred, real = start.copy(), start.copy()
    for ev in selected:
        pred = step_fn(pred[None], [ev])[0]
        real = oracle_step(real[None], [ev])[0]
    row = dict(mode=name, seconds=time.time()-t0, found=plan is not None,
               expanded=info["expanded"], plan_length=len(selected), best_h=info.get("best_h"),
               predicted_goal=bool(M.at_goal(pred, goal)), reference_goal=bool(M.at_goal(real, goal)),
               states_diagnostic=measure(pred, real, goal, start),
               commands=[dict(e=int(e), x=x.tolist()) for e,x in selected])
    report["searches"].append(row)
    print("SEARCH", name, {k:v for k,v in row.items() if k not in ("commands","states_diagnostic")},flush=True)
M.step_batch, M.heuristic = orig_step, orig_h
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
