#!/usr/bin/env python3
"""Unified backend, step 5: one closed loop for every task family (official OGBench tasks).

Reader -> entity states of the current frame and of the goal image -> batched weighted A* over events
(u_wm.Model.plan) -> the skill executes the first event (e, x) -> the event ends when identity e is at
rest again (reader at-rest head for m frames, free of the agent) and either it moved by more than one
object width from the event start (place identity: its appearance changed) or it reached its target
x within the goal tolerance; or when another identity moved / changed and is at rest (a knock) -> replan. Timeout per event: replay the mean post-event action of
the play data, then replan. No privileged input; simulator states are logged for analysis only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import save_json
from u_reader import make_reader
from u_skill import make_skill
from u_wm import Model


def run(a, jobs):
    os.environ.setdefault("LP_NUM_THREADS", "1")
    import gymnasium
    import ogbench  # noqa: F401
    import torch

    torch.set_num_threads(1)
    dev = a.device
    M = Model(torch.load(a.model, map_location="cpu", weights_only=False), dev)
    K = M.K
    rk = torch.load(a.reader, map_location="cpu", weights_only=False)
    reader = make_reader(K, agent=rk.get("agent", False)).to(dev).eval(); reader.load_state_dict(rk["reader"])
    sk = torch.load(a.skill, map_location="cpu", weights_only=False)
    pi = make_skill(K, chunk=sk["chunk"], cond=sk.get("cond", "full")).to(dev).eval(); pi.load_state_dict(sk["skill"])
    amu, asd, gap = sk["action_mean"], sk["action_std"], sk["hist_gap"]
    e2k = None
    if a.cube_skill is not None:
        # ATTRIBUTION ARM (not the unified method): the cube-specific label-free skill (k, q) in this loop;
        # identities are matched to its objects by chromaticity of their colour clusters
        from train_cube_skill import make_cube_skill
        ck = torch.load(a.cube_skill, map_location="cpu", weights_only=False)
        pi = make_cube_skill(ck["K"], chunk=ck["chunk"], lo=ck["lo"], hi=ck["hi"], support=ck.get("support", False)).to(dev).eval()
        pi.load_state_dict(ck["skill"])
        amu, asd, gap = ck["action_mean"], ck["action_std"], ck["hist_gap"]
        cd = json.loads((a.cube_discover / "discover.json").read_text())
        g = np.array([np.mean(x, 0) for x in cd["group_rgb"]], np.float64) / 255.0
        ud = json.loads((a.model.parent.parent / "front" / "discover.json").read_text())
        raw = {r["cluster"]: np.array(r["rgb"]) / 255.0 for r in ud["raw_clusters"]}
        tcl = {t["type"]: t["clusters"] for t in ud["types"]}
        chroma = lambda c: c / c.sum(-1, keepdims=True)
        e2k = []
        for idn in ud["identities"]:
            c = np.mean([raw[c_] for t in idn["types"] for c_ in tcl[t] if c_ in raw], 0)
            e2k.append(int(np.argmin(np.linalg.norm(chroma(g) - chroma(c), axis=-1))))
        print({"cube_skill_map": e2k}, flush=True)
    recover = np.zeros(5)
    if a.cache is not None:
        ev = np.load(a.events / "events_train.npz")
        acts = np.load(a.cache / "train_actions.npy", mmap_mode="r")
        idx = (ev["t"][:, None] + 1 + np.arange(5)[None]).ravel()
        recover = np.clip(np.asarray(acts[idx[idx < len(acts)]]).mean(0), -1, 1)
    app_tol = M.app_tol

    has_agent = bool(rk.get("agent", False))
    vg, ug = np.mgrid[0:64, 0:64]

    def observe(frame):
        """-> state (K, 6), at rest (K,): the rest head, and -- with an agent head -- free of the agent (no agent
        pixel within half an object width), the same definition as the rest labels (u_events contact-free)."""
        with torch.no_grad():
            x = torch.as_tensor(frame, device=dev).permute(2, 0, 1)[None].float() / 255.0
            out = reader(x, return_agent=has_agent)
            st, rest = out[0], out[1]
            st = st[0].float().cpu().numpy(); st[:, 5] = 1 / (1 + np.exp(-st[:, 5]))
            at_rest = rest[0].float().cpu().numpy() > 0
            if has_agent:
                am = out[2][0].float().cpu().numpy() > 0
                if am.any():
                    d = np.hypot(ug[am][:, None] - st[:, 0][None], vg[am][:, None] - st[:, 1][None]).min(0)
                    at_rest &= d > M.sc.thr_pos / 2
            return st, at_rest

    def changed(s1, s0):
        """A move is a displacement of more than one object width (reader jitter under the gripper stays below
        it); a place identity changes by its appearance."""
        return (np.linalg.norm(s1[:, :2] - s0[:, :2], axis=-1) > M.sc.thr_pos) | (np.abs(s1[:, 2:5] - s0[:, 2:5]).max(-1) > app_tol)

    def arrived(s1, e, x):
        return np.linalg.norm(s1[e, :2] - np.asarray(x)[:2]) <= M.sc.tol_pos and np.abs(s1[e, 2:5] - np.asarray(x)[2:5]).max() <= app_tol[e]

    env = gymnasium.make(a.env)
    episodes = []
    for task, epi in jobs:
        seed = a.seed * 10000 + task * 100 + epi
        ob, info = env.reset(seed=seed, options=dict(task_id=task, render_goal=False))
        if ob.mean() < 20 or info["goal"].mean() < 20:
            raise RuntimeError("rendering looks broken")
        G, _ = observe(info["goal"])
        S, rest = observe(ob)
        hist = [S]
        t_plan = time.time()
        plan, pinfo = M.plan(S, G, max_expansions=a.max_expansions)
        rec = {"task": task, "episode": epi, "first_plan": None if plan is None else len(plan), "plan_info": {k: v for k, v in pinfo.items() if k != "best_plan"},
               "plan_sec": round(time.time() - t_plan, 2), "events": [], "replans": 0, "timeouts": 0, "success": False, "steps": 0}
        if plan is None:
            plan = pinfo.get("best_plan") or []
        frames, queue = [ob] * (gap + 1), []
        start_state, rest_count, k_steps = S.copy(), np.zeros(K, int), 0
        done = False
        while not done:
            if plan:
                e, x = plan[0]
                if not queue:
                    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                        px = torch.as_tensor(np.concatenate([frames[0], frames[-1]], -1), device=dev).permute(2, 0, 1)[None].float() / 255.0
                        if e2k is not None:
                            ch = pi(px, torch.tensor([e2k[e]], device=dev), torch.as_tensor(np.asarray(x)[None, :2], device=dev).float()).float().cpu().numpy()[0]
                        else:
                            ch = pi(px, torch.tensor([e], device=dev), torch.as_tensor(start_state[e][None], device=dev).float(),
                                    torch.as_tensor(np.asarray(x)[None], device=dev).float()).float().cpu().numpy()[0]
                    queue = list(np.clip(ch[: a.exec_steps] * asd + amu, -1, 1))
                action = queue.pop(0)
            else:
                action = np.zeros(5)
            ob, _, term, trunc, info = env.step(action)
            rec["steps"] += 1; k_steps += 1
            frames = frames[1:] + [ob]
            if info["success"]:
                rec["success"] = True
            done = term or trunc or rec["success"]
            S, rest = observe(ob)
            hist = (hist + [S])[-3:]
            rest_count = np.where(rest, rest_count + 1, 0)
            end = False
            if plan:
                e, x = plan[0]
                ch_ = changed(S, start_state)
                settled = rest_count >= a.m
                if (settled[e] and (ch_[e] or arrived(S, e, x))) or any(ch_[j] and settled[j] for j in range(K) if j != e):
                    end = True
                    rec["events"].append({"e": int(e), "x": np.round(np.asarray(x), 3).tolist(), "final": np.round(S[e], 3).tolist(),
                                          "steps": k_steps, "others_changed": int(sum(ch_[j] for j in range(K) if j != e))})
            if end or k_steps > a.timeout:
                if not end:
                    queue = [recover.copy() for _ in range(a.recover_steps)]
                    rec["timeouts"] += 1
                else:
                    queue = []
                S_plan = np.mean(hist, 0)
                start_state, k_steps, rest_count = S_plan.copy(), 0, np.zeros(K, int)
                if not done:
                    plan, pinfo = M.plan(S_plan, G, max_expansions=a.max_expansions)
                    if plan is None:
                        plan = pinfo.get("best_plan") or []
                    rec["replans"] += 1
        episodes.append(rec)
        print(json.dumps({k: rec[k] for k in ("task", "episode", "first_plan", "success", "steps", "replans", "timeouts")}), flush=True)
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="e.g. visual-cube-triple-v0, visual-puzzle-4x5-v0, visual-scene-v0")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--reader", type=Path, required=True)
    ap.add_argument("--skill", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=250)
    ap.add_argument("--exec-steps", type=int, default=4)
    ap.add_argument("--recover-steps", type=int, default=8)
    ap.add_argument("--max-expansions", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cube-skill", type=Path, default=None, help="ATTRIBUTION: cube-specific skill checkpoint (train_cube_skill.py)")
    ap.add_argument("--cube-discover", type=Path, default=None, help="its discover dir (object colours)")
    ap.add_argument("--device", default="cuda", help="cpu for smoke tests")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    jobs = [(t, e) for t in a.tasks for e in range(a.episode_start, a.episode_start + a.episodes)]
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        chunks = [jobs[i::a.workers] for i in range(a.workers)]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            episodes = [r for part in pool.starmap(run, [(a, c) for c in chunks if c]) for r in part]
    else:
        episodes = run(a, jobs)
    episodes.sort(key=lambda r: (r["task"], r["episode"]))
    summary = {"arm": "UNIFIED: SAM 2 entities -> reader + entity event WM + A* + event skill (no privileged input)", "env": a.env,
               "success": float(np.mean([e["success"] for e in episodes])),
               "by_task": {t: float(np.mean([e["success"] for e in episodes if e["task"] == t])) for t in a.tasks},
               "first_plan_found": float(np.mean([e["first_plan"] is not None for e in episodes])),
               "minutes": round((time.time() - t0) / 60, 1), "args": {k: str(v) for k, v in vars(a).items()}}
    save_json(a.out / "u_closed_loop.json", {"summary": summary, "episodes": episodes})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
