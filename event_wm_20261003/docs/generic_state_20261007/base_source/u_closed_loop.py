#!/usr/bin/env python3
"""Unified backend, step 5: one closed loop for every task family (official OGBench tasks).

Reader -> entity states of the current frame and of the goal image -> batched weighted A* over events
(u_wm.Model.plan) -> the skill executes the first event (e, x) -> the event ends when identity e is at
rest again (reader at-rest head for m frames, free of the agent) and either it moved by more than one
object width from the event start (place identity: its appearance changed) or it reached its target
x within the goal tolerance; or when another identity moved / changed and is at rest (a knock) -> replan. Timeout per event: replay the mean post-event action of
the play data, then replan. No privileged input; simulator states are logged for analysis only.
--state-layout (state track, s_entities.py): the env observation is the state vector; entity states come from its
object blocks instead of the reader, and "at rest" = unchanged since the previous step (noise radii) and the end
effector farther than half an object width (the event-extraction definition); the skill reads observation vectors.
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
    state = a.state_layout is not None
    if not state:
        rk = torch.load(a.reader, map_location="cpu", weights_only=False)
        reader = make_reader(K, agent=rk.get("agent", False)).to(dev).eval(); reader.load_state_dict(rk["reader"])
    sk = torch.load(a.skill, map_location="cpu", weights_only=False)
    pi = make_skill(K, chunk=sk["chunk"], cond=sk.get("cond", "full"), obs_dim=sk.get("obs_dim")).to(dev).eval(); pi.load_state_dict(sk["skill"])
    if state:
        obs_mu, obs_sd = np.tile(sk["obs_mean"], 2), np.tile(sk["obs_std"], 2)
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

    has_agent = (not state) and bool(rk.get("agent", False))
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

    if state:
        from s_entities import state_entities
        Ls = json.loads(Path(a.state_layout).read_text())
        last = {"S": None}

        def read_goal(frame):
            return state_entities(np.asarray(frame, np.float32)[None], Ls)[0][0]

        def observe(frame):
            """State track: entity states from the observation's object blocks; at rest = unchanged since the previous
            step and the end effector clear of the entity (half an object width)."""
            S_, eff = state_entities(np.asarray(frame, np.float32)[None], Ls)
            S_, eff = S_[0], eff[0]
            P_ = S_ if last["S"] is None else last["S"]
            still = (np.linalg.norm(S_[:, :2] - P_[:, :2], axis=-1) <= Ls["r_pos"]) & (np.abs(S_[:, 2:5] - P_[:, 2:5]).max(-1) <= Ls["r_app"])
            last["S"] = S_
            return S_, still & (np.linalg.norm(S_[:, :2] - eff[None], axis=-1) > M.sc.thr_pos / 2)
    else:
        def read_goal(frame):
            return observe(frame)[0]

    def changed(s1, s0):
        """A move is a displacement of more than one object width (reader jitter under the gripper stays below
        it); a place identity changes by its appearance."""
        return (np.linalg.norm(s1[:, :2] - s0[:, :2], axis=-1) > M.sc.thr_pos) | (np.abs(s1[:, 2:5] - s0[:, 2:5]).max(-1) > app_tol)

    def arrived(s1, e, x):
        return np.linalg.norm(s1[e, :2] - np.asarray(x)[:2]) <= M.sc.tol_pos and np.abs(s1[e, 2:5] - np.asarray(x)[2:5]).max() <= app_tol[e]

    env = gymnasium.make(a.env)
    u = env.unwrapped

    def sim_state(goal=False):
        """PRIVILEGED simulator state, logged for analysis only (never read by the loop): cube positions / button
        states (goal=True: the task's targets). None if the env exposes neither."""
        try:
            s = {}
            if hasattr(u, "_num_cubes"):
                s["cubes"] = np.round(np.stack([u._data.mocap_pos[u._cube_target_mocap_ids[i]] if goal else u._data.joint(f"object_joint_{i}").qpos[:3]
                                                for i in range(u._num_cubes)]), 4).tolist()
            if hasattr(u, "_cur_button_states"):
                s["buttons"] = np.asarray(u._target_button_states if goal else u._cur_button_states).astype(int).tolist()
            return s or None
        except Exception:
            return None

    episodes = []
    for task, epi in jobs:
        seed = a.seed * 10000 + task * 100 + epi
        ob, info = env.reset(seed=seed, options=dict(task_id=task, render_goal=False))
        if not state and (ob.mean() < 20 or info["goal"].mean() < 20):
            raise RuntimeError("rendering looks broken")
        G = read_goal(info["goal"])
        if state:
            last["S"] = None
        S, rest = observe(ob)
        hist = [S]
        t_plan = time.time()
        plan, pinfo = M.plan(S, G, max_expansions=a.max_expansions)
        rec = {"task": task, "episode": epi, "first_plan": None if plan is None else len(plan), "plan_info": {k: v for k, v in pinfo.items() if k != "best_plan"},
               "plan_sec": round(time.time() - t_plan, 2), "events": [], "replans": 0, "timeouts": 0, "success": False, "steps": 0,
               "sim_start": sim_state(), "sim_goal": sim_state(goal=True), "read_start": np.round(S, 3).tolist(), "read_goal": np.round(G, 3).tolist(),
               "timeout_log": []}
        rec["first_events"] = [{"e": int(e), "x": np.asarray(x).tolist()} for e, x in (plan or pinfo.get("best_plan") or [])]
        if plan is None:
            plan = pinfo.get("best_plan") or []
        frames, queue = [ob] * (gap + 1), []
        start_state, rest_count, k_steps = S.copy(), np.zeros(K, int), 0
        predicted_next = None
        done = False
        while not done:
            if plan:
                e, x = plan[0]
                if not queue:
                    if k_steps == 0:
                        predicted_next = M.step(start_state, [(e, x)])[0]
                    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                        if state:
                            px = torch.as_tensor((np.concatenate([frames[0], frames[-1]], -1) - obs_mu) / obs_sd, device=dev)[None].float()
                        else:
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
                                          "steps": k_steps, "others_changed": int(sum(ch_[j] for j in range(K) if j != e)),
                                          "t": rec["steps"], "read_start": np.round(start_state, 3).tolist(),
                                          "wm_predicted_next": None if predicted_next is None else np.round(predicted_next, 3).tolist(),
                                          "read_end": np.round(S, 3).tolist(), "sim_end": sim_state()})
            if end or k_steps > a.timeout:
                if not end:
                    queue = [recover.copy() for _ in range(a.recover_steps)]
                    rec["timeouts"] += 1
                    if plan:
                        rec["timeout_log"].append({"e": int(plan[0][0]), "x": np.round(np.asarray(plan[0][1]), 3).tolist(), "t": rec["steps"],
                                                   "read_end": np.round(S, 3).tolist(), "sim_end": sim_state()})
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
        save_json(a.out / "episodes" / f"task{task}_episode{epi}.json", rec)
        print(json.dumps({k: rec[k] for k in ("task", "episode", "first_plan", "success", "steps", "replans", "timeouts")}), flush=True)
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="e.g. visual-cube-triple-v0, visual-puzzle-4x5-v0, visual-scene-v0")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--reader", type=Path, default=None, help="pixel track: the entity reader")
    ap.add_argument("--state-layout", type=Path, default=None, help="state track: layout.json from s_entities.py (no reader)")
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
    arm = ("UNIFIED, STATE TRACK: object-factored state observation -> entity event WM + A* + event skill on state vectors"
           if a.state_layout is not None else "UNIFIED: SAM 2 entities -> reader + entity event WM + A* + event skill (no privileged input)")
    summary = {"arm": arm, "env": a.env,
               "success": float(np.mean([e["success"] for e in episodes])),
               "by_task": {t: float(np.mean([e["success"] for e in episodes if e["task"] == t])) for t in a.tasks},
               "first_plan_found": float(np.mean([e["first_plan"] is not None for e in episodes])),
               "minutes": round((time.time() - t0) / 60, 1), "args": {k: str(v) for k, v in vars(a).items()}}
    save_json(a.out / "u_closed_loop.json", {"summary": summary, "episodes": episodes})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
