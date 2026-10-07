#!/usr/bin/env python3
"""Closed-loop evaluation on the official OGBench visual-puzzle tasks.

Method (all learned, pixels only): encode the current and goal frames to event codes, plan
an event sequence with batch weighted A* (event WM + cost-to-go), execute the first event
with the skill policy, detect the event as a stable code change, then replan.

Attribution arms (debug, labelled PRIVILEGED in the output):
  --low scripted   execute each planned event with the scripted press controller on the true
                   button that the event type maps to (isolates the learned high level);
  --high oracle    plan with the true GF(2) minimal press set, mapped to event types
                   (isolates the learned low level).
Every episode also logs privileged diagnostics: the true button pressed by each detected
event versus the commanded one.
--random-goals N (overfitting check): instead of the 5 fixed official tasks, N fresh problems with
random start and goal configurations (official ones excluded), passed to OGBench as task_info so the
goal frame is rendered exactly as for the official tasks. Difficulty k = number of random distinct
presses, cycled over 1..d_max where d_max = the largest d* among the official tasks of that size
(same step-limit regime as the official protocol); d* is the GF(2) minimum (scoring only).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import load_code, save_json, size_of, tokens
from inventory import PressController
from lightsout import min_press_set
from planner import bwas, make_costtogo, make_event_wm
from train_skill import make_skill
from train_skill2 import make_skill2
from train_skill3 import make_skill3
from train_reader import make_reader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="e.g. visual-puzzle-4x5-play-v0 (dataset name)")
    ap.add_argument("--planner", type=Path, required=True)
    ap.add_argument("--skill", type=Path, default=None)
    ap.add_argument("--code", type=Path, default=None, help="override the planner's code (e.g. refine_code.py output)")
    ap.add_argument("--reader", type=Path, default=None, help="CNN code reader on pixels (train_reader.py); same bit order")
    ap.add_argument("--episodes", type=int, default=20, help="per task")
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--low", choices=["skill", "scripted"], default="skill")
    ap.add_argument("--high", choices=["learned", "oracle"], default="learned")
    ap.add_argument("--tau0", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=80, help="steps per event before replanning")
    ap.add_argument("--stable", type=int, default=3, help="per-bit debounce frames (build_events uses 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1, help="parallel episode workers (rendering is CPU-bound)")
    ap.add_argument("--max-expansions", type=int, default=50000, help="search budget per plan")
    ap.add_argument("--lam", type=float, default=0.6, help="weighted A*: f = lam * g + h")
    ap.add_argument("--exec-steps", type=int, default=4, help="skill v2: actions executed per predicted chunk")
    ap.add_argument("--recover-steps", type=int, default=0,
                    help="on a timeout, replay the mean post-event (lift-off) action of the play data for N steps")
    ap.add_argument("--events", type=Path, default=None, help="build_events output (for --recover-steps)")
    ap.add_argument("--cache", type=Path, default=None, help="cache/<env> (for --recover-steps)")
    ap.add_argument("--random-goals", type=int, default=0, help="N random-goal problems instead of the official tasks")
    ap.add_argument("--pending-max", type=int, default=30,
                    help="frames a code change that no event explains is ignored before it is accepted (0 = no filter)")
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    jobs = [(0, e) for e in range(a.random_goals)] if a.random_goals else [(t, e) for t in a.tasks for e in range(a.episodes)]
    t_start = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        chunks = [jobs[i::a.workers] for i in range(a.workers)]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            episodes = [r for part in pool.starmap(run, [(a, c) for c in chunks if c]) for r in part]
    else:
        episodes = run(a, jobs)
    episodes.sort(key=lambda r: (r["task"], r["episode"]))
    summary = {"arms": {"low": a.low, "high": a.high, "privileged": a.low == "scripted" or a.high == "oracle"},
               "success": float(np.mean([r["success"] for r in episodes])),
               "by_task": {t: float(np.mean([r["success"] for r in episodes if r["task"] == t])) for t in a.tasks}
               if not a.random_goals else None,
               "minutes": round((time.time() - t_start) / 60, 1), "args": {k: str(v) for k, v in vars(a).items()}}
    if a.random_goals:
        ds = np.array([r["dstar"] for r in episodes]); sc = np.array([r["success"] for r in episodes], float)
        summary["n"] = len(episodes)
        summary["by_dstar_bin"] = {f"{lo}-{hi}": [int(((ds >= lo) & (ds <= hi)).sum()), float(sc[(ds >= lo) & (ds <= hi)].mean())]
                                   for lo, hi in ((1, 5), (6, 10), (11, 15), (16, 20), (21, 30)) if ((ds >= lo) & (ds <= hi)).any()}
        summary["optimal_presses_frac"] = float(np.mean([r["success"] and len(r["events"]) == r["dstar"] for r in episodes]))
    save_json(a.out / "closed_loop.json", {"summary": summary, "episodes": episodes})
    print(json.dumps(summary), flush=True)


def run(a, jobs):
    """Run the given (task, episode) pairs; models are loaded once per worker."""
    os.environ.setdefault("LP_NUM_THREADS", "1")
    import gymnasium
    import ogbench  # noqa: F401
    import torch

    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    np.random.seed(a.seed)
    dev = "cuda"
    rows, cols = size_of(a.env)
    pk = torch.load(a.planner, map_location="cpu", weights_only=False)
    E, K = pk["events"], pk["bits"]
    wm = make_event_wm(K, E).to(dev).eval()
    wm.load_state_dict(pk["wm"])
    h = make_costtogo(K, pk.get("h_hidden", 512), pk.get("h_xor", False)).to(dev).eval()
    h.load_state_dict(pk["h"])
    mask = np.asarray(pk["bit_mask"], bool)
    type_to_button = pk["type_to_button"]                       # privileged: scripted arm + diagnostics
    button_to_type = {}
    for e, bt in enumerate(type_to_button):
        button_to_type.setdefault(bt, e)
    enc, code_logits, _ = load_code(a.code or Path(pk["code"]), dev)   # --code: refined code, same bit order
    if a.low == "skill":
        sk = torch.load(a.skill, map_location="cpu", weights_only=False)
        ver = sk.get("version", 1)
        v2 = ver == 2
        if ver == 3:
            pi = make_skill3(sk["events"], sk["tmaps"], chunk=sk["chunk"])
        elif v2:
            pi = make_skill2(sk["events"], sk["tmaps"], chunk=sk["chunk"])
        else:
            pi = make_skill(sk["events"])
        pi = pi.to(dev).eval()
        pi.load_state_dict(sk["skill"])
        gap = sk.get("hist_gap", 2)
        if ver < 3:
            stmu, stsd = sk["token_mean"].to(dev), sk["token_std"].to(dev)
        amu, asd = sk["action_mean"], sk["action_std"]
        no_tau = bool(sk.get("no_tau", False))

    reader = None
    if a.reader is not None:                                    # CNN code reader on pixels (train_reader.py)
        rk = torch.load(a.reader, map_location="cpu", weights_only=False)
        reader = make_reader(rk["bits"]).to(dev).eval()
        reader.load_state_dict(rk["reader"])

    def read_logits(ob):
        with torch.no_grad():
            if reader is not None:
                px = torch.as_tensor(ob, device=dev).permute(2, 0, 1)[None].float() / 255.0
                return reader(px).float()
            return code_logits(tokens(enc, ob[None], dev)).float()

    def observe(ob):
        with torch.no_grad():
            tok = tokens(enc, ob[None], dev)
            code = (read_logits(ob) > 0).cpu().numpy()[0].astype(np.uint8)[mask]
        return tok, code

    def plan_from(code, goal_code, true_state, goal_state):
        if a.high == "oracle":
            x = min_press_set(true_state, goal_state, rows, cols)
            return [button_to_type[i] for i in np.nonzero(x)[0] if i in button_to_type], {"found": True, "expanded": 0}
        path, info = bwas(code, goal_code, wm, h, dev, lam=a.lam, batch=256, max_expansions=a.max_expansions)
        if path is None:                                     # no plan found: greedy event by cost-to-go
            with torch.no_grad():
                ch = wm.successors(torch.as_tensor(code, device=dev)[None])[0]
                hv = h(ch, torch.as_tensor(goal_code, device=dev)[None].expand_as(ch))
            path, info["greedy"] = [int(hv.argmin())], True
        return path, info

    env = gymnasium.make(a.env.replace("-play", ""))
    recover = None
    if a.recover_steps > 0:
        # Mean action over the 5 frames after each detected event: the lift-off that follows a press (label-free).
        ev = np.load(a.events / "events_train.npz")
        acts = np.load(a.cache / "train_actions.npy", mmap_mode="r")
        idx = (ev["t"][:, None] + 1 + np.arange(5)[None]).ravel()
        recover = np.clip(np.asarray(acts[idx[idx < len(acts)]]).mean(0), -1, 1)
    episodes = []
    if a.random_goals:
        from lightsout import solver
        A_ = solver(rows, cols)[0]
        official = []
        for t in range(1, 6):
            env.reset(seed=0, options=dict(task_id=t))
            uu = env.unwrapped
            official.append((uu._cur_button_states.astype(np.uint8).copy(), uu._target_button_states.astype(np.uint8).copy()))
        d_max = max(int(min_press_set(i_, g_, rows, cols).sum()) for i_, g_ in official)

        def random_task(e):
            rg = np.random.default_rng(1_000_003 + 1000 * a.seed + e)
            k_ = 1 + e % d_max
            while True:
                init = rg.integers(0, 2, rows * cols).astype(np.uint8)
                x = np.zeros(rows * cols, np.uint8); x[rg.choice(rows * cols, k_, replace=False)] = 1
                goal = (init ^ (A_ @ x % 2)).astype(np.uint8)
                if not any((init == i_).all() and (goal == g_).all() for i_, g_ in official):
                    return dict(init_button_states=init.astype(np.int64), goal_button_states=goal.astype(np.int64))
    for task, epi in jobs:
        if True:
            opts = dict(task_info=random_task(epi)) if a.random_goals else dict(task_id=task)
            ob, info = env.reset(seed=a.seed * 10000 + task * 100 + epi, options=dict(**opts, render_goal=False))
            u = env.unwrapped
            if ob.mean() < 20 or info["goal"].mean() < 20:   # black frames (CPU-node OSMesa, job 56924)
                raise RuntimeError(f"rendering looks broken: frame mean {ob.mean():.1f}, goal mean {info['goal'].mean():.1f}")
            goal_state = u._target_button_states.copy()
            _, goal_code = observe(info["goal"])
            gl = read_logits(info["goal"]).cpu().numpy()[0][mask]     # goal-code confidence (diagnostics)
            tok, cur = observe(ob)
            path, pinfo = plan_from(cur, goal_code, u._cur_button_states.copy(), goal_state)
            rec = {"task": task, "episode": epi, "dstar": int(min_press_set(u._cur_button_states, goal_state, rows, cols).sum()),
                   "first_plan": None if path is None else len(path), "plan_found": pinfo["found"],
                   "events": [], "replans": 0, "success": False, "steps": 0,
                   "init_state": u._cur_button_states.tolist(), "goal_state": goal_state.tolist(),
                   "init_code": cur.tolist(), "goal_code": goal_code.tolist(), "goal_logits": np.round(gl, 3).tolist()}
            k, ctl = 0, None
            hist, queue = [tok] * (3 if a.low != "skill" else gap + 1), []   # skill v2: token history, action chunk
            frames = [ob] * (3 if a.low != "skill" else gap + 1)               # skill v3: raw frame history
            deb = cur.copy()                                   # per-bit debounced code
            pend_val, pend_n = cur.copy(), np.zeros_like(cur, dtype=np.int64)
            pending = 0
            rec["ignored_frames"], rec["forced_events"] = 0, 0
            prev_true = u._cur_button_states.copy()
            done = False
            while not done:
                if not path:
                    action = np.zeros(5)
                    action[4] = 1
                else:
                    e = path[0]
                    if a.low == "skill" and ver == 3:
                        if not queue:
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                px = torch.as_tensor(np.concatenate([frames[0], frames[-1]], -1), device=dev)
                                px = px.permute(2, 0, 1)[None].float() / 255.0
                                ch = pi(px, torch.tensor([e], device=dev)).float().cpu().numpy()[0]
                            queue = list(np.clip(ch[: a.exec_steps] * asd + amu, -1, 1))
                        action = queue.pop(0)
                    elif a.low == "skill" and v2:
                        if not queue:
                            with torch.no_grad():
                                th = (torch.stack([hist[0], hist[-1]], 1) - stmu) / stsd          # (1, 2, 64, 192)
                                ch = pi(th, torch.tensor([e], device=dev)).cpu().numpy()[0]       # (chunk, 5)
                            queue = list(np.clip(ch[: a.exec_steps] * asd + amu, -1, 1))
                        action = queue.pop(0)
                    elif a.low == "skill":
                        tau = 0 if no_tau else max(a.tau0 - k, 0)
                        with torch.no_grad():
                            out = pi((tok - stmu) / stsd, torch.tensor([e], device=dev),
                                     torch.tensor([tau], device=dev)).cpu().numpy()[0]
                        action = np.clip(out * asd + amu, -1, 1)
                    else:
                        if ctl is None:
                            ctl = PressController(env, type_to_button[e])
                        action = ctl.act()
                        if action is None:                       # pressed and lifted; wait for the code
                            action = np.zeros(5)
                            action[4] = 1
                ob, _, term, trunc, info = env.step(action)
                rec["steps"] += 1
                k += 1
                if info["success"]:
                    rec["success"] = True
                done = term or trunc or rec["success"]
                tok, c = observe(ob)
                hist = hist[1:] + [tok]
                frames = frames[1:] + [ob]
                # Per-bit debounce, as in build_events: a bit flips after holding its new value `stable` frames.
                diff = c != deb
                pend_n = np.where(diff & (c == pend_val), pend_n + 1, np.where(diff, 1, 0))
                pend_val = np.where(diff, c, deb)
                flipb = pend_n >= a.stable
                if flipb.any():
                    deb = np.where(flipb, c, deb).astype(np.uint8)
                    pend_n[flipb] = 0
                # Event-consistency filter: accept a change only if the event WM can produce it from the
                # confirmed code (occlusion flips and partial reveals wait); force it after pending_max frames.
                event = False
                if (deb != cur).any():
                    with torch.no_grad():
                        succ = wm.successors(torch.as_tensor(cur, device=dev)[None])[0].cpu().numpy()
                    if (succ == deb[None]).all(1).any() or a.pending_max == 0:
                        event = True
                    else:
                        pending += 1
                        rec["ignored_frames"] += 1
                        if pending > a.pending_max:
                            event = True
                            rec["forced_events"] += 1
                else:
                    pending = 0
                if event:
                    new_code = deb.copy()
                    pending = 0
                if event or k > a.timeout:
                    if event:
                        true_now = u._cur_button_states.copy()
                        tog = np.nonzero(true_now != prev_true)[0].tolist()
                        rec["events"].append({"step": rec["steps"], "commanded": None if not path else int(path[0]),
                                              "commanded_button": None if not path else int(type_to_button[path[0]]),
                                              "true_toggled": tog})
                        prev_true = true_now
                        cur = new_code
                    k, ctl, queue = 0, None, []
                    if not event and recover is not None:
                        queue = [recover.copy() for _ in range(a.recover_steps)]   # lift off, then retry
                        rec["recoveries"] = rec.get("recoveries", 0) + 1
                    if not done:
                        path, _ = plan_from(cur, goal_code, u._cur_button_states.copy(), goal_state)
                        rec["replans"] += 1
            episodes.append(rec)
            print(json.dumps({x: rec[x] for x in ("task", "episode", "dstar", "first_plan", "success", "steps", "replans")}),
                  flush=True)
    return episodes


if __name__ == "__main__":
    main()
