#!/usr/bin/env python3
"""Closed loop on the official OGBench visual-puzzle tasks with the scene-memory-v2 pipeline (pixels + actions only, no
labels): read the current and the goal frame with the sm2 reader (sm2_reader.py), plan an event sequence with batch
weighted A* in the event WM + cost-to-go of sm2_planner.py, execute the first event with skill v3 (sm2_skill.py),
detect the event as a debounced bit change that the WM can explain from the current code (otherwise accepted after
--pending-max frames), then replan. closed_loop.py without the token code.

Attribution arms (PRIVILEGED, flagged in the output):
  --low scripted   press, with the scripted controller, the true button that the planned type maps to (majority over
                   TRAIN events, planner_eval.json) -> isolates the learned high level;
  --high oracle    plan the GF(2) minimal press set, mapped to event types -> isolates the learned low level.
Privileged diagnostics per episode: the true lights toggled by each detected event and the commanded type's button.
--random-goals N: fresh problems (official ones excluded), as closed_loop.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import save_json, size_of
from lightsout import min_press_set
from planner import bwas, make_costtogo, make_event_wm
from train_reader import make_reader
from train_skill3 import make_skill3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="dataset name, e.g. visual-puzzle-4x5-play-v0")
    ap.add_argument("--planner", type=Path, required=True, help="sm2_planner.py planner.pt (planner_eval.json next to it)")
    ap.add_argument("--reader", type=Path, required=True, help="sm2_reader.py reader.pt")
    ap.add_argument("--skill", type=Path, default=None, help="sm2_skill.py skill_best.pt")
    ap.add_argument("--episodes", type=int, default=6, help="per task")
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--low", choices=["skill", "scripted"], default="skill")
    ap.add_argument("--high", choices=["learned", "oracle"], default="learned")
    ap.add_argument("--timeout", type=int, default=80, help="steps per event before replanning")
    ap.add_argument("--stable", type=int, default=3, help="per-bit debounce frames (sm2_reader events use 3)")
    ap.add_argument("--pending-max", type=int, default=30,
                    help="frames a code change that no event explains is ignored before it is accepted (0 = no filter)")
    ap.add_argument("--max-expansions", type=int, default=20000, help="search budget per plan")
    ap.add_argument("--lam", type=float, default=0.6, help="weighted A*: f = lam * g + h")
    ap.add_argument("--exec-steps", type=int, default=4, help="actions executed per predicted chunk")
    ap.add_argument("--random-goals", type=int, default=0, help="N random-goal problems instead of the official tasks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    if a.low == "skill" and a.skill is None:
        raise SystemExit("--low skill needs --skill")
    t0 = time.time()
    jobs = [(0, e) for e in range(a.random_goals)] if a.random_goals else [(t, e) for t in a.tasks for e in range(a.episodes)]
    episodes = run(a, jobs)
    summary = {"arms": {"low": a.low, "high": a.high, "privileged": a.low == "scripted" or a.high == "oracle"},
               "success": float(np.mean([r["success"] for r in episodes])),
               "by_task": None if a.random_goals else {t: float(np.mean([r["success"] for r in episodes if r["task"] == t])) for t in a.tasks},
               "first_plan_found": float(np.mean([r["plan_found"] for r in episodes])),
               "minutes": round((time.time() - t0) / 60, 1), "args": {k: str(v) for k, v in vars(a).items()}}
    if a.random_goals:
        ds = np.array([r["dstar"] for r in episodes]); sc = np.array([r["success"] for r in episodes], float)
        summary["n"] = len(episodes)
        summary["by_dstar_bin"] = {f"{lo}-{hi}": [int(((ds >= lo) & (ds <= hi)).sum()), float(sc[(ds >= lo) & (ds <= hi)].mean())]
                                   for lo, hi in ((1, 5), (6, 10), (11, 15), (16, 20), (21, 30)) if ((ds >= lo) & (ds <= hi)).any()}
    save_json(a.out / "sm2_loop.json", {"summary": summary, "episodes": episodes})
    print(json.dumps(summary), flush=True)


def run(a, jobs):
    os.environ.setdefault("LP_NUM_THREADS", "1")
    import gymnasium
    import ogbench  # noqa: F401
    import torch

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    rows, cols = size_of(a.env)
    pk = torch.load(a.planner, map_location="cpu", weights_only=False)
    E, K = pk["events"], pk["bits"]
    wm = make_event_wm(K, E).to(dev).eval()
    wm.load_state_dict(pk["wm"])
    h = make_costtogo(K, pk["h_hidden"], pk["h_xor"]).to(dev).eval()
    h.load_state_dict(pk["h"])
    pev = json.loads((a.planner.parent / "planner_eval.json").read_text())
    type_to_button = {int(k): int(v) for k, v in pev.get("type_to_button", {}).items()}     # PRIVILEGED: arms + diagnostics
    button_to_type = {}
    for e, bt in sorted(type_to_button.items()):
        if bt >= 0:
            button_to_type.setdefault(bt, e)
    rk = torch.load(a.reader, map_location="cpu", weights_only=False)
    if rk["bits"] != K:
        raise SystemExit(f"reader bits {rk['bits']} != planner bits {K}")
    reader = make_reader(K).to(dev).eval()
    reader.load_state_dict(rk["reader"])
    if a.low == "skill":
        sk = torch.load(a.skill, map_location="cpu", weights_only=False)
        if sk["events"] != E:
            raise SystemExit(f"skill events {sk['events']} != planner events {E}")
        pi = make_skill3(sk["events"], sk["tmaps"], chunk=sk["chunk"]).to(dev).eval()
        pi.load_state_dict(sk["skill"])
        gap, amu, asd = sk["hist_gap"], sk["action_mean"], sk["action_std"]
    else:
        from inventory import PressController
        gap = 2

    def read(ob):
        with torch.no_grad():
            px = torch.as_tensor(ob, device=dev).permute(2, 0, 1)[None].float().div_(255)
            lg = reader(px).float()
        return (lg > 0).cpu().numpy()[0].astype(np.uint8), lg.cpu().numpy()[0]

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

    episodes = []
    for task, epi in jobs:
        opts = dict(task_info=random_task(epi)) if a.random_goals else dict(task_id=task)
        ob, info = env.reset(seed=a.seed * 10000 + task * 100 + epi, options=dict(**opts, render_goal=False))
        u = env.unwrapped
        if ob.mean() < 20 or info["goal"].mean() < 20:
            raise RuntimeError(f"rendering looks broken: frame mean {ob.mean():.1f}, goal mean {info['goal'].mean():.1f}")
        goal_state = u._target_button_states.copy()
        goal_code, gl = read(info["goal"])
        cur, _ = read(ob)
        path, pinfo = plan_from(cur, goal_code, u._cur_button_states.copy(), goal_state)
        rec = {"task": task, "episode": epi, "dstar": int(min_press_set(u._cur_button_states, goal_state, rows, cols).sum()),
               "first_plan": None if path is None else len(path), "plan_found": bool(pinfo["found"]),
               "first_plan_expanded": int(pinfo.get("expanded", 0)), "events": [], "replans": 0, "success": False, "steps": 0,
               "init_state": u._cur_button_states.astype(int).tolist(), "goal_state": goal_state.astype(int).tolist(),
               "init_code": cur.tolist(), "goal_code": goal_code.tolist(), "goal_logit_min_abs": float(np.abs(gl).min()),
               "ignored_frames": 0, "forced_events": 0, "timeouts": 0}
        k, ctl, queue = 0, None, []
        frames = [ob] * (gap + 1)
        deb = cur.copy()                                       # per-bit debounced code
        pend_val, pend_n = cur.copy(), np.zeros(K, np.int64)
        pending = 0
        prev_true = u._cur_button_states.copy()
        done = False
        while not done:
            if not path:
                action = np.zeros(5); action[4] = 1
            elif a.low == "skill":
                if not queue:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        px = torch.as_tensor(np.concatenate([frames[0], frames[-1]], -1), device=dev).permute(2, 0, 1)[None].float() / 255.0
                        ch = pi(px, torch.tensor([path[0]], device=dev)).float().cpu().numpy()[0]
                    queue = list(np.clip(ch[: a.exec_steps] * asd + amu, -1, 1))
                action = queue.pop(0)
            else:
                if ctl is None and type_to_button.get(path[0], -1) >= 0:
                    ctl = PressController(env, type_to_button[path[0]])
                action = None if ctl is None else ctl.act()
                if action is None:                               # pressed and lifted; wait for the code
                    action = np.zeros(5); action[4] = 1
            ob, _, term, trunc, info = env.step(action)
            rec["steps"] += 1
            k += 1
            if info["success"]:
                rec["success"] = True
            done = term or trunc or rec["success"]
            c, _ = read(ob)
            frames = frames[1:] + [ob]
            diff = c != deb                                      # per-bit debounce, as in sm2_reader events
            pend_n = np.where(diff & (c == pend_val), pend_n + 1, np.where(diff, 1, 0))
            pend_val = np.where(diff, c, deb)
            flipb = pend_n >= a.stable
            if flipb.any():
                deb = np.where(flipb, c, deb).astype(np.uint8)
                pend_n[flipb] = 0
            event = False                                        # WM-consistency filter (closed_loop.py)
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
            if event or k > a.timeout:
                if event:
                    true_now = u._cur_button_states.copy()
                    rec["events"].append({"step": rec["steps"], "commanded": None if not path else int(path[0]),
                                          "commanded_button": None if not path else type_to_button.get(int(path[0]), -1),
                                          "true_toggled": np.nonzero(true_now != prev_true)[0].tolist(),
                                          "code_changed_bits": int((deb != cur).sum())})
                    prev_true = true_now
                    cur = deb.copy()
                    pending = 0
                else:
                    rec["timeouts"] += 1
                k, ctl, queue = 0, None, []
                if not done:
                    path, _ = plan_from(cur, goal_code, u._cur_button_states.copy(), goal_state)
                    rec["replans"] += 1
        rec["final_true_lights_wrong"] = int((u._cur_button_states != goal_state).sum())
        episodes.append(rec)
        print(json.dumps({x: rec[x] for x in ("task", "episode", "dstar", "first_plan", "plan_found", "success", "steps",
                                               "replans", "timeouts", "forced_events", "final_true_lights_wrong")}), flush=True)
    return episodes


if __name__ == "__main__":
    main()
