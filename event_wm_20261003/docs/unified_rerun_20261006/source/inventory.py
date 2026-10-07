#!/usr/bin/env python3
"""Data inventory and evaluation harness for OGBench visual puzzle.

1. Play data (train split): press events per episode, steps between presses, lights toggled
   per event. Uses the privileged `button_states` array; it describes the data and is not an
   input to the method.
2. Evaluation harness: for every official evaluation task, reset the visual env, store the
   initial and goal observations (offline evaluation of the code and the planner), and
   execute a minimal press set with a scripted press controller. PRIVILEGED: the controller
   reads button positions from the simulator. This checks the harness and whether the press
   budget fits the episode limit; it is not a result of the method.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from lightsout import min_press_set

SIZES = {"3x3": (3, 3), "4x4": (4, 4), "4x5": (4, 5), "4x6": (4, 6)}


def data_stats(path: Path) -> dict:
    z = np.load(path)
    bs, term = z["button_states"], z["terminals"]
    ep = np.concatenate([[0], np.cumsum(term[:-1])])
    same = ep[1:] == ep[:-1]
    flips = (bs[1:] != bs[:-1]).sum(1)
    ev = np.nonzero((flips > 0) & same)[0]          # toggle between frame t and t + 1
    per_ep = np.bincount(ep[ev], minlength=int(ep[-1]) + 1)
    gaps = np.diff(ev)[ep[ev[1:]] == ep[ev[:-1]]]
    q = lambda x: {p: float(np.percentile(x, p)) for p in (5, 25, 50, 75, 95)}
    return {"frames": int(len(term)), "episodes": int(ep[-1]) + 1, "events": int(len(ev)),
            "events_per_episode": q(per_ep), "steps_between_events": q(gaps),
            "lights_toggled_per_event": {int(k): int(v) for k, v in zip(*np.unique(flips[ev], return_counts=True))}}


class PressController:
    """Scripted press of one button (ButtonMarkovOracle phases 1-3, gripper closed, no wander)."""

    def __init__(self, env, button):
        self.u, self.button, self.pressed = env.unwrapped, button, False
        self.start = self.u._cur_button_states.copy()

    def act(self):
        u = self.u
        eff = u.compute_ob_info()["proprio/effector_pos"]
        top = u._data.site_xpos[u._button_site_ids[self.button]].copy()
        above, bottom = top + np.array([0, 0, 0.06]), top - np.array([0, 0, 0.022])
        if not self.pressed and (u._cur_button_states != self.start).any():
            self.pressed = True
        if self.pressed:
            if eff[2] > 0.16:
                return None                                   # lifted: done
            target = np.array([above[0], above[1], 0.32])
        elif np.linalg.norm(above[:2] - eff[:2]) > 0.04:
            target = above
        else:
            target = bottom
        diff = target - eff
        n = np.linalg.norm(diff)
        diff = diff if n >= 0.4 else diff / (n + 1e-6) * 0.4
        a = np.zeros(5)
        a[:3] = diff * 5
        a[4] = 1
        return np.clip(a, -1, 1)


def button_xy(env, i):
    u = env.unwrapped
    return u._data.site_xpos[u._button_site_ids[i]][:2].copy()


def run_task(env, rows, cols, task_id, store):
    ob, info = env.reset(options=dict(task_id=task_id, render_goal=False))
    u = env.unwrapped
    init, goal = u._cur_button_states.copy(), u._target_button_states.copy()
    store.append(dict(task_id=task_id, init_ob=ob, goal_ob=info["goal"], init=init, goal=goal))
    todo = list(np.nonzero(min_press_set(init, goal, rows, cols))[0])
    steps, per_press, success = 0, [], False
    while todo:
        eff = u.compute_ob_info()["proprio/effector_pos"][:2]
        nxt = min(todo, key=lambda i: np.linalg.norm(button_xy(env, i) - eff))   # greedy nearest button
        todo.remove(nxt)
        ctl, n = PressController(env, nxt), 0
        while True:
            a = ctl.act()
            if a is None:
                break
            ob, _, term, trunc, info = env.step(a)
            steps, n = steps + 1, n + 1
            success = bool(info["success"])
            if term or trunc:
                todo = []
                break
        per_press.append(n)
    return {"task_id": task_id, "presses": int(min_press_set(init, goal, rows, cols).sum()), "steps": steps,
            "success": success, "steps_per_press": per_press}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sizes", nargs="+", default=list(SIZES))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import gymnasium
    import ogbench  # noqa: F401  (registers the manipspace envs)

    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for size in a.sizes:
        rows, cols = SIZES[size]
        t0 = time.time()
        rec = {"data": data_stats(a.data / f"visual-puzzle-{size}-play-v0.npz")}
        env = gymnasium.make(f"visual-puzzle-{size}-v0")
        store, runs = [], []
        for task_id in range(1, 6):
            runs.append(run_task(env, rows, cols, task_id, store))
        rec["scripted_executor_privileged"] = runs
        rec["max_episode_steps"] = int(env.spec.max_episode_steps)
        rec["minutes"] = round((time.time() - t0) / 60, 2)
        np.savez_compressed(a.out / f"tasks_{size}.npz",
                            **{k: np.stack([s[k] for s in store]) for k in store[0]})
        report[size] = rec
        print(size, json.dumps(rec), flush=True)
        (a.out / "inventory.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
