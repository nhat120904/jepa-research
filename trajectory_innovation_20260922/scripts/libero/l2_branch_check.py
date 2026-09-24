"""LIBERO L2: branching fidelity and candidate diversity (docs/LIBERO_QUALIFICATION_PROTOCOL.md, Amendment 2).

Per root (task, init) the live env follows P0 (seeded candidate 0). At every CHECK_EVERY-th decision:
- fidelity (hard requirement): the live state is restored twice into a second env instance and candidate 0's chunk
  is stepped; the two restored runs must repeat exactly and match the live env stepping the same chunk
  (max |dqpos| <= 1e-6 per step);
- diversity (reported only): all 8 candidates run from the same state; relative chunk spread and the largest
  pairwise end-position distance of any task object across the 8 branches.
"""

import argparse
import json
import os
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np

from ti_wm.libero_runtime import (
    Policy, candidate_seeds, load_state, make_env, object_positions, run_chunk, save_state, scene_state,
)

K, CHECK_EVERY, FIDELITY_TOL = 8, 3, 1e-6
ROOT_BASE = 100_000


def root_id(task, init):
    """LIBERO root ids are disjoint from PushT roots."""
    return ROOT_BASE + 100 * task + init


def max_dev(a, b):
    n = min(len(a), len(b))
    return float(max((np.abs(a[i] - b[i]).max() for i in range(n)), default=0.0)) if n else float("nan")


def check(policy_bank, live, worker, state):
    """Fidelity runs and the 8-branch diversity panel from one saved state."""
    reps = []
    for _ in range(2):
        load_state(worker, state)
        reps.append(run_chunk(worker, policy_bank[0])[4])
    ends, scenes = [], []
    for k in range(K):
        load_state(worker, state)
        run_chunk(worker, policy_bank[k])
        ends.append(object_positions(worker))
        scenes.append(scene_state(worker))
    flat = policy_bank.reshape(K, -1)
    pair = [np.linalg.norm(flat[i] - flat[j]) for i, j in combinations(range(K), 2)]
    spread = {name: max(np.linalg.norm(ends[i][name] - ends[j][name]) for i, j in combinations(range(K), 2))
              for name in ends[0]}
    scene = max(float(np.abs(scenes[i]["scene_qpos"] - scenes[j]["scene_qpos"]).max()) for i, j in combinations(range(K), 2))
    eef = max(float(np.linalg.norm(scenes[i]["eef"] - scenes[j]["eef"])) for i, j in combinations(range(K), 2))
    return reps, {"chunk_rel_spread": float(np.mean(pair) / max(np.linalg.norm(flat, axis=1).mean(), 1e-9)),
                  "object_spread_m": float(max(spread.values(), default=0.0)), "object_spread_by_name": spread,
                  "scene_qpos_spread": scene, "eef_spread_m": eef}


def run_root(policy, task, init):
    root = root_id(task, init)
    live, worker = make_env(policy.cfg, task, init), make_env(policy.cfg, task, init)
    try:
        obs, _ = live.reset(seed=0)
        worker.reset(seed=0)
        text, max_steps = live.task_description, live._max_episode_steps
        t, d, success, checks = 0, 0, False, []
        while not success and t < max_steps:
            bank = policy.bank(obs, text, candidate_seeds(root, d, K))[:, : max_steps - t]
            item = None
            if d % CHECK_EVERY == 0:
                state = save_state(live)
                reps, item = check(bank, live, worker, state)
            obs, success, n, _, live_trace = run_chunk(live, bank[0])
            if item is not None:
                item.update({"d": d, "t": t, "repeat_max_dqpos": max_dev(reps[0], reps[1]),
                             "live_max_dqpos": max_dev(reps[0], live_trace)})
                item["object_spread_by_name"] = {k: float(v) for k, v in item["object_spread_by_name"].items()}
                checks.append(item)
            t += n
            d += 1
        return {"task": task, "init": init, "root": root, "language": text, "success": success, "steps": t,
                "decisions": d, "checks": checks}
    finally:
        live.close()
        worker.close()


def main(run, setup, tasks, inits, device):
    assert os.environ.get("SLURM_JOB_ID"), "Run through sbatch"
    ckpt = json.loads((setup / "checkpoints.json").read_text())["HuggingFaceVLA/smolvla_libero"]["path"]
    t0 = time.perf_counter()
    policy = Policy(ckpt, device)
    out = run / f"l2_tasks_{'-'.join(map(str, tasks))}.jsonl"
    with open(out, "w") as stream:
        for task in tasks:
            for init in inits:
                t1 = time.perf_counter()
                try:
                    rec = run_root(policy, task, init)
                except Exception:
                    rec = {"task": task, "init": init, "error": traceback.format_exc()}
                rec["seconds"] = time.perf_counter() - t1
                stream.write(json.dumps(rec) + "\n")
                stream.flush()
                summary = {k: rec.get(k) for k in ("task", "init", "success", "steps", "decisions", "seconds")}
                if rec.get("checks"):
                    summary["worst_live_dqpos"] = max(c["live_max_dqpos"] for c in rec["checks"])
                    summary["worst_repeat_dqpos"] = max(c["repeat_max_dqpos"] for c in rec["checks"])
                    summary["object_spread_gt_1cm"] = float(np.mean([c["object_spread_m"] > 0.01 for c in rec["checks"]]))
                    summary["scene_qpos_spread_gt_0.01"] = float(np.mean([c["scene_qpos_spread"] > 0.01 for c in rec["checks"]]))
                    summary["median_eef_spread_cm"] = 100 * float(np.median([c["eef_spread_m"] for c in rec["checks"]]))
                print(json.dumps(summary if "error" not in rec else rec), flush=True)
    print(f"total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--setup", type=Path, required=True)
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--inits", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    main(a.run, a.setup, a.tasks, a.inits, a.device)
