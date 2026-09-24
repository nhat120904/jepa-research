"""LIBERO L3: closed-loop oracle headroom (docs/LIBERO_QUALIFICATION_PROTOCOL.md, Amendment 2).

Arms on the same root (task, init) and the same seeded nested bank:
- P0: candidate 0 at every decision;
- ORACLE8: at every decision all 8 candidates run their chunk on a clone (exact, L2) and the candidate with the highest
  privileged BDDL goal progress is executed on the live env (candidate 0 wins ties).
SmolVLA runs on CPU here, as at L2; the paired contrast uses the same device for both arms.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from ti_wm.contract import select_candidate
from ti_wm.libero_runtime import (
    Policy, candidate_seeds, goal_progress, goal_reference, load_state, make_env, run_chunk, save_state,
)

K = 8
ROOT_BASE = 100_000


def root_id(task, init):
    return ROOT_BASE + 100 * task + init


def episode(policy, live, worker, task, init, arm):
    root = root_id(task, init)
    live.init_state_id = init
    obs, _ = live.reset(seed=0)
    ref = goal_reference(live)
    start = goal_progress(live, ref)
    text, max_steps = live.task_description, live._max_episode_steps
    t, d, success, decisions = 0, 0, False, []
    while not success and t < max_steps:
        bank = policy.bank(obs, text, candidate_seeds(root, d, 1 if arm == "P0" else K))[:, : max_steps - t]
        rec = {"d": d, "t": t}
        chosen = 0
        if arm == "ORACLE8":
            state = save_state(live)
            prog = []
            for k in range(K):
                load_state(worker, state)
                _, succ, _, _, _ = run_chunk(worker, bank[k])
                prog.append(1.0 if succ else goal_progress(worker, ref))
            rec["progress"] = prog
            chosen = select_candidate(prog)
        obs, success, n, _, _ = run_chunk(live, bank[chosen])
        rec["chosen"] = int(chosen)
        decisions.append(rec)
        t += n
        d += 1
    return {"success": success, "steps": t, "decisions": decisions, "progress_start": start,
            "progress_end": 1.0 if success else goal_progress(live, ref)}


def main(run, setup, tasks, inits, device, arms):
    assert os.environ.get("SLURM_JOB_ID"), "Run through sbatch"
    ckpt = json.loads((setup / "checkpoints.json").read_text())["HuggingFaceVLA/smolvla_libero"]["path"]
    policy = Policy(ckpt, device)
    out = run / f"l3_tasks_{'-'.join(map(str, tasks))}.jsonl"
    with open(out, "w") as stream:
        for task in tasks:
            live, worker = make_env(policy.cfg, task, 0), make_env(policy.cfg, task, 0)
            live.reset(seed=0)
            worker.reset(seed=0)
            try:
                for init in inits:
                    t0 = time.perf_counter()
                    rec = {"task": task, "init": init, "root": root_id(task, init), "language": live.task_description}
                    try:
                        for arm in arms:
                            rec[arm] = episode(policy, live, worker, task, init, arm)
                    except Exception:
                        rec["error"] = traceback.format_exc()
                    rec["seconds"] = time.perf_counter() - t0
                    stream.write(json.dumps(rec) + "\n")
                    stream.flush()
                    print(json.dumps({"task": task, "init": init, "seconds": round(rec["seconds"]),
                                      **({a: rec[a]["success"] for a in arms if a in rec}),
                                      **({"error": rec["error"][-400:]} if "error" in rec else {})}), flush=True)
            finally:
                live.close()
                worker.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--setup", type=Path, required=True)
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--device", default="cpu")
    p.add_argument("--arms", default="P0,ORACLE8")
    a = p.parse_args()
    main(a.run, a.setup, a.tasks, a.inits, a.device, a.arms.split(","))
