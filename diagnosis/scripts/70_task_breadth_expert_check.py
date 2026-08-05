"""Gate 1 for the task-breadth extension (see
docs/plans/<date>-generality-extension-design.md): does the scripted expert
actually succeed on each candidate task, and does it succeed within the
100-step cap that `rollout_expert` uses by default in scripts/29 and 30?

`rollout_expert` (scripts/18_closed_loop_eval.py:238) hardcodes
`max_steps=100` at both call sites; peg-insert-side's expert only fires
success in 4/16 goal rollouts under that cap (results/metaworld_precision_ladder.csv),
because MetaWorld's own `max_path_length` is 500. This script reruns the
IDENTICAL make_env + rollout_expert path used by scripts/29/30, but with a
raised `max_steps` so we can see whether a task's expert eventually succeeds
at all (mechanism: policy competence) versus never reaches the cap
(mechanism: budget). No model, no GPU compute beyond MuJoCo rendering, no
CEM search — this is a cheap pre-flight check before committing the full
terminal-cost-ladder budget to a new task.

Gate 1 criterion (pre-registered): a task is ELIGIBLE for the terminal-cost
ladder only if >=75% of `--episodes` seeds succeed within max_steps=100 AND
the same holds within max_steps=500 (i.e. the 100-step cap is not the
bottleneck). A task that only succeeds under the raised cap is flagged
INELIGIBLE-BUDGET (needs a code change to raise the cap, not a data problem);
a task that never succeeds even at 500 steps is INELIGIBLE-POLICY.

    python scripts/70_task_breadth_expert_check.py \
        --tasks mw-door-open mw-drawer-close mw-button-press mw-window-close \
                mw-assembly mw-peg-insert-side \
        --episodes 16 --seed0 70000 \
        --out results/task_breadth_expert_check.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util as _ilu
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_cl = _load("closed_loop_eval", "18_closed_loop_eval.py")
make_env, rollout_expert = _cl.make_env, _cl.rollout_expert


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=70000)
    ap.add_argument("--cap-a", type=int, default=100,
                     help="the cap scripts/29 and 30 actually use")
    ap.add_argument("--cap-b", type=int, default=500,
                     help="MetaWorld's own max_path_length, as an upper check")
    ap.add_argument("--out", default="results/task_breadth_expert_check.csv")
    ap.add_argument("--out-md", default="results/task_breadth_expert_check.md")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "seed", "succ_step_capA", "succ_step_capB",
              "within_capA", "within_capB"]
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for task in args.tasks:
            for e in range(args.episodes):
                seed = args.seed0 + e
                t0 = time.time()
                env, init_state = make_env(task, seed)
                _, _, succ_b = rollout_expert(env, init_state, task, max_steps=args.cap_b)
                env.close()
                succ_a = succ_b if (succ_b is not None and succ_b <= args.cap_a) else None
                r = {
                    "task": task, "seed": seed,
                    "succ_step_capA": succ_a, "succ_step_capB": succ_b,
                    "within_capA": int(succ_a is not None),
                    "within_capB": int(succ_b is not None),
                }
                w.writerow(r); f.flush(); rows.append(r)
                print(f"  {task:20s} seed={seed} succ_capA={succ_a} "
                      f"succ_capB={succ_b} ({(time.time()-t0):.1f}s)", flush=True)

    # Per-task summary + eligibility verdict.
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    lines = ["# Task-breadth Gate 1: scripted-expert competence check", "",
             f"seed0={args.seed0}, episodes={args.episodes}, "
             f"cap_a={args.cap_a} (production cap), cap_b={args.cap_b} "
             "(MetaWorld max_path_length upper check).", "",
             "| task | within capA | within capB | verdict |",
             "|---|---|---|---|"]
    verdicts = {}
    for task, rs in by_task.items():
        n = len(rs)
        rate_a = sum(r["within_capA"] for r in rs) / n
        rate_b = sum(r["within_capB"] for r in rs) / n
        if rate_a >= 0.75 and rate_b >= 0.75:
            verdict = "ELIGIBLE"
        elif rate_b >= 0.75:
            verdict = "INELIGIBLE-BUDGET (raise cap_a)"
        else:
            verdict = "INELIGIBLE-POLICY"
        verdicts[task] = verdict
        lines.append(f"| {task} | {rate_a*100:.0f}% ({sum(r['within_capA'] for r in rs)}/{n}) "
                      f"| {rate_b*100:.0f}% ({sum(r['within_capB'] for r in rs)}/{n}) "
                      f"| {verdict} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
