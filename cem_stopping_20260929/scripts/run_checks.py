#!/usr/bin/env python3
"""Smoke checks: upstream equivalence, prefix property, exact replay, tree == live."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))
sys.path.insert(0, str(PROJECT / "scripts"))

from run_tree import BASE_SEED, MANIFEST_SEED, N_MANIFEST  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["cube", "reacher"])
    p.add_argument("--roots", nargs="+", type=int, default=[0, 1])
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def unit_tests() -> dict:
    import test_rules

    names = [k for k in dir(test_rules) if k.startswith("test_")]
    for name in names:
        getattr(test_rules, name)()
    return {"passed": names}


def solver_equivalence(swm, task, root) -> dict:
    import torch

    from cemstop.recorder import PlanRecorder
    from cemstop.rules import CHECKPOINTS

    init, goal = task.rows(root)
    task.restore(root, init, goal)
    prepared = task.prepared(task.render(), np.asarray(goal["goal"]))
    seed = 424242
    rec = PlanRecorder(CHECKPOINTS)
    final = task.plan(prepared, seed, rec).numpy()
    rec_again = PlanRecorder(CHECKPOINTS)
    again = task.plan(prepared, seed, rec_again).numpy()
    result = {"repeat_max_abs": float(np.max(np.abs(final - again))),
              "repeat_costs_max_abs": float(np.max(np.abs(rec.cost_matrix() - rec_again.cost_matrix())))}
    for n in (30, 10, 3):
        plain = swm.planning.CEMSolver(cost=task.solver.cost, batch_size=1, num_samples=300,
                                       n_steps=n, topk=30, var_scale=1.0, device="cuda", seed=seed)
        plain.configure(action_space=task.world.envs.action_space, n_envs=1, config=task.policy.cfg)
        with torch.inference_mode():
            out = plain.solve(dict(prepared))["actions"][0].numpy()
        result[f"upstream_n{n}_vs_checkpoint_max_abs"] = float(np.max(np.abs(out - rec.means[n])))
    ok = all(v == 0.0 for v in result.values())
    return {"ok": ok, **result}


def tree_vs_live(task, root, arrays) -> dict:
    from cemstop.evaluate import TreeRoot, outcome
    from cemstop.rules import CHECKPOINTS, Band, Converge, Fixed, Gap
    from cemstop.tree import run_live

    tr = TreeRoot(root=root.root, costs1=arrays["costs1"], costs2=arrays["costs2"],
                  p1_term=arrays["p1_term"], leaf_success=arrays["leaf_success"],
                  leaf_dist=arrays["leaf_dist"], checkpoints=CHECKPOINTS, extra={})
    rows = []
    ok = True
    for rule in (Fixed(30), Fixed(10), Fixed(3), Fixed(1), Converge(0.01), Gap(0.01), Band(0.1)):
        live = run_live(task, root, BASE_SEED, rule)
        off = outcome(rule, tr)
        k1 = CHECKPOINTS[off["i1"]]
        k2 = None if off["i2"] is None else CHECKPOINTS[off["i2"]]
        same_costs1 = bool(np.array_equal(live["costs1"], arrays["costs1"]))
        same_costs2 = (live["costs2"] is None) == (k2 is None) and (
            live["costs2"] is None or bool(np.array_equal(live["costs2"], arrays["costs2"][off["i1"]])))
        match = (live["k1"] == k1 and live["k2"] == k2 and live["success"] == off["success"]
                 and live["final_distance"] == off["distance"] and same_costs1 and same_costs2)
        ok &= match
        rows.append({"rule": f"{rule.name}({rule.param():g})", "live_k": [live["k1"], live["k2"]],
                     "tree_k": [k1, k2], "live_success": live["success"],
                     "tree_success": off["success"], "live_distance": live["final_distance"],
                     "tree_distance": off["distance"], "costs1_equal": same_costs1,
                     "costs2_equal": same_costs2, "match": match})
    return {"ok": ok, "rules": rows}


def main():
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model and physics work must run under sbatch")
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import stable_worldmodel as swm

    from cemstop.tasks import TASKS, build_roots
    from cemstop.tree import run_tree

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"unit_tests": unit_tests()}
    print("UNIT TESTS OK", flush=True)
    all_ok = True
    for name in args.tasks:
        entry = {}
        t0 = time.perf_counter()
        task = TASKS[name](swm)
        entry["setup_seconds"] = time.perf_counter() - t0
        try:
            roots = build_roots(task.dataset, N_MANIFEST, MANIFEST_SEED)
            entry["solver"] = solver_equivalence(swm, task, roots[args.roots[0]])
            print(name, "solver", json.dumps(entry["solver"]), flush=True)
            entry["roots"] = {}
            for rid in args.roots:
                t = time.perf_counter()
                arrays, summary = run_tree(task, roots[rid], BASE_SEED, label=(rid == args.roots[0]))
                np.savez_compressed(args.out_dir / f"{name}_root_{rid:04d}.npz", **arrays)
                live = tree_vs_live(task, roots[rid], arrays)
                entry["roots"][rid] = {"summary": summary, "tree_vs_live": live,
                                       "seconds": time.perf_counter() - t}
                print(name, rid, json.dumps({"fixed": summary["fixed_k_success"],
                                             "start_diff": summary["start_image_vs_dataset"],
                                             "plan_s": summary["plan_seconds"][:2],
                                             "sim_s": summary["sim_seconds"],
                                             "live_ok": live["ok"]}), flush=True)
            entry["ok"] = entry["solver"]["ok"] and all(
                r["tree_vs_live"]["ok"] for r in entry["roots"].values())
        except Exception:
            entry["ok"] = False
            entry["error"] = traceback.format_exc()
            print(entry["error"], flush=True)
        finally:
            task.close()
        all_ok &= entry["ok"]
        report[name] = entry
        (args.out_dir / "checks.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("SMOKE_OK" if all_ok else "SMOKE_FAILED", flush=True)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
