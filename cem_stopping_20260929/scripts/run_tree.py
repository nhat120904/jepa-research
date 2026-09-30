#!/usr/bin/env python3
"""Record stopping trees for a contiguous range of root ids (resumable)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

MANIFEST_SEED = 20260929
BASE_SEED = 5_000_000
N_MANIFEST = 300


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("cube", "reacher"), required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--last", type=int, required=True, help="exclusive")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--offset", type=int, default=0, help="shard offset within [first, last)")
    p.add_argument("--label-roots", type=int, default=20, help="label plan-1 candidates for root < N")
    p.add_argument("--allow-test", action="store_true",
                   help="required for root ids >= 100 (test split)")
    p.add_argument("--goal-source", choices=("dataset", "render"), default="dataset")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model and physics work must run under sbatch")
    if args.last > 100 and not args.allow_test:
        raise RuntimeError("test roots (id >= 100) need --allow-test after rules are frozen")
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import stable_worldmodel as swm

    from cemstop.tasks import TASKS, build_roots
    from cemstop.tree import run_tree

    task = TASKS[args.task](swm)
    task.goal_source = args.goal_source
    roots = build_roots(task.dataset, N_MANIFEST, MANIFEST_SEED)
    out = args.out_dir / args.task
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    payload = {"task": args.task, "manifest_seed": MANIFEST_SEED, "base_seed": BASE_SEED,
               "goal_source": args.goal_source, "roots": [asdict(r) for r in roots]}
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if (previous["roots"] != payload["roots"]
                or previous.get("goal_source", "dataset") != args.goal_source):
            raise RuntimeError("manifest or goal source changed between runs")
    else:
        manifest.write_text(json.dumps(payload, indent=1) + "\n")

    ids = list(range(args.first + args.offset, args.last, args.stride))
    try:
        for rid in ids:
            path = out / f"root_{rid:04d}.npz"
            if path.exists():
                continue
            t = time.perf_counter()
            arrays, summary = run_tree(task, roots[rid], BASE_SEED, label=rid < args.label_roots)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            tmp.replace(path)
            (out / f"root_{rid:04d}.json").write_text(json.dumps(summary, indent=1) + "\n")
            print(json.dumps({"root": rid, "fixed_k_success": summary["fixed_k_success"],
                              "any_leaf": summary["any_leaf_success"],
                              "seconds": round(time.perf_counter() - t, 1),
                              "plan_s": round(float(np.sum(summary["plan_seconds"])), 1),
                              "sim_s": round(summary["sim_seconds"], 1)}), flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()
