#!/usr/bin/env python3
"""Renderer-domain check: dataset frames versus our renders of the same state.

For dev roots, render the start state and the goal state with the evaluation
renderer and compare with the dataset frames, in pixels and in the LeWM latent
used by the planning cost. The latent gaps are reported next to the cost scale
of the recorded trees (first-population median and final elite-mean cost).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from run_tree import MANIFEST_SEED, N_MANIFEST  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["reacher", "cube"])
    p.add_argument("--roots", type=int, default=20)
    p.add_argument("--tree-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model and physics work must run under sbatch")
    import torch
    from PIL import Image

    import stable_worldmodel as swm

    from cemstop.tasks import TASKS, build_roots

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.tasks:
        task = TASKS[name](swm)
        roots = build_roots(task.dataset, N_MANIFEST, MANIFEST_SEED)
        rows = []
        try:
            for root in roots[: args.roots]:
                init, goal = task.rows(root)
                task.restore(root, init, goal)
                r_start = task.render()
                # Goal state rendered with the same renderer.
                goal_init = {"qpos": goal["goal_qpos"], "qvel": goal["goal_qvel"]}
                task.restore(root, goal_init, goal)
                r_goal = task.render()
                imgs = np.stack([np.asarray(init["pixels"]), r_start, np.asarray(goal["goal"]), r_goal])
                with torch.inference_mode():
                    info = task.policy._prepare_info({"pixels": imgs[:, None]})
                    emb = task.model.encode({"pixels": info["pixels"].to(task.device)})["emb"][:, -1]
                    emb = emb.float().reshape(len(imgs), -1).cpu().numpy()
                ds_start, rd_start, ds_goal, rd_goal = emb

                def sq(a, b):
                    return float(np.sum((a - b) ** 2))

                tree = np.load(args.tree_dir / name / f"root_{root.root:04d}.npz")
                costs1 = tree["costs1"].astype(np.float64)
                final_elite = float(np.mean(np.sort(costs1[-1])[:30]))
                pix = lambda a, b: float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))  # noqa: E731
                rows.append({
                    "root": root.root,
                    "pix_start_ds_vs_render": pix(imgs[0], imgs[1]),
                    "pix_goal_ds_vs_render": pix(imgs[2], imgs[3]),
                    "lat_start_ds_vs_render": sq(ds_start, rd_start),
                    "lat_goal_ds_vs_render": sq(ds_goal, rd_goal),
                    "lat_render_start_to_ds_goal": sq(rd_start, ds_goal),
                    "lat_render_start_to_render_goal": sq(rd_start, rd_goal),
                    "tree_first_pop_median_cost": float(np.median(costs1[0])),
                    "tree_final_elite_mean_cost": final_elite,
                })
                if root.root < 3:
                    Image.fromarray(np.concatenate(list(imgs), axis=1)).save(
                        args.out_dir / f"{name}_root{root.root}_dsstart_rstart_dsgoal_rgoal.png")
        finally:
            task.close()
        keys = [k for k in rows[0] if k != "root"]
        report[name] = {"rows": rows,
                        "median": {k: float(np.median([r[k] for r in rows])) for k in keys}}
        print(name, json.dumps(report[name]["median"], indent=1), flush=True)
    (args.out_dir / "render_check.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
