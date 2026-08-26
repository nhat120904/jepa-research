#!/usr/bin/env python3
"""Arm-independent viability filter for the CEM-interaction test.

The planner can only matter where the task is not already solved before it
acts.  ``cube.yaml`` sets ``terminate_at_goal: True``, so on a start state that
already satisfies the success predicate the episode ends on the first primitive
action and no arm's choice can change the outcome.

The filter is the environment's own predicate, evaluated on the pre-action
start state (``cube_env.py:_compute_successes``: a cube succeeds iff
``||obj_pos - tar_pos|| <= 0.04``; ``cube_distance`` in script 72 computes the
same norm).  A snapshot enters the analysis iff its start state is NOT already
successful.

This depends only on the restored start state and the goal, never on any model,
so it cannot favour an arm.  ``max(executed_steps) > 1`` over the shared
population is recorded alongside as a corroborating classification, not as the
definition: that quantity also depends on 300 sampled actions, so a state could
in principle be unsolved yet have every candidate terminate early for an
unrelated reason.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "diagnosis"
sys.path.insert(0, str(REPO))

SUCCESS_THRESHOLD_M = 0.04  # cube_env.py:1339


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--populations-dir", type=Path, required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--last", type=int, required=True)
    p.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "vb_audit")
    corrected = load_module(DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "vb_corrected")

    import stable_worldmodel as swm
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])

    rows: list[dict[str, Any]] = []
    for order in range(args.first, args.last + 1):
        snapshot = audit.Snapshot(**manifest[order])
        if snapshot.order != order:
            raise RuntimeError(f"manifest order mismatch at {order}")
        init_rows, goal_rows, _ = _extract_init_goal(
            dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset)
        init_row, goal_row = init_rows[0], goal_rows[0]
        target = np.asarray(audit.goal_field(goal_row, "block_0_pos"), dtype=np.float64)

        world, raw_env, _, _ = corrected.make_world(swm, snapshot)
        try:
            corrected.restore_complete(
                raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit)
            start_distance = float(audit.cube_distance(raw_env, target))
        finally:
            world.close()

        pop = np.load(args.populations_dir / f"snapshot_{order:03d}/populations.npz")
        max_exec = int(pop["executed_steps"][0].max())
        physical = np.asarray(pop["physical_distance_m"][0], dtype=np.float64)

        rows.append({
            "snapshot": order,
            "episode": snapshot.episode,
            "start_cube_distance_m": start_distance,
            "start_state_already_successful": bool(start_distance <= SUCCESS_THRESHOLD_M),
            "viable": bool(start_distance > SUCCESS_THRESHOLD_M),
            "corroborating_max_executed_steps": max_exec,
            "corroborating_population_spread_m": float(physical.max() - physical.min()),
        })
        print(f"snapshot {order:3d}  start_dist {start_distance:.5f}  "
              f"viable {rows[-1]['viable']}  max_exec {max_exec}")

    viable = [r["snapshot"] for r in rows if r["viable"]]
    by_exec = [r["snapshot"] for r in rows if r["corroborating_max_executed_steps"] > 1]
    summary = {
        "success_threshold_m": SUCCESS_THRESHOLD_M,
        "n_snapshots": len(rows),
        "n_viable": len(viable),
        "n_excluded_already_successful": len(rows) - len(viable),
        "viable_snapshots": viable,
        "corroboration_max_executed_steps_gt_1": {
            "n": len(by_exec),
            "agrees_with_start_state_filter": by_exec == viable,
            "only_in_start_state_filter": sorted(set(viable) - set(by_exec)),
            "only_in_executed_steps_filter": sorted(set(by_exec) - set(viable)),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
