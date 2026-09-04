#!/usr/bin/env python3
"""Is the single frame insufficient because the automaton latches progress?

`advance_milestones` only ever increases a stage, but the scene can regress:
opening then closing the window leaves `window_stage=2` while the current image
looks like `window_stage=1`.  If that is why a frame observer under-reads, its
under-reads should concentrate after a regressing skill has been deployed.

Control: the same rate measured on decisions the observer got exactly right.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path


REGRESSORS = {
    "window": ("window_close", "toggle_button_1"),
    "cube": ("drawer_close", "toggle_button_0"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("must run inside a Slurm compute job")

    stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "after_regressor": 0})
    )
    for shard in sorted(args.eval_root.glob("*/result.json"), key=lambda p: int(p.parent.name)):
        payload = json.loads(shard.read_text())
        for row in payload["results"]:
            if all(r["exact_q_correct"] is None for r in row["replans"]):
                continue
            arm = row["arm"]
            executed: list[str] = []
            for replan in row["replans"]:
                true_state, pred = replan["true_state"], replan["planning_state"]
                for branch, stage_key in (("cube", "cube_stage"), ("window", "window_stage")):
                    delta = int(pred[stage_key]) - int(true_state[stage_key])
                    if delta == 0:
                        bucket = "exact_control"
                    elif delta < 0:
                        bucket = "under_read"
                    else:
                        bucket = "over_read"
                    seen = any(skill in REGRESSORS[branch] for skill in executed)
                    cell = stats[arm][f"{branch}_{bucket}"]
                    cell["n"] += 1
                    cell["after_regressor"] += int(seen)
                executed.append(replan["selected_skill"])

    summary = {
        "protocol": "scene_latching_signature_v1",
        "regressing_skills": REGRESSORS,
        "rates": {
            arm: {
                key: {
                    "n": cell["n"],
                    "after_regressor": cell["after_regressor"],
                    "rate": (cell["after_regressor"] / cell["n"]) if cell["n"] else None,
                }
                for key, cell in sorted(buckets.items())
            }
            for arm, buckets in sorted(stats.items())
        },
        "scope": "descriptive audit of a completed run; exploratory, not preregistered",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "latching_signature.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary["rates"], sort_keys=True))


if __name__ == "__main__":
    main()
