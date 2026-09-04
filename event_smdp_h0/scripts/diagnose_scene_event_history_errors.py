#!/usr/bin/env python3
"""Why does a more accurate frame observer plan worse?

Static exact-q accuracy treats every misread alike.  Planning does not: the
automaton only advances through causally ordered milestones, so an observer that
under-reads progress makes the planner redo a step it has already achieved,
while an observer that over-reads progress makes it skip a prerequisite that
nothing later will supply.  This classifies every deployed misread by direction
and reports success conditioned on that direction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path


PROTOCOL = "scene_event_history_error_direction_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def direction(predicted: dict, true: dict) -> str:
    keys = ("cube_stage", "window_stage")
    deltas = [int(predicted[key]) - int(true[key]) for key in keys]
    if all(delta == 0 for delta in deltas):
        return "exact"
    if all(delta <= 0 for delta in deltas):
        return "behind"
    if all(delta >= 0 for delta in deltas):
        return "ahead"
    return "mixed"


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("error diagnosis must run inside a Slurm compute job")

    shards = sorted(args.eval_root.glob("*/result.json"), key=lambda p: int(p.parent.name))
    if not shards:
        raise FileNotFoundError(f"no shards under {args.eval_root}")

    decisions: dict[str, Counter] = defaultdict(Counter)
    episodes: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    stage_errors: dict[str, Counter] = defaultdict(Counter)
    visited: dict[str, Counter] = defaultdict(Counter)

    for shard in shards:
        payload = json.loads(shard.read_text())
        for row in payload["results"]:
            # Include every arm whose planning state is inferred rather than
            # read from the simulator, which covers the untrained open-loop
            # tracking arm as well as the learned observers.
            if all(
                replan["exact_q_correct"] is None for replan in row["replans"]
            ):
                continue
            arm = row["arm"]
            seen: set[str] = set()
            for replan in row["replans"]:
                label = direction(replan["planning_state"], replan["true_state"])
                decisions[arm][label] += 1
                seen.add(label)
                key = (
                    int(replan["true_state"]["cube_stage"]),
                    int(replan["true_state"]["window_stage"]),
                )
                visited[arm][str(list(key))] += 1
                if label != "exact":
                    stage_errors[arm][
                        f"{list(key)}->"
                        f"{[int(replan['planning_state']['cube_stage']), int(replan['planning_state']['window_stage'])]}"
                    ] += 1
            worst = (
                "ahead_or_mixed"
                if ({"ahead", "mixed"} & seen)
                else ("behind_only" if "behind" in seen else "exact_only")
            )
            episodes[arm][worst].append(bool(row["success"]))

    summary = {
        "protocol": PROTOCOL,
        "num_shards": len(shards),
        "decision_error_direction": {
            arm: {
                "counts": dict(counter),
                "n": sum(counter.values()),
                "exact_rate": counter["exact"] / max(sum(counter.values()), 1),
            }
            for arm, counter in sorted(decisions.items())
        },
        "success_by_worst_error_in_episode": {
            arm: {
                label: {
                    "n": len(values),
                    "success_rate": (sum(values) / len(values)) if values else None,
                }
                for label, values in sorted(buckets.items())
            }
            for arm, buckets in sorted(episodes.items())
        },
        "top_confusions": {
            arm: counter.most_common(6) for arm, counter in sorted(stage_errors.items())
        },
        "visited_true_states": {
            arm: counter.most_common(8) for arm, counter in sorted(visited.items())
        },
        "scope": "descriptive audit of a completed evaluation run; no new claim",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "decision_error_direction": summary["decision_error_direction"],
                "success_by_worst_error_in_episode": summary[
                    "success_by_worst_error_in_episode"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
