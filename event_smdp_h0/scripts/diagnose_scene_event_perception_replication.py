#!/usr/bin/env python3
"""Mechanistic error audit for the learned-q replication."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


PROTOCOL = "scene_event_perception_replication_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob("*/result.json"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")

    episodes: dict[tuple[int, int, int], dict] = {}
    oracle: dict[tuple[int, int], int] = {}
    terminal: dict[tuple[int, int], int] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError(f"unexpected protocol in {path}")
        task_id = int(payload["task_id"])
        reset_seed = int(payload["reset_seed"])
        for row in payload["results"]:
            if row["arm"] == "learned_latent":
                episodes[(task_id, reset_seed, int(row["observer_seed"]))] = row
            elif row["arm"] == "oracle_event":
                oracle[(task_id, reset_seed)] = int(row["success"])
            elif row["arm"] == "abstract_terminal":
                terminal[(task_id, reset_seed)] = int(row["success"])

    stage_rows: list[dict] = []
    episode_rows: list[dict] = []
    confusions: Counter[tuple[int, int, int, int, int, int]] = Counter()
    decision_totals: Counter[tuple[int, int, int]] = Counter()
    decision_correct: Counter[tuple[int, int, int]] = Counter()
    by_stage_total: Counter[tuple[int, int, int, int]] = Counter()
    by_stage_correct: Counter[tuple[int, int, int, int]] = Counter()
    for (task_id, reset_seed, observer_seed), row in sorted(episodes.items()):
        replans = row["replans"]
        exact = [int(replan["exact_q_correct"]) for replan in replans]
        episode_rows.append(
            {
                "task_id": task_id,
                "reset_seed": reset_seed,
                "observer_seed": observer_seed,
                "success": int(row["success"]),
                "oracle_success": oracle[(task_id, reset_seed)],
                "terminal_success": terminal[(task_id, reset_seed)],
                "num_replans": len(replans),
                "num_q_errors": len(exact) - sum(exact),
                "any_q_error": int(not all(exact)),
                "first_q_error_decision": next(
                    (
                        int(replan["decision"])
                        for replan in replans
                        if not replan["exact_q_correct"]
                    ),
                    -1,
                ),
            }
        )
        for replan in replans:
            true = replan["true_state"]
            pred = replan["planning_state"]
            decision = int(replan["decision"])
            correct = int(replan["exact_q_correct"])
            decision_totals[(task_id, observer_seed, decision)] += 1
            decision_correct[(task_id, observer_seed, decision)] += correct
            stage_key = (
                task_id,
                observer_seed,
                int(true["cube_stage"]),
                int(true["window_stage"]),
            )
            by_stage_total[stage_key] += 1
            by_stage_correct[stage_key] += correct
            if not correct:
                confusions[
                    (
                        task_id,
                        observer_seed,
                        int(true["cube_stage"]),
                        int(true["window_stage"]),
                        int(pred["cube_stage"]),
                        int(pred["window_stage"]),
                    )
                ] += 1

    for (task_id, observer_seed, cube_stage, window_stage), total in sorted(
        by_stage_total.items()
    ):
        correct = by_stage_correct[(task_id, observer_seed, cube_stage, window_stage)]
        stage_rows.append(
            {
                "task_id": task_id,
                "observer_seed": observer_seed,
                "true_cube_stage": cube_stage,
                "true_window_stage": window_stage,
                "n_visits": total,
                "exact_q_accuracy": ratio(correct, total),
            }
        )

    decision_rows = [
        {
            "task_id": task_id,
            "observer_seed": observer_seed,
            "decision": decision,
            "n_visits": total,
            "exact_q_accuracy": ratio(
                decision_correct[(task_id, observer_seed, decision)], total
            ),
        }
        for (task_id, observer_seed, decision), total in sorted(decision_totals.items())
    ]
    confusion_rows = [
        {
            "task_id": key[0],
            "observer_seed": key[1],
            "true_cube_stage": key[2],
            "true_window_stage": key[3],
            "pred_cube_stage": key[4],
            "pred_window_stage": key[5],
            "count": count,
        }
        for key, count in confusions.most_common()
    ]

    condition_rows: list[dict] = []
    for task_id in (4, 5, 0):
        for observer_seed in (0, 1, 2):
            subset = [
                row
                for row in episode_rows
                if row["observer_seed"] == observer_seed
                and (task_id == 0 or row["task_id"] == task_id)
            ]
            for condition, selected in (
                ("no_q_error", [row for row in subset if not row["any_q_error"]]),
                ("any_q_error", [row for row in subset if row["any_q_error"]]),
            ):
                condition_rows.append(
                    {
                        "task_id": task_id,
                        "observer_seed": observer_seed,
                        "condition": condition,
                        "n": len(selected),
                        "success_rate": ratio(
                            sum(row["success"] for row in selected), len(selected)
                        ),
                    }
                )

    disagreement_rows: list[dict] = []
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in episode_rows:
        grouped[(row["task_id"], row["reset_seed"])].append(row)
    for (task_id, reset_seed), rows in sorted(grouped.items()):
        successes = [row["success"] for row in sorted(rows, key=lambda x: x["observer_seed"])]
        disagreement_rows.append(
            {
                "task_id": task_id,
                "reset_seed": reset_seed,
                "seed0_success": successes[0],
                "seed1_success": successes[1],
                "seed2_success": successes[2],
                "observer_outcome_disagreement": int(len(set(successes)) > 1),
                "oracle_success": oracle[(task_id, reset_seed)],
            }
        )

    summary = {
        "protocol": "scene_event_perception_replication_diagnostic_v1",
        "num_episodes": len(episode_rows),
        "num_resets": len(grouped),
        "top_confusions": confusion_rows[:20],
        "outcome_disagreement_rate": ratio(
            sum(row["observer_outcome_disagreement"] for row in disagreement_rows),
            len(disagreement_rows),
        ),
        "condition_rows": condition_rows,
        "interpretation": (
            "post-replication diagnostic only; it does not alter the locked FAIL verdict "
            "or license a confirmatory claim"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for filename, rows in (
        ("episodes.csv", episode_rows),
        ("by_stage.csv", stage_rows),
        ("by_decision.csv", decision_rows),
        ("confusions.csv", confusion_rows),
        ("conditioned_success.csv", condition_rows),
        ("observer_disagreement.csv", disagreement_rows),
    ):
        with (args.out_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out_dir / "DIAGNOSIS.md").write_text(
        "# Learned-q replication failure diagnosis\n\n"
        "This is a post-hoc mechanistic audit and does not revise the locked replication "
        "verdict.  See `summary.json` and the stage/confusion tables.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
