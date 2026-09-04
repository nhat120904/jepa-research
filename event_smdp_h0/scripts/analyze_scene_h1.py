#!/usr/bin/env python3
"""Aggregate the paired Scene H1 learnability pilot on a compute node."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--primary-budget", type=int, default=28)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def exact_mcnemar(event_only: int, terminal_only: int) -> float:
    n = event_only + terminal_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(event_only, terminal_only) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob("*/result.json"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")
    records: dict[tuple, dict] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["protocol"] != "scene_h1_learnability_v1":
            raise RuntimeError(f"unexpected protocol in {path}")
        for row in payload["results"]:
            key = (
                int(row["task_id"]),
                int(row["reset_seed"]),
                int(row["model_seed"]),
                str(row["feature_view"]),
                str(row["head"]),
                int(row["budget"]),
            )
            if key in records:
                raise RuntimeError(f"duplicate evaluation cell: {key}")
            records[key] = row

    rng = np.random.default_rng(args.seed)
    rates: list[dict] = []
    grouped: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for row in records.values():
        grouped[
            (
                str(row["feature_view"]),
                str(row["head"]),
                int(row["task_id"]),
                int(row["budget"]),
            )
        ].append(row)
        grouped[
            (
                str(row["feature_view"]),
                str(row["head"]),
                0,
                int(row["budget"]),
            )
        ].append(row)
    for (view, head, task_id, budget), rows in sorted(grouped.items()):
        success = np.asarray([int(row["success"]) for row in rows], dtype=float)
        rates.append(
            {
                "feature_view": view,
                "head": head,
                "task_id": task_id,
                "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                "budget": budget,
                "n": len(rows),
                "success_rate": float(success.mean()),
                "ci95": bootstrap_ci(success, args.bootstrap, rng),
                "mean_replans": float(np.mean([row["num_replans"] for row in rows])),
            }
        )

    contrasts: list[dict] = []
    heads = ("event_bce", "event_time")
    views = sorted({str(row["feature_view"]) for row in records.values()})
    budgets = sorted({int(row["budget"]) for row in records.values()})
    for view in views:
        for head in heads:
            for task_id in (4, 5, 0):
                for budget in budgets:
                    keys = sorted(
                        {
                            (int(row["task_id"]), int(row["reset_seed"]), int(row["model_seed"]))
                            for row in records.values()
                            if row["feature_view"] == view
                            and int(row["budget"]) == budget
                            and (task_id == 0 or int(row["task_id"]) == task_id)
                        }
                    )
                    differences: list[int] = []
                    event_only = 0
                    terminal_only = 0
                    for actual_task, reset_seed, model_seed in keys:
                        terminal = records[
                            (actual_task, reset_seed, model_seed, view, "terminal", budget)
                        ]
                        event = records[
                            (actual_task, reset_seed, model_seed, view, head, budget)
                        ]
                        t_success = int(terminal["success"])
                        e_success = int(event["success"])
                        differences.append(e_success - t_success)
                        event_only += int(e_success == 1 and t_success == 0)
                        terminal_only += int(e_success == 0 and t_success == 1)
                    diff = np.asarray(differences, dtype=float)
                    contrasts.append(
                        {
                            "feature_view": view,
                            "event_head": head,
                            "task_id": task_id,
                            "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                            "budget": budget,
                            "n": len(diff),
                            "paired_difference": float(diff.mean()),
                            "paired_ci95": bootstrap_ci(diff, args.bootstrap, rng),
                            "event_only": event_only,
                            "terminal_only": terminal_only,
                            "mcnemar_exact_p": exact_mcnemar(event_only, terminal_only),
                        }
                    )

    primary = [
        row
        for row in contrasts
        if row["task_id"] == 0 and row["budget"] == args.primary_budget
    ]
    by_view: dict[str, dict] = {}
    for view in views:
        candidates = [row for row in primary if row["feature_view"] == view]
        by_view[view] = max(candidates, key=lambda row: row["paired_difference"])

    def passes(row: dict) -> bool:
        event_rate = next(
            rate["success_rate"]
            for rate in rates
            if rate["feature_view"] == row["feature_view"]
            and rate["head"] == row["event_head"]
            and rate["task_id"] == 0
            and rate["budget"] == row["budget"]
        )
        return (
            event_rate >= 0.50
            and row["paired_difference"] >= 0.10
            and row["paired_ci95"][0] > 0.0
        )

    latent_pass = passes(by_view["latent"])
    privileged_pass = passes(by_view["privileged"])
    if latent_pass:
        verdict = "H1_PILOT_GO_SEARCH_EXPLOITATION_AUDIT"
    elif privileged_pass:
        verdict = "H1_REPRESENTATION_CEILING"
    else:
        verdict = "H1_NO_LEARNED_EVENT_ADVANTAGE"

    summary = {
        "protocol": "scene_h1_learnability_v1",
        "num_shards": len(paths),
        "num_rows": len(records),
        "primary_budget": args.primary_budget,
        "primary_contrasts": by_view,
        "rates": rates,
        "contrasts": contrasts,
        "verdict": verdict,
        "scope": (
            "pilot gate with one or more learned seeds; current event state is an "
            "oracle monitor at real replans, and duration is auxiliary when duration_cost=0"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.out_dir / "rates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rates[0]))
        writer.writeheader()
        writer.writerows(rates)
    with (args.out_dir / "contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    (args.out_dir / "DECISION.md").write_text(
        "# Scene H1 learned-event pilot\n\n"
        f"Verdict: **{verdict}**\n\n"
        "This pilot does not validate SearchCal.  It only decides whether a "
        "learned event evaluator retains enough of the oracle H0 advantage to "
        "justify the H2 selected-error audit.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
