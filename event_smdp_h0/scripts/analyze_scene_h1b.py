#!/usr/bin/env python3
"""Analyze the H1b closure intervention and compare to locked H1 terminal."""

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
    parser.add_argument("--h1b-root", type=Path, required=True)
    parser.add_argument("--h1-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=16)
    parser.add_argument("--primary-budget", type=int, default=28)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def exact_mcnemar(event_only: int, terminal_only: int) -> float:
    n = event_only + terminal_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(event_only, terminal_only) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def load_rows(root: Path, protocol: str, expected: int) -> list[dict]:
    paths = sorted(root.glob("*/result.json"))
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} shards under {root}, found {len(paths)}")
    rows: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("protocol") != protocol:
            raise RuntimeError(f"unexpected protocol in {path}")
        rows.extend(payload["results"])
    return rows


def trace_audit(rows: list[dict], budget: int) -> dict:
    selected = [row for row in rows if int(row["budget"]) == budget]
    rewards: list[float] = []
    first_progress: list[int] = []
    deployed_progress: list[int] = []
    for row in selected:
        for replan in row["replans"]:
            rewards.extend(
                float(item["predicted_reward"])
                for item in replan["search"].get("evaluations", [])
            )
            deployed_progress.append(int(bool(replan["deployed"]["new_events"])))
        first_progress.append(int(bool(row["replans"][0]["deployed"]["new_events"])))
    reward_array = np.asarray(rewards, dtype=np.float64)
    return {
        "num_candidate_scores": int(len(reward_array)),
        "fraction_candidates_predicted_success": float(np.mean(reward_array >= 0.999)),
        "mean_predicted_reward": float(reward_array.mean()),
        "first_action_progress_rate": float(np.mean(first_progress)),
        "deployed_action_progress_rate": float(np.mean(deployed_progress)),
    }


def main() -> None:
    args = parse_args()
    h1b = load_rows(args.h1b_root, "scene_h1b_abstract_closure_v1", args.expected_shards)
    h1_all = load_rows(args.h1_root, "scene_h1_learnability_v1", args.expected_shards)
    baseline = [
        row
        for row in h1_all
        if row["feature_view"] == "latent" and row["head"] == "terminal"
    ]
    b_index = {
        (int(row["task_id"]), int(row["reset_seed"]), int(row["budget"])): row
        for row in baseline
    }
    rng = np.random.default_rng(args.seed)
    rates: list[dict] = []
    contrasts: list[dict] = []
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in h1b:
        grouped[(int(row["task_id"]), int(row["budget"]))].append(row)
        grouped[(0, int(row["budget"]))].append(row)
    for (task_id, budget), rows in sorted(grouped.items()):
        success = np.asarray([int(row["success"]) for row in rows], dtype=float)
        rates.append(
            {
                "task_id": task_id,
                "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                "budget": budget,
                "n": len(rows),
                "success_rate": float(success.mean()),
                "ci95": bootstrap_ci(success, args.bootstrap, rng),
                "mean_replans": float(np.mean([row["num_replans"] for row in rows])),
            }
        )
        differences: list[int] = []
        event_only = 0
        terminal_only = 0
        for row in rows:
            terminal = b_index[(int(row["task_id"]), int(row["reset_seed"]), budget)]
            event_success = int(row["success"])
            terminal_success = int(terminal["success"])
            differences.append(event_success - terminal_success)
            event_only += int(event_success == 1 and terminal_success == 0)
            terminal_only += int(event_success == 0 and terminal_success == 1)
        difference = np.asarray(differences, dtype=float)
        contrasts.append(
            {
                "task_id": task_id,
                "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                "budget": budget,
                "n": len(rows),
                "paired_difference": float(difference.mean()),
                "paired_ci95": bootstrap_ci(difference, args.bootstrap, rng),
                "event_only": event_only,
                "terminal_only": terminal_only,
                "mcnemar_exact_p": exact_mcnemar(event_only, terminal_only),
            }
        )
    primary_rate = next(
        row for row in rates if row["task_id"] == 0 and row["budget"] == args.primary_budget
    )
    primary_contrast = next(
        row
        for row in contrasts
        if row["task_id"] == 0 and row["budget"] == args.primary_budget
    )
    passed = (
        primary_rate["success_rate"] >= 0.50
        and primary_contrast["paired_difference"] >= 0.10
        and primary_contrast["paired_ci95"][0] > 0.0
    )
    verdict = "H1B_ABSTRACT_CLOSURE_PASS" if passed else "H1B_ABSTRACT_CLOSURE_FAIL"
    h1_event_time = [
        row
        for row in h1_all
        if row["feature_view"] == "latent" and row["head"] == "event_time"
    ]
    summary = {
        "protocol": "scene_h1b_abstract_closure_v1",
        "verdict": verdict,
        "primary_budget": args.primary_budget,
        "primary_rate": primary_rate,
        "primary_contrast": primary_contrast,
        "rates": rates,
        "contrasts": contrasts,
        "trace_audit": {
            "h1_latent_contextual_event_time": trace_audit(h1_event_time, args.primary_budget),
            "h1b_abstract_smdp": trace_audit(h1b, args.primary_budget),
        },
        "scope": (
            "post-H1 mechanism intervention; current event state remains simulator-monitored "
            "at physical replans and this result cannot be described as learned event perception"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for name, rows in (("rates.csv", rates), ("contrasts.csv", contrasts)):
        with (args.out_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out_dir / "DECISION.md").write_text(
        "# Scene H1b abstract-closure audit\n\n"
        f"Verdict: **{verdict}**\n\n"
        "H1 remains the locked learned contextual-model result.  H1b only tests "
        "whether recursive feature dynamics caused its closed-loop collapse.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

