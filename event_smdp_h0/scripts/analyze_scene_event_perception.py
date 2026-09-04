#!/usr/bin/env python3
"""Analyze the paired non-oracle current-q Scene pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--terminal-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=16)
    parser.add_argument("--primary-budget", type=int, default=112)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def result_paths(root: Path, expected: int) -> list[Path]:
    paths = sorted(root.glob("*/result.json"))
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} shards under {root}, found {len(paths)}")
    return paths


def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    learned: dict[tuple[int, int, str, int], dict] = {}
    oracle: dict[tuple[int, int, int], dict] = {}
    terminal: dict[tuple[int, int, int], dict] = {}
    for path in result_paths(args.input_root, args.expected_shards):
        payload = json.loads(path.read_text())
        if payload.get("protocol") != "scene_event_perception_v1":
            raise RuntimeError(f"unexpected learned protocol in {path}")
        for row in payload["results"]:
            key = (
                int(row["task_id"]),
                int(row["reset_seed"]),
                str(row["feature_view"]),
                int(row["budget"]),
            )
            learned[key] = row
    for path in result_paths(args.oracle_root, args.expected_shards):
        payload = json.loads(path.read_text())
        for row in payload["closed_loop"]:
            oracle[(int(payload["task_id"]), int(payload["reset_seed"]), int(row["budget"]))] = row
    for path in result_paths(args.terminal_root, args.expected_shards):
        payload = json.loads(path.read_text())
        for row in payload["results"]:
            terminal[(int(row["task_id"]), int(row["reset_seed"]), int(row["budget"]))] = row

    rng = np.random.default_rng(args.seed)
    views = sorted({key[2] for key in learned})
    budgets = sorted({key[3] for key in learned})
    rates: list[dict] = []
    contrasts: list[dict] = []
    for view in views:
        for budget in budgets:
            for task_id in (4, 5, 0):
                keys = sorted(
                    key
                    for key in learned
                    if key[2] == view
                    and key[3] == budget
                    and (task_id == 0 or key[0] == task_id)
                )
                success = np.asarray([int(learned[key]["success"]) for key in keys], dtype=float)
                exact_q = [
                    float(replan["exact_q_correct"])
                    for key in keys
                    for replan in learned[key]["replans"]
                ]
                cube = [
                    float(replan["cube_correct"])
                    for key in keys
                    for replan in learned[key]["replans"]
                ]
                window = [
                    float(replan["window_correct"])
                    for key in keys
                    for replan in learned[key]["replans"]
                ]
                rates.append(
                    {
                        "feature_view": view,
                        "task_id": task_id,
                        "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                        "budget": budget,
                        "n": len(keys),
                        "success_rate": float(success.mean()),
                        "success_ci95": bootstrap_ci(success, args.bootstrap, rng),
                        "visited_exact_q_accuracy": float(np.mean(exact_q)),
                        "visited_cube_accuracy": float(np.mean(cube)),
                        "visited_window_accuracy": float(np.mean(window)),
                    }
                )
                terminal_success = np.asarray(
                    [int(terminal[(key[0], key[1], budget)]["success"]) for key in keys],
                    dtype=float,
                )
                oracle_success = np.asarray(
                    [int(oracle[(key[0], key[1], budget)]["success"]) for key in keys],
                    dtype=float,
                )
                gain = success - terminal_success
                retention = success - oracle_success
                contrasts.append(
                    {
                        "feature_view": view,
                        "task_id": task_id,
                        "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                        "budget": budget,
                        "n": len(keys),
                        "paired_gain_over_terminal": float(gain.mean()),
                        "gain_ci95": bootstrap_ci(gain, args.bootstrap, rng),
                        "paired_gap_to_oracle_q": float(retention.mean()),
                        "oracle_gap_ci95": bootstrap_ci(retention, args.bootstrap, rng),
                    }
                )
    primary: dict[str, dict] = {}
    passed: dict[str, bool] = {}
    for view in views:
        rate = next(
            row
            for row in rates
            if row["feature_view"] == view
            and row["task_id"] == 0
            and row["budget"] == args.primary_budget
        )
        contrast = next(
            row
            for row in contrasts
            if row["feature_view"] == view
            and row["task_id"] == 0
            and row["budget"] == args.primary_budget
        )
        primary[view] = {"rate": rate, "contrast": contrast}
        passed[view] = (
            rate["success_rate"] >= 0.75
            and contrast["paired_gain_over_terminal"] >= 0.50
            and contrast["gain_ci95"][0] > 0.0
            and contrast["paired_gap_to_oracle_q"] >= -0.25
        )
    if passed.get("latent", False):
        verdict = "LEARNED_EVENT_PERCEPTION_PILOT_PASS"
    elif passed.get("privileged", False):
        verdict = "VISUAL_EVENT_REPRESENTATION_CEILING"
    else:
        verdict = "SINGLE_OBSERVATION_EVENT_STATE_FAIL"
    summary = {
        "protocol": "scene_event_perception_v1",
        "verdict": verdict,
        "primary_budget": args.primary_budget,
        "passes": passed,
        "primary": primary,
        "rates": rates,
        "contrasts": contrasts,
        "scope": (
            "pilot on one observer/transition seed and 16 resets; q is inferred from each "
            "current observation but its automaton and supervised labels remain hand specified"
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
        "# Scene learned event-state perception gate\n\n"
        f"Verdict: **{verdict}**\n\n"
        "A positive latent verdict licenses multi-seed, larger-reset replication; "
        "it does not remove the hand-specified event-label limitation.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

