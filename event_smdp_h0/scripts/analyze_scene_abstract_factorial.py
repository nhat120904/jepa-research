#!/usr/bin/env python3
"""Paired event-feedback ablation under one shared abstract transition."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--terminal-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=16)
    parser.add_argument("--primary-budget", type=int, default=112)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def exact_mcnemar(event_only: int, terminal_only: int) -> float:
    n = event_only + terminal_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(event_only, terminal_only) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def result_paths(root: Path, expected: int) -> list[Path]:
    paths = sorted(root.glob("*/result.json"))
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} shards under {root}, found {len(paths)}")
    return paths


def main() -> None:
    args = parse_args()
    event: dict[tuple[int, int, int], dict] = {}
    terminal: dict[tuple[int, int, int], dict] = {}
    event_hashes: set[str] = set()
    terminal_hashes: set[str] = set()
    for path in result_paths(args.event_root, args.expected_shards):
        payload = json.loads(path.read_text())
        if payload.get("protocol") != "scene_h2_search_width_audit_v1":
            raise RuntimeError(f"unexpected event protocol in {path}")
        event_hashes.add(payload["checkpoint"]["sha256"])
        for row in payload["closed_loop"]:
            key = (int(payload["task_id"]), int(payload["reset_seed"]), int(row["budget"]))
            event[key] = row
    for path in result_paths(args.terminal_root, args.expected_shards):
        payload = json.loads(path.read_text())
        if payload.get("protocol") != "scene_abstract_factorial_v1":
            raise RuntimeError(f"unexpected abstract-terminal protocol in {path}")
        terminal_hashes.add(payload["checkpoint"]["sha256"])
        for row in payload["results"]:
            key = (int(row["task_id"]), int(row["reset_seed"]), int(row["budget"]))
            terminal[key] = row
    if event_hashes != terminal_hashes or len(event_hashes) != 1:
        raise RuntimeError("event and terminal arms did not use one identical checkpoint")
    if set(event) != set(terminal):
        raise RuntimeError("paired cells differ between factorial arms")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for budget in sorted({key[2] for key in event}):
        for task_id in (4, 5, 0):
            keys = sorted(
                key
                for key in event
                if key[2] == budget and (task_id == 0 or key[0] == task_id)
            )
            e = np.asarray([int(event[key]["success"]) for key in keys], dtype=float)
            t = np.asarray([int(terminal[key]["success"]) for key in keys], dtype=float)
            difference = e - t
            event_only = int(np.sum((e == 1) & (t == 0)))
            terminal_only = int(np.sum((e == 0) & (t == 1)))
            rows.append(
                {
                    "task_id": task_id,
                    "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                    "budget": budget,
                    "n": len(keys),
                    "event_success_rate": float(e.mean()),
                    "event_ci95": bootstrap_ci(e, args.bootstrap, rng),
                    "terminal_success_rate": float(t.mean()),
                    "terminal_ci95": bootstrap_ci(t, args.bootstrap, rng),
                    "paired_difference": float(difference.mean()),
                    "paired_ci95": bootstrap_ci(difference, args.bootstrap, rng),
                    "event_only": event_only,
                    "terminal_only": terminal_only,
                    "mcnemar_exact_p": exact_mcnemar(event_only, terminal_only),
                }
            )
    primary = next(
        row for row in rows if row["task_id"] == 0 and row["budget"] == args.primary_budget
    )
    passes = (
        primary["paired_difference"] >= 0.10
        and primary["paired_ci95"][0] > 0.0
        and primary["mcnemar_exact_p"] < 0.05
    )
    verdict = (
        "EVENT_FEEDBACK_CAUSAL_GAIN_UNDER_SHARED_MODEL"
        if passes
        else "NO_EVENT_FEEDBACK_GAIN_UNDER_SHARED_MODEL"
    )
    summary = {
        "protocol": "scene_abstract_factorial_v1",
        "verdict": verdict,
        "checkpoint_sha256": next(iter(event_hashes)),
        "primary_budget": args.primary_budget,
        "primary": primary,
        "rows": rows,
        "scope": (
            "same learned abstract transition checkpoint, UCT, skills, horizons, budgets, "
            "and reset seeds; arms differ only in event-progress versus terminal-probability feedback"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.out_dir / "paired.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "DECISION.md").write_text(
        "# Scene learned abstract-model factorial\n\n"
        f"Verdict: **{verdict}**\n\n"
        "The two arms share one checkpoint and differ only in planner feedback.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

