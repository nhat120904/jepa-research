#!/usr/bin/env python3
"""Compare H1b Event-SMDP and terminal-head success across search widths."""

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
    parser.add_argument("--seed", type=int, default=20260907)
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


def paths(root: Path, expected: int) -> list[Path]:
    result = sorted(root.glob("*/result.json"))
    if len(result) != expected:
        raise RuntimeError(f"expected {expected} shards under {root}, found {len(result)}")
    return result


def main() -> None:
    args = parse_args()
    event: dict[tuple[int, int, int], dict] = {}
    terminal: dict[tuple[int, int, int], dict] = {}
    for path in paths(args.event_root, args.expected_shards):
        payload = json.loads(path.read_text())
        if payload.get("protocol") != "scene_h2_search_width_audit_v1":
            raise RuntimeError(f"unexpected event protocol in {path}")
        for row in payload["closed_loop"]:
            key = (int(payload["task_id"]), int(payload["reset_seed"]), int(row["budget"]))
            event[key] = row
    for path in paths(args.terminal_root, args.expected_shards):
        payload = json.loads(path.read_text())
        if payload.get("protocol") != "scene_h1_learnability_v1":
            raise RuntimeError(f"unexpected terminal protocol in {path}")
        for row in payload["results"]:
            if row["feature_view"] != "latent" or row["head"] != "terminal":
                raise RuntimeError(f"non-terminal row in terminal sweep {path}")
            key = (int(row["task_id"]), int(row["reset_seed"]), int(row["budget"]))
            terminal[key] = row
    if set(event) != set(terminal):
        raise RuntimeError("event and terminal paired cells do not match")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    budgets = sorted({key[2] for key in event})
    for budget in budgets:
        for task_id in (4, 5, 0):
            keys = sorted(
                key
                for key in event
                if key[2] == budget and (task_id == 0 or key[0] == task_id)
            )
            event_success = np.asarray([int(event[key]["success"]) for key in keys], dtype=float)
            terminal_success = np.asarray(
                [int(terminal[key]["success"]) for key in keys], dtype=float
            )
            differences = event_success - terminal_success
            event_only = int(np.sum((event_success == 1) & (terminal_success == 0)))
            terminal_only = int(np.sum((event_success == 0) & (terminal_success == 1)))
            rows.append(
                {
                    "task_id": task_id,
                    "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                    "budget": budget,
                    "n": len(keys),
                    "event_success_rate": float(event_success.mean()),
                    "event_ci95": bootstrap_ci(event_success, args.bootstrap, rng),
                    "terminal_success_rate": float(terminal_success.mean()),
                    "terminal_ci95": bootstrap_ci(terminal_success, args.bootstrap, rng),
                    "paired_difference": float(differences.mean()),
                    "paired_ci95": bootstrap_ci(differences, args.bootstrap, rng),
                    "event_only": event_only,
                    "terminal_only": terminal_only,
                    "mcnemar_exact_p": exact_mcnemar(event_only, terminal_only),
                }
            )
    primary = next(
        row for row in rows if row["task_id"] == 0 and row["budget"] == args.primary_budget
    )
    retained = (
        primary["paired_difference"] >= 0.10
        and primary["paired_ci95"][0] > 0.0
        and primary["mcnemar_exact_p"] < 0.05
    )
    verdict = (
        "EVENT_SMDP_ADVANTAGE_RETAINS_AT_WIDE_SEARCH"
        if retained
        else "NO_EVENT_SMDP_ADVANTAGE_AT_WIDE_SEARCH"
    )
    summary = {
        "protocol": "scene_wide_budget_baseline_v1",
        "verdict": verdict,
        "primary_budget": args.primary_budget,
        "primary": primary,
        "rows": rows,
        "scope": (
            "paired deterministic reruns on the same 16 reset seeds; latent terminal head "
            "uses the same recursive skill dynamics, UCT, horizon, and query budgets"
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
        "# Scene wide-budget terminal baseline\n\n"
        f"Verdict: **{verdict}**\n\n"
        "This comparison determines whether the structured Event-SMDP advantage "
        "survives when terminal-head UCT receives the same wide search budget.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

