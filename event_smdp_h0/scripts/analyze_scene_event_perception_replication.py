#!/usr/bin/env python3
"""Analyze the fresh-reset, three-observer-seed perception replication."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PROTOCOL = "scene_event_perception_replication_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=128)
    parser.add_argument("--expected-observer-seeds", default="0,1,2")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> float:
    left_only = int(np.sum((left == 1) & (right == 0)))
    right_only = int(np.sum((left == 0) & (right == 1)))
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob("*/result.json"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(
            f"expected {args.expected_shards} shards under {args.input_root}, found {len(paths)}"
        )
    expected_observer_seeds = tuple(
        sorted({int(value) for value in args.expected_observer_seeds.split(",") if value})
    )
    expected_reset_keys = {
        *((4, 84400 + local) for local in range(64)),
        *((5, 84500 + local) for local in range(64)),
    }
    baseline: dict[tuple[int, int, str], dict] = {}
    learned: dict[tuple[int, int, int], dict] = {}
    transition_hashes: set[str] = set()
    observer_hashes: dict[int, set[str]] = {seed: set() for seed in expected_observer_seeds}
    seen_reset_keys: set[tuple[int, int]] = set()
    budgets: set[int] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError(f"unexpected protocol in {path}")
        reset_key = (int(payload["task_id"]), int(payload["reset_seed"]))
        if reset_key in seen_reset_keys:
            raise RuntimeError(f"duplicate physical reset {reset_key}")
        seen_reset_keys.add(reset_key)
        transition_hashes.add(payload["transition"]["sha256"])
        budgets.add(int(payload["budget"]))
        if tuple(payload["observer_seeds"]) != expected_observer_seeds:
            raise RuntimeError(f"observer seed mismatch in {path}")
        for observer_seed in expected_observer_seeds:
            observer_hashes[observer_seed].add(
                payload["observers"][str(observer_seed)]["sha256"]
            )
        for row in payload["results"]:
            task_id = int(row["task_id"])
            reset_seed = int(row["reset_seed"])
            arm = str(row["arm"])
            if arm == "learned_latent":
                key = (task_id, reset_seed, int(row["observer_seed"]))
                learned[key] = row
            else:
                baseline[(task_id, reset_seed, arm)] = row
    if seen_reset_keys != expected_reset_keys:
        missing = sorted(expected_reset_keys - seen_reset_keys)
        extra = sorted(seen_reset_keys - expected_reset_keys)
        raise RuntimeError(f"locked reset mismatch; missing={missing}, extra={extra}")
    if len(transition_hashes) != 1 or budgets != {112}:
        raise RuntimeError(
            f"expected one transition checkpoint and K=112, got {transition_hashes}, {budgets}"
        )
    if any(len(hashes) != 1 for hashes in observer_hashes.values()):
        raise RuntimeError(f"observer checkpoint changed within run: {observer_hashes}")

    rng = np.random.default_rng(args.seed)
    reset_keys = sorted(seen_reset_keys)
    oracle = {
        key: int(baseline[(key[0], key[1], "oracle_event")]["success"])
        for key in reset_keys
    }
    terminal = {
        key: int(baseline[(key[0], key[1], "abstract_terminal")]["success"])
        for key in reset_keys
    }
    seed_rows: list[dict] = []
    for observer_seed in expected_observer_seeds:
        for task_id in (4, 5, 0):
            keys = [key for key in reset_keys if task_id == 0 or key[0] == task_id]
            success = np.asarray(
                [int(learned[(key[0], key[1], observer_seed)]["success"]) for key in keys],
                dtype=float,
            )
            oracle_values = np.asarray([oracle[key] for key in keys], dtype=float)
            terminal_values = np.asarray([terminal[key] for key in keys], dtype=float)
            q_values = [
                float(replan["exact_q_correct"])
                for key in keys
                for replan in learned[(key[0], key[1], observer_seed)]["replans"]
            ]
            seed_rows.append(
                {
                    "observer_seed": observer_seed,
                    "task_id": task_id,
                    "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                    "n": len(keys),
                    "success_rate": float(success.mean()),
                    "success_ci95": bootstrap_ci(success, args.bootstrap, rng),
                    "visited_exact_q_accuracy": float(np.mean(q_values)),
                    "gain_over_terminal": float((success - terminal_values).mean()),
                    "gain_ci95": bootstrap_ci(success - terminal_values, args.bootstrap, rng),
                    "gap_to_oracle_q": float((success - oracle_values).mean()),
                    "oracle_gap_ci95": bootstrap_ci(success - oracle_values, args.bootstrap, rng),
                    "mcnemar_vs_terminal_p": exact_mcnemar(success, terminal_values),
                }
            )

    aggregate_rows: list[dict] = []
    for task_id in (4, 5, 0):
        keys = [key for key in reset_keys if task_id == 0 or key[0] == task_id]
        learned_per_reset = np.asarray(
            [
                np.mean(
                    [
                        int(learned[(key[0], key[1], observer_seed)]["success"])
                        for observer_seed in expected_observer_seeds
                    ]
                )
                for key in keys
            ],
            dtype=float,
        )
        oracle_values = np.asarray([oracle[key] for key in keys], dtype=float)
        terminal_values = np.asarray([terminal[key] for key in keys], dtype=float)
        aggregate_rows.append(
            {
                "task_id": task_id,
                "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                "n_resets": len(keys),
                "n_observer_seeds": len(expected_observer_seeds),
                "mean_learned_success": float(learned_per_reset.mean()),
                "learned_ci95_reset_clustered": bootstrap_ci(
                    learned_per_reset, args.bootstrap, rng
                ),
                "oracle_success": float(oracle_values.mean()),
                "terminal_success": float(terminal_values.mean()),
                "mean_gain_over_terminal": float(
                    (learned_per_reset - terminal_values).mean()
                ),
                "gain_ci95_reset_clustered": bootstrap_ci(
                    learned_per_reset - terminal_values, args.bootstrap, rng
                ),
                "mean_gap_to_oracle_q": float(
                    (learned_per_reset - oracle_values).mean()
                ),
                "oracle_gap_ci95_reset_clustered": bootstrap_ci(
                    learned_per_reset - oracle_values, args.bootstrap, rng
                ),
            }
        )

    pooled_seed_rows = [row for row in seed_rows if row["task_id"] == 0]
    pooled = next(row for row in aggregate_rows if row["task_id"] == 0)
    task_rows = [row for row in aggregate_rows if row["task_id"] in (4, 5)]
    checks = {
        "every_seed_pooled_success_ge_75pct": all(
            row["success_rate"] >= 0.75 for row in pooled_seed_rows
        ),
        "each_task_mean_success_ge_65pct": all(
            row["mean_learned_success"] >= 0.65 for row in task_rows
        ),
        "pooled_gain_ge_50pct_and_ci_above_zero": (
            pooled["mean_gain_over_terminal"] >= 0.50
            and pooled["gain_ci95_reset_clustered"][0] > 0.0
        ),
        "every_seed_oracle_gap_within_25pct": all(
            row["gap_to_oracle_q"] >= -0.25 for row in pooled_seed_rows
        ),
    }
    passed = all(checks.values())
    verdict = (
        "LEARNED_EVENT_PERCEPTION_REPLICATION_PASS"
        if passed
        else "LEARNED_EVENT_PERCEPTION_REPLICATION_FAIL"
    )
    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "checks": checks,
        "budget": 112,
        "transition_sha256": next(iter(transition_hashes)),
        "observer_sha256": {
            str(seed): next(iter(observer_hashes[seed]))
            for seed in expected_observer_seeds
        },
        "seed_rows": seed_rows,
        "aggregate_rows": aggregate_rows,
        "scope": (
            "three latent observer initializations, one frozen abstract transition, and "
            "128 fresh physical resets; inference still uses a hand-specified event automaton"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for filename, rows in (
        ("per_seed.csv", seed_rows),
        ("aggregate.csv", aggregate_rows),
    ):
        with (args.out_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    interpretation = (
        "All locked replication checks passed."
        if passed
        else "At least one locked replication check failed; this is not a confirmatory PASS."
    )
    (args.out_dir / "DECISION.md").write_text(
        "# Scene learned event-state perception replication\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"At K=112, reset-clustered mean learned success is "
        f"{100.0 * pooled['mean_learned_success']:.2f}%, versus "
        f"{100.0 * pooled['terminal_success']:.2f}% terminal and "
        f"{100.0 * pooled['oracle_success']:.2f}% simulator-q event planning.\n\n"
        f"{interpretation}  Even a PASS would validate learned current-q perception, "
        "not discovery of the event automaton.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
