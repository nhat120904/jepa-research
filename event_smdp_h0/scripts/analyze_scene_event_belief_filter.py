#!/usr/bin/env python3
"""Analyze the fresh-reset prediction--correction event-filter pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROTOCOL = "scene_event_belief_filter_pilot_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob("*/result.json"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")
    expected_resets = {
        *((4, 85400 + local) for local in range(16)),
        *((5, 85501 + local) for local in range(16)),
    }
    rows: dict[tuple[int, int, str, int | None], dict] = {}
    reset_keys: set[tuple[int, int]] = set()
    transition_hashes: set[str] = set()
    observer_hashes: dict[int, set[str]] = {seed: set() for seed in (0, 1, 2)}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError(f"unexpected protocol in {path}")
        reset_key = (int(payload["task_id"]), int(payload["reset_seed"]))
        if reset_key in reset_keys:
            raise RuntimeError(f"duplicate reset {reset_key}")
        reset_keys.add(reset_key)
        transition_hashes.add(payload["transition"]["sha256"])
        for observer_seed in (0, 1, 2):
            observer_hashes[observer_seed].add(
                payload["observers"][str(observer_seed)]["sha256"]
            )
        for row in payload["results"]:
            rows[
                (
                    int(row["task_id"]),
                    int(row["reset_seed"]),
                    str(row["arm"]),
                    None if row["observer_seed"] is None else int(row["observer_seed"]),
                )
            ] = row
    if reset_keys != expected_resets:
        raise RuntimeError("fresh reset set does not match the locked protocol")
    if len(transition_hashes) != 1 or any(
        len(hashes) != 1 for hashes in observer_hashes.values()
    ):
        raise RuntimeError("checkpoint identity changed within the pilot")

    rng = np.random.default_rng(args.seed)
    per_seed: list[dict] = []
    for observer_seed in (0, 1, 2):
        for task_id in (4, 5, 0):
            keys = sorted(key for key in reset_keys if task_id == 0 or key[0] == task_id)
            fresh = np.asarray(
                [int(rows[(key[0], key[1], "fresh_q", observer_seed)]["success"]) for key in keys],
                dtype=float,
            )
            filtered = np.asarray(
                [
                    int(rows[(key[0], key[1], "filtered_q", observer_seed)]["success"])
                    for key in keys
                ],
                dtype=float,
            )
            observation_correct = [
                float(replan["observation_q_correct"])
                for key in keys
                for replan in rows[(key[0], key[1], "filtered_q", observer_seed)]["replans"]
            ]
            planning_correct = [
                float(replan["planning_q_correct"])
                for key in keys
                for replan in rows[(key[0], key[1], "filtered_q", observer_seed)]["replans"]
            ]
            gain = filtered - fresh
            per_seed.append(
                {
                    "observer_seed": observer_seed,
                    "task_id": task_id,
                    "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                    "n": len(keys),
                    "fresh_success": float(fresh.mean()),
                    "filtered_success": float(filtered.mean()),
                    "paired_gain": float(gain.mean()),
                    "gain_ci95": bootstrap_ci(gain, args.bootstrap, rng),
                    "filtered_observation_q_accuracy": float(np.mean(observation_correct)),
                    "filtered_planning_q_accuracy": float(np.mean(planning_correct)),
                }
            )

    aggregate: list[dict] = []
    for task_id in (4, 5, 0):
        keys = sorted(key for key in reset_keys if task_id == 0 or key[0] == task_id)
        fresh_per_reset = np.asarray(
            [
                np.mean(
                    [
                        int(rows[(key[0], key[1], "fresh_q", seed)]["success"])
                        for seed in (0, 1, 2)
                    ]
                )
                for key in keys
            ],
            dtype=float,
        )
        filtered_per_reset = np.asarray(
            [
                np.mean(
                    [
                        int(rows[(key[0], key[1], "filtered_q", seed)]["success"])
                        for seed in (0, 1, 2)
                    ]
                )
                for key in keys
            ],
            dtype=float,
        )
        oracle = np.asarray(
            [int(rows[(key[0], key[1], "oracle_event", None)]["success"]) for key in keys],
            dtype=float,
        )
        terminal = np.asarray(
            [
                int(rows[(key[0], key[1], "abstract_terminal", None)]["success"])
                for key in keys
            ],
            dtype=float,
        )
        gain = filtered_per_reset - fresh_per_reset
        aggregate.append(
            {
                "task_id": task_id,
                "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                "n_resets": len(keys),
                "fresh_mean_success": float(fresh_per_reset.mean()),
                "filtered_mean_success": float(filtered_per_reset.mean()),
                "filtered_ci95": bootstrap_ci(filtered_per_reset, args.bootstrap, rng),
                "paired_gain": float(gain.mean()),
                "gain_ci95": bootstrap_ci(gain, args.bootstrap, rng),
                "oracle_success": float(oracle.mean()),
                "terminal_success": float(terminal.mean()),
            }
        )

    task5 = next(row for row in aggregate if row["task_id"] == 5)
    seed1_task5 = next(
        row for row in per_seed if row["observer_seed"] == 1 and row["task_id"] == 5
    )
    pooled_filtered = [row for row in per_seed if row["task_id"] == 0]
    checks = {
        "filtered_task5_mean_success_ge_75pct": task5["filtered_mean_success"] >= 0.75,
        "task5_gain_ge_15pct_and_ci_above_zero": (
            task5["paired_gain"] >= 0.15 and task5["gain_ci95"][0] > 0.0
        ),
        "seed1_task5_gain_ge_25pct": seed1_task5["paired_gain"] >= 0.25,
        "every_filtered_seed_pooled_success_ge_75pct": all(
            row["filtered_success"] >= 0.75 for row in pooled_filtered
        ),
    }
    passed = all(checks.values())
    verdict = "EVENT_BELIEF_FILTER_PILOT_PASS" if passed else "EVENT_BELIEF_FILTER_PILOT_FAIL"
    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "checks": checks,
        "transition_sha256": next(iter(transition_hashes)),
        "observer_sha256": {
            str(seed): next(iter(observer_hashes[seed])) for seed in (0, 1, 2)
        },
        "aggregate": aggregate,
        "per_seed": per_seed,
        "scope": (
            "post-diagnostic but preregistered fresh-reset pilot; a PASS licenses a larger "
            "confirmatory filter evaluation only"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for filename, data in (("aggregate.csv", aggregate), ("per_seed.csv", per_seed)):
        with (args.out_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (args.out_dir / "DECISION.md").write_text(
        "# Scene prediction--correction event-belief filter\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"On task 5, filtered mean success is "
        f"{100.0 * task5['filtered_mean_success']:.2f}% versus "
        f"{100.0 * task5['fresh_mean_success']:.2f}% from single-frame q.\n\n"
        "This is an exploratory fresh-reset pilot following the replication failure audit.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
