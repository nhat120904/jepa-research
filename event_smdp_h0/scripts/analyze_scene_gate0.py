#!/usr/bin/env python3
"""Aggregate paired OGBench-Scene H0 shards on a compute node."""

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
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def exact_mcnemar(discordant_event: int, discordant_terminal: int) -> float:
    n = discordant_event + discordant_terminal
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(discordant_event, discordant_terminal) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob("*/result.json"))
    if len(paths) != args.expected:
        raise RuntimeError(f"expected {args.expected} shards, found {len(paths)}")

    paired: dict[tuple[int, int, int], dict[str, dict]] = defaultdict(dict)
    protocol = None
    for path in paths:
        payload = json.loads(path.read_text())
        protocol = protocol or payload["protocol"]
        if payload["protocol"] != protocol:
            raise RuntimeError("mixed protocols")
        for row in payload["results"]:
            key = (int(row["task_id"]), int(row["reset_seed"]), int(row["budget_per_replan"]))
            arm = str(row["arm"])
            if arm in paired[key]:
                raise RuntimeError(f"duplicate arm for {key}: {arm}")
            paired[key][arm] = row

    rows: list[dict] = []
    for key, arms in sorted(paired.items()):
        if set(arms) != {"terminal_only", "event_state"}:
            raise RuntimeError(f"incomplete pair: {key}")
        terminal = arms["terminal_only"]
        event = arms["event_state"]
        rows.append(
            {
                "task_id": key[0],
                "reset_seed": key[1],
                "budget": key[2],
                "terminal_success": int(bool(terminal["success"])),
                "event_success": int(bool(event["success"])),
                "paired_difference": int(bool(event["success"])) - int(bool(terminal["success"])),
                "terminal_replans": int(terminal["num_replans"]),
                "event_replans": int(event["num_replans"]),
                "terminal_eval_skill_calls": int(terminal["eval_skill_calls"]),
                "event_eval_skill_calls": int(event["eval_skill_calls"]),
            }
        )

    rng = np.random.default_rng(args.seed)
    groups: list[dict] = []
    group_keys = sorted({(row["task_id"], row["budget"]) for row in rows})
    group_keys += [(0, budget) for budget in sorted({row["budget"] for row in rows})]
    for task_id, budget in group_keys:
        sample = [
            row for row in rows
            if row["budget"] == budget and (task_id == 0 or row["task_id"] == task_id)
        ]
        differences = np.asarray([row["paired_difference"] for row in sample], dtype=float)
        boot = np.empty(args.bootstrap, dtype=float)
        for index in range(args.bootstrap):
            boot[index] = rng.choice(differences, size=len(differences), replace=True).mean()
        e_only = sum(row["event_success"] == 1 and row["terminal_success"] == 0 for row in sample)
        t_only = sum(row["event_success"] == 0 and row["terminal_success"] == 1 for row in sample)
        groups.append(
            {
                "task_id": task_id,
                "task_label": "pooled" if task_id == 0 else f"task_{task_id}",
                "budget": budget,
                "n": len(sample),
                "terminal_success_rate": float(np.mean([row["terminal_success"] for row in sample])),
                "event_success_rate": float(np.mean([row["event_success"] for row in sample])),
                "paired_difference": float(differences.mean()),
                "paired_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
                "event_only": e_only,
                "terminal_only": t_only,
                "mcnemar_exact_p": exact_mcnemar(e_only, t_only),
            }
        )

    pooled = [group for group in groups if group["task_id"] == 0]
    best = max(pooled, key=lambda group: group["paired_difference"])
    verdict = (
        "PILOT_GO_LEARNED_EVENT_WM"
        if best["paired_difference"] >= 0.125 and best["event_only"] > best["terminal_only"]
        else "PILOT_NO_CLEAR_SUCCESS_ROOM"
    )
    summary = {
        "protocol": protocol,
        "num_shards": len(paths),
        "num_pairs": len(rows),
        "verdict": verdict,
        "groups": groups,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.out_dir / "paired.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "DECISION.md").write_text(
        "# OGBench-Scene H0 pilot decision\n\n"
        f"Verdict: **{verdict}**\n\n"
        "This is an oracle causal-room result, not learned-world-model evidence.\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

