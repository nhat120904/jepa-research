#!/usr/bin/env python3
"""Locked verdict for the skill-failure dose-response sweep."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np


PROTOCOL = "scene_skill_failure_analysis_v1"
BOOTSTRAP = 10000
SEEDS = (0, 1, 2)
SINGLE = ("oracle_event", "openloop_transition")
LEARNED = ("frame_full", "action_only_full", "obs_history_full", "history_full")
SOURCES = (*SINGLE, *LEARNED)
RATES = (0.00, 0.10, 0.20, 0.30)
DEAD_RECKONERS = ("action_only_full", "openloop_transition")
REFERENCE = "obs_history_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("skill-failure analysis must run inside a Slurm compute job")

    shards = sorted(args.eval_root.glob("*/result.json"), key=lambda p: int(p.parent.name))
    if len(shards) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(shards)}")

    cell: dict[int, dict[tuple[float, str], dict[object, dict]]] = {}
    for shard in shards:
        payload = json.loads(shard.read_text())
        seed = int(payload["reset_seed"])
        entry: dict[tuple[float, str], dict[object, dict]] = defaultdict(dict)
        for row in payload["results"]:
            slot = "single" if row["observer_seed"] is None else int(row["observer_seed"])
            exact = [r["exact_q_correct"] for r in row["replans"]]
            over = sum(
                1
                for r in row["replans"]
                if r["exact_q_correct"] is False
                and (
                    int(r["planning_state"]["cube_stage"]) > int(r["true_state"]["cube_stage"])
                    or int(r["planning_state"]["window_stage"]) > int(r["true_state"]["window_stage"])
                )
            )
            under = sum(
                1
                for r in row["replans"]
                if r["exact_q_correct"] is False
                and int(r["planning_state"]["cube_stage"]) <= int(r["true_state"]["cube_stage"])
                and int(r["planning_state"]["window_stage"]) <= int(r["true_state"]["window_stage"])
            )
            entry[(float(row["failure_rate"]), row["state_source"])][slot] = {
                "success": bool(row["success"]),
                "plan": list(row["deployed_skills"]),
                "exhausted": bool(row["exhausted_budget"]),
                "over_reads": over,
                "under_reads": under,
                "exact_q_rate": (
                    float(np.mean([float(bool(v)) for v in exact]))
                    if exact and exact[0] is not None
                    else float("nan")
                ),
                "beyond_trained_history": sum(
                    int(bool(r.get("beyond_trained_history"))) for r in row["replans"]
                ),
            }
        cell[seed] = entry
    seeds = sorted(cell)

    # p = 0 must reproduce the ablation run.
    mismatches: list[dict[str, object]] = []
    compared = 0
    for shard in sorted(args.ablation_root.glob("*/result.json"), key=lambda p: int(p.parent.name)):
        payload = json.loads(shard.read_text())
        if int(payload["task_id"]) != 5:
            continue
        seed = int(payload["reset_seed"])
        if seed not in cell:
            continue
        for row in payload["results"]:
            if row["arm"] not in SOURCES:
                continue
            slot = "single" if row["observer_seed"] is None else int(row["observer_seed"])
            ours = cell[seed].get((0.0, row["arm"]), {}).get(slot)
            if ours is None:
                continue
            compared += 1
            if ours["success"] != bool(row["success"]) or ours["plan"] != list(
                row["deployed_skills"]
            ):
                mismatches.append({"reset": seed, "arm": row["arm"], "slot": str(slot)})

    def vector(rate: float, source: str, field: str = "success") -> np.ndarray:
        slots = SEEDS if source in LEARNED else ("single",)
        rows = []
        for seed in seeds:
            values = [float(cell[seed][(rate, source)][s][field]) for s in slots]
            rows.append(float(np.mean(values)))
        return np.asarray(rows, dtype=np.float64)

    rates_table: dict[str, dict] = {}
    for source in SOURCES:
        for rate in RATES:
            mean, low, high = ci(vector(rate, source), args.bootstrap_seed)
            rates_table[f"{source}@{rate:.2f}"] = {
                "state_source": source,
                "failure_rate": rate,
                "mean_success": mean,
                "ci_low": low,
                "ci_high": high,
                "exhausted_rate": float(vector(rate, source, "exhausted").mean()),
                "over_reads": float(vector(rate, source, "over_reads").mean()),
                "under_reads": float(vector(rate, source, "under_reads").mean()),
                "exact_q_rate": float(np.nanmean(vector(rate, source, "exact_q_rate"))),
                "beyond_trained_history": float(
                    vector(rate, source, "beyond_trained_history").mean()
                ),
            }

    degradation: dict[str, dict] = {}
    for source in SOURCES:
        delta = vector(0.00, source) - vector(0.30, source)
        mean, low, high = ci(delta, args.bootstrap_seed + 1)
        degradation[source] = {
            "mean_points": 100 * mean,
            "ci_low_points": 100 * low,
            "ci_high_points": 100 * high,
        }

    contrasts: dict[str, dict] = {}
    for source in DEAD_RECKONERS:
        delta = (vector(0.00, source) - vector(0.30, source)) - (
            vector(0.00, REFERENCE) - vector(0.30, REFERENCE)
        )
        mean, low, high = ci(delta, args.bootstrap_seed + 2)
        contrasts[f"EXTRA_DEGRADATION_{source.upper()}"] = {
            "mean_points": 100 * mean,
            "ci_low_points": 100 * low,
            "ci_high_points": 100 * high,
        }

    confirmed = all(row["ci_low_points"] > 0 for row in contrasts.values())
    refuted = any(row["ci_high_points"] <= 0 for row in contrasts.values())
    if mismatches:
        verdict = "NONDETERMINISTIC_EVAL"
    elif confirmed:
        verdict = "MECHANISM_CONFIRMED"
    elif refuted:
        verdict = "MECHANISM_REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "num_resets": len(seeds),
        "reproduction": {
            "rows_compared": compared,
            "mismatches": mismatches,
            "exact": not mismatches,
        },
        "rates": rates_table,
        "degradation_0_to_30": degradation,
        "contrasts": contrasts,
        "scope": "task-5 only, 64 resets, feedback fixed at branch_w050",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Scene skill-failure dose response",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"{len(seeds)} task-5 resets.  Reproduction of the p=0.00 column against the "
        f"ablation run: {compared} rows, {len(mismatches)} mismatches.",
        "",
        "| State source | p=0.00 | p=0.10 | p=0.20 | p=0.30 | degradation (0.00 - 0.30) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for source in SOURCES:
        cells = " | ".join(
            f"{100 * rates_table[f'{source}@{rate:.2f}']['mean_success']:.2f}%" for rate in RATES
        )
        deg = degradation[source]
        lines.append(
            f"| `{source}` | {cells} | {deg['mean_points']:+.2f} "
            f"[{deg['ci_low_points']:+.2f}, {deg['ci_high_points']:+.2f}] |"
        )
    lines += ["", "| Contrast | points | 95% CI |", "|---|---:|---|"]
    for name, row in contrasts.items():
        lines.append(
            f"| `{name}` | {row['mean_points']:+.2f} | "
            f"[{row['ci_low_points']:+.2f}, {row['ci_high_points']:+.2f}] |"
        )
    lines += [
        "",
        "| State source | p | exact-q | over-reads | under-reads | timeout |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCES:
        for rate in RATES:
            row = rates_table[f"{source}@{rate:.2f}"]
            exact = "n/a" if np.isnan(row["exact_q_rate"]) else f"{100 * row['exact_q_rate']:.1f}%"
            lines.append(
                f"| `{source}` | {rate:.2f} | {exact} | {row['over_reads']:.2f} | "
                f"{row['under_reads']:.2f} | {100 * row['exhausted_rate']:.1f}% |"
            )
    (args.out_dir / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "contrasts": contrasts}, sort_keys=True))


if __name__ == "__main__":
    main()
