#!/usr/bin/env python3
"""Apply the locked 3x2 verdict over event-state source and planner feedback."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.scene_feedback import FEEDBACKS  # noqa: E402


PROTOCOL = "scene_state_vs_feedback_analysis_v1"
BOOTSTRAP = 10000
SEEDS = (0, 1, 2)
ORACLE = "oracle"
LEARNED_SOURCES = ("frame_full", "obs_history_full")
STATE_SOURCES = (*LEARNED_SOURCES, ORACLE)


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
        raise RuntimeError("grid analysis must run inside a Slurm compute job")

    shards = sorted(args.eval_root.glob("*/result.json"), key=lambda p: int(p.parent.name))
    if len(shards) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(shards)}")

    table: dict[tuple[int, int], dict[str, dict[object, bool]]] = {}
    plans: dict[tuple[int, int], dict[tuple[str, object], list[str]]] = {}
    for shard in shards:
        payload = json.loads(shard.read_text())
        key = (int(payload["task_id"]), int(payload["reset_seed"]))
        entry: dict[str, dict[object, bool]] = defaultdict(dict)
        plan: dict[tuple[str, object], list[str]] = {}
        for row in payload["results"]:
            seed = row["observer_seed"]
            slot = "single" if seed is None else int(seed)
            entry[row["arm"]][slot] = bool(row["success"])
            plan[(row["arm"], slot)] = list(row["deployed_skills"])
        table[key] = entry
        plans[key] = plan
    keys = sorted(table)

    # Reproduction of the event_progress column against the ablation run.
    ablation: dict[tuple[int, int], dict[str, dict[object, bool]]] = {}
    ablation_plans: dict[tuple[int, int], dict[tuple[str, object], list[str]]] = {}
    for shard in sorted(args.ablation_root.glob("*/result.json"), key=lambda p: int(p.parent.name)):
        payload = json.loads(shard.read_text())
        key = (int(payload["task_id"]), int(payload["reset_seed"]))
        entry: dict[str, dict[object, bool]] = defaultdict(dict)
        plan: dict[tuple[str, object], list[str]] = {}
        for row in payload["results"]:
            seed = row["observer_seed"]
            slot = "single" if seed is None else int(seed)
            entry[row["arm"]][slot] = bool(row["success"])
            plan[(row["arm"], slot)] = list(row["deployed_skills"])
        ablation[key] = entry
        ablation_plans[key] = plan

    reproduction_mismatches: list[dict[str, object]] = []
    compared = 0
    alias = {
        "frame_full": "frame_full",
        "obs_history_full": "obs_history_full",
        ORACLE: "oracle_event",
    }
    for key in keys:
        if key not in ablation:
            continue
        for source, other in alias.items():
            slots = SEEDS if source in LEARNED_SOURCES else ("single",)
            for slot in slots:
                compared += 1
                grid_arm = f"{source}__event_progress"
                same = (
                    table[key][grid_arm][slot] == ablation[key][other][slot]
                    and plans[key][(grid_arm, slot)] == ablation_plans[key][(other, slot)]
                )
                if not same:
                    reproduction_mismatches.append(
                        {"reset": list(key), "source": source, "slot": str(slot)}
                    )

    # Task 4 must be identical between the two feedbacks by construction.
    task4_mismatches: list[dict[str, object]] = []
    for key in keys:
        if key[0] != 4:
            continue
        for source in STATE_SOURCES:
            slots = SEEDS if source in LEARNED_SOURCES else ("single",)
            for slot in slots:
                left = plans[key][(f"{source}__event_progress", slot)]
                right = plans[key][(f"{source}__automaton_potential", slot)]
                if left != right:
                    task4_mismatches.append(
                        {"reset": list(key), "source": source, "slot": str(slot)}
                    )

    def vector(source: str, feedback: str, task: int | None = None) -> np.ndarray:
        arm = f"{source}__{feedback}"
        rows = []
        for key in keys:
            if task is not None and key[0] != task:
                continue
            cell = table[key][arm]
            rows.append(
                float(np.mean([float(cell[s]) for s in SEEDS]))
                if source in LEARNED_SOURCES
                else float(cell["single"])
            )
        return np.asarray(rows, dtype=np.float64)

    rates = {}
    for source in STATE_SOURCES:
        for feedback in FEEDBACKS:
            values = vector(source, feedback)
            mean, low, high = ci(values, args.bootstrap_seed)
            rates[f"{source}__{feedback}"] = {
                "state_source": source,
                "feedback": feedback,
                "mean_success": mean,
                "ci_low": low,
                "ci_high": high,
                "per_task": {
                    str(task): float(vector(source, feedback, task).mean())
                    for task in sorted({k[0] for k in keys})
                },
            }

    contrasts = {}
    for feedback in FEEDBACKS:
        for source in LEARNED_SOURCES:
            delta = vector(source, feedback) - vector(ORACLE, feedback)
            mean, low, high = ci(delta, args.bootstrap_seed + 1)
            contrasts[f"STATE_GAP_{source.upper()}__{feedback}"] = {
                "mean_points": 100 * mean,
                "ci_low_points": 100 * low,
                "ci_high_points": 100 * high,
            }
    for source in STATE_SOURCES:
        # The feedback contrast is task-5 only; task 4 is identical by construction.
        delta = vector(source, "event_progress", 5) - vector(source, "automaton_potential", 5)
        mean, low, high = ci(delta, args.bootstrap_seed + 2)
        contrasts[f"FEEDBACK_GAP_{source.upper()}__task5"] = {
            "mean_points": 100 * mean,
            "ci_low_points": 100 * low,
            "ci_high_points": 100 * high,
        }

    def gap_negative(source: str, feedback: str) -> bool:
        row = contrasts[f"STATE_GAP_{source.upper()}__{feedback}"]
        return row["ci_high_points"] < 0

    def gap_covers_zero(source: str, feedback: str) -> bool:
        row = contrasts[f"STATE_GAP_{source.upper()}__{feedback}"]
        return row["ci_low_points"] <= 0 <= row["ci_high_points"]

    state_constraint = all(
        gap_negative("frame_full", f) and gap_covers_zero("obs_history_full", f)
        for f in FEEDBACKS
    )
    feedback_matters = all(
        not (
            contrasts[f"FEEDBACK_GAP_{s.upper()}__task5"]["ci_low_points"]
            <= 0
            <= contrasts[f"FEEDBACK_GAP_{s.upper()}__task5"]["ci_high_points"]
        )
        for s in STATE_SOURCES
    )

    if reproduction_mismatches or task4_mismatches:
        verdict = "NONDETERMINISTIC_EVAL"
    elif state_constraint and feedback_matters:
        verdict = "BOTH_MATTER"
    elif state_constraint:
        verdict = "STATE_ESTIMATION_IS_THE_CONSTRAINT"
    elif feedback_matters:
        verdict = "FEEDBACK_ALSO_MATTERS"
    else:
        verdict = "INCONCLUSIVE"

    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "num_resets": len(keys),
        "reproduction": {
            "rows_compared": compared,
            "mismatches": reproduction_mismatches,
            "exact": not reproduction_mismatches,
        },
        "task4_feedback_identity": {
            "mismatches": task4_mismatches,
            "holds": not task4_mismatches,
        },
        "gate": {
            "state_estimation_is_the_constraint": state_constraint,
            "feedback_also_matters": feedback_matters,
        },
        "rates": rates,
        "contrasts": contrasts,
        "scope": (
            "3x2 grid on the ablation reset band; event_progress cells are exact "
            "reproductions, the feedback contrast is task-5 only"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Scene 3x2: event-state source against planner feedback",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"{len(keys)} paired resets at K=112.  Reproduction of the `event_progress` "
        f"column against the ablation run: {compared} rows, "
        f"{len(reproduction_mismatches)} mismatches.  Task-4 feedback identity: "
        f"{len(task4_mismatches)} mismatches.",
        "",
        "| Feedback | `frame_full` | `obs_history_full` | simulator q |",
        "|---|---:|---:|---:|",
    ]
    for feedback in FEEDBACKS:
        cells = " | ".join(
            f"{100 * rates[f'{s}__{feedback}']['mean_success']:.2f}%" for s in STATE_SOURCES
        )
        lines.append(f"| `{feedback}` | {cells} |")
    lines += ["", "| Contrast | points | 95% CI |", "|---|---:|---|"]
    for name, row in contrasts.items():
        lines.append(
            f"| `{name}` | {row['mean_points']:+.2f} | "
            f"[{row['ci_low_points']:+.2f}, {row['ci_high_points']:+.2f}] |"
        )
    (args.out_dir / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "contrasts": contrasts}, sort_keys=True))


if __name__ == "__main__":
    main()
