#!/usr/bin/env python3
"""Locked verdict for the feedback-robustness sweep.

Primary endpoint is the learned-versus-oracle gap per feedback, not raw success.
"""

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

from event_smdp_h0.scene_feedback import SWEEP_FEEDBACKS  # noqa: E402


PROTOCOL = "scene_feedback_robustness_analysis_v1"
BOOTSTRAP = 10000
SEEDS = (0, 1, 2)
ORACLE = "oracle"
LEARNED_SOURCES = ("frame_full", "obs_history_full")
STATE_SOURCES = (*LEARNED_SOURCES, ORACLE)
PRESERVE_FLOOR_POINTS = -5.0
GENERAL_MIN = 5
KNIFE_EDGE_MAX = 2
ANCHORS = {"branch_w050": "event_progress", "branch_w062": "automaton_potential"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--grid-root", type=Path, required=True)
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
        raise RuntimeError("sweep analysis must run inside a Slurm compute job")

    shards = sorted(args.eval_root.glob("*/result.json"), key=lambda p: int(p.parent.name))
    if len(shards) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(shards)}")

    success: dict[int, dict[str, dict[object, bool]]] = {}
    extra: dict[int, dict[str, dict[object, dict[str, float]]]] = {}
    plans: dict[int, dict[tuple[str, object], list[str]]] = {}
    for shard in shards:
        payload = json.loads(shard.read_text())
        seed = int(payload["reset_seed"])
        cell: dict[str, dict[object, bool]] = defaultdict(dict)
        aux: dict[str, dict[object, dict[str, float]]] = defaultdict(dict)
        plan: dict[tuple[str, object], list[str]] = {}
        for row in payload["results"]:
            slot = "single" if row["observer_seed"] is None else int(row["observer_seed"])
            cell[row["arm"]][slot] = bool(row["success"])
            exact = [r["exact_q_correct"] for r in row["replans"]]
            aux[row["arm"]][slot] = {
                "exhausted": float(row["exhausted_budget"]),
                "repeated_skill_rate": float(row["repeated_skill_rate"]),
                "exact_q_rate": (
                    float(np.mean([float(bool(v)) for v in exact])) if exact[0] is not None else float("nan")
                ),
            }
            plan[(row["arm"], slot)] = list(row["deployed_skills"])
        success[seed] = cell
        extra[seed] = aux
        plans[seed] = plan
    seeds = sorted(success)

    # Anchor reproduction against the task-5 half of the 3x2 grid.
    grid_plans: dict[int, dict[tuple[str, object], list[str]]] = {}
    grid_success: dict[int, dict[tuple[str, object], bool]] = {}
    for shard in sorted(args.grid_root.glob("*/result.json"), key=lambda p: int(p.parent.name)):
        payload = json.loads(shard.read_text())
        if int(payload["task_id"]) != 5:
            continue
        seed = int(payload["reset_seed"])
        grid_plans[seed] = {
            (row["arm"], "single" if row["observer_seed"] is None else int(row["observer_seed"])):
            list(row["deployed_skills"])
            for row in payload["results"]
        }
        grid_success[seed] = {
            (row["arm"], "single" if row["observer_seed"] is None else int(row["observer_seed"])):
            bool(row["success"])
            for row in payload["results"]
        }

    mismatches: list[dict[str, object]] = []
    compared = 0
    for seed in seeds:
        if seed not in grid_plans:
            continue
        for sweep_name, grid_name in ANCHORS.items():
            for source in STATE_SOURCES:
                slots = SEEDS if source in LEARNED_SOURCES else ("single",)
                for slot in slots:
                    compared += 1
                    ours = (f"{source}__{sweep_name}", slot)
                    theirs = (f"{source}__{grid_name}", slot)
                    if (
                        plans[seed][ours] != grid_plans[seed][theirs]
                        or success[seed][ours[0]][slot] != grid_success[seed][theirs]
                    ):
                        mismatches.append(
                            {"reset": seed, "feedback": sweep_name, "source": source, "slot": str(slot)}
                        )

    def vector(source: str, feedback: str, field: str | None = None) -> np.ndarray:
        arm = f"{source}__{feedback}"
        rows = []
        for seed in seeds:
            slots = SEEDS if source in LEARNED_SOURCES else ("single",)
            if field is None:
                values = [float(success[seed][arm][s]) for s in slots]
            else:
                values = [float(extra[seed][arm][s][field]) for s in slots]
            rows.append(float(np.mean(values)))
        return np.asarray(rows, dtype=np.float64)

    rates: dict[str, dict] = {}
    for feedback in SWEEP_FEEDBACKS:
        for source in STATE_SOURCES:
            mean, low, high = ci(vector(source, feedback), args.bootstrap_seed)
            rates[f"{source}__{feedback}"] = {
                "state_source": source,
                "feedback": feedback,
                "mean_success": mean,
                "ci_low": low,
                "ci_high": high,
                "exhausted_rate": float(vector(source, feedback, "exhausted").mean()),
                "repeated_skill_rate": float(
                    vector(source, feedback, "repeated_skill_rate").mean()
                ),
                "exact_q_rate": float(
                    np.nanmean(vector(source, feedback, "exact_q_rate"))
                ),
            }

    gaps: dict[str, dict] = {}
    preserved: list[str] = []
    for feedback in SWEEP_FEEDBACKS:
        delta = vector("obs_history_full", feedback) - vector(ORACLE, feedback)
        mean, low, high = ci(delta, args.bootstrap_seed + 1)
        keeps = low > PRESERVE_FLOOR_POINTS / 100.0
        gaps[feedback] = {
            "mean_points": 100 * mean,
            "ci_low_points": 100 * low,
            "ci_high_points": 100 * high,
            "preserves": bool(keeps),
        }
        if keeps:
            preserved.append(feedback)

    if mismatches:
        verdict = "NONDETERMINISTIC_EVAL"
    elif len(preserved) >= GENERAL_MIN:
        verdict = "FEEDBACK_TOLERANCE_IS_GENERAL"
    elif len(preserved) <= KNIFE_EDGE_MAX:
        verdict = "EVENT_PROGRESS_IS_A_KNIFE_EDGE"
    else:
        verdict = "PARTIAL"

    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "num_resets": len(seeds),
        "anchor_reproduction": {
            "rows_compared": compared,
            "mismatches": mismatches,
            "exact": not mismatches,
        },
        "gate": {
            "preserve_floor_points": PRESERVE_FLOOR_POINTS,
            "general_min": GENERAL_MIN,
            "knife_edge_max": KNIFE_EDGE_MAX,
            "num_preserved": len(preserved),
            "preserved": preserved,
        },
        "rates": rates,
        "learned_vs_oracle_gap": gaps,
        "scope": "task-5 only, 64 resets, one automaton and one planner",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Scene feedback-robustness sweep",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"{len(seeds)} task-5 resets.  Anchor reproduction against the 3x2 grid: "
        f"{compared} rows, {len(mismatches)} mismatches.",
        "",
        "| Feedback | oracle | `obs_history_full` | `frame_full` | gap (obs - oracle) | preserves |",
        "|---|---:|---:|---:|---|:--:|",
    ]
    for feedback in SWEEP_FEEDBACKS:
        gap = gaps[feedback]
        lines.append(
            f"| `{feedback}` | "
            f"{100 * rates[f'oracle__{feedback}']['mean_success']:.2f}% | "
            f"{100 * rates[f'obs_history_full__{feedback}']['mean_success']:.2f}% | "
            f"{100 * rates[f'frame_full__{feedback}']['mean_success']:.2f}% | "
            f"{gap['mean_points']:+.2f} [{gap['ci_low_points']:+.2f}, {gap['ci_high_points']:+.2f}] | "
            f"{'yes' if gap['preserves'] else 'no'} |"
        )
    lines += [
        "",
        "| Feedback | source | timeout rate | repeated-skill rate | exact-q |",
        "|---|---|---:|---:|---:|",
    ]
    for feedback in SWEEP_FEEDBACKS:
        for source in ("oracle", "obs_history_full"):
            row = rates[f"{source}__{feedback}"]
            lines.append(
                f"| `{feedback}` | `{source}` | {100 * row['exhausted_rate']:.1f}% | "
                f"{100 * row['repeated_skill_rate']:.1f}% | "
                + (
                    "n/a |"
                    if source == "oracle"
                    else f"{100 * row['exact_q_rate']:.1f}% |"
                )
            )
    (args.out_dir / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "gaps": gaps, "preserved": preserved}, sort_keys=True))


if __name__ == "__main__":
    main()
