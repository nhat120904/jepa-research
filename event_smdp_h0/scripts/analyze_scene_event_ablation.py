#!/usr/bin/env python3
"""Apply the locked dead-reckoning verdict to the input-ablation run.

Also verifies that the two reused arms reproduce the confirmatory factorial
exactly; a mismatch invalidates every paired contrast in the project, so no
other verdict may be issued when one is found.
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

from event_smdp_h0.scene_event_history import ABLATION_ARMS  # noqa: E402


PROTOCOL = "scene_event_ablation_analysis_v1"
BOOTSTRAP = 10000
OPENLOOP = "openloop_transition"
SEEDS = (0, 1, 2)
DEAD_RECKONING_TOLERANCE_POINTS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def load_run(root: Path) -> dict[tuple[int, int], dict]:
    shards = sorted(root.glob("*/result.json"), key=lambda p: int(p.parent.name))
    table: dict[tuple[int, int], dict] = {}
    for shard in shards:
        payload = json.loads(shard.read_text())
        key = (int(payload["task_id"]), int(payload["reset_seed"]))
        if key in table:
            raise RuntimeError(f"duplicate reset {key} under {root}")
        table[key] = payload
    return table


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("ablation analysis must run inside a Slurm compute job")

    run = load_run(args.eval_root)
    if len(run) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(run)}")
    reference = load_run(args.reference_root)

    invariance_failures = [
        str(key)
        for key, payload in run.items()
        if not all(payload.get("ablation_invariance_checks", {}).values())
    ]
    if invariance_failures:
        raise RuntimeError(f"ablation invariance failed on {invariance_failures}")

    success: dict[tuple[int, int], dict[str, dict[object, bool]]] = {}
    skills: dict[tuple[int, int], dict[tuple[str, object], list[str]]] = {}
    for key, payload in run.items():
        entry: dict[str, dict[object, bool]] = defaultdict(dict)
        plans: dict[tuple[str, object], list[str]] = {}
        for row in payload["results"]:
            seed = row["observer_seed"]
            slot = "single" if seed is None else int(seed)
            entry[row["arm"]][slot] = bool(row["success"])
            plans[(row["arm"], slot)] = list(row["deployed_skills"])
        success[key] = entry
        skills[key] = plans

    # Reproduction of the two reused arms against the confirmatory factorial.
    mismatches: list[dict[str, object]] = []
    compared = 0
    for key, payload in reference.items():
        if key not in success:
            continue
        for row in payload["results"]:
            seed = row["observer_seed"]
            if row["arm"] not in ("frame_full", "history_full") or seed is None:
                continue
            compared += 1
            slot = int(seed)
            same_success = success[key][row["arm"]][slot] == bool(row["success"])
            same_plan = skills[key][(row["arm"], slot)] == list(row["deployed_skills"])
            if not (same_success and same_plan):
                mismatches.append(
                    {
                        "reset": list(key),
                        "arm": row["arm"],
                        "seed": slot,
                        "success_matches": same_success,
                        "plan_matches": same_plan,
                    }
                )

    keys = sorted(success)
    learned = set(ABLATION_ARMS)

    def arm_vector(arm: str) -> np.ndarray:
        rows = []
        for key in keys:
            cell = success[key][arm]
            if arm in learned:
                rows.append(float(np.mean([float(cell[s]) for s in SEEDS])))
            else:
                rows.append(float(cell["single"]))
        return np.asarray(rows, dtype=np.float64)

    all_arms = ["oracle_event", "abstract_terminal", OPENLOOP, *ABLATION_ARMS]
    rates = {}
    for arm in all_arms:
        vector = arm_vector(arm)
        mean, low, high = bootstrap_ci(vector, args.bootstrap_seed)
        rates[arm] = {
            "mean_success": mean,
            "ci_low": low,
            "ci_high": high,
            "per_model_seed": (
                {
                    str(seed): float(
                        np.mean([float(success[key][arm][seed]) for key in keys])
                    )
                    for seed in SEEDS
                }
                if arm in learned
                else None
            ),
            "per_task": {
                str(task): float(
                    np.mean([vector[i] for i, key in enumerate(keys) if key[0] == task])
                )
                for task in sorted({key[0] for key in keys})
            },
        }

    contrast_specs = {
        "VISION_GIVEN_ACTIONS": ("history_full", "action_only_full"),
        "ACTIONS_GIVEN_VISION": ("history_full", "obs_history_full"),
        "ACTION_ONLY_VS_OPENLOOP": ("action_only_full", OPENLOOP),
        "OBS_HISTORY_VS_FRAME": ("obs_history_full", "frame_full"),
        "OPENLOOP_VS_HISTORY_FULL": (OPENLOOP, "history_full"),
        "HISTORY_FULL_VS_FRAME_FULL": ("history_full", "frame_full"),
        "OPENLOOP_VS_FRAME_FULL": (OPENLOOP, "frame_full"),
    }
    contrasts = {}
    for name, (left, right) in contrast_specs.items():
        delta = arm_vector(left) - arm_vector(right)
        mean, low, high = bootstrap_ci(delta, args.bootstrap_seed + 1)
        contrasts[name] = {
            "left": left,
            "right": right,
            "mean_points": 100.0 * mean,
            "ci_low_points": 100.0 * low,
            "ci_high_points": 100.0 * high,
        }

    def within_tolerance(name: str) -> bool:
        row = contrasts[name]
        return bool(
            row["ci_low_points"] <= 0.0 <= row["ci_high_points"]
            and abs(row["mean_points"]) <= DEAD_RECKONING_TOLERANCE_POINTS
        )

    if mismatches:
        verdict = "NONDETERMINISTIC_EVAL"
    elif (
        contrasts["VISION_GIVEN_ACTIONS"]["ci_low_points"] > 0
        and -contrasts["OPENLOOP_VS_HISTORY_FULL"]["ci_high_points"] > 0
    ):
        verdict = "DEAD_RECKONING_REFUTED"
    elif within_tolerance("VISION_GIVEN_ACTIONS") or within_tolerance(
        "OPENLOOP_VS_HISTORY_FULL"
    ):
        verdict = "DEAD_RECKONING_SUFFICIENT"
    else:
        verdict = "PARTIAL"

    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "num_resets": len(keys),
        "reproduction": {
            "arms_checked": ["frame_full", "history_full"],
            "rows_compared": compared,
            "mismatches": mismatches,
            "exact": not mismatches,
        },
        "ablation_invariance_verified_shards": len(run),
        "gate": {
            "dead_reckoning_tolerance_points": DEAD_RECKONING_TOLERANCE_POINTS,
            "vision_given_actions_ci_low": contrasts["VISION_GIVEN_ACTIONS"][
                "ci_low_points"
            ],
            "history_full_beats_openloop_ci_low": -contrasts[
                "OPENLOOP_VS_HISTORY_FULL"
            ]["ci_high_points"],
        },
        "rates": rates,
        "contrasts": contrasts,
        "scope": (
            "input-ablation decomposition on the confirmatory reset band; the two "
            "reused arms are a determinism check, not a fresh measurement"
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.out_dir / "rates.csv").open("w") as handle:
        handle.write("arm,mean_success,ci_low,ci_high,seed0,seed1,seed2,task4,task5\n")
        for arm in all_arms:
            row = rates[arm]
            per_seed = row["per_model_seed"] or {}
            per_task = row["per_task"]
            handle.write(
                f"{arm},{row['mean_success']:.4f},{row['ci_low']:.4f},{row['ci_high']:.4f},"
                + ",".join(f"{per_seed.get(str(s), float('nan')):.4f}" for s in SEEDS)
                + f",{per_task.get('4', float('nan')):.4f},{per_task.get('5', float('nan')):.4f}\n"
            )
    with (args.out_dir / "contrasts.csv").open("w") as handle:
        handle.write("contrast,left,right,mean_points,ci_low_points,ci_high_points\n")
        for name, row in contrasts.items():
            handle.write(
                f"{name},{row['left']},{row['right']},{row['mean_points']:.2f},"
                f"{row['ci_low_points']:.2f},{row['ci_high_points']:.2f}\n"
            )

    lines = [
        "# Scene event-observer input ablation",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"{len(keys)} paired resets at K=112.  Reproduction of `frame_full` and "
        f"`history_full` against the confirmatory factorial: {compared} rows compared, "
        f"{len(mismatches)} mismatches.",
        "",
        "| Arm | mean success | 95% CI | seed 0 | seed 1 | seed 2 | task 4 | task 5 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for arm in all_arms:
        row = rates[arm]
        per_seed = row["per_model_seed"]
        cells = (
            " | ".join(f"{100 * per_seed[str(s)]:.2f}%" for s in SEEDS)
            if per_seed
            else "n/a | n/a | n/a"
        )
        lines.append(
            f"| `{arm}` | {100 * row['mean_success']:.2f}% | "
            f"[{100 * row['ci_low']:.2f}, {100 * row['ci_high']:.2f}] | {cells} | "
            f"{100 * row['per_task'].get('4', float('nan')):.2f}% | "
            f"{100 * row['per_task'].get('5', float('nan')):.2f}% |"
        )
    lines += ["", "| Contrast | points | 95% CI |", "|---|---:|---|"]
    for name, row in contrasts.items():
        lines.append(
            f"| `{name}` = {row['left']} - {row['right']} | {row['mean_points']:+.2f} | "
            f"[{row['ci_low_points']:+.2f}, {row['ci_high_points']:+.2f}] |"
        )
    (args.out_dir / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "contrasts": contrasts}, sort_keys=True))


if __name__ == "__main__":
    main()
