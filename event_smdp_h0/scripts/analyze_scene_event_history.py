#!/usr/bin/env python3
"""Apply the locked coverage/history verdict to a completed evaluation run.

Success is paired by reset.  Each learned arm contributes the mean success over
its three model seeds at that reset, so the bootstrap resamples reset clusters
rather than treating 3 x N seed-reset rows as independent.
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

from event_smdp_h0.scene_event_history import ARMS  # noqa: E402


PROTOCOL = "scene_event_history_analysis_v1"
BOOTSTRAP = 10000
FRAME_FULL_MEAN_BAR = 0.85
PER_SEED_BAR = 0.75
HISTORY_EFFECT_BAR = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
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
        raise RuntimeError("history analysis must run inside a Slurm compute job")

    shards = sorted(args.eval_root.glob("*/result.json"), key=lambda p: int(p.parent.name))
    if len(shards) != args.expected_shards:
        raise RuntimeError(
            f"expected {args.expected_shards} shards, found {len(shards)}"
        )

    # reset key -> arm -> {seed: success}
    table: dict[tuple[int, int], dict[str, dict[str | int, bool]]] = {}
    transition_hashes: set[str] = set()
    observer_hashes: set[str] = set()
    beyond_history = 0
    total_learned_decisions = 0
    for shard in shards:
        payload = json.loads(shard.read_text())
        transition_hashes.add(payload["transition"]["sha256"])
        observer_hashes.update(v["sha256"] for v in payload["observers"].values())
        key = (int(payload["task_id"]), int(payload["reset_seed"]))
        if key in table:
            raise RuntimeError(f"duplicate reset {key}")
        entry: dict[str, dict[str | int, bool]] = defaultdict(dict)
        for row in payload["results"]:
            seed = row["observer_seed"]
            entry[row["arm"]][ "oracle" if seed is None else int(seed)] = bool(row["success"])
            if seed is not None:
                for replan in row["replans"]:
                    total_learned_decisions += 1
                    beyond_history += int(bool(replan.get("beyond_trained_history")))
        table[key] = entry

    if len(transition_hashes) != 1:
        raise RuntimeError(f"transition checkpoint drift: {sorted(transition_hashes)}")
    if len(observer_hashes) != 4 * 3:
        raise RuntimeError(f"expected 12 observer checkpoints, saw {len(observer_hashes)}")

    keys = sorted(table)
    seeds = (0, 1, 2)

    def arm_vector(arm: str) -> np.ndarray:
        """Per-reset success, averaged over model seeds for learned arms."""

        rows = []
        for key in keys:
            cell = table[key][arm]
            if arm in ARMS:
                rows.append(float(np.mean([float(cell[s]) for s in seeds])))
            else:
                rows.append(float(cell["oracle"]))
        return np.asarray(rows, dtype=np.float64)

    rng = np.random.default_rng(args.bootstrap_seed)
    all_arms = ["oracle_event", "abstract_terminal", *ARMS]
    rates = {}
    for arm in all_arms:
        vector = arm_vector(arm)
        mean, low, high = bootstrap_ci(vector, np.random.default_rng(args.bootstrap_seed))
        per_seed = (
            {
                str(seed): float(
                    np.mean([float(table[key][arm][seed]) for key in keys])
                )
                for seed in seeds
            }
            if arm in ARMS
            else None
        )
        per_task = {
            str(task): float(
                np.mean([vector[i] for i, key in enumerate(keys) if key[0] == task])
            )
            for task in sorted({key[0] for key in keys})
        }
        rates[arm] = {
            "mean_success": mean,
            "ci_low": low,
            "ci_high": high,
            "per_model_seed": per_seed,
            "per_task": per_task,
        }

    contrast_specs = {
        "COVERAGE": ("frame_full", "frame_canonical"),
        "HISTORY": ("history_full", "frame_full"),
        "HISTORY_UNDER_CANONICAL": ("history_canonical", "frame_canonical"),
        "COVERAGE_UNDER_HISTORY": ("history_full", "history_canonical"),
        "FRAME_FULL_VS_TERMINAL": ("frame_full", "abstract_terminal"),
        "HISTORY_FULL_VS_TERMINAL": ("history_full", "abstract_terminal"),
        "FRAME_FULL_VS_ORACLE": ("frame_full", "oracle_event"),
        "HISTORY_FULL_VS_ORACLE": ("history_full", "oracle_event"),
    }
    contrasts = {}
    for name, (left, right) in contrast_specs.items():
        delta = arm_vector(left) - arm_vector(right)
        mean, low, high = bootstrap_ci(delta, np.random.default_rng(args.bootstrap_seed + 1))
        contrasts[name] = {
            "left": left,
            "right": right,
            "mean_points": 100.0 * mean,
            "ci_low_points": 100.0 * low,
            "ci_high_points": 100.0 * high,
        }

    def passes_bar(arm: str) -> bool:
        return bool(
            rates[arm]["mean_success"] >= FRAME_FULL_MEAN_BAR
            and min(rates[arm]["per_model_seed"].values()) >= PER_SEED_BAR
        )

    frame_full_pass = passes_bar("frame_full") and contrasts["COVERAGE"]["ci_low_points"] > 0
    history_full_pass = passes_bar("history_full")
    history_adds = bool(
        contrasts["HISTORY"]["ci_low_points"] > 0
        and contrasts["HISTORY"]["mean_points"] >= 100.0 * HISTORY_EFFECT_BAR
    )
    if frame_full_pass and history_adds:
        verdict = "COVERAGE_IS_THE_FIX_AND_HISTORY_ADDS_VALUE"
    elif frame_full_pass:
        verdict = "COVERAGE_IS_THE_FIX"
    elif history_full_pass:
        verdict = "HISTORY_REQUIRED_CONFIRMED"
    else:
        verdict = "BOTH_FAIL"

    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "num_resets": len(keys),
        "resets": [list(key) for key in keys],
        "transition_sha256": sorted(transition_hashes)[0],
        "num_observer_checkpoints": len(observer_hashes),
        "gate": {
            "frame_full_mean_bar": FRAME_FULL_MEAN_BAR,
            "per_model_seed_bar": PER_SEED_BAR,
            "history_effect_bar_points": 100.0 * HISTORY_EFFECT_BAR,
            "frame_full_passes": frame_full_pass,
            "history_full_passes_same_bar": history_full_pass,
            "history_adds_value": history_adds,
        },
        "rates": rates,
        "contrasts": contrasts,
        "history_distribution": {
            "learned_decisions": total_learned_decisions,
            "beyond_trained_history": beyond_history,
        },
        "scope": (
            "paired fresh-reset planning success at K=112 under a frozen abstract "
            "transition model; learned arms never read simulator q"
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.out_dir / "rates.csv").open("w") as handle:
        handle.write("arm,mean_success,ci_low,ci_high,seed0,seed1,seed2\n")
        for arm in all_arms:
            row = rates[arm]
            per_seed = row["per_model_seed"] or {}
            handle.write(
                f"{arm},{row['mean_success']:.4f},{row['ci_low']:.4f},"
                f"{row['ci_high']:.4f},"
                + ",".join(f"{per_seed.get(str(s), float('nan')):.4f}" for s in seeds)
                + "\n"
            )
    with (args.out_dir / "contrasts.csv").open("w") as handle:
        handle.write("contrast,left,right,mean_points,ci_low_points,ci_high_points\n")
        for name, row in contrasts.items():
            handle.write(
                f"{name},{row['left']},{row['right']},{row['mean_points']:.2f},"
                f"{row['ci_low_points']:.2f},{row['ci_high_points']:.2f}\n"
            )
    with (args.out_dir / "paired.csv").open("w") as handle:
        handle.write("task_id,reset_seed," + ",".join(all_arms) + "\n")
        vectors = {arm: arm_vector(arm) for arm in all_arms}
        for index, key in enumerate(keys):
            handle.write(
                f"{key[0]},{key[1]},"
                + ",".join(f"{vectors[arm][index]:.4f}" for arm in all_arms)
                + "\n"
            )
    lines = [
        "# Scene event-observer coverage/history factorial",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"{len(keys)} paired fresh resets at K=112.",
        "",
        "| Arm | mean success | 95% CI | seed 0 | seed 1 | seed 2 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for arm in all_arms:
        row = rates[arm]
        per_seed = row["per_model_seed"]
        seed_cells = (
            " | ".join(f"{100 * per_seed[str(s)]:.2f}%" for s in seeds)
            if per_seed
            else "n/a | n/a | n/a"
        )
        lines.append(
            f"| `{arm}` | {100 * row['mean_success']:.2f}% | "
            f"[{100 * row['ci_low']:.2f}, {100 * row['ci_high']:.2f}] | {seed_cells} |"
        )
    lines += ["", "| Contrast | points | 95% CI |", "|---|---:|---|"]
    for name, row in contrasts.items():
        lines.append(
            f"| `{name}` = {row['left']} - {row['right']} | "
            f"{row['mean_points']:+.2f} | "
            f"[{row['ci_low_points']:+.2f}, {row['ci_high_points']:+.2f}] |"
        )
    (args.out_dir / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "resets"}, sort_keys=True))


if __name__ == "__main__":
    main()
