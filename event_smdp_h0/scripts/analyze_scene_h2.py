#!/usr/bin/env python3
"""Cluster-aware analysis for the Scene H2 search-width audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=16)
    parser.add_argument("--reference-budget", type=int, default=14)
    parser.add_argument("--max-budget", type=int, default=112)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def root_metrics(candidates: list[dict], selected: dict) -> dict[str, float]:
    predicted_probability = np.asarray(
        [row["predicted_success_probability"] for row in candidates], dtype=float
    )
    true_success = np.asarray([int(row["true_success"]) for row in candidates], dtype=float)
    signed = np.asarray([row["reward_signed_error"] for row in candidates], dtype=float)
    absolute = np.asarray([row["reward_absolute_error"] for row in candidates], dtype=float)
    successful = true_success == 1
    recall = (
        float(np.mean(predicted_probability[successful] >= 0.5))
        if successful.any()
        else float("nan")
    )
    selected_true = float(selected["true_success"])
    selected_probability = float(selected["predicted_success_probability"])
    return {
        "candidate_success_rate": float(true_success.mean()),
        "candidate_predicted_success_mean": float(predicted_probability.mean()),
        "candidate_calibration_gap": float((predicted_probability - true_success).mean()),
        "candidate_brier": float(
            np.mean((predicted_probability - true_success) ** 2)
        ),
        "candidate_reward_signed_error": float(signed.mean()),
        "candidate_reward_absolute_error": float(absolute.mean()),
        "successful_candidate_recall": recall,
        "selected_success": selected_true,
        "selected_predicted_success": selected_probability,
        "selected_calibration_gap": selected_probability - selected_true,
        "selected_brier": float(selected["success_brier"]),
        "selected_log_loss": float(selected["success_log_loss"]),
        "selected_reward_signed_error": float(selected["reward_signed_error"]),
        "selected_reward_absolute_error": float(selected["reward_absolute_error"]),
        "selected_transition_nll": float(selected["teacher_forced_transition_nll"]),
        "selected_mode_step_accuracy": float(selected["mode_step_accuracy"]),
        "selected_reward_regret": float(
            max(row["true_reward"] for row in candidates) - selected["true_reward"]
        ),
        "selection_absolute_error_bias": float(
            selected["reward_absolute_error"] - absolute.mean()
        ),
    }


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array[np.isfinite(array)].mean()) if np.isfinite(array).any() else float("nan")


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob("*/result.json"))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")
    payloads = [json.loads(path.read_text()) for path in paths]
    if any(row.get("protocol") != "scene_h2_search_width_audit_v1" for row in payloads):
        raise RuntimeError("unexpected H2 protocol")
    rng = np.random.default_rng(args.seed)
    per_reset: dict[tuple[int, int], dict[int, dict[str, float]]] = {}
    closed_loop: dict[int, list[float]] = defaultdict(list)
    root_rows: list[dict] = []
    for payload in payloads:
        reset_key = (int(payload["task_id"]), int(payload["reset_seed"]))
        accum: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for root in payload["roots"]:
            candidate_lookup = {
                tuple(row["sequence"]): row for row in root["candidates"]
            }
            for search in root["searches"]:
                budget = int(search["budget"])
                candidates = [
                    candidate_lookup[tuple(sequence)]
                    for sequence in search["candidate_sequences"]
                ]
                selected = search["root_selected_truth"]
                metrics = root_metrics(candidates, selected)
                root_rows.append(
                    {
                        "task_id": reset_key[0],
                        "reset_seed": reset_key[1],
                        "root_index": int(root["root_index"]),
                        "budget": budget,
                        **metrics,
                    }
                )
                for key, value in metrics.items():
                    accum[budget][key].append(value)
        per_reset[reset_key] = {
            budget: {key: finite_mean(values) for key, values in metrics.items()}
            for budget, metrics in accum.items()
        }
        for row in payload["closed_loop"]:
            closed_loop[int(row["budget"])].append(float(row["success"]))

    metric_names = sorted(next(iter(next(iter(per_reset.values())).values())).keys())
    budgets = sorted(next(iter(per_reset.values())).keys())
    aggregate: list[dict] = []
    for budget in budgets:
        row: dict[str, object] = {"budget": budget, "n_resets": len(per_reset)}
        for metric in metric_names:
            values = np.asarray(
                [per_reset[key][budget][metric] for key in sorted(per_reset)], dtype=float
            )
            finite = values[np.isfinite(values)]
            row[metric] = float(finite.mean()) if len(finite) else float("nan")
            row[f"{metric}_ci95"] = (
                bootstrap_ci(finite, args.bootstrap, rng) if len(finite) else [float("nan")] * 2
            )
        successes = np.asarray(closed_loop[budget], dtype=float)
        row["closed_loop_success_rate"] = float(successes.mean())
        row["closed_loop_success_ci95"] = bootstrap_ci(
            successes, args.bootstrap, rng
        )
        aggregate.append(row)

    def paired_delta(metric: str) -> dict[str, object]:
        values = np.asarray(
            [
                per_reset[key][args.max_budget][metric]
                - per_reset[key][args.reference_budget][metric]
                for key in sorted(per_reset)
            ],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        return {
            "metric": metric,
            "reference_budget": args.reference_budget,
            "max_budget": args.max_budget,
            "n": len(values),
            "mean_delta": float(values.mean()),
            "ci95": bootstrap_ci(values, args.bootstrap, rng),
        }

    overestimate_delta = paired_delta("selected_reward_signed_error")
    brier_delta = paired_delta("selected_brier")
    calibration_delta = paired_delta("selected_calibration_gap")
    reward_exploitation = (
        overestimate_delta["mean_delta"] >= 0.10
        and overestimate_delta["ci95"][0] > 0.0
    )
    probability_exploitation = (
        brier_delta["mean_delta"] >= 0.05 and brier_delta["ci95"][0] > 0.0
    )
    verdict = (
        "H2_SEARCH_INDUCED_MISCALIBRATION"
        if reward_exploitation or probability_exploitation
        else "H2_NO_SEARCH_INDUCED_MISCALIBRATION"
    )
    summary = {
        "protocol": "scene_h2_search_width_audit_v1",
        "verdict": verdict,
        "num_shards": len(paths),
        "num_roots": len({(row["task_id"], row["reset_seed"], row["root_index"]) for row in root_rows}),
        "budgets": budgets,
        "aggregate": aggregate,
        "paired_width_deltas": {
            "selected_reward_overestimate": overestimate_delta,
            "selected_brier": brier_delta,
            "selected_calibration_gap": calibration_delta,
        },
        "gate": {
            "reward_exploitation": reward_exploitation,
            "probability_exploitation": probability_exploitation,
            "rule": (
                "from K=14 to K=112, paired selected reward overestimate rises by >=0.10 "
                "or selected Brier rises by >=0.05, with bootstrap lower bound >0"
            ),
        },
        "scope": (
            "exact simulator truth for every candidate sequence at canonical milestone roots; "
            "reset-clustered intervals; current q remains simulator-monitored"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    with (args.out_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (args.out_dir / "DECISION.md").write_text(
        "# Scene H2 search-width audit\n\n"
        f"Verdict: **{verdict}**\n\n"
        "SearchCal is licensed only by the positive verdict; otherwise the "
        "structured model should be improved or evaluated without adding calibration.\n"
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()

