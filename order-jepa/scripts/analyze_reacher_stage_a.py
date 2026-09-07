#!/usr/bin/env python3
"""Aggregate LeWM-Reacher ORDER Stage-A shards and apply fixed go/stop gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from order_jepa.core import clustered_bootstrap_mean, selection_regret  # noqa: E402
from order_jepa.lewm_reacher import OFFICIAL_REPO  # noqa: E402


def metric(values: np.ndarray, clusters: np.ndarray, n_resamples: int, seed: int) -> dict:
    mean, lo, hi = clustered_bootstrap_mean(
        values, clusters, n_resamples=n_resamples, seed=seed
    )
    return {"mean": mean, "ci95": [lo, hi], "n": int(np.isfinite(values).sum())}


def metric_text(value: dict) -> str:
    return (
        f"{value['mean']:.4f} "
        f"[{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}]"
    )


def main() -> None:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Stage-A aggregation must run on a Slurm compute node")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-anchors", type=int, default=200)
    parser.add_argument("--min-relevant-pairs", type=int, default=100)
    parser.add_argument("--physical-margin", type=float, default=0.10)
    parser.add_argument("--min-order-effect", type=float, default=0.50)
    parser.add_argument("--representation-accuracy", type=float, default=0.65)
    parser.add_argument("--minimum-accuracy-gap", type=float, default=0.10)
    parser.add_argument("--reset-tolerance", type=float, default=1e-9)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    paths = sorted(args.scores_dir.glob("anchor_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    if not records:
        raise FileNotFoundError(f"no score shards in {args.scores_dir}")
    provenance_keys = {
        (
            row["provenance"]["implementation"],
            row["provenance"]["checkpoint_repo"],
            row["provenance"]["source_commit"],
            row["provenance"]["checkpoint_sha256"],
            row["provenance"]["config_sha256"],
        )
        for row in records
    }
    provenance_ok = (
        len(provenance_keys) == 1
        and next(iter(provenance_keys))[0] == "lucas-maes/le-wm:LeWM"
        and next(iter(provenance_keys))[1] == OFFICIAL_REPO
    )
    episodes = np.asarray([row["episode"] for row in records], dtype=np.int64)

    ordinary_coverage = []
    ordinary_true_success = []
    ordinary_pred_success = []
    ordinary_true_regret = []
    ordinary_pred_regret = []
    witness_success = []
    for row in records:
        witness = [candidate for candidate in row["candidates"] if candidate["kind"] == "witness"]
        if len(witness) != 1:
            raise RuntimeError(f"anchor {row['anchor_id']} does not have exactly one witness")
        witness_success.append(float(witness[0]["success"]))
        candidates = [
            candidate
            for candidate in row["candidates"]
            if candidate["kind"].startswith("ordinary_")
        ]
        physical = np.asarray([candidate["physical_cost"] for candidate in candidates])
        true_cost = np.asarray([candidate["true_latent_cost"] for candidate in candidates])
        pred_cost = np.asarray([candidate["predicted_latent_cost"] for candidate in candidates])
        success = np.asarray([candidate["success"] for candidate in candidates], dtype=bool)
        true_index, true_regret = selection_regret(true_cost, physical)
        pred_index, pred_regret = selection_regret(pred_cost, physical)
        ordinary_coverage.append(float(np.any(success)))
        ordinary_true_success.append(float(success[true_index]))
        ordinary_pred_success.append(float(success[pred_index]))
        ordinary_true_regret.append(true_regret)
        ordinary_pred_regret.append(pred_regret)

    pairs = []
    pair_episode = []
    for row in records:
        for pair in row["order_pairs"]:
            pairs.append(pair)
            pair_episode.append(row["episode"])
    pair_episode = np.asarray(pair_episode, dtype=np.int64)
    effect = np.asarray([pair["qpos_order_effect"] for pair in pairs], dtype=np.float64)
    noise = np.asarray([pair["repeat_object_noise"] for pair in pairs], dtype=np.float64)
    physical_delta = np.asarray([pair["physical_delta"] for pair in pairs], dtype=np.float64)
    true_delta = np.asarray([pair["true_latent_delta"] for pair in pairs], dtype=np.float64)
    pred_delta = np.asarray([pair["predicted_latent_delta"] for pair in pairs], dtype=np.float64)
    relevant = (
        (effect >= args.min_order_effect)
        & (effect > 3.0 * noise)
        & (np.abs(physical_delta) >= args.physical_margin)
    )
    true_correct = (np.sign(true_delta) == np.sign(physical_delta)).astype(np.float64)
    pred_correct = (np.sign(pred_delta) == np.sign(physical_delta)).astype(np.float64)
    accuracy_gap = true_correct - pred_correct

    ordinary_true_regret = np.asarray(ordinary_true_regret, dtype=np.float64)
    ordinary_pred_regret = np.asarray(ordinary_pred_regret, dtype=np.float64)
    reset_physics = max(row["reset"]["physics_max_abs"] for row in records)
    reset_pixels = max(row["reset"]["pixel_max_abs"] for row in records)
    repeat_endpoint = max(row["reset"]["repeat_endpoint_max_abs"] for row in records)

    summary = {
        "schema": "order-jepa-reacher-stage-a-decision-v1",
        "anchors_found": len(records),
        "expected_anchors": args.expected_anchors,
        "episodes": int(np.unique(episodes).size),
        "pairs_total": len(pairs),
        "pairs_relevant": int(np.sum(relevant)),
        "provenance": records[0]["provenance"],
        "provenance_consistent_and_official": provenance_ok,
        "reset": {
            "physics_max_abs": reset_physics,
            "pixel_max_abs": reset_pixels,
            "repeat_endpoint_max_abs": repeat_endpoint,
        },
        "physical_order_effect": metric(effect, pair_episode, args.n_bootstrap, args.seed),
        "witness_success_rate": metric(
            np.asarray(witness_success), episodes, args.n_bootstrap, args.seed + 1
        ),
        "ordinary_panel": {
            "excludes_witness_and_diagnostic_pairs": True,
            "coverage_rate": metric(
                np.asarray(ordinary_coverage), episodes, args.n_bootstrap, args.seed + 2
            ),
            "true_latent_success_rate": metric(
                np.asarray(ordinary_true_success), episodes, args.n_bootstrap, args.seed + 3
            ),
            "predicted_latent_success_rate": metric(
                np.asarray(ordinary_pred_success), episodes, args.n_bootstrap, args.seed + 4
            ),
            "true_latent_regret": metric(
                ordinary_true_regret, episodes, args.n_bootstrap, args.seed + 5
            ),
            "predicted_latent_regret": metric(
                ordinary_pred_regret, episodes, args.n_bootstrap, args.seed + 6
            ),
            "dynamics_excess_regret": metric(
                ordinary_pred_regret - ordinary_true_regret,
                episodes,
                args.n_bootstrap,
                args.seed + 7,
            ),
        },
        "relevant_pair_true_physical_accuracy": metric(
            true_correct[relevant], pair_episode[relevant], args.n_bootstrap, args.seed + 8
        ),
        "relevant_pair_predicted_physical_accuracy": metric(
            pred_correct[relevant], pair_episode[relevant], args.n_bootstrap, args.seed + 9
        ),
        "relevant_pair_accuracy_gap": metric(
            accuracy_gap[relevant], pair_episode[relevant], args.n_bootstrap, args.seed + 10
        ),
        "relevant_order_normalized_error": metric(
            np.asarray([pair["normalized_error"] for pair in pairs])[relevant],
            pair_episode[relevant],
            args.n_bootstrap,
            args.seed + 11,
        ),
        "relevant_order_cosine": metric(
            np.asarray([pair["cosine"] for pair in pairs])[relevant],
            pair_episode[relevant],
            args.n_bootstrap,
            args.seed + 12,
        ),
        "relevant_order_amplitude_ratio": metric(
            np.asarray([pair["amplitude_ratio"] for pair in pairs])[relevant],
            pair_episode[relevant],
            args.n_bootstrap,
            args.seed + 13,
        ),
        "thresholds": {
            "min_relevant_pairs": args.min_relevant_pairs,
            "physical_margin": args.physical_margin,
            "min_order_effect": args.min_order_effect,
            "representation_accuracy": args.representation_accuracy,
            "minimum_accuracy_gap": args.minimum_accuracy_gap,
            "reset_tolerance": args.reset_tolerance,
        },
    }
    gates = {
        "complete": len(records) == args.expected_anchors,
        "official_source_single_checkpoint": provenance_ok,
        "reset_deterministic": reset_physics <= args.reset_tolerance
        and reset_pixels == 0
        and repeat_endpoint <= args.reset_tolerance,
        "witness_coverage": summary["witness_success_rate"]["mean"] >= 0.95,
        "enough_order_signal": summary["pairs_relevant"] >= args.min_relevant_pairs,
        "latent_metric_is_physically_useful": summary[
            "relevant_pair_true_physical_accuracy"
        ]["ci95"][0]
        >= args.representation_accuracy,
        "predictor_has_order_ranking_gap": summary["relevant_pair_accuracy_gap"][
            "ci95"
        ][0]
        >= args.minimum_accuracy_gap,
        "predictor_adds_ordinary_selection_regret": summary["ordinary_panel"][
            "dynamics_excess_regret"
        ]["ci95"][0]
        > 0,
    }
    summary["gates"] = gates
    summary["decision"] = "GO_STAGE_B_SCREEN" if all(gates.values()) else "STOP_OR_FIX_STAGE_A"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# ORDER-JEPA LeWM-Reacher Stage-A decision",
        "",
        f"**Decision: `{summary['decision']}`**",
        "",
        "This is a qualification-screen decision, not a trained-method result.",
        "",
        "## Gate table",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {'YES' if passed else 'NO'} |" for name, passed in gates.items())
    lines.extend(
        [
            "",
            "## Key estimates",
            "",
            f"- Anchors: {len(records)}/{args.expected_anchors}; relevant pairs: {summary['pairs_relevant']}/{len(pairs)}.",
            "- Mean normalized physical order effect: "
            f"{metric_text(summary['physical_order_effect'])}.",
            "- True-latent physical ranking accuracy: "
            f"{metric_text(summary['relevant_pair_true_physical_accuracy'])}.",
            "- Predicted physical ranking accuracy: "
            f"{metric_text(summary['relevant_pair_predicted_physical_accuracy'])}.",
            "- Accuracy gap (true minus predicted): "
            f"{metric_text(summary['relevant_pair_accuracy_gap'])}.",
            "- Ordinary-panel true/predicted selected success: "
            f"{metric_text(summary['ordinary_panel']['true_latent_success_rate'])} / "
            f"{metric_text(summary['ordinary_panel']['predicted_latent_success_rate'])}.",
            "- Ordinary-panel dynamics excess regret: "
            f"{metric_text(summary['ordinary_panel']['dynamics_excess_regret'])}.",
            "- Relevant order-vector cosine and amplitude ratio: "
            f"{metric_text(summary['relevant_order_cosine'])} / "
            f"{metric_text(summary['relevant_order_amplitude_ratio'])}.",
        ]
    )
    (args.out_dir / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
