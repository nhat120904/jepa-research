#!/usr/bin/env python3
"""Aggregate Stage-A shards and make the pre-registered train/stop decision."""

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
from order_jepa.original_dino_wm import PINNED_COMMIT  # noqa: E402


def require_compute_node() -> None:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Stage-A aggregation must run through scripts/slurm_analyze.sh")


def metric(values: np.ndarray, clusters: np.ndarray, n_resamples: int, seed: int) -> dict:
    mean, lo, hi = clustered_bootstrap_mean(
        values, clusters, n_resamples=n_resamples, seed=seed
    )
    return {"mean": mean, "ci95": [lo, hi], "n": int(np.isfinite(values).sum())}


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-anchors", type=int, default=200)
    parser.add_argument("--min-relevant-pairs", type=int, default=100)
    parser.add_argument("--physical-margin", type=float, default=0.10)
    parser.add_argument("--min-object-effect", type=float, default=0.25)
    parser.add_argument("--representation-accuracy", type=float, default=0.65)
    parser.add_argument("--minimum-accuracy-gap", type=float, default=0.10)
    parser.add_argument("--reset-tolerance", type=float, default=1e-6)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    require_compute_node()

    paths = sorted(args.scores_dir.glob("anchor_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    if not records:
        raise FileNotFoundError(f"no score shards in {args.scores_dir}")
    provenance_keys = {
        (
            row["provenance"]["implementation"],
            row["provenance"]["source_commit"],
            row["provenance"]["checkpoint_sha256"],
            row["provenance"]["config_sha256"],
        )
        for row in records
    }
    provenance_ok = (
        len(provenance_keys) == 1
        and next(iter(provenance_keys))[0] == "gaoyuezhou/dino_wm"
        and next(iter(provenance_keys))[1] == PINNED_COMMIT
    )

    episodes = np.asarray([row["episode"] for row in records], dtype=np.int64)
    excess_regret = np.asarray(
        [row["selection"]["dynamics_excess_regret"] for row in records], dtype=np.float64
    )
    true_regret = np.asarray(
        [row["selection"]["true_latent_regret"] for row in records], dtype=np.float64
    )
    pred_regret = np.asarray(
        [row["selection"]["predicted_latent_regret"] for row in records], dtype=np.float64
    )
    coverage = np.asarray(
        [row["selection"]["oracle_has_success"] for row in records], dtype=np.float64
    )

    # The manifest includes one explicitly labeled feasible witness so that
    # task feasibility can be certified.  Selection quality must also be
    # reported on the ordinary panel without that witness; otherwise picking
    # the demonstrated goal suffix makes regret nearly tautological.
    ordinary_coverage = []
    ordinary_true_success = []
    ordinary_pred_success = []
    ordinary_true_regret = []
    ordinary_pred_regret = []
    for row in records:
        candidates = [c for c in row["candidates"] if c["kind"] != "witness"]
        physical = np.asarray([c["physical_cost"] for c in candidates], dtype=np.float64)
        true_cost = np.asarray(
            [c["true_latent_cost"] for c in candidates], dtype=np.float64
        )
        pred_cost = np.asarray(
            [c["predicted_latent_cost"] for c in candidates], dtype=np.float64
        )
        success = np.asarray([c["success"] for c in candidates], dtype=bool)
        true_index, true_panel_regret = selection_regret(true_cost, physical)
        pred_index, pred_panel_regret = selection_regret(pred_cost, physical)
        ordinary_coverage.append(float(np.any(success)))
        ordinary_true_success.append(float(success[true_index]))
        ordinary_pred_success.append(float(success[pred_index]))
        ordinary_true_regret.append(true_panel_regret)
        ordinary_pred_regret.append(pred_panel_regret)
    ordinary_coverage_np = np.asarray(ordinary_coverage, dtype=np.float64)
    ordinary_true_success_np = np.asarray(ordinary_true_success, dtype=np.float64)
    ordinary_pred_success_np = np.asarray(ordinary_pred_success, dtype=np.float64)
    ordinary_true_regret_np = np.asarray(ordinary_true_regret, dtype=np.float64)
    ordinary_pred_regret_np = np.asarray(ordinary_pred_regret, dtype=np.float64)
    ordinary_excess_regret_np = ordinary_pred_regret_np - ordinary_true_regret_np
    reset_physics = max(row["reset"]["physics_max_abs"] for row in records)
    reset_pixels = max(row["reset"]["pixel_max_abs"] for row in records)
    repeat_endpoint = max(row["reset"]["repeat_endpoint_max_abs"] for row in records)

    pairs = []
    pair_episode = []
    for row in records:
        for pair in row["order_pairs"]:
            pairs.append(pair)
            pair_episode.append(row["episode"])
    pair_episode_np = np.asarray(pair_episode, dtype=np.int64)
    object_effect = np.asarray([pair["object_effect"] for pair in pairs], dtype=np.float64)
    noise = np.asarray([pair["repeat_object_noise"] for pair in pairs], dtype=np.float64)
    physical_delta = np.asarray([pair["physical_delta"] for pair in pairs], dtype=np.float64)
    true_delta = np.asarray([pair["true_latent_delta"] for pair in pairs], dtype=np.float64)
    pred_delta = np.asarray([pair["predicted_latent_delta"] for pair in pairs], dtype=np.float64)
    relevant = (
        (object_effect >= args.min_object_effect)
        & (object_effect > 3.0 * noise)
        & (np.abs(physical_delta) >= args.physical_margin)
    )
    relevant_count = int(np.sum(relevant))
    true_correct = (np.sign(true_delta) == np.sign(physical_delta)).astype(np.float64)
    pred_correct = (np.sign(pred_delta) == np.sign(physical_delta)).astype(np.float64)
    accuracy_gap = true_correct - pred_correct

    summary = {
        "schema": "order-jepa-stage-a-decision-v1",
        "anchors_found": len(records),
        "expected_anchors": args.expected_anchors,
        "episodes": int(np.unique(episodes).size),
        "pairs_total": len(pairs),
        "pairs_relevant": relevant_count,
        "provenance": records[0]["provenance"],
        "provenance_consistent_and_original_source": provenance_ok,
        "reset": {
            "physics_max_abs": reset_physics,
            "pixel_max_abs": reset_pixels,
            "repeat_endpoint_max_abs": repeat_endpoint,
        },
        "coverage_rate": metric(coverage, episodes, args.n_bootstrap, args.seed),
        "true_latent_regret": metric(true_regret, episodes, args.n_bootstrap, args.seed + 1),
        "predicted_latent_regret": metric(pred_regret, episodes, args.n_bootstrap, args.seed + 2),
        "dynamics_excess_regret": metric(
            excess_regret, episodes, args.n_bootstrap, args.seed + 3
        ),
        "ordinary_panel": {
            "excludes_labeled_witness": True,
            "coverage_rate": metric(
                ordinary_coverage_np, episodes, args.n_bootstrap, args.seed + 20
            ),
            "true_latent_success_rate": metric(
                ordinary_true_success_np, episodes, args.n_bootstrap, args.seed + 21
            ),
            "predicted_latent_success_rate": metric(
                ordinary_pred_success_np, episodes, args.n_bootstrap, args.seed + 22
            ),
            "true_latent_regret": metric(
                ordinary_true_regret_np, episodes, args.n_bootstrap, args.seed + 23
            ),
            "predicted_latent_regret": metric(
                ordinary_pred_regret_np, episodes, args.n_bootstrap, args.seed + 24
            ),
            "dynamics_excess_regret": metric(
                ordinary_excess_regret_np, episodes, args.n_bootstrap, args.seed + 25
            ),
        },
        "relevant_pair_true_physical_accuracy": metric(
            true_correct[relevant], pair_episode_np[relevant], args.n_bootstrap, args.seed + 4
        ),
        "relevant_pair_predicted_physical_accuracy": metric(
            pred_correct[relevant], pair_episode_np[relevant], args.n_bootstrap, args.seed + 5
        ),
        "relevant_pair_accuracy_gap": metric(
            accuracy_gap[relevant], pair_episode_np[relevant], args.n_bootstrap, args.seed + 6
        ),
        "relevant_order_normalized_error": metric(
            np.asarray([pair["normalized_error"] for pair in pairs])[relevant],
            pair_episode_np[relevant],
            args.n_bootstrap,
            args.seed + 7,
        ),
        "relevant_order_cosine": metric(
            np.asarray([pair["cosine"] for pair in pairs])[relevant],
            pair_episode_np[relevant],
            args.n_bootstrap,
            args.seed + 8,
        ),
        "relevant_order_amplitude_ratio": metric(
            np.asarray([pair["amplitude_ratio"] for pair in pairs])[relevant],
            pair_episode_np[relevant],
            args.n_bootstrap,
            args.seed + 9,
        ),
        "thresholds": {
            "min_relevant_pairs": args.min_relevant_pairs,
            "physical_margin": args.physical_margin,
            "min_object_effect": args.min_object_effect,
            "representation_accuracy": args.representation_accuracy,
            "minimum_accuracy_gap": args.minimum_accuracy_gap,
            "reset_tolerance": args.reset_tolerance,
        },
    }
    gates = {
        "complete": len(records) == args.expected_anchors,
        "original_source_single_checkpoint": provenance_ok,
        "reset_deterministic": reset_physics <= args.reset_tolerance
        and reset_pixels == 0
        and repeat_endpoint <= args.reset_tolerance,
        "witness_coverage": summary["coverage_rate"]["mean"] >= 0.95,
        "enough_order_signal": relevant_count >= args.min_relevant_pairs,
        "latent_metric_is_physically_useful": summary[
            "relevant_pair_true_physical_accuracy"
        ]["ci95"][0]
        >= args.representation_accuracy,
        "predictor_has_order_ranking_gap": summary["relevant_pair_accuracy_gap"]["ci95"][0]
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
        "# ORDER-JEPA Stage-A decision",
        "",
        f"**Decision: `{summary['decision']}`**",
        "",
        "This is a qualification-screen decision, not a method result.",
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
            f"- Anchors: {len(records)}/{args.expected_anchors}; relevant pairs: {relevant_count}/{len(pairs)}.",
            f"- True-latent physical ranking accuracy: {fmt(summary['relevant_pair_true_physical_accuracy']['mean'])} "
            f"[{fmt(summary['relevant_pair_true_physical_accuracy']['ci95'][0])}, {fmt(summary['relevant_pair_true_physical_accuracy']['ci95'][1])}].",
            f"- Predicted physical ranking accuracy: {fmt(summary['relevant_pair_predicted_physical_accuracy']['mean'])} "
            f"[{fmt(summary['relevant_pair_predicted_physical_accuracy']['ci95'][0])}, {fmt(summary['relevant_pair_predicted_physical_accuracy']['ci95'][1])}].",
            f"- Accuracy gap (true minus predicted): {fmt(summary['relevant_pair_accuracy_gap']['mean'])} "
            f"[{fmt(summary['relevant_pair_accuracy_gap']['ci95'][0])}, {fmt(summary['relevant_pair_accuracy_gap']['ci95'][1])}].",
            f"- Dynamics excess regret: {fmt(summary['dynamics_excess_regret']['mean'])} "
            f"[{fmt(summary['dynamics_excess_regret']['ci95'][0])}, {fmt(summary['dynamics_excess_regret']['ci95'][1])}].",
            f"- Ordinary-panel dynamics excess regret (witness excluded): "
            f"{fmt(summary['ordinary_panel']['dynamics_excess_regret']['mean'])} "
            f"[{fmt(summary['ordinary_panel']['dynamics_excess_regret']['ci95'][0])}, "
            f"{fmt(summary['ordinary_panel']['dynamics_excess_regret']['ci95'][1])}].",
            f"- Relevant order-vector cosine: {fmt(summary['relevant_order_cosine']['mean'])}; "
            f"amplitude ratio: {fmt(summary['relevant_order_amplitude_ratio']['mean'])}.",
            "",
            "Stage B remains closed unless every gate passes. The witness panel certifies feasibility only; selection regret is gated on the ordinary panel.",
        ]
    )
    (args.out_dir / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": summary["decision"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
