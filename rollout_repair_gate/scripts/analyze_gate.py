#!/usr/bin/env python3
"""Locked snapshot-clustered analysis for the predictor gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rollout_repair_gate.core import ARMS, spearman_no_tie_assumption


SEEDS = (11, 23, 47)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "n": int(len(values)),
    }


def arm_labels(arm: str) -> list[str]:
    return [f"{arm}_seed{seed}" for seed in SEEDS]


def load_fixed(root: Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    paths = sorted(root.glob("snapshot_*.npz"))
    if len(paths) != 32:
        raise RuntimeError(f"expected 32 fixed-pool files, got {len(paths)}")
    labels: list[str] | None = None
    metrics: dict[str, dict[str, list]] = {}
    for path in paths:
        with np.load(path) as data:
            current_labels = [str(value) for value in data["labels"].tolist()]
            predicted = data["predicted_cost"].astype(np.float64)
            prediction_mse = data["prediction_mse"].astype(np.float64)
            true_cost = data["true_dataset_cost"].astype(np.float64)
            physical = data["physical_distance_m"].astype(np.float64)
            valid = data["valid_horizon"].astype(bool)
        if labels is None:
            labels = current_labels
            metrics = {
                label: {
                    "rho_latent": [],
                    "rho_physical": [],
                    "latent_mse": [],
                    "selected_distance": [],
                    "selected_success": [],
                }
                for label in labels
            }
        if current_labels != labels:
            raise RuntimeError("fixed-pool label mismatch")
        for model_index, label in enumerate(labels):
            rho_latent = np.empty(
                (predicted.shape[1], predicted.shape[-1]), dtype=np.float64
            )
            rho_physical = np.empty_like(rho_latent)
            mse = np.empty_like(rho_latent)
            selected_distance, selected_success = [], []
            for population in range(predicted.shape[1]):
                for horizon in range(predicted.shape[-1]):
                    keep = valid[population, :, horizon]
                    rho_latent[population, horizon] = spearman_no_tie_assumption(
                        predicted[model_index, population, keep, horizon],
                        true_cost[population, keep, horizon],
                    )
                    rho_physical[population, horizon] = spearman_no_tie_assumption(
                        predicted[model_index, population, keep, horizon],
                        physical[population, keep, horizon],
                    )
                    mse[population, horizon] = np.mean(
                        prediction_mse[model_index, population, keep, horizon]
                    )
                selected = int(np.argmin(predicted[model_index, population, :, -1]))
                value = float(physical[population, selected, -1])
                selected_distance.append(value)
                selected_success.append(float(value <= 0.04))
            # Two populations share a snapshot and are averaged before bootstrap.
            metrics[label]["rho_latent"].append(np.nanmean(rho_latent, axis=0))
            metrics[label]["rho_physical"].append(np.nanmean(rho_physical, axis=0))
            metrics[label]["latent_mse"].append(np.nanmean(mse, axis=0))
            metrics[label]["selected_distance"].append(float(np.mean(selected_distance)))
            metrics[label]["selected_success"].append(float(np.mean(selected_success)))
    assert labels is not None
    packed = {
        label: {name: np.asarray(values) for name, values in row.items()}
        for label, row in metrics.items()
    }
    return labels, packed


def load_fresh(root: Path, expected_labels: list[str]) -> dict[str, dict[str, np.ndarray]]:
    paths = sorted(root.glob("snapshot_*.json"))
    if len(paths) != 32:
        raise RuntimeError(f"expected 32 fresh-CEM files, got {len(paths)}")
    rows = {label: {"distance": [], "success": []} for label in expected_labels}
    for path in paths:
        payload = json.loads(path.read_text())
        mapping = {row["label"]: row for row in payload}
        if set(mapping) != set(expected_labels):
            raise RuntimeError(f"fresh label mismatch in {path}")
        for label in expected_labels:
            rows[label]["distance"].append(float(mapping[label]["physical_distance_m"]))
            rows[label]["success"].append(float(mapping[label]["success"]))
    return {
        label: {name: np.asarray(values) for name, values in row.items()}
        for label, row in rows.items()
    }


def mean_seed_metric(source: dict[str, dict[str, np.ndarray]], arm: str, metric: str) -> np.ndarray:
    return np.mean(np.stack([source[label][metric] for label in arm_labels(arm)]), axis=0)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    labels, fixed = load_fixed(args.fixed_dir)
    expected = ["native"] + [f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS]
    if set(labels) != set(expected):
        raise RuntimeError(f"unexpected labels: {labels}")
    fresh = load_fresh(args.fresh_dir, labels)

    signatures = set()
    for path in sorted(args.checkpoint_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        signatures.add(json.dumps(payload["same_compute_signature"], sort_keys=True))
    if len(signatures) != 1:
        raise RuntimeError("same-compute signature gate failed")

    report: dict[str, object] = {
        "same_compute_signature": json.loads(next(iter(signatures))),
        "fixed_pool": {},
        "fresh_cem": {},
        "contrasts": {},
    }
    for label in labels:
        report["fixed_pool"][label] = {
            "selected_distance_m": interval(fixed[label]["selected_distance"], rng, args.bootstrap),
            "selected_success": interval(fixed[label]["selected_success"], rng, args.bootstrap),
            "rho_latent_by_horizon": [
                interval(fixed[label]["rho_latent"][:, horizon], rng, args.bootstrap)
                for horizon in range(fixed[label]["rho_latent"].shape[1])
            ],
            "rho_physical_by_horizon": [
                interval(fixed[label]["rho_physical"][:, horizon], rng, args.bootstrap)
                for horizon in range(fixed[label]["rho_physical"].shape[1])
            ],
            "latent_mse_by_horizon": [
                interval(fixed[label]["latent_mse"][:, horizon], rng, args.bootstrap)
                for horizon in range(fixed[label]["latent_mse"].shape[1])
            ],
        }
        report["fresh_cem"][label] = {
            "distance_m": interval(fresh[label]["distance"], rng, args.bootstrap),
            "success": interval(fresh[label]["success"], rng, args.bootstrap),
        }

    for arm in ARMS:
        fixed_distance = mean_seed_metric(fixed, arm, "selected_distance")
        fixed_success = mean_seed_metric(fixed, arm, "selected_success")
        fixed_rho_latent = mean_seed_metric(fixed, arm, "rho_latent")
        fixed_rho_physical = mean_seed_metric(fixed, arm, "rho_physical")
        fixed_mse = mean_seed_metric(fixed, arm, "latent_mse")
        fresh_distance = mean_seed_metric(fresh, arm, "distance")
        fresh_success = mean_seed_metric(fresh, arm, "success")
        report["fixed_pool"][arm] = {
            "selected_distance_m": interval(fixed_distance, rng, args.bootstrap),
            "selected_success": interval(fixed_success, rng, args.bootstrap),
            "rho_latent_by_horizon": [
                interval(fixed_rho_latent[:, horizon], rng, args.bootstrap)
                for horizon in range(fixed_rho_latent.shape[1])
            ],
            "rho_physical_by_horizon": [
                interval(fixed_rho_physical[:, horizon], rng, args.bootstrap)
                for horizon in range(fixed_rho_physical.shape[1])
            ],
            "latent_mse_by_horizon": [
                interval(fixed_mse[:, horizon], rng, args.bootstrap)
                for horizon in range(fixed_mse.shape[1])
            ],
        }
        report["fresh_cem"][arm] = {
            "distance_m": interval(fresh_distance, rng, args.bootstrap),
            "success": interval(fresh_success, rng, args.bootstrap),
        }

    comparisons = (
        ("multistep_expert_vs_one_step", "multistep_expert", "one_step_expert"),
        ("offpolicy_vs_multistep_expert", "multistep_offpolicy", "multistep_expert"),
        ("offpolicy_vs_one_step", "multistep_offpolicy", "one_step_expert"),
    )
    for name, method, baseline in comparisons:
        fixed_gain = mean_seed_metric(fixed, baseline, "selected_distance") - mean_seed_metric(
            fixed, method, "selected_distance"
        )
        fresh_gain = mean_seed_metric(fresh, baseline, "distance") - mean_seed_metric(
            fresh, method, "distance"
        )
        success_gain = mean_seed_metric(fresh, method, "success") - mean_seed_metric(
            fresh, baseline, "success"
        )
        report["contrasts"][name] = {
            "fixed_distance_gain_m": interval(fixed_gain, rng, args.bootstrap),
            "fresh_distance_gain_m": interval(fresh_gain, rng, args.bootstrap),
            "fresh_success_gain": interval(success_gain, rng, args.bootstrap),
        }

    offpolicy_rho5 = report["fixed_pool"]["multistep_offpolicy"][
        "rho_latent_by_horizon"
    ][4]["mean"]
    primary = report["contrasts"]["offpolicy_vs_one_step"]
    gates = {
        "rho5_ge_0_50": bool(offpolicy_rho5 >= 0.50),
        "fixed_gain_ci_clean": bool(primary["fixed_distance_gain_m"]["ci_low"] > 0),
        "fresh_gain_ci_clean": bool(primary["fresh_distance_gain_m"]["ci_low"] > 0),
        "fresh_success_no_harm": bool(primary["fresh_success_gain"]["mean"] >= -0.03),
    }
    native_rho1 = report["fixed_pool"]["native"]["rho_latent_by_horizon"][0]["mean"]
    if all(gates.values()):
        verdict = "PASS_DYNAMICS_REPAIR"
    elif gates["rho5_ge_0_50"] and gates["fixed_gain_ci_clean"] and not gates["fresh_gain_ci_clean"]:
        verdict = "FIXED_POOL_ONLY_REOPTIMIZATION_FAILURE"
    elif report["contrasts"]["multistep_expert_vs_one_step"]["fresh_distance_gain_m"]["ci_low"] > 0:
        verdict = "ROLLOUT_MODE_ONLY"
    else:
        verdict = "STOP_DYNAMICS_REPAIR"
    report["gates"] = gates
    report["native_rho1_lt_0_20"] = bool(native_rho1 < 0.20)
    report["verdict"] = verdict
    report["bootstrap"] = {
        "unit": "snapshot",
        "draws": args.bootstrap,
        "seed": args.seed,
        "seed_aggregation": "mean three training seeds within snapshot before bootstrap",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decision.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Deployment-matched predictor gate",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## Locked gates",
        "",
    ]
    lines.extend(f"- {name}: **{value}**" for name, value in gates.items())
    lines.extend(
        [
            "",
            f"Native off-policy rho at k=1: **{native_rho1:.3f}**.",
            "",
            "Full snapshot-clustered intervals and per-seed results are in `decision.json`.",
        ]
    )
    (args.out_dir / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "gates": gates, "native_rho1": native_rho1}, indent=2))


if __name__ == "__main__":
    main()
