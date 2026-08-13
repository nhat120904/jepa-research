#!/usr/bin/env python3
"""Snapshot-clustered aggregation of the OGBench matched-refit intervention.

Reads the per-snapshot shards written by ``84_ogb_matched_refit.py`` and reports
the paired latent-minus-physical differences at the final CEM iteration, with a
snapshot bootstrap and a sign count, mirroring the MetaWorld estimator in
``55_analyze_shared_population_branch.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_DELTAS = [
    "delta_best_task_distance_m",
    "delta_best_shaped_cost",
    "delta_selected_task_distance_m",
    "delta_mean_task_distance_m",
]

ITERATION_CURVES = [
    "best_task_distance_m",
    "selected_task_distance_m",
    "latent_physical_spearman",
    "latent_top10pct_recall_physical",
    "n_success",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--require-shards", type=int, default=32)
    return parser.parse_args()


def bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    estimate = float(values.mean())
    if len(values) == 1:
        return {"estimate": estimate, "ci_low": estimate, "ci_high": estimate}
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high)}


def read_iterations(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    shards = sorted(
        (path for path in args.shard_root.iterdir() if (path / "summary.json").exists()),
        key=lambda path: int(path.name),
    )
    if len(shards) < args.require_shards:
        raise SystemExit(
            f"found {len(shards)} shards, required {args.require_shards}"
        )

    summaries: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for shard in shards:
        summary = json.loads((shard / "summary.json").read_text())
        if not summary["gate"]["pass"]:
            raise SystemExit(f"shard {shard.name} did not pass its gates")
        summaries.append(summary)
        iteration_rows.extend(read_iterations(shard / "iteration_metrics.csv"))

    configs = {json.dumps(s["config"], sort_keys=True) for s in summaries}
    config = json.loads(next(iter(configs)))
    if len(configs) != 1:
        differing = {
            key
            for key in config
            if len({json.dumps(json.loads(c)[key], sort_keys=True) for c in configs}) > 1
        }
        if differing - {"snapshot_index"}:
            raise SystemExit(f"shards disagree on config keys {sorted(differing)}")

    primary: dict[str, Any] = {}
    for name in PRIMARY_DELTAS:
        values = np.asarray([s["primary"][name] for s in summaries], dtype=float)
        stats = bootstrap_mean(values, rng, args.bootstrap)
        stats["n_snapshots"] = int(len(values))
        stats["n_favor_physical"] = int(np.sum(values > 0))
        stats["n_favor_latent"] = int(np.sum(values < 0))
        stats["n_tied"] = int(np.sum(values == 0))
        stats["median"] = float(np.median(values))
        primary[name] = stats

    success_latent = np.asarray(
        [s["primary"]["latent_final_best_task_distance_m"] for s in summaries]
    )
    success_physical = np.asarray(
        [s["primary"]["physical_final_best_task_distance_m"] for s in summaries]
    )
    levels = {
        "latent_final_best_task_distance_m": bootstrap_mean(
            success_latent, rng, args.bootstrap
        ),
        "physical_final_best_task_distance_m": bootstrap_mean(
            success_physical, rng, args.bootstrap
        ),
        "start_distance_m": bootstrap_mean(
            np.asarray([s["start_distance_m"] for s in summaries]), rng, args.bootstrap
        ),
    }

    final_iter = int(config["cem_iterations"]) - 1
    curves: dict[str, dict[str, dict[str, Any]]] = {}
    for branch in ("latent", "physical"):
        curves[branch] = {}
        for iteration in (0, final_iter):
            selected = [
                row
                for row in iteration_rows
                if row["branch"] == branch and int(row["iter"]) == iteration
            ]
            curves[branch][str(iteration)] = {
                metric: bootstrap_mean(
                    np.asarray([float(row[metric]) for row in selected]),
                    rng,
                    args.bootstrap,
                )
                for metric in ITERATION_CURVES
            }

    result = {
        "config": config,
        "n_shards": len(shards),
        "primary_deltas_latent_minus_physical": primary,
        "levels": levels,
        "iteration_curves": curves,
        "bootstrap_draws": args.bootstrap,
        "note": (
            "positive delta favours physical-cost refitting; unit of "
            "uncertainty is the evaluation snapshot"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )

    with (args.out_dir / "snapshot_deltas.csv").open("w", newline="") as handle:
        fields = ["snapshot", *PRIMARY_DELTAS, "delta_success_any",
                  "latent_final_best_task_distance_m",
                  "physical_final_best_task_distance_m", "start_distance_m"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                "snapshot": summary["snapshot"]["order"],
                "start_distance_m": summary["start_distance_m"],
                **{key: summary["primary"][key] for key in fields[1:-1]},
            })

    lines = [
        "# OGBench-Cube matched-refit intervention",
        "",
        f"Shards: {len(shards)}; CEM {config['num_samples']}x{config['cem_iterations']}, "
        f"topk {config['topk']}, horizon {config['horizon']}, w_hand {config['w_hand']}.",
        "",
        "Latent-minus-physical differences at the final iteration; positive "
        "favours physical-cost refitting.",
        "",
        "| quantity | mean [95% CI] | median | snapshots favouring physical |",
        "|---|---|---|---|",
    ]
    for name, stats in primary.items():
        scale = 100.0 if name.endswith("_m") else 1.0
        unit = " cm" if name.endswith("_m") else ""
        lines.append(
            f"| {name} | {stats['estimate'] * scale:.2f} "
            f"[{stats['ci_low'] * scale:.2f}, {stats['ci_high'] * scale:.2f}]{unit} | "
            f"{stats['median'] * scale:.2f}{unit} | "
            f"{stats['n_favor_physical']}/{stats['n_snapshots']} |"
        )
    lines += ["", "Levels (metres):", ""]
    for name, stats in levels.items():
        lines.append(
            f"- {name}: {stats['estimate']:.4f} "
            f"[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]"
        )
    (args.out_dir / "MATCHED_REFIT_REPORT.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(result["primary_deltas_latent_minus_physical"], indent=2))
    print(f"wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
