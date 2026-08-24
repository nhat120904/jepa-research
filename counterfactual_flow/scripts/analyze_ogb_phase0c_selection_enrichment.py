#!/usr/bin/env python3
"""Measure whether CEM final populations enrich proxy-deceptive selections.

The persisted Stage-0 artifacts contain candidate populations but not the
solver's final proposal mean.  Therefore this audit evaluates the proxy argmin
*within each recorded population*.  It must not be interpreted as the exact
deployed CEM action.  All physical outcomes were already evaluated in Phase 0.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--min-physical-gap-m", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return ranks / max(len(values) - 1, 1)


def mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("bootstrap inputs must be one-dimensional")
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_finite": 0}
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_finite": int(len(values)),
    }


def read_population(path: Path, source: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["source"] == source]
    if not rows:
        raise RuntimeError(f"no {source} rows in {path}")
    return rows


def population_metrics(
    snapshot: int, source: str, rows: list[dict[str, str]], top_frac: float, min_gap: float
) -> dict[str, Any]:
    proxy = np.asarray([float(row["learned_proxy_cost"]) for row in rows])
    physical = np.asarray([float(row["physical_distance_m"]) for row in rows])
    success = np.asarray([int(row["success"]) for row in rows], dtype=int)
    proxy_order = np.argsort(proxy, kind="stable")
    physical_order = np.argsort(physical, kind="stable")
    proxy_rank = rank_fraction(proxy)
    regret = physical - physical[physical_order[0]]
    n_elite = max(1, int(np.ceil(top_frac * len(rows))))
    elite = proxy_order[:n_elite]
    non_elite = proxy_order[n_elite:]
    selected = int(proxy_order[0])

    # A selected inversion requires a physically better action that the proxy
    # ranks below the selected candidate by strict order, not merely high regret.
    corrective = (proxy_rank > proxy_rank[selected]) & (
        physical <= physical[selected] - min_gap
    )
    selected_verified_inversion = bool(regret[selected] >= min_gap and corrective.any())

    dp = proxy[:, None] - proxy[None, :]
    dt = physical[:, None] - physical[None, :]
    upper = np.triu(np.ones((len(rows), len(rows)), dtype=bool), k=1)
    comparable = upper & (dp != 0.0) & (dt != 0.0)
    inversion_fraction = float(np.mean((dp[comparable] * dt[comparable]) < 0.0))

    return {
        "snapshot": snapshot,
        "source": source,
        "n_candidates": len(rows),
        "selected_candidate": int(rows[selected]["candidate"]),
        "selected_physical_distance_m": float(physical[selected]),
        "selected_physical_regret_m": float(regret[selected]),
        "selected_physical_rank_fraction": float(rank_fraction(physical)[selected]),
        "selected_success": int(success[selected]),
        "selected_verified_inversion": int(selected_verified_inversion),
        "physical_oracle_distance_m": float(physical[physical_order[0]]),
        "physical_oracle_success": int(success.any()),
        "elite_mean_physical_regret_m": float(regret[elite].mean()),
        "nonelite_mean_physical_regret_m": float(regret[non_elite].mean()),
        "elite_excess_regret_m": float(regret[elite].mean() - regret[non_elite].mean()),
        "elite_false_rate_above_physical_median": float(np.mean(physical[elite] > np.median(physical))),
        "elite_physical_top_recall": float(
            len(set(elite.tolist()) & set(physical_order[:n_elite].tolist())) / n_elite
        ),
        "proxy_physical_inversion_fraction": inversion_fraction,
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.top_frac < 1 or args.min_physical_gap_m <= 0 or args.bootstrap <= 0:
        raise ValueError("invalid thresholds or bootstrap count")
    paths = sorted(args.shards.glob("*/candidates.csv"), key=lambda p: int(p.parent.name))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} candidate files, found {len(paths)}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        snapshot = int(path.parent.name)
        for source in ("cem_initial", "cem_final"):
            rows.append(population_metrics(
                snapshot, source, read_population(path, source), args.top_frac, args.min_physical_gap_m
            ))

    initial = {row["snapshot"]: row for row in rows if row["source"] == "cem_initial"}
    final = {row["snapshot"]: row for row in rows if row["source"] == "cem_final"}
    if set(initial) != set(final) or len(initial) != args.expected_shards:
        raise RuntimeError("initial/final snapshot pairing is incomplete")
    paired: list[dict[str, Any]] = []
    paired_metric_names = (
        "selected_physical_distance_m",
        "selected_physical_regret_m",
        "selected_physical_rank_fraction",
        "elite_mean_physical_regret_m",
        "elite_excess_regret_m",
        "elite_false_rate_above_physical_median",
        "elite_physical_top_recall",
        "proxy_physical_inversion_fraction",
        "selected_verified_inversion",
    )
    for snapshot in sorted(initial):
        record = {"snapshot": snapshot}
        for name in paired_metric_names:
            record[f"delta_final_minus_initial_{name}"] = float(final[snapshot][name] - initial[snapshot][name])
        paired.append(record)

    rng = np.random.default_rng(args.seed)
    source_summary: dict[str, dict[str, dict[str, float]]] = {}
    summary_names = (
        "selected_physical_distance_m", "selected_physical_regret_m",
        "selected_physical_rank_fraction", "selected_success", "selected_verified_inversion",
        "elite_mean_physical_regret_m", "elite_excess_regret_m",
        "elite_false_rate_above_physical_median", "elite_physical_top_recall",
        "proxy_physical_inversion_fraction",
    )
    for source in ("cem_initial", "cem_final"):
        subset = [row for row in rows if row["source"] == source]
        source_summary[source] = {
            name: mean_ci(np.asarray([row[name] for row in subset]), rng, args.bootstrap)
            for name in summary_names
        }
    paired_summary = {
        name: mean_ci(
            np.asarray([row[f"delta_final_minus_initial_{name}"] for row in paired]),
            rng,
            args.bootstrap,
        )
        for name in paired_metric_names
    }
    selected_regret_delta = paired_summary["selected_physical_regret_m"]
    verdict = (
        "FINAL_POOL_PROXY_ARGMIN_MORE_PHYSICALLY_REGRETFUL"
        if selected_regret_delta["ci_low"] > 0.0
        else "NO_CI_CLEAN_FINAL_POOL_SELECTION_REGRET_ENRICHMENT"
    )
    report = {
        "scope": (
            "post-hoc population-level audit of persisted Phase-0 candidates. "
            "The selected action is proxy argmin within a recorded population, not the unpersisted CEM proposal mean."
        ),
        "definitions": {
            "elite": f"lowest {args.top_frac:.0%} proxy-cost candidates",
            "selected_verified_inversion": (
                "proxy argmin has at least 2 cm physical regret and a same-population, "
                "proxy-rejected candidate that is physically better by at least 2 cm"
            ),
            "paired_delta": "final population metric minus initial population metric within snapshot",
        },
        "config": {
            "n_snapshots": args.expected_shards,
            "top_frac": args.top_frac,
            "min_physical_gap_m": args.min_physical_gap_m,
            "bootstrap_draws": args.bootstrap,
            "bootstrap_seed": args.seed,
        },
        "source_summary": source_summary,
        "paired_final_minus_initial": paired_summary,
        "verdict": verdict,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "population_selection_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "paired_final_minus_initial.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# Phase 0c — CEM population selection enrichment",
        "",
        f"- **Verdict:** {verdict}",
        f"- Proxy-argmin physical regret, initial: {source_summary['cem_initial']['selected_physical_regret_m']['mean'] * 100:.2f} cm",
        f"- Proxy-argmin physical regret, final: {source_summary['cem_final']['selected_physical_regret_m']['mean'] * 100:.2f} cm",
        (
            "- Paired final−initial proxy-argmin regret: "
            f"{selected_regret_delta['mean'] * 100:+.2f} cm "
            f"[{selected_regret_delta['ci_low'] * 100:+.2f}, {selected_regret_delta['ci_high'] * 100:+.2f}]"
        ),
        (
            "- Selected verified inversion rate, initial/final: "
            f"{source_summary['cem_initial']['selected_verified_inversion']['mean'] * 100:.1f}% / "
            f"{source_summary['cem_final']['selected_verified_inversion']['mean'] * 100:.1f}%"
        ),
        "",
        "This is a paired population comparison, not evidence about the unpersisted final CEM proposal mean or downstream policy learning.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
