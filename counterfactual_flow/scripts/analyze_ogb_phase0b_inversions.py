#!/usr/bin/env python3
"""Verify same-population proxy/physics ordinal inversions from Phase-0 pools.

This analysis never loads LeWM or MuJoCo.  It uses the persisted candidate
tables produced by Phase 0, where every candidate was already evaluated from
the same restored simulator state.  The statistical unit is a snapshot, not a
candidate pair.
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
    parser.add_argument("--proxy-good-top-frac", type=float, default=0.10)
    parser.add_argument("--min-physical-gap-m", type=float, default=0.02)
    parser.add_argument("--max-control-gap-m", type=float, default=0.01)
    return parser.parse_args()


def rank_fraction(values: np.ndarray) -> np.ndarray:
    """Stable fractional rank where 0 is the smallest (best) value."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return ranks / max(len(values) - 1, 1)


def read_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_population(
    snapshot: int,
    source: str,
    rows: list[dict[str, Any]],
    proxy_good_top_frac: float,
    min_physical_gap_m: float,
    max_control_gap_m: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a population-level summary and its strongest verified pair.

    A strong inversion is a pair (i, j) from the same CEM population for which
    the proxy strictly prefers i, physical execution prefers j by at least the
    configured gap, and i is in the proxy's best decile.  The score is a
    scale-free proxy-rank margin times the physical margin in metres.
    """
    candidate = np.asarray([int(row["candidate"]) for row in rows], dtype=int)
    proxy = np.asarray([float(row["learned_proxy_cost"]) for row in rows])
    physical = np.asarray([float(row["physical_distance_m"]) for row in rows])
    success = np.asarray([int(row["success"]) for row in rows], dtype=int)
    proxy_rank = rank_fraction(proxy)
    regret = physical - float(physical.min())
    proxy_good = proxy_rank <= proxy_good_top_frac
    physical_bad = regret >= min_physical_gap_m

    # The 2x2 is exhaustive: "physical good" means below the 2 cm hard-case
    # threshold used by Phase 0; all other candidates are physical-hard.
    cell_counts = {
        "proxy_good_physical_good": int(np.sum(proxy_good & ~physical_bad)),
        "proxy_good_physical_bad": int(np.sum(proxy_good & physical_bad)),
        "proxy_bad_physical_good": int(np.sum(~proxy_good & ~physical_bad)),
        "proxy_bad_physical_bad": int(np.sum(~proxy_good & physical_bad)),
    }

    best_pair: dict[str, Any] | None = None
    n_strong_pairs = 0
    for i in np.flatnonzero(proxy_good & physical_bad):
        # lower cost is better, so a positive proxy margin means the planner
        # prefers i; a positive physical margin means j is physically better.
        proxy_margin_rank = proxy_rank - proxy_rank[i]
        physical_margin_m = physical[i] - physical
        valid = (proxy_margin_rank > 0.0) & (physical_margin_m >= min_physical_gap_m)
        n_strong_pairs += int(np.sum(valid))
        if not np.any(valid):
            continue
        score = proxy_margin_rank * physical_margin_m
        j = int(np.argmax(np.where(valid, score, -np.inf)))
        pair = {
            "snapshot": snapshot,
            "source": source,
            "deceptive_candidate": int(candidate[i]),
            "corrective_candidate": int(candidate[j]),
            "deceptive_proxy_cost": float(proxy[i]),
            "corrective_proxy_cost": float(proxy[j]),
            "deceptive_proxy_rank_fraction": float(proxy_rank[i]),
            "corrective_proxy_rank_fraction": float(proxy_rank[j]),
            "proxy_rank_margin": float(proxy_margin_rank[j]),
            "deceptive_physical_distance_m": float(physical[i]),
            "corrective_physical_distance_m": float(physical[j]),
            "deceptive_regret_m": float(regret[i]),
            "corrective_regret_m": float(regret[j]),
            "physical_margin_m": float(physical_margin_m[j]),
            "inversion_score_rank_m": float(score[j]),
            "deceptive_success": int(success[i]),
            "corrective_success": int(success[j]),
        }
        if best_pair is None or pair["inversion_score_rank_m"] > best_pair["inversion_score_rank_m"]:
            best_pair = pair

    matched_control: dict[str, Any] | None = None
    if best_pair is not None:
        i = int(np.flatnonzero(candidate == best_pair["deceptive_candidate"])[0])
        hard_non_deceptive = np.flatnonzero(~proxy_good & physical_bad)
        if len(hard_non_deceptive):
            control = int(hard_non_deceptive[np.argmin(np.abs(regret[hard_non_deceptive] - regret[i]))])
            control_gap = float(abs(regret[control] - regret[i]))
            matched_control = {
                "candidate": int(candidate[control]),
                "proxy_rank_fraction": float(proxy_rank[control]),
                "physical_distance_m": float(physical[control]),
                "physical_regret_m": float(regret[control]),
                "absolute_regret_gap_m": control_gap,
                "within_tolerance": bool(control_gap <= max_control_gap_m),
            }
            best_pair["matched_proxy_rejected_hard_control"] = matched_control

    summary: dict[str, Any] = {
        "snapshot": snapshot,
        "source": source,
        "n_candidates": int(len(rows)),
        "pool_best_physical_distance_m": float(physical.min()),
        "pool_success_available": int(success.any()),
        "cell_counts": cell_counts,
        "n_proxy_good_physical_bad": cell_counts["proxy_good_physical_bad"],
        "n_strong_inversion_pairs": n_strong_pairs,
        "has_strong_inversion": bool(best_pair is not None),
        "has_matched_proxy_rejected_hard_control": bool(
            matched_control is not None and matched_control["within_tolerance"]
        ),
    }
    if best_pair is not None:
        summary.update({
            "strongest_inversion_score_rank_m": best_pair["inversion_score_rank_m"],
            "strongest_inversion_physical_margin_m": best_pair["physical_margin_m"],
            "strongest_inversion_proxy_rank_margin": best_pair["proxy_rank_margin"],
            "strongest_inversion_deceptive_regret_m": best_pair["deceptive_regret_m"],
        })
        if matched_control is not None:
            summary["matched_control_regret_gap_m"] = matched_control["absolute_regret_gap_m"]
    return summary, best_pair


def main() -> None:
    args = parse_args()
    if not 0.0 < args.proxy_good_top_frac < 1.0:
        raise ValueError("--proxy-good-top-frac must lie in (0, 1)")
    if args.min_physical_gap_m <= 0.0 or args.max_control_gap_m <= 0.0:
        raise ValueError("physical thresholds must be positive")

    candidate_paths = sorted(args.shards.glob("*/candidates.csv"), key=lambda p: int(p.parent.name))
    if len(candidate_paths) != args.expected_shards:
        raise RuntimeError(
            f"expected {args.expected_shards} candidate tables, found {len(candidate_paths)}"
        )

    summaries: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for path in candidate_paths:
        snapshot = int(path.parent.name)
        rows = read_candidates(path)
        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_source.setdefault(row["source"], []).append(row)
        required = {"cem_initial", "cem_final"}
        if set(by_source) != required:
            raise RuntimeError(f"{path} has sources {sorted(by_source)}, expected {sorted(required)}")
        for source in sorted(by_source):
            summary, pair = analyze_population(
                snapshot=snapshot,
                source=source,
                rows=by_source[source],
                proxy_good_top_frac=args.proxy_good_top_frac,
                min_physical_gap_m=args.min_physical_gap_m,
                max_control_gap_m=args.max_control_gap_m,
            )
            summaries.append(summary)
            if pair is not None:
                pairs.append(pair)

    final_rows = [row for row in summaries if row["source"] == "cem_final"]
    final_pairs = [row for row in pairs if row["source"] == "cem_final"]
    matched_final = [
        row for row in final_rows if row["has_matched_proxy_rejected_hard_control"]
    ]
    report = {
        "scope": (
            "post-hoc verification on persisted Phase-0 same-state candidate pools; "
            "not policy or query-efficiency evidence"
        ),
        "definition": {
            "proxy_good": f"proxy rank <= {args.proxy_good_top_frac}",
            "physical_hard": f"physical regret >= {args.min_physical_gap_m} m",
            "strong_inversion": (
                "same-population pair where proxy prefers deceptive candidate, "
                "but corrective candidate is physically better by at least the physical gap"
            ),
            "score": "proxy-rank margin times physical margin in metres",
            "matched_control": (
                "same-population proxy-rejected/physical-hard candidate matched on regret; "
                "this is a control, not a claim that it participates in no inversion whatsoever"
            ),
        },
        "thresholds": {
            "proxy_good_top_frac": args.proxy_good_top_frac,
            "min_physical_gap_m": args.min_physical_gap_m,
            "max_control_gap_m": args.max_control_gap_m,
        },
        "n_snapshots": args.expected_shards,
        "n_population_rows": len(summaries),
        "final_population": {
            "n_snapshots_with_strong_inversion": len(final_pairs),
            "strong_inversion_coverage": len(final_pairs) / len(final_rows),
            "n_with_matched_proxy_rejected_hard_control": len(matched_final),
            "verified_inversion_gate": "GO" if len(final_pairs) >= 8 and len(matched_final) >= 8 else "NO_GO",
        },
        "initial_population": {
            "n_snapshots_with_strong_inversion": sum(row["source"] == "cem_initial" for row in pairs),
            "strong_inversion_coverage": sum(row["source"] == "cem_initial" for row in pairs) / args.expected_shards,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "population_summary.csv").open("w", newline="") as handle:
        fields = sorted({key for row in summaries for key in row if key != "cell_counts"})
        fields += [
            "proxy_good_physical_good", "proxy_good_physical_bad",
            "proxy_bad_physical_good", "proxy_bad_physical_bad",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            flat = {key: value for key, value in row.items() if key != "cell_counts"}
            flat.update(row["cell_counts"])
            writer.writerow(flat)
    with (args.out_dir / "strongest_inversion_pairs.jsonl").open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, sort_keys=True) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    final = report["final_population"]
    report_lines = [
        "# Phase 0b — verified same-population proxy inversions",
        "",
        f"- **Gate:** {final['verified_inversion_gate']}",
        f"- Final CEM populations with a strong inversion: {final['n_snapshots_with_strong_inversion']}/{args.expected_shards}",
        f"- Final CEM populations with a matched proxy-rejected hard control: {final['n_with_matched_proxy_rejected_hard_control']}/{args.expected_shards}",
        f"- Initial-population inversion coverage: {report['initial_population']['n_snapshots_with_strong_inversion']}/{args.expected_shards}",
        "",
        "Each reported inversion compares two candidates from the identical restored state and the same CEM population.",
        "This establishes the presence of proxy/physics ordering reversals, not that mining them improves policy learning.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
