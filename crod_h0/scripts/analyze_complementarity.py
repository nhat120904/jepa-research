#!/usr/bin/env python3
"""Cluster-bootstrap the 32-state fully-labelled complementarity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def ratio(counts: np.ndarray, numerator: str, denominator: str) -> float:
    fields = {
        "informative": 0,
        "native_wrong": 1,
        "auxiliary_wrong": 2,
        "both_wrong": 3,
        "disagreement": 4,
        "native_wrong_and_disagreement": 5,
        "native_rejected": 6,
        "corrective": 7,
        "directional_support": 8,
        "corrective_in_directional_support": 9,
    }
    den = counts[:, fields[denominator]].sum()
    return float(counts[:, fields[numerator]].sum() / den) if den else float("nan")


def main() -> None:
    args = parse_args()
    paths = sorted(args.shards.glob("*/summary.json"), key=lambda p: int(p.parent.name))
    if len(paths) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} shards, found {len(paths)}")
    names = (
        "informative",
        "native_wrong",
        "auxiliary_wrong",
        "both_wrong",
        "disagreement",
        "native_wrong_and_disagreement",
        "native_rejected",
        "corrective",
        "directional_support",
        "corrective_in_directional_support",
    )
    records = [json.loads(path.read_text()) for path in paths]
    counts = np.asarray([[record["counts"][name] for name in names] for record in records])
    definitions = {
        "p_native_wrong": ("native_wrong", "informative"),
        "p_auxiliary_wrong": ("auxiliary_wrong", "informative"),
        "p_both_wrong": ("both_wrong", "informative"),
        "p_native_wrong_given_disagreement": (
            "native_wrong_and_disagreement",
            "disagreement",
        ),
        "p_corrective_given_native_rejected": ("corrective", "native_rejected"),
        "p_corrective_given_directional_support": (
            "corrective_in_directional_support",
            "directional_support",
        ),
    }
    point = {name: ratio(counts, *terms) for name, terms in definitions.items()}
    rng = np.random.default_rng(args.seed + 771)
    boot = {name: [] for name in definitions}
    for _ in range(args.bootstrap):
        sampled = counts[rng.integers(0, len(counts), size=len(counts))]
        for name, terms in definitions.items():
            value = ratio(sampled, *terms)
            if np.isfinite(value):
                boot[name].append(value)
    estimates = {
        name: {
            "estimate": point[name],
            "ci_low": float(np.quantile(boot[name], 0.025)),
            "ci_high": float(np.quantile(boot[name], 0.975)),
        }
        for name in definitions
    }
    enrichment_samples = np.asarray(boot["p_corrective_given_directional_support"])
    base_samples = np.asarray(boot["p_corrective_given_native_rejected"])
    paired_n = min(len(enrichment_samples), len(base_samples))
    enrichment_gain = enrichment_samples[:paired_n] - base_samples[:paired_n]
    report = {
        "scope": (
            "Candidate-level pairwise errors with snapshot-clustered bootstrap on "
            "the 32 fully physics-labelled Phase-0d final populations."
        ),
        "n_snapshots": args.expected_shards,
        "total_informative_pairs": int(counts[:, 0].sum()),
        "estimates": estimates,
        "directional_enrichment_gain": {
            "estimate": float(
                point["p_corrective_given_directional_support"]
                - point["p_corrective_given_native_rejected"]
            ),
            "ci_low": float(np.quantile(enrichment_gain, 0.025)),
            "ci_high": float(np.quantile(enrichment_gain, 0.975)),
        },
        "mechanism_supported": bool(np.quantile(enrichment_gain, 0.025) > 0),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    gain = report["directional_enrichment_gain"]
    lines = [
        "# CROD complementarity audit",
        "",
        f"- Mechanism supported: **{report['mechanism_supported']}**",
        (
            "- Directional rescue enrichment over all native-rejected candidates: "
            f"{gain['estimate'] * 100:+.1f} pp "
            f"[{gain['ci_low'] * 100:+.1f}, {gain['ci_high'] * 100:+.1f}]"
        ),
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
