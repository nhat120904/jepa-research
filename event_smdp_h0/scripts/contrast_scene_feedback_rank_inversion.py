#!/usr/bin/env python3
"""Exploratory: does oracle-state ranking of feedbacks predict learned-state ranking?"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("must run inside a Slurm compute job")
    summary = json.loads(args.summary.read_text())
    feedbacks = sorted({row["feedback"] for row in summary["rates"].values()})
    oracle = np.array(
        [summary["rates"][f"oracle__{f}"]["mean_success"] for f in feedbacks]
    )
    learned = np.array(
        [summary["rates"][f"obs_history_full__{f}"]["mean_success"] for f in feedbacks]
    )
    ro, rl = rank(oracle), rank(learned)
    spearman = float(np.corrcoef(ro, rl)[0, 1])
    pearson = float(np.corrcoef(oracle, learned)[0, 1])
    best_oracle = feedbacks[int(oracle.argmax())]
    out = {
        "protocol": "scene_feedback_rank_inversion_v1",
        "scope": "exploratory, not preregistered; n=8 feedbacks",
        "feedbacks": feedbacks,
        "oracle_success": {f: float(v) for f, v in zip(feedbacks, oracle)},
        "learned_success": {f: float(v) for f, v in zip(feedbacks, learned)},
        "spearman_rank_correlation": spearman,
        "pearson_correlation": pearson,
        "best_under_oracle": {
            "feedback": best_oracle,
            "oracle_success": float(oracle.max()),
            "learned_success": float(learned[int(oracle.argmax())]),
            "learned_rank_from_worst": int(rl[int(oracle.argmax())]),
        },
        "best_under_learned": {
            "feedback": feedbacks[int(learned.argmax())],
            "learned_success": float(learned.max()),
            "oracle_success": float(oracle[int(learned.argmax())]),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rank_inversion.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
