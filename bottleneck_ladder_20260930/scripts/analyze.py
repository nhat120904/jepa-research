#!/usr/bin/env python3
"""Summarise the oracle ladder (CPU)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def boot(x, draws=10_000, seed=0):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("bulk analysis runs under sbatch")
    report = {}
    for task in ("cube", "reacher"):
        rows = [json.loads(f.read_text()) for f in sorted((a.run_dir / task).glob("root_*.json"))]
        if not rows:
            continue
        hist = [r for r in rows if r["has_history"]]
        informative = [r for r in hist if 0 < r["top1_random"] < 1]
        rep = {"roots": len(rows), "with_history": len(hist),
               "informative_roots(mixed candidates)": len(informative)}
        for label, sub in (("all_with_history", hist), ("informative", informative)):
            d = {}
            for lvl in ("h1", "h3", "true"):
                d[f"spearman_{lvl}"] = boot([r[f"rho_{lvl}"] for r in sub])
                d[f"top1_success_{lvl}"] = boot([r[f"top1_{lvl}"] for r in sub])
            d["top1_success_random"] = boot([r["top1_random"] for r in sub])
            d["oracle_any"] = boot([r["oracle_any"] for r in sub])
            d["top1_h3_minus_h1"] = boot([r["top1_h3"] - r["top1_h1"] for r in sub])
            d["top1_true_minus_h1"] = boot([r["top1_true"] - r["top1_h1"] for r in sub])
            d["pred_error_median_h1"] = float(np.median([r["err_h1_median"] for r in sub]))
            d["pred_error_median_h3"] = float(np.median([r["err_h3_median"] for r in sub]))
            rep[label] = d
        h1 = np.array([np.mean(r["closed_h1"]) for r in hist])
        h3 = np.array([np.mean(r["closed_h3"]) for r in hist])
        rep["closed_loop"] = {"h1": boot(h1), "h3": boot(h3), "h3_minus_h1": boot(h3 - h1),
                              "per_seed_h1": np.mean([r["closed_h1"] for r in hist], 0).round(3).tolist(),
                              "per_seed_h3": np.mean([r["closed_h3"] for r in hist], 0).round(3).tolist()}
        rep["replay_qpos_err_median"] = float(np.median([r["replay_qpos_err"] for r in hist]))
        report[task] = rep
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
