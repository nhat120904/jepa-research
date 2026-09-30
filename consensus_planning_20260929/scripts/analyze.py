#!/usr/bin/env python3
"""Analyse the consensus-planning probe (development roots only)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

M = 10
AGG = ("medoid3", "avg3", "bestcost3", "latmedoid3", "medoid30")


def boot(x, draws=10_000, seed=0):
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def analyse(rows: list[dict], npz: list[dict]) -> dict:
    s30 = np.array([[r["arms"][f"single30_{s}"]["success"] for s in range(M)] for r in rows], float)
    s3 = np.array([[r["arms"][f"single3_{s}"]["success"] for s in range(M)] for r in rows], float)
    agg = {k: np.array([r["arms"][k]["success"] for r in rows], float) for k in AGG}
    p30, p3 = s30.mean(1), s3.mean(1)
    rng = np.random.default_rng(1)
    # Success of a standard 50-episode evaluation under one CEM seed, over seeds and root draws.
    fifty = [s30[rng.choice(len(rows), 50, replace=False), rng.integers(0, M)].mean() for _ in range(5000)]
    fifty_fixed_roots = [s30[:50, s].mean() for s in range(M)]
    out = {
        "roots": len(rows),
        "single30_expected": boot(p30), "single3_expected": boot(p3),
        "single30_per_seed": s30.mean(0).round(3).tolist(),
        "single3_per_seed": s3.mean(0).round(3).tolist(),
        "seed_spread_single30": {"min": float(s30.mean(0).min()), "max": float(s30.mean(0).max()),
                                 "sd": float(s30.mean(0).std(ddof=1))},
        "fifty_episode_eval_sd_roots_and_seed": float(np.std(fifty)),
        "fifty_fixed_roots_seed_range": [float(min(fifty_fixed_roots)), float(max(fifty_fixed_roots))],
        "roots_seed_mixed_single30": int(np.sum((p30 > 0) & (p30 < 1))),
        "roots_always_succeed_single30": int(np.sum(p30 == 1)),
        "roots_never_succeed_single30": int(np.sum(p30 == 0)),
        "oracle_any_seed_single30": float((s30.max(1)).mean()),
        "aggregates": {},
    }
    for k, v in agg.items():
        out["aggregates"][k] = {"success": boot(v), "vs_single30_expected": boot(v - p30),
                                "vs_single3_expected": boot(v - p3),
                                "vs_single30_seed0": boot(v - s30[:, 0])}
    # Plan-1 open loop: aggregate distance versus the seed-average distance.
    ol3 = np.stack([z["ol3_dist"] for z in npz])
    ol30 = np.stack([z["ol30_dist"] for z in npz])
    olagg = {k: np.array([r["plan1_open_loop_aggregates"][k][0] for r in rows]) for k in AGG}
    out["plan1_open_loop_distance"] = {
        "single3_mean": float(ol3.mean()), "single30_mean": float(ol30.mean()),
        "single3_oracle_min": float(ol3.min(1).mean()),
        **{k: float(v.mean()) for k, v in olagg.items()},
        "medoid3_minus_single3_mean": boot(olagg["medoid3"] - ol3.mean(1)),
        "latmedoid3_minus_single3_mean": boot(olagg["latmedoid3"] - ol3.mean(1)),
        "bestcost3_minus_single3_mean": boot(olagg["bestcost3"] - ol3.mean(1)),
        "avg3_minus_single3_mean": boot(olagg["avg3"] - ol3.mean(1)),
    }
    # Does the model's cost rank the M plan-1 candidates by open-loop distance?
    rho = []
    for z in npz:
        c, d = z["cost3"], z["ol3_dist"]
        if np.ptp(d) > 1e-6:
            rc, rd = np.argsort(np.argsort(c)), np.argsort(np.argsort(d))
            rho.append(np.corrcoef(rc, rd)[0, 1])
    out["plan1_cost_vs_distance_spearman_over_seeds"] = {"mean": float(np.mean(rho)), "roots": len(rho)}
    out["seconds_per_root_mean"] = float(np.mean([r["seconds"] for r in rows]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("bulk analysis runs under sbatch")
    report = {}
    for task in ("cube", "reacher"):
        files = sorted((args.run_dir / task).glob("root_*.json"))
        if not files:
            continue
        rows = [json.loads(f.read_text()) for f in files]
        npz = [dict(np.load(f.with_suffix(".npz"))) for f in files]
        report[task] = analyse(rows, npz)
        report[task]["complete_0_99"] = sorted(r["root"] for r in rows) == list(range(100))
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for t, r in report.items():
        print(t, json.dumps({"s30": r["single30_expected"]["mean"], "s3": r["single3_expected"]["mean"],
                             "seed_spread": r["seed_spread_single30"],
                             **{k: round(v["success"]["mean"], 3) for k, v in r["aggregates"].items()}}))


if __name__ == "__main__":
    main()
