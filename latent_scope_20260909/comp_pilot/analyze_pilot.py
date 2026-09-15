"""Sanity baselines for a finished pilot run: constant predictor and prediction-target correlation (CPU, sbatch only)."""

import json
import os
import sys
from pathlib import Path

import numpy as np

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("run inside sbatch")
run = Path(sys.argv[1])
rows = []
for arm in ("segment_nocomp", "segment_comp", "frame_rollout", "map_union"):
    rows += json.loads((run / arm / "test_rows.json").read_text())
out = {}
groups = {}
for r in rows:
    strata = ["contact" if r["has_contact"] else "no_contact"]
    if r["hard"] is not None and r["has_contact"]:
        strata.append("hard" if r["hard"] else "easy_contact")
    for s in strata:
        groups.setdefault((r["eval"], s), {}).setdefault((r["arm"], r["method"]), []).append(r)
for (ev, st), by in sorted(groups.items()):
    any_rows = next(iter(by.values()))
    for metric in ("count_empty", "count_increment"):
        true = np.array([r[f"true_{metric}"] for r in any_rows])
        entry = {
            "windows": len(true),
            "true_mean": float(true.mean()),
            "true_std": float(true.std()),
            "const_mean_mae_optimistic": float(np.abs(true - true.mean()).mean()),
            "const_median_mae_optimistic": float(np.abs(true - np.median(true)).mean()),
            "arms": {},
        }
        for (arm, method), rs in sorted(by.items()):
            t = np.array([r[f"true_{metric}"] for r in rs])
            p = np.array([r[f"pred_{metric}"] for r in rs])
            corr = float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else None
            entry["arms"][f"{arm}:{method}"] = {"mae": float(np.abs(t - p).mean()), "pearson_r": corr, "pred_mean": float(p.mean())}
        out[f"{ev}|{st}|{metric}"] = entry
(run / "sanity_baselines.json").write_text(json.dumps(out, indent=2))
for key, e in out.items():
    if "count_empty" in key and ("contact" in key or "hard" in key):
        print(f"{key:40s} n={e['windows']:5d} mean={e['true_mean']:.2f} sd={e['true_std']:.2f} const_mae={e['const_mean_mae_optimistic']:.3f}")
        for a, v in e["arms"].items():
            print(f"    {a:32s} mae={v['mae']:.3f} r={v['pearson_r'] if v['pearson_r'] is None else round(v['pearson_r'], 3)} pred_mean={v['pred_mean']:.2f}")
