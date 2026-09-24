"""S1 diagnostic (CPU): is sibling identity recoverable from goal16 at all? Rules a training bug in or out.

Pixel nearest neighbour: for each held-out decision and each sibling k, predict argmin_j ||goal16_k - end_j||^2.
Also: how often are sibling end frames / goal frames pixel-identical, and within-bank Spearman(cov8, cov24).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

assert os.environ.get("SLURM_JOB_ID"), "Run through sbatch"
run, out = Path(sys.argv[1]), Path(sys.argv[2])
shards = sorted(p for p in run.glob("shard_*.npz") if 1500 <= int(p.stem.split("_")[1]) <= 1599)
d = {k: np.concatenate([np.load(p)[k] for p in shards]) for k in ("end", "goal16", "phys8", "cov8", "cov24", "ctx")}
n, K = d["cov8"].shape
end = d["end"].reshape(n, K, -1).astype(np.float32)
goal = d["goal16"].reshape(n, K, -1).astype(np.float32)
ctx = d["ctx"].reshape(n, 1, -1).astype(np.float32)
phys = d["phys8"]
hits = hits_delta = total = 0
distinct_end, distinct_goal, end_eq_ctx = [], [], []
rho = []
for i in range(n):
    same = np.abs(phys[i][:, None] - phys[i][None]).max(-1) < 1e-3
    distinct_end.append(len({e.tobytes() for e in d["end"][i]}))
    distinct_goal.append(len({g.tobytes() for g in d["goal16"][i]}))
    end_eq_ctx.append(float(np.mean([np.array_equal(d["end"][i][k], d["ctx"][i]) for k in range(K)])))
    if same.all():
        continue
    dist = ((goal[i][:, None] - end[i][None]) ** 2).mean(-1)          # (k goal, j end)
    ddist = (((goal[i] - ctx[i])[:, None] - (end[i] - ctx[i])[None]) ** 2).mean(-1)
    for k in range(K):
        hits += bool(same[k, dist[k].argmin()])
        hits_delta += bool(same[k, ddist[k].argmin()])
        total += 1
    if np.ptp(d["cov8"][i]) > 0 and np.ptp(d["cov24"][i]) > 0:
        rho.append(spearmanr(d["cov8"][i], d["cov24"][i]).statistic)
report = {"decisions": int(n), "pixel_nn_top1": hits / total, "pixel_nn_delta_top1": hits_delta / total,
          "chance": 1 / K, "n_queries": total,
          "median_distinct_end_frames": float(np.median(distinct_end)),
          "median_distinct_goal_frames": float(np.median(distinct_goal)),
          "mean_frac_end_equal_ctx": float(np.mean(end_eq_ctx)),
          "within_bank_spearman_cov8_cov24": float(np.mean(rho)), "n_rho": len(rho)}
out.mkdir(parents=True, exist_ok=True)
(out / "signal_check.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
