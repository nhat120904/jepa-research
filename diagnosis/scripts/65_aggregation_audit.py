"""Cost-aggregation audit on already-mined CEM populations (K=2, offline).

Question: on the populations CEM actually produced, does *aggregating* two
costs select better than the better single cost?  The two members are the
planner's own latent-L2 (``proxy_cost``) and the state-probe cost
(``decoded_stateprobe_cost``), both recorded per candidate by scripts/51, so
the comparison runs on one identical population per row — no re-mining.

Scope, stated up front because it bounds every number below:

* **K = 2.**  A median over voters is undefined, so this cannot test
  median-of-ranks / rank aggregation with three or more heterogeneous members.
  It tests Borda and value aggregation of *these two* costs only.
* **One-shot selection**, not closed-loop success.  It measures the pick made
  inside a population, not the task outcome.
* Both members read the same frozen encoder, so they are not independent
  voters in the sense robust-aggregation theory assumes.

The script also reports the residual statistic ``r_k = z(c_k) - z(c*)`` against
a conditional-independence null.  That statistic was used in an earlier
analysis to argue that optimisation pressure couples the two costs; the null
shows the effect is almost entirely mechanical (both residuals contain
``-z(c*)``), so the residual correlation is reported here only to document that
it is **not** usable evidence.  Direct selection regret is the load-bearing
measure.

    .venv/bin/python scripts/65_aggregation_audit.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CELLS = ["dino_push", "dino_pick", "jepa_push", "jepa_pick"]


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so Borda does not depend on input order."""
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    sorted_x = x[order]
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or sorted_x[i] != sorted_x[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def load_populations(results: Path, cell: str, min_size: int):
    path = results / f"cem_preselection_{cell}_l2_candidates.csv.gz"
    raw = collections.defaultdict(list)
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            raw[(row["seed"], row["replan"], int(row["iter"]))].append(row)
    populations = []
    for (seed, replan, iteration), rows in raw.items():
        rows.sort(key=lambda r: int(r["candidate"]))
        truth = np.array([float(r["true_shaped_cost"]) for r in rows])
        if len(rows) < min_size or truth.std() < 1e-9:
            continue
        spread = truth.mean() - truth.min()
        if spread < 1e-9:
            continue
        populations.append({
            "seed": seed, "iter": iteration,
            "c1": np.array([float(r["proxy_cost"]) for r in rows]),
            "c2": np.array([float(r["decoded_stateprobe_cost"]) for r in rows]),
            "truth": truth, "spread": spread,
        })
    return populations


def selections(c1: np.ndarray, c2: np.ndarray):
    z1, z2 = zscore(c1), zscore(c2)
    r1, r2 = average_ranks(c1), average_ranks(c2)
    return {
        "l2": int(np.argmin(c1)),
        "stateprobe": int(np.argmin(c2)),
        "mean_z": int(np.argmin(0.5 * (z1 + z2))),
        "max_z (veto)": int(np.argmin(np.maximum(z1, z2))),
        "min_z": int(np.argmin(np.minimum(z1, z2))),
        "borda": int(np.argmin(0.5 * (r1 + r2))),
        "rank_max": int(np.argmin(np.maximum(r1, r2))),
    }


def residual_null(c1, c2, truth, draws, rng):
    z1, z2, zt = zscore(c1), zscore(c2), zscore(truth)
    rho2 = float(np.corrcoef(z2, zt)[0, 1])
    observed = float(np.corrcoef(z1 - zt, z2 - zt)[0, 1])
    nulls = []
    for _ in range(draws):
        noise = zscore(rng.standard_normal(len(zt)))
        surrogate = zscore(rho2 * zt + np.sqrt(max(0.0, 1 - rho2 ** 2)) * noise)
        nulls.append(float(np.corrcoef(z1 - zt, surrogate - zt)[0, 1]))
    return observed, float(np.mean(nulls))


def cluster_ci(by_seed, boots, rng):
    keys = sorted(by_seed)
    values = np.array([np.mean(by_seed[k]) for k in keys], float)
    draws = [rng.choice(values, len(values)).mean() for _ in range(boots)]
    return values.mean(), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--min-population", type=int, default=20)
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--boots", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/aggregation_audit.md")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    results = ROOT / args.results
    lines = ["# Cost-aggregation audit on mined CEM populations (K=2)", "",
             "Scope: K=2 so median-of-ranks is undefined; one-shot selection, "
             "not closed-loop success; both members read the same frozen "
             "encoder. Normalised regret 1.00 = expected regret of a uniformly "
             "random candidate in the same population, 0.00 = oracle. "
             "CIs are a cluster bootstrap over episode seeds.", ""]

    for cell in args.cells:
        populations = load_populations(results, cell, args.min_population)
        if not populations:
            lines += [f"## {cell}", "", "no usable population", ""]
            continue
        iters = sorted({p["iter"] for p in populations})
        lines += [f"## {cell}  ({len(populations)} populations, "
                  f"{len({p['seed'] for p in populations})} episode seeds)", "",
                  "| iter | arm | normalised regret [95% CI] |", "|---|---|---|"]
        for iteration in (iters[0], iters[-1]):
            subset = [p for p in populations if p["iter"] == iteration]
            per_arm = collections.defaultdict(lambda: collections.defaultdict(list))
            for pop in subset:
                for arm, pick in selections(pop["c1"], pop["c2"]).items():
                    regret = (pop["truth"][pick] - pop["truth"].min()) / pop["spread"]
                    per_arm[arm][pop["seed"]].append(regret)
            for arm in ["l2", "stateprobe", "mean_z", "max_z (veto)", "min_z",
                        "borda", "rank_max"]:
                mean, lo, hi = cluster_ci(per_arm[arm], args.boots, rng)
                lines.append(f"| {iteration} | {arm} | {mean:.3f} [{lo:.3f}, {hi:.3f}] |")
        lines += ["",
                  "Residual statistic (reported as a negative control, not as "
                  "evidence — see module docstring):", "",
                  "| iter | observed corr(r1,r2) | conditional-independence null | excess |",
                  "|---|---|---|---|"]
        for iteration in (iters[0], iters[-1]):
            subset = [p for p in populations if p["iter"] == iteration]
            pairs = [residual_null(p["c1"], p["c2"], p["truth"], args.null_draws, rng)
                     for p in subset]
            observed = float(np.mean([a for a, _ in pairs]))
            null = float(np.mean([b for _, b in pairs]))
            lines.append(f"| {iteration} | {observed:+.3f} | {null:+.3f} | "
                         f"{observed - null:+.3f} |")
        lines.append("")
        print(f"  {cell}: {len(populations)} populations")

    (ROOT / args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
