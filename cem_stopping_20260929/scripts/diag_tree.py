#!/usr/bin/env python3
"""Follow-up diagnostics on recorded dev trees (CPU only).

1. Outcome stability across K: is per-root success a smooth function of the
   iteration count, or does it flip between neighbouring checkpoints?
2. Can the planner's own cost rank its iterates? Within each plan-1, compare
   the proxy cost of the checkpoint means with their executed plan-1 distance.
3. A mean-shift stopping rule (elite-mean displacement vs its sampling error),
   tuned with the same 5-fold CV as the other rules.
4. Random-checkpoint baseline (expected success of a uniformly random k).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from analyze import cross_validated, rate  # noqa: E402
from cemstop.evaluate import load_task, paired_bootstrap  # noqa: E402
from cemstop.rules import CHECKPOINTS, TOPK, choose  # noqa: E402


@dataclass(frozen=True)
class Shift:
    """Stop when the elite mean moved less than c standard errors since the last checkpoint.

    Needs the checkpoint means and elite std, so it is evaluated with a custom
    path function rather than ``choose`` on costs alone.
    """
    c: float
    name: str = "shift"

    def param(self) -> float:
        return self.c


def shift_choice(rule: Shift, means: np.ndarray, std: np.ndarray) -> int:
    prev = None
    for i, k in enumerate(CHECKPOINTS):
        if k == CHECKPOINTS[-1]:
            return i
        if prev is not None:
            move = float(np.sqrt(np.mean((means[i] - means[prev]) ** 2)))
            se = float(std[k - 1]) / np.sqrt(TOPK)
            if move < rule.c * se:
                return i
        prev = i
    raise AssertionError


def shift_outcomes(rule: Shift, roots) -> dict:
    succ, iters = [], []
    for tr in roots:
        e = tr.extra
        i1 = shift_choice(rule, e["means1"], e["std1"])
        if tr.p1_term[i1]:
            succ.append(tr.leaf_success[i1, 0])
            iters.append(CHECKPOINTS[i1])
            continue
        i2 = shift_choice(rule, e["means2"][i1], e["std2"][i1])
        succ.append(tr.leaf_success[i1, i2])
        iters.append(CHECKPOINTS[i1] + CHECKPOINTS[i2])
    return {"success": np.array(succ, bool), "iterations": np.array(iters)}


def spearman(a, b) -> float:
    a = np.argsort(np.argsort(a))
    b = np.argsort(np.argsort(b))
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("bulk analysis runs under sbatch")
    report = {}
    for name in ("cube", "reacher"):
        roots = [tr for tr in load_task(args.run_dir / name) if tr.root < 100]
        diag = np.stack([np.diag(tr.leaf_success) for tr in roots]).astype(float)
        mixed = ~(diag.all(1) | (~diag.astype(bool)).all(1))
        flips = np.abs(np.diff(diag, axis=1))
        # Success correlation between checkpoints across roots.
        corr = np.corrcoef(diag.T)
        res = {
            "roots": len(roots),
            "diagonal_mixed_roots": int(mixed.sum()),
            "adjacent_flip_rate_mixed": float(flips[mixed].mean()) if mixed.any() else None,
            "success_corr_k10_k30": float(corr[CHECKPOINTS.index(10), CHECKPOINTS.index(30)]),
            "success_corr_k15_k20": float(corr[CHECKPOINTS.index(15), CHECKPOINTS.index(20)]),
            "random_checkpoint_expected_success": float(np.mean(
                [tr.leaf_success.mean() for tr in roots])),
        }
        # Proxy cost of checkpoint mean k (k < 30) is candidate 0 of iteration k + 1.
        rhos, picks = [], []
        idx = [i for i, k in enumerate(CHECKPOINTS) if k < 30]
        for tr in roots:
            proxy = np.array([tr.costs1[CHECKPOINTS[i], 0] for i in idx], np.float64)
            dist = tr.extra["p1_dist"][idx]
            if np.ptp(dist) > 1e-6:
                rhos.append(spearman(proxy, dist))
            picks.append(idx[int(np.argmin(proxy))])
        res["iterate_proxy_vs_plan1_distance_spearman"] = {
            "mean": float(np.nanmean(rhos)), "roots": int(np.sum(np.isfinite(rhos)))}
        res["proxy_argmin_checkpoint_histogram"] = {
            int(CHECKPOINTS[i]): int(np.sum(np.array(picks) == i)) for i in idx}
        # Mean-shift rule, 5-fold CV with the same fold assignment as analyze.py.
        grid = [Shift(c) for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
        table = []
        for rule in grid:
            o = shift_outcomes(rule, roots)
            table.append({"c": rule.c, "success": float(o["success"].mean()),
                          "iterations": float(o["iterations"].mean())})
        rng = np.random.default_rng(0)
        assign = rng.permutation(len(roots)) % 5
        cv_s = np.zeros(len(roots), bool)
        cv_i = np.zeros(len(roots))
        chosen = []
        for f in range(5):
            tr_idx = np.where(assign != f)[0]
            te_idx = np.where(assign == f)[0]
            best = max(grid, key=lambda r: (
                shift_outcomes(r, [roots[i] for i in tr_idx])["success"].mean(),
                -shift_outcomes(r, [roots[i] for i in tr_idx])["iterations"].mean()))
            chosen.append(best.c)
            o = shift_outcomes(best, [roots[i] for i in te_idx])
            cv_s[te_idx] = o["success"]
            cv_i[te_idx] = o["iterations"]
        fixed_cv = cross_validated("fixed", roots, 5)
        res["shift_rule"] = {"in_sample": table, "cv_success": rate(cv_s),
                             "cv_iterations": float(cv_i.mean()), "chosen": chosen,
                             "vs_fixed_cv": paired_bootstrap(cv_s, fixed_cv["success"])}
        report[name] = res
    args.out.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
