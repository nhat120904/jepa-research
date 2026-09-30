#!/usr/bin/env python3
"""Analyse recorded stopping trees.

Development mode (default) reads roots 0-99 only: fixed-K curves, oracle,
rule tuning with 5-fold cross-validation inside dev (an honest estimate before
the test split is opened), and ranking diagnostics on labelled roots.
Test mode needs a frozen-rules JSON written from a development analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cemstop.evaluate import (  # noqa: E402
    evaluate, kendall_with_zone, load_task, oracle, paired_bootstrap, tune,
)
from cemstop.rules import CHECKPOINTS, TOPK, Band, Converge, Fixed, Gap, first_iqr, grid  # noqa: E402

ZONE = {"cube": 0.01, "reacher": 0.0125}
FAMILIES = ("fixed", "converge", "gap", "band")
RULE_TYPES = {"fixed": Fixed, "converge": Converge, "gap": Gap, "band": Band}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["cube", "reacher"])
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--frozen", type=Path, help="frozen rules JSON (required for test)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true",
                   help="explicit exploratory report on incomplete roots; never a final result")
    return p.parse_args()


def rate(x: np.ndarray, draws: int = 10_000, seed: int = 0) -> dict:
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def cross_validated(family: str, roots, folds: int, seed: int = 0) -> dict:
    """Tune within folds of dev roots; return out-of-fold outcomes per root."""
    rng = np.random.default_rng(seed)
    assign = rng.permutation(len(roots)) % folds
    succ = np.zeros(len(roots), bool)
    iters = np.zeros(len(roots), np.int32)
    chosen = []
    for f in range(folds):
        train = [r for r, a in zip(roots, assign) if a != f]
        test_idx = np.where(assign == f)[0]
        rule, _ = tune(grid()[family], train)
        chosen.append(rule.param())
        res = evaluate(rule, [roots[i] for i in test_idx])
        succ[test_idx] = res["success"]
        iters[test_idx] = res["iterations"]
    return {"success": succ, "iterations": iters, "chosen_params": chosen}


def ranking_diagnostics(roots, zone: float) -> dict:
    labelled = [tr for tr in roots if "label_cost" in tr.extra]
    if not labelled:
        return {"labelled_roots": 0}
    its = [int(k) for k in labelled[0].extra["label_iters"]]
    per_iter = {k: [] for k in its}
    gaps, correct, root_of = [], [], []
    for tr in labelled:
        scale, degenerate = first_iqr(tr.costs1)
        for a, k in enumerate(its):
            res = kendall_with_zone(tr.extra["label_cost"][a], tr.extra["label_dist"][a], zone)
            per_iter[k].append(res["accuracy"])
            if not degenerate and res["pairs"]:
                gaps.append(res["abs_cost_gap"] / scale)
                correct.append(res["correct"])
                root_of.append(np.full(res["pairs"], tr.root))
    out = {"labelled_roots": len(labelled), "zone": zone,
           "pairwise_accuracy_by_iteration": {
               k: {"mean": float(np.nanmean(v)), "roots_with_pairs": int(np.sum(np.isfinite(v)))}
               for k, v in per_iter.items()}}
    if gaps:
        g = np.concatenate(gaps)
        c = np.concatenate(correct)
        edges = np.array([0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, np.inf])
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (g >= lo) & (g < hi)
            bins.append({"lo": float(lo), "hi": float(hi), "pairs": int(m.sum()),
                         "accuracy": float(c[m].mean()) if m.any() else None})
        out["margin_reliability"] = bins
        # Calibrated tau: smallest normalised gap above which inversions <= 10%.
        order = np.argsort(g)
        g_sorted, wrong = g[order], ~c[order]
        tail_wrong = np.cumsum(wrong[::-1])[::-1] / np.arange(len(g), 0, -1)
        ok = np.where(tail_wrong <= 0.10)[0]
        out["calibrated_tau_alpha10"] = float(g_sorted[ok[0]]) if len(ok) else None
    return out


def iteration_diagnostics(roots) -> dict:
    """How the plan-1 population and executed plan-1 outcome change with k."""
    gap_norm = {k: [] for k in CHECKPOINTS}
    for tr in roots:
        scale, degenerate = first_iqr(tr.costs1)
        if degenerate:
            continue
        for k in CHECKPOINTS:
            s = np.sort(tr.costs1[k - 1].astype(np.float64))
            gap_norm[k].append((s[TOPK] - s[TOPK - 1]) / scale)
    p1_dist = np.stack([tr.extra["p1_dist"] for tr in roots])
    p1_term = np.stack([tr.p1_term for tr in roots])
    return {
        "gap_over_iqr1_median": {k: float(np.median(v)) for k, v in gap_norm.items() if v},
        "plan1_distance_mean_by_k": {k: float(p1_dist[:, i].mean()) for i, k in enumerate(CHECKPOINTS)},
        "plan1_success_by_k": {k: float(p1_term[:, i].mean()) for i, k in enumerate(CHECKPOINTS)},
    }


def analyse_task(name: str, roots, args, frozen: dict | None) -> dict:
    res = {"task": name, "roots": len(roots)}
    diag = np.stack([np.diag(tr.leaf_success) for tr in roots])
    res["fixed_k_success"] = {k: rate(diag[:, i]) for i, k in enumerate(CHECKPOINTS)}
    res["fixed_k_distance"] = {k: float(np.mean([tr.leaf_dist[i, i] for tr in roots]))
                               for i, k in enumerate(CHECKPOINTS)}
    res["leaf_success_matrix"] = np.mean([tr.leaf_success for tr in roots], axis=0).round(3).tolist()
    o = oracle(roots)
    res["oracle"] = {"any_leaf": rate(o["any_leaf_success"]),
                     "best_fixed_k_per_root": rate(o["best_fixed_k_per_root_success"])}
    i10, i30 = CHECKPOINTS.index(10), CHECKPOINTS.index(30)
    res["k10_vs_k30"] = paired_bootstrap(diag[:, i10], diag[:, i30])
    res["fixed_vs_k30"] = {k: paired_bootstrap(diag[:, i], diag[:, i30])
                           for i, k in enumerate(CHECKPOINTS)}
    every = np.array([tr.leaf_success.all() for tr in roots])
    none = np.array([not tr.leaf_success.any() for tr in roots])
    res["root_types"] = {"all_leaves_succeed": int(every.sum()), "no_leaf_succeeds": int(none.sum()),
                         "mixed": int((~every & ~none).sum()),
                         # success on the very first env step under every choice
                         "solved_at_first_step": int(sum(
                             bool(tr.leaf_success.all() and tr.extra["leaf_steps"].max() <= 1)
                             for tr in roots))}
    res["iterations"] = iteration_diagnostics(roots)
    res["ranking"] = ranking_diagnostics(roots, ZONE[name])

    if args.split == "dev":
        res["tuned_in_sample"] = {}
        res["cross_validated"] = {}
        for fam in FAMILIES:
            best, table = tune(grid()[fam], roots)
            res["tuned_in_sample"][fam] = {
                "best_param": best.param(),
                "table": [{k: v for k, v in row.items() if k != "rule"} for row in table]}
            cv = cross_validated(fam, roots, args.folds)
            res["cross_validated"][fam] = {
                "success": rate(cv["success"]), "iterations": float(cv["iterations"].mean()),
                "chosen_params": cv["chosen_params"]}
            res["cross_validated"][fam]["_success"] = cv["success"]
        base = res["cross_validated"]["fixed"]["_success"]
        for fam in FAMILIES[1:]:
            res["cross_validated"][fam]["vs_fixed_cv"] = paired_bootstrap(
                res["cross_validated"][fam]["_success"], base)
            res["cross_validated"][fam]["vs_k30"] = paired_bootstrap(
                res["cross_validated"][fam]["_success"], diag[:, i30])
        tau = res["ranking"].get("calibrated_tau_alpha10")
        if tau is not None:
            ev = evaluate(Gap(tau), roots)
            res["gap_calibrated"] = {"tau": tau, "success": rate(ev["success"]),
                                     "iterations": float(ev["iterations"].mean())}
        for fam in FAMILIES:
            res["cross_validated"][fam].pop("_success")
    else:
        spec = frozen[name]
        res["frozen_eval"] = {}
        outcomes = {}
        for label, (fam, param) in spec.items():
            rule = RULE_TYPES[fam](int(param) if fam == "fixed" else float(param))
            ev = evaluate(rule, roots)
            outcomes[label] = ev
            res["frozen_eval"][label] = {"rule": [fam, param], "success": rate(ev["success"]),
                                         "iterations": float(ev["iterations"].mean())}
        ref = outcomes["fixed_dev"]["success"]
        for label, ev in outcomes.items():
            res["frozen_eval"][label]["vs_fixed_dev"] = paired_bootstrap(ev["success"], ref)
    return res


def main():
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("bulk analysis runs under sbatch")
    frozen = None
    if args.split == "test":
        if args.frozen is None:
            raise RuntimeError("test analysis needs --frozen")
        frozen = json.loads(args.frozen.read_text())
    report = {}
    for name in args.tasks:
        # Do not open held-out outcomes merely to filter them afterwards.
        from cemstop.evaluate import load_root
        expected = set(range(100)) if args.split == "dev" else set(range(100, 300))
        paths = [p for p in sorted((args.run_dir / name).glob("root_*.npz"))
                 if int(p.stem.split("_")[-1]) in expected]
        found = {int(p.stem.split("_")[-1]) for p in paths}
        incomplete = [p.name for p in paths if not p.with_suffix('.json').exists()]
        if (found != expected or incomplete) and not args.allow_partial:
            raise RuntimeError(f"Incomplete {name} {args.split}: missing roots {sorted(expected-found)}, "
                               f"missing summaries {incomplete}; use --allow-partial only for exploration")
        roots = [load_root(p) for p in paths]
        if not roots:
            continue
        report[name] = analyse_task(name, roots, args, frozen)
        report[name]["data_status"] = "COMPLETE" if found == expected and not incomplete else "PARTIAL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({t: {"fixed_k": {k: round(v["mean"], 3) for k, v in r["fixed_k_success"].items()},
                          "oracle_any": round(r["oracle"]["any_leaf"]["mean"], 3)}
                      for t, r in report.items()}, indent=1))


if __name__ == "__main__":
    main()
