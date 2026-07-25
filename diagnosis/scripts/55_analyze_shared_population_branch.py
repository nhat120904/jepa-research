"""Seed-clustered analysis for the shared-noise branched-CEM audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPORT_METRICS = [
    "population_obj_decode_error_median_cm",
    "own_elite_obj_decode_error_median_cm",
    "true_elite_obj_decode_error_median_cm",
    "own_selected_true_regret",
    "best_true_state_cost",
    "stateprobe_true_spearman",
    "stateprobe_true_topk_overlap",
    "coverage_success_end",
    "paired_action_rmse_vs_first_branch",
]

BRANCH_DIFF_METRICS = [
    "population_obj_decode_error_median_cm",
    "best_true_state_cost",
    "coverage_success_end",
    "own_selected_true_regret",
]


def _source(path: str) -> str:
    name = Path(path).name
    suffix = "_iterations.csv"
    return name[: -len(suffix)] if name.endswith(suffix) else Path(path).stem


def _boot(values: np.ndarray, rng: np.random.Generator,
          n_bootstrap: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, estimate, estimate
    draws = rng.choice(values, (n_bootstrap, len(values)), replace=True).mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def aggregate(rows: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    keys = ["source", "task", "generating_branch", "iter"]
    for key, g in rows.groupby(keys, sort=False):
        seed_mean = g.groupby("seed", sort=False)[REPORT_METRICS].mean(numeric_only=True)
        for metric in REPORT_METRICS:
            est, lo, hi = _boot(seed_mean[metric].to_numpy(float), rng, n_bootstrap)
            out.append({**dict(zip(keys, key)), "n_seed": len(seed_mean),
                        "metric": metric, "estimate": est,
                        "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


def paired_branch_differences(rows: pd.DataFrame, n_bootstrap: int,
                              seed: int) -> pd.DataFrame:
    """Stateprobe minus true-state on paired seed/replan/iteration rows."""
    rng = np.random.default_rng(seed)
    idx = ["source", "task", "seed", "replan", "iter"]
    probe = rows[rows.generating_branch == "stateprobe"].set_index(idx)
    truth = rows[rows.generating_branch == "true_state"].set_index(idx)
    common = probe.index.intersection(truth.index)
    if not len(common):
        raise ValueError("no paired stateprobe/true_state branch rows")
    paired = probe.loc[common, BRANCH_DIFF_METRICS] - truth.loc[common, BRANCH_DIFF_METRICS]
    paired = paired.reset_index()
    out = []
    for key, g in paired.groupby(["source", "task", "iter"], sort=False):
        seed_mean = g.groupby("seed", sort=False)[BRANCH_DIFF_METRICS].mean(numeric_only=True)
        for metric in BRANCH_DIFF_METRICS:
            est, lo, hi = _boot(seed_mean[metric].to_numpy(float), rng, n_bootstrap)
            out.append({**dict(zip(["source", "task", "iter"], key)),
                        "n_seed": len(seed_mean), "metric": metric,
                        "stateprobe_minus_true": est, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


def immediate_selection(rows: pd.DataFrame, n_bootstrap: int,
                        seed: int) -> pd.DataFrame:
    """At shared iteration 0, compare proxy-selected and truth-selected elites."""
    rng = np.random.default_rng(seed)
    first = rows[rows["iter"] == 0].copy()
    first = first[first.generating_branch == "stateprobe"]
    first["proxy_minus_true_elite_error_median_cm"] = (
        first.own_elite_obj_decode_error_median_cm
        - first.true_elite_obj_decode_error_median_cm)
    metrics = ["proxy_minus_true_elite_error_median_cm",
               "stateprobe_selected_true_regret"]
    out = []
    for key, g in first.groupby(["source", "task"], sort=False):
        seed_mean = g.groupby("seed", sort=False)[metrics].mean(numeric_only=True)
        for metric in metrics:
            est, lo, hi = _boot(seed_mean[metric].to_numpy(float), rng, n_bootstrap)
            out.append({**dict(zip(["source", "task"], key)),
                        "n_seed": len(seed_mean), "metric": metric,
                        "estimate": est, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


def _block(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_csv(index=False, float_format="%.4f") + "```"


def write_report(summary: pd.DataFrame, branch_diff: pd.DataFrame,
                 immediate: pd.DataFrame, out: Path) -> None:
    final = summary[summary["iter"] == summary.groupby(
        ["source", "task", "generating_branch"])["iter"].transform("max")]
    final = final[final.metric.isin([
        "population_obj_decode_error_median_cm", "best_true_state_cost",
        "coverage_success_end", "own_selected_true_regret"])]
    lines = [
        "# Shared-noise branched-CEM audit", "",
        "Iteration 0 is candidate-for-candidate identical across branches. Later",
        "iterations use common Gaussian noise transformed by branch-specific CEM",
        "means and variances. All physical quantities come from simulator rollouts.",
        "", "## Immediate selection on the shared iteration-0 population", "",
        _block(immediate), "", "## Stateprobe minus true-state branch by iteration", "",
        _block(branch_diff), "", "## Final branch state", "", _block(final), "",
        "## Interpretation gates", "",
        "- Immediate error-pocket selection requires positive, CI-clean proxy-minus-true",
        "  elite error and positive proxy-selected physical regret at iteration 0.",
        "- Adaptive amplification requires later stateprobe populations to have worse",
        "  physical best cost/coverage or higher decode error than the paired true-state",
        "  populations. Iteration-0 equality is validated in the raw iteration rows.",
        "- A positive result identifies the encoder--probe--cost composition, not the",
        "  encoder alone. The true-state carrier determines visited MPC snapshots, so",
        "  branch comparisons are within-snapshot mechanisms, not two closed-loop policies.",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", nargs="+", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=55001)
    ap.add_argument("--out-prefix", default="results/shared_population_branch_audit")
    args = ap.parse_args()

    frames = []
    for path in args.iterations:
        df = pd.read_csv(path)
        df.insert(0, "source", _source(path))
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)
    iter0 = rows[rows["iter"] == 0]
    if not (iter0.identical_to_first_branch == 1).all():
        raise ValueError("iteration-0 candidate equality invariant failed")

    summary = aggregate(rows, args.n_bootstrap, args.bootstrap_seed)
    branch_diff = paired_branch_differences(
        rows, args.n_bootstrap, args.bootstrap_seed + 1)
    immediate = immediate_selection(rows, args.n_bootstrap, args.bootstrap_seed + 2)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    branch_diff.to_csv(prefix.with_name(prefix.name + "_branch_differences.csv"), index=False)
    immediate.to_csv(prefix.with_name(prefix.name + "_immediate.csv"), index=False)
    write_report(summary, branch_diff, immediate, prefix.with_suffix(".md"))
    print(f"wrote shared-branch audit under {prefix.parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
