"""Analyze whether CEM selection enriches object-readout error.

The input candidate dumps contain every candidate *before* elite selection.
For each identical population we compare the proxy top-k, the simulator-state
top-k, and the rejected candidates.  Inference is clustered by episode seed;
replans and CEM iterations are never treated as independent episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["source", "task", "cost", "seed", "replan", "iter"]
METRICS = [
    "population_error_mean_cm", "population_error_median_cm",
    "population_within5", "proxy_elite_error_mean_cm",
    "proxy_elite_error_median_cm", "proxy_elite_within5",
    "true_elite_error_mean_cm", "true_elite_error_median_cm",
    "true_elite_within5", "rejected_error_mean_cm",
    "proxy_argmin_error_cm", "true_argmin_error_cm",
    "proxy_elite_enrichment_cm", "proxy_elite_vs_rejected_cm",
    "proxy_elite_median_enrichment_cm",
    "proxy_vs_true_elite_excess_cm", "proxy_error_spearman",
    "proxy_vs_true_elite_median_excess_cm",
    "population_optimism_mean_m", "proxy_elite_optimism_mean_m",
    "true_elite_optimism_mean_m", "proxy_elite_optimism_enrichment_m",
    "proxy_true_spearman", "proxy_true_topk_overlap",
    "selected_true_shaped_regret_m",
]


def _source(path: str) -> str:
    name = Path(path).name
    for suffix in ("_candidates.csv.gz", "_candidates.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(pd.Series(a).rank(method="average").corr(
        pd.Series(b).rank(method="average")))


def summarize_population(g: pd.DataFrame, topk_frac: float) -> dict[str, float]:
    g = g.sort_values("candidate")
    n = len(g)
    k = max(1, int(np.ceil(topk_frac * n)))
    proxy = g.proxy_cost.to_numpy(float)
    truth = g.true_shaped_cost.to_numpy(float)
    err = g.obj_decode_error_cm.to_numpy(float)
    optimism = g.stateprobe_optimism_m.to_numpy(float)
    if not (np.isfinite(proxy).all() and np.isfinite(truth).all()
            and np.isfinite(err).all()):
        raise ValueError("candidate costs and decode errors must be finite")
    proxy_order = np.argsort(proxy, kind="mergesort")
    truth_order = np.argsort(truth, kind="mergesort")
    pidx, tidx = proxy_order[:k], truth_order[:k]
    rejected = proxy_order[k:]
    pset, tset = set(pidx.tolist()), set(tidx.tolist())

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    def median(x):
        return float(np.median(x)) if len(x) else float("nan")

    return {
        "n_candidates": n,
        "topk": k,
        "population_error_mean_cm": mean(err),
        "population_error_median_cm": median(err),
        "population_within5": mean(err < 5.0),
        "proxy_elite_error_mean_cm": mean(err[pidx]),
        "proxy_elite_error_median_cm": median(err[pidx]),
        "proxy_elite_within5": mean(err[pidx] < 5.0),
        "true_elite_error_mean_cm": mean(err[tidx]),
        "true_elite_error_median_cm": median(err[tidx]),
        "true_elite_within5": mean(err[tidx] < 5.0),
        "rejected_error_mean_cm": mean(err[rejected]),
        "proxy_argmin_error_cm": float(err[proxy_order[0]]),
        "true_argmin_error_cm": float(err[truth_order[0]]),
        "proxy_elite_enrichment_cm": mean(err[pidx]) - mean(err),
        "proxy_elite_vs_rejected_cm": mean(err[pidx]) - mean(err[rejected]),
        "proxy_elite_median_enrichment_cm": median(err[pidx]) - median(err),
        "proxy_vs_true_elite_excess_cm": mean(err[pidx]) - mean(err[tidx]),
        "proxy_vs_true_elite_median_excess_cm": median(err[pidx]) - median(err[tidx]),
        "population_optimism_mean_m": mean(optimism),
        "proxy_elite_optimism_mean_m": mean(optimism[pidx]),
        "true_elite_optimism_mean_m": mean(optimism[tidx]),
        "proxy_elite_optimism_enrichment_m": (
            mean(optimism[pidx]) - mean(optimism)),
        # Negative means lower proxy costs coincide with larger decode errors.
        "proxy_error_spearman": _spearman(proxy, err),
        "proxy_true_spearman": _spearman(proxy, truth),
        "proxy_true_topk_overlap": len(pset & tset) / k,
        "selected_true_shaped_regret_m": float(
            truth[proxy_order[0]] - truth[truth_order[0]]),
    }


def _bootstrap_seed_mean(values: np.ndarray, rng: np.random.Generator,
                         n_bootstrap: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, estimate, estimate
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def aggregate(pop: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in pop.groupby(["source", "task", "cost", "iter"], sort=False):
        seed_means = g.groupby("seed", sort=False)[METRICS].mean(numeric_only=True)
        for metric in METRICS:
            est, lo, hi = _bootstrap_seed_mean(
                seed_means[metric].to_numpy(float), rng, n_bootstrap)
            rows.append(dict(zip(["source", "task", "cost", "iter"], key),
                             n_seed=len(seed_means), metric=metric,
                             estimate=est, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def first_final(pop: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in pop.groupby(["source", "task", "cost"], sort=False):
        first_it, final_it = int(g["iter"].min()), int(g["iter"].max())
        seed_it = g.groupby(["seed", "iter"], sort=False)[METRICS].mean(numeric_only=True)
        for metric in METRICS:
            diffs = []
            for episode_seed in seed_it.index.get_level_values("seed").unique():
                sg = seed_it.xs(episode_seed, level="seed")
                if first_it in sg.index and final_it in sg.index:
                    diffs.append(float(sg.loc[final_it, metric] - sg.loc[first_it, metric]))
            est, lo, hi = _bootstrap_seed_mean(np.asarray(diffs), rng, n_bootstrap)
            rows.append(dict(zip(["source", "task", "cost"], key),
                             first_iter=first_it, final_iter=final_it,
                             n_seed=len(diffs), metric=metric,
                             final_minus_first=est, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def _csv_block(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_csv(index=False, float_format="%.4f") + "```"


def write_report(summary: pd.DataFrame, delta: pd.DataFrame, out: Path) -> None:
    primary = [
        "population_error_mean_cm", "proxy_elite_error_mean_cm",
        "true_elite_error_mean_cm", "proxy_elite_enrichment_cm",
        "proxy_elite_median_enrichment_cm",
        "proxy_vs_true_elite_excess_cm", "proxy_error_spearman",
        "proxy_elite_optimism_enrichment_m",
        "proxy_true_spearman", "selected_true_shaped_regret_m",
    ]
    first = summary[summary.metric.isin(primary)].copy()
    first = first[first["iter"] == first.groupby(
        ["source", "task", "cost"])["iter"].transform("min")]
    final = summary[summary.metric.isin(primary)].copy()
    final = final[final["iter"] == final.groupby(
        ["source", "task", "cost"])["iter"].transform("max")]
    dsel = delta[delta.metric.isin(
        ["proxy_elite_error_mean_cm", "proxy_elite_enrichment_cm"])]
    lines = [
        "# CEM pre-selection error-pocket audit", "",
        "Every row compares selections on the identical simulator-rolled candidate",
        "population. `proxy_elite_enrichment_cm > 0` means the proxy top-k has",
        "larger object-decode error than the unselected population. A CI-clean",
        "effect at iteration 0 is the primary immediate-selection test. It diagnoses",
        "the encoder+readout cost composition, not the encoder alone.", "",
        "## First iteration (already after the first top-k selection)", "",
        _csv_block(first), "", "## Final iteration", "", _csv_block(final), "",
        "## Final minus first", "", _csv_block(dsel), "",
        "## Pre-registered interpretation", "",
        "- Primary support: iteration-0 median proxy-elite error enrichment and mean",
        "  optimism-enrichment CIs are both entirely above zero.",
        "- Stronger specificity: proxy elites have more error than simulator-truth elites",
        "  on the same population, with CI-clean positive excess.",
        "- Absolute proxy-elite error rises from first to final, but elite-versus-population",
        "  enrichment decreases as the proposal population also shifts. Do not claim",
        "  monotonic growth in relative selection enrichment.",
        "- A null primary test means the random-reference-to-elite gap cannot be attributed",
        "  to top-k selection without another explanation (population/horizon shift).",
        "- None of these tests alone assigns the error specifically to the encoder rather",
        "  than the probe head or their induced cost geometry.",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--topk-frac", type=float, default=0.1)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=53001)
    ap.add_argument("--out-prefix", default="results/cem_preselection_audit")
    args = ap.parse_args()

    required = {"task", "cost", "seed", "replan", "iter", "candidate",
                "proxy_cost", "true_shaped_cost", "obj_decode_error_cm",
                "stateprobe_optimism_m"}
    frames = []
    for path in args.candidates:
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df.insert(0, "source", _source(path))
        frames.append(df)
    candidates = pd.concat(frames, ignore_index=True)

    rows = []
    for key, g in candidates.groupby(KEYS, sort=False):
        rows.append({**dict(zip(KEYS, key)),
                     **summarize_population(g, args.topk_frac)})
    pop = pd.DataFrame(rows)
    summary = aggregate(pop, args.n_bootstrap, args.bootstrap_seed)
    delta = first_final(pop, args.n_bootstrap, args.bootstrap_seed + 1)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pop.to_csv(prefix.with_name(prefix.name + "_populations.csv"), index=False)
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    delta.to_csv(prefix.with_name(prefix.name + "_first_final.csv"), index=False)
    write_report(summary, delta, prefix.with_suffix(".md"))
    print(f"wrote CEM pre-selection audit under {prefix.parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
