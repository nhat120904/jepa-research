"""Seed-clustered analysis of oracle candidate coverage versus proxy selection.

The report answers two separate questions on the same candidate populations:

* Did the proposal cover an exact simulator-successful / physically progressive
  candidate?
* When it did, did the latent proxy select it?

Replans and search iterations are nested observations, never independent sample
units.  All intervals therefore resample episode seeds as clusters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._coverage_selection_metrics import coverage_selection_summary


METRICS = [
    "coverage_success_any", "selected_success_any", "missed_success_any",
    "coverage_success_end", "selected_success_end", "missed_success_end",
    "best_true_progress", "selected_true_progress", "selected_physical_regret",
    "proxy_true_spearman", "proxy_true_topk_overlap",
    "best_true_shaped", "selected_true_shaped", "selected_shaped_regret",
    "proxy_shaped_spearman", "proxy_shaped_topk_overlap",
]


def source_name(path: str | Path) -> str:
    name = Path(path).name
    for suffix in ("_iterations.csv", "_candidates.csv.gz", "_episodes.csv"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(path).stem


def load_iterations(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        d = pd.read_csv(path)
        d.insert(0, "source", source_name(path))
        frames.append(d)
    if not frames:
        raise ValueError("no iteration files supplied")
    out = pd.concat(frames, ignore_index=True, sort=False)
    # These are derived selection-failure events rather than runner fields.
    out["missed_success_any"] = (
        (out.coverage_success_any == 1) & (out.selected_success_any == 0)).astype(int)
    out["missed_success_end"] = (
        (out.coverage_success_end == 1) & (out.selected_success_end == 0)).astype(int)
    required = {"source", "task", "cost", "seed", "replan", "iter", *METRICS}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"iteration dump missing columns: {missing}")
    return out


def cluster_bootstrap(df: pd.DataFrame, column: str, *, n_boot: int,
                      rng: np.random.Generator) -> tuple[float, float, float]:
    groups = [g[column].dropna().to_numpy(float) for _, g in df.groupby("seed")]
    groups = [g for g in groups if len(g)]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    point = float(np.concatenate(groups).mean())
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        boot[b] = np.concatenate(picked).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize(iterations: pd.DataFrame, *, n_boot: int,
              bootstrap_seed: int) -> pd.DataFrame:
    rows = []
    group_cols = ["source", "task", "cost", "iter"]
    for group_index, (key, g) in enumerate(iterations.groupby(group_cols, sort=False)):
        row = dict(zip(group_cols, key))
        row.update(n_seed=int(g.seed.nunique()), n_replan=int(len(g)),
                   n_candidates_per_search=int(g.n_candidates.mode().iloc[0]),
                   topk=int(g.topk.mode().iloc[0]))
        for metric_index, metric in enumerate(METRICS):
            rng = np.random.default_rng(bootstrap_seed + 1009 * group_index + metric_index)
            point, lo, hi = cluster_bootstrap(g, metric, n_boot=n_boot, rng=rng)
            row[metric] = point
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def final_iterations(iterations: pd.DataFrame) -> pd.DataFrame:
    key = ["source", "task", "cost", "seed", "replan"]
    max_iter = iterations.groupby(key, dropna=False).iter.transform("max")
    return iterations[iterations.iter == max_iter].copy()


def validate_candidates(iterations: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    """Recompute core metrics from raw candidates and compare to iteration rows."""
    checks = []
    lookup_cols = ["source", "task", "cost", "seed", "replan", "iter"]
    indexed = iterations.set_index(lookup_cols)
    for path in paths:
        source = source_name(path)
        candidates = pd.read_csv(path)
        candidates.insert(0, "source", source)
        required = {
            *lookup_cols, "candidate", "proxy_cost", "true_progress_cost",
            "success_any", "success_end", "proxy_selected", "true_progress_best",
        }
        missing = sorted(required - set(candidates.columns))
        if missing:
            raise ValueError(f"candidate dump {path} missing columns: {missing}")
        for key, g in candidates.groupby(lookup_cols, sort=False):
            g = g.sort_values("candidate")
            recorded = indexed.loc[key]
            if isinstance(recorded, pd.DataFrame):
                raise ValueError(f"duplicate iteration summary key: {key}")
            recomputed = coverage_selection_summary(
                g.proxy_cost, g.true_progress_cost, g.success_any, g.success_end,
                topk_frac=float(recorded.topk) / len(g),
            )
            numeric_checks = [
                "n_candidates", "coverage_success_any", "coverage_success_end",
                "selected_success_any", "selected_success_end", "best_true_progress",
                "selected_true_progress", "selected_physical_regret",
                "proxy_true_spearman", "proxy_true_topk_overlap", "selected_index",
            ]
            max_abs = 0.0
            for col in numeric_checks:
                a, b = float(recorded[col]), float(recomputed[col])
                if np.isnan(a) and np.isnan(b):
                    continue
                max_abs = max(max_abs, abs(a - b))
            structural_ok = (
                len(g) == int(recorded.n_candidates)
                and int(g.proxy_selected.sum()) == 1
                and int(g.true_progress_best.sum()) == 1
                and int(g.loc[g.proxy_selected == 1, "candidate"].iloc[0])
                == int(recorded.selected_index)
            )
            checks.append({
                **dict(zip(lookup_cols, key)), "n_candidates": len(g),
                "structural_ok": int(structural_ok), "max_abs_metric_diff": max_abs,
                "metric_match": int(max_abs < 1e-9),
            })
    return pd.DataFrame(checks)


def write_report(path: Path, final: pd.DataFrame, validation: pd.DataFrame | None) -> None:
    lines = [
        "# Oracle candidate coverage versus proxy selection", "",
        "Every metric compares proxy and simulator truth on the **same sampled",
        "candidates**. `coverage_success_*` asks whether at least one exact MetaWorld",
        "successful candidate was present; `selected_success_*` asks whether the proxy",
        "argmin was successful; `missed_success_*` is their selection-failure event.", "",
        "`selected_physical_regret` is task-distance(proxy argmin) minus the best task",
        "distance in that population. It is opportunity regret within the sampled set,",
        "not global planning regret. Spearman and top-k overlap measure full/ranked-set",
        "agreement. Intervals are percentile bootstraps clustered by episode seed.", "",
        "## Final search iteration", "",
    ]
    display = [
        "source", "task", "cost", "iter", "n_seed", "n_replan",
        "coverage_success_end", "coverage_success_end_ci_lo", "coverage_success_end_ci_hi",
        "selected_success_end", "selected_success_end_ci_lo", "selected_success_end_ci_hi",
        "missed_success_end", "selected_physical_regret",
        "selected_physical_regret_ci_lo", "selected_physical_regret_ci_hi",
        "proxy_true_spearman", "proxy_true_topk_overlap",
    ]
    if len(final):
        try:
            lines.append(final[display].to_markdown(index=False))
        except ImportError:
            # `tabulate` is optional; preserve report generation in minimal envs.
            lines.extend(["```text", final[display].to_string(index=False), "```"])
    else:
        lines.append("_No rows._")
    lines.extend(["", "## Raw-dump validation", ""])
    if validation is None:
        lines.append("Raw candidate validation was not requested.")
    else:
        ok = bool(len(validation) and validation.structural_ok.all()
                  and validation.metric_match.all())
        lines.append(
            f"Validated {len(validation)} candidate populations: "
            f"**{'PASS' if ok else 'FAIL'}**."
        )
    lines.extend(["", "## Interpretation guardrails", "",
                  "- Low coverage implicates the proposal/search budget; it does not test proxy ranking.",
                  "- High coverage plus frequent missed success or positive regret implicates selection.",
                  "- Candidate-horizon success does not guarantee closed-loop endpoint success after MPC replanning.",
                  "- Results are specific to the tested checkpoint, proxy, task, and CEM budget.", ""])
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", nargs="+", required=True)
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--out-prefix", default="results/oracle_coverage_selection_analysis")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260713)
    args = ap.parse_args()

    iterations = load_iterations(args.iterations)
    curves = summarize(iterations, n_boot=args.n_bootstrap,
                       bootstrap_seed=args.bootstrap_seed)
    final_raw = final_iterations(iterations)
    final = summarize(final_raw, n_boot=args.n_bootstrap,
                      bootstrap_seed=args.bootstrap_seed + 1_000_000)
    validation = validate_candidates(iterations, args.candidates) if args.candidates else None

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(prefix.with_name(prefix.name + "_by_iteration.csv"), index=False)
    final.to_csv(prefix.with_name(prefix.name + "_final.csv"), index=False)
    if validation is not None:
        validation.to_csv(prefix.with_name(prefix.name + "_validation.csv"), index=False)
    write_report(prefix.with_suffix(".md"), final, validation)
    print(f"wrote coverage-selection analysis under {prefix.parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
