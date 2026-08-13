"""Task-aligned reanalysis of exact-dynamics CEM candidate audits.

The original audit used the simulator-state analogue of the deployed shaped
cost as its physical reference.  That is the right matched-unit comparator for
residual analysis, but it is not itself a task outcome.  This script keeps that
matched-unit score only to construct the residual-shuffle null, then evaluates
every selected candidate with two simulator-derived task outcomes:

* terminal object-to-goal distance; and
* the environment's terminal success indicator.

It also reanalyses shared-noise CEM branches with the same task outcomes.  All
confidence intervals resample independent episode seeds after averaging the
seven within-seed MPC replans.  Run on a Slurm compute node because the input
candidate dumps are large.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _source(path: str) -> str:
    name = Path(path).name
    for suffix in ("_candidates.csv.gz", "_candidates.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _stable_seed(base: int, key: tuple[object, ...]) -> int:
    digest = hashlib.blake2b(
        "|".join(map(str, key)).encode("utf-8"), digest_size=8
    ).digest()
    return (base + int.from_bytes(digest, "little")) % (2**63 - 1)


def _bootstrap(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    if len(values) == 1:
        return point, point, point
    draws = rng.choice(values, (n_bootstrap, len(values)), replace=True).mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def _two_sided_sign_p(values: np.ndarray) -> tuple[int, int, float]:
    values = values[np.isfinite(values) & (values != 0)]
    n = int(len(values))
    if n == 0:
        return 0, 0, float("nan")
    k = int(np.count_nonzero(values > 0))
    tail_k = min(k, n - k)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(tail_k + 1)) / 2**n)
    return k, n, float(p)


def _exact_signflip_mean_p(values: np.ndarray) -> float:
    """Two-sided paired randomization p-value for the mean contrast."""
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n == 0 or n > 20:
        return float("nan")
    observed = abs(float(values.mean()))
    signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.count_nonzero(permuted >= observed - 1e-15) / len(permuted))


def _preselection_population(
    group: pd.DataFrame, n_permutations: int, rng: np.random.Generator
) -> dict[str, float]:
    group = group.sort_values("candidate")
    proxy = group.proxy_cost.to_numpy(float)
    shaped_truth = group.true_shaped_cost.to_numpy(float)
    distance = group.obj_goal_dist.to_numpy(float)
    success = group.success_end.to_numpy(float)
    if not all(np.isfinite(v).all() for v in (proxy, shaped_truth, distance, success)):
        raise ValueError("non-finite candidate score or task outcome")

    residual = proxy - shaped_truth
    selected = int(np.argmin(proxy))
    best_distance = float(distance.min())
    success_available = float(success.max())

    null_distance_regret = np.empty(n_permutations, dtype=np.float64)
    null_success_shortfall = np.empty(n_permutations, dtype=np.float64)
    for b in range(n_permutations):
        shuffled = shaped_truth + residual[rng.permutation(len(group))]
        chosen = int(np.argmin(shuffled))
        null_distance_regret[b] = distance[chosen] - best_distance
        null_success_shortfall[b] = success_available - success[chosen]

    actual_distance_regret = float(distance[selected] - best_distance)
    actual_success_shortfall = float(success_available - success[selected])
    return {
        "n_candidates": float(len(group)),
        "selected_obj_goal_dist_m": float(distance[selected]),
        "best_obj_goal_dist_m": best_distance,
        "actual_distance_regret_m": actual_distance_regret,
        "null_mean_distance_regret_m": float(null_distance_regret.mean()),
        "actual_minus_null_distance_regret_m": float(
            actual_distance_regret - null_distance_regret.mean()
        ),
        "null_mc_se_distance_regret_m": float(
            null_distance_regret.std(ddof=1) / np.sqrt(n_permutations)
        ),
        "selected_success_end": float(success[selected]),
        "success_available": success_available,
        "actual_success_shortfall": actual_success_shortfall,
        "null_mean_success_shortfall": float(null_success_shortfall.mean()),
        "actual_minus_null_success_shortfall": float(
            actual_success_shortfall - null_success_shortfall.mean()
        ),
        "null_mc_se_success_shortfall": float(
            null_success_shortfall.std(ddof=1) / np.sqrt(n_permutations)
        ),
    }


def analyse_preselection(
    paths: list[str], n_permutations: int, permutation_seed: int,
    n_bootstrap: int, bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "task", "cost", "seed", "replan", "iter", "candidate", "proxy_cost",
        "true_shaped_cost", "obj_goal_dist", "success_end",
    }
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=lambda c: c in required)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        if set(frame.cost.unique()) != {"stateprobe"}:
            raise ValueError(f"{path}: expected stateprobe candidates only")
        frame.insert(0, "source", _source(path))
        frames.append(frame)
    candidates = pd.concat(frames, ignore_index=True)

    keys = ["source", "task", "seed", "replan", "iter"]
    rows = []
    for key, group in candidates.groupby(keys, sort=False):
        rows.append({
            **dict(zip(keys, key)),
            **_preselection_population(
                group, n_permutations,
                np.random.default_rng(_stable_seed(permutation_seed, key)),
            ),
        })
    populations = pd.DataFrame(rows)

    metrics = [
        "selected_obj_goal_dist_m", "best_obj_goal_dist_m",
        "actual_distance_regret_m", "null_mean_distance_regret_m",
        "actual_minus_null_distance_regret_m", "selected_success_end",
        "success_available", "actual_success_shortfall",
        "null_mean_success_shortfall", "actual_minus_null_success_shortfall",
    ]
    rng = np.random.default_rng(bootstrap_seed)
    out = []
    for key, group in populations.groupby(["source", "task", "iter"], sort=False):
        seed_means = group.groupby("seed", sort=False)[metrics].mean()
        for metric in metrics:
            point, lo, hi = _bootstrap(
                seed_means[metric].to_numpy(float), rng, n_bootstrap
            )
            positive, nonzero, sign_p = _two_sided_sign_p(
                seed_means[metric].to_numpy(float)
            )
            out.append({
                **dict(zip(["source", "task", "iter"], key)),
                "n_seed": int(len(seed_means)), "metric": metric,
                "estimate": point, "ci_lo": lo, "ci_hi": hi,
                "positive_seed_means": positive, "nonzero_seed_means": nonzero,
                "sign_test_p": sign_p,
                "signflip_mean_p": _exact_signflip_mean_p(
                    seed_means[metric].to_numpy(float)
                ),
            })
    return populations, pd.DataFrame(out)


def analyse_branches(
    paths: list[str], n_bootstrap: int, bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "task", "seed", "replan", "iter", "generating_branch", "candidate",
        "true_state_cost", "obj_goal_dist", "success_end",
    }
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=lambda c: c in required)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        frame.insert(0, "source", _source(path))
        frames.append(frame)
    candidates = pd.concat(frames, ignore_index=True)

    rows = []
    keys = ["source", "task", "seed", "replan", "iter", "generating_branch"]
    for key, group in candidates.groupby(keys, sort=False):
        rows.append({
            **dict(zip(keys, key)),
            "best_true_state_cost_m": float(group.true_state_cost.min()),
            "best_obj_goal_dist_m": float(group.obj_goal_dist.min()),
            "success_available": float(group.success_end.max()),
            "n_candidates": int(len(group)),
        })
    populations = pd.DataFrame(rows)

    comparison_rows = []
    rng = np.random.default_rng(bootstrap_seed)
    pair_index = ["source", "task", "seed", "replan", "iter"]
    for source, source_rows in populations.groupby("source", sort=False):
        branches = sorted(set(source_rows.generating_branch) - {"true_state"})
        if "true_state" not in set(source_rows.generating_branch):
            continue
        for branch in branches:
            left = source_rows[source_rows.generating_branch == branch].set_index(pair_index)
            right = source_rows[source_rows.generating_branch == "true_state"].set_index(pair_index)
            common = left.index.intersection(right.index)
            if not len(common):
                continue
            metrics = [
                "best_true_state_cost_m", "best_obj_goal_dist_m", "success_available"
            ]
            paired = left.loc[common, metrics] - right.loc[common, metrics]
            paired = paired.reset_index()
            paired["branch"] = branch
            for (task, iteration), group in paired.groupby(["task", "iter"], sort=False):
                seed_means = group.groupby("seed", sort=False)[
                    ["best_true_state_cost_m", "best_obj_goal_dist_m", "success_available"]
                ].mean()
                for metric in (
                    "best_true_state_cost_m", "best_obj_goal_dist_m", "success_available"
                ):
                    values = seed_means[metric].to_numpy(float)
                    point, lo, hi = _bootstrap(values, rng, n_bootstrap)
                    positive, nonzero, sign_p = _two_sided_sign_p(values)
                    comparison_rows.append({
                        "source": source, "task": task, "branch": branch,
                        "reference_branch": "true_state", "iter": int(iteration),
                        "n_seed": int(len(seed_means)), "metric": metric,
                        "branch_minus_reference": point, "ci_lo": lo, "ci_hi": hi,
                        "positive_seed_means": positive,
                        "nonzero_seed_means": nonzero, "sign_test_p": sign_p,
                        "signflip_mean_p": _exact_signflip_mean_p(values),
                    })
    return populations, pd.DataFrame(comparison_rows)


def _csv_block(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_csv(index=False, float_format="%.5f") + "```"


def write_report(
    path: Path, pre_summary: pd.DataFrame, branch_summary: pd.DataFrame,
    pre_populations: pd.DataFrame,
) -> None:
    first = pre_summary[
        pre_summary["iter"]
        == pre_summary.groupby(["source", "task"])["iter"].transform("min")
    ]
    final_branch = branch_summary[
        branch_summary["iter"]
        == branch_summary.groupby(["source", "task", "branch"])["iter"].transform("max")
    ] if len(branch_summary) else branch_summary
    max_mc = pre_populations[[
        "null_mc_se_distance_regret_m", "null_mc_se_success_shortfall"
    ]].max()
    lines = [
        "# Task-aligned exact-dynamics cost audit", "",
        "All outcomes below come directly from simulator rollouts. The residual-shuffle",
        "null is still constructed in matched shaped-cost units, but the selected",
        "candidate is evaluated by terminal object-to-goal distance and environment",
        "success. CIs are percentile bootstraps over independent episode seeds after",
        "averaging within-seed replans; they quantify seed uncertainty. With the requested",
        f"permutations, the largest population-level null Monte Carlo SE was "
        f"{max_mc.iloc[0]:.6g} m for distance regret and {max_mc.iloc[1]:.6g} for success shortfall.",
        "", "## Initial identical-population selection", "", _csv_block(first),
        "", "## Final shared-noise branch comparison", "", _csv_block(final_branch),
        "", "## Interpretation", "",
        "- Positive distance differences mean the deployed-cost branch retained a worse",
        "  best physical task outcome than the privileged true-state branch.",
        "- Negative success-availability differences mean it retained fewer successful",
        "  candidates. Exact sign tests use one mean per independent seed and exclude ties.",
        "- These are optimizer-conditioned diagnostics of fixed cost compositions, not",
        "  closed-loop policy comparisons or evidence that privileged state is deployable.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preselection-candidates", nargs="+", required=True)
    parser.add_argument("--branch-candidates", nargs="+", required=True)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--permutation-seed", type=int, default=78001)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=78002)
    parser.add_argument("--out-prefix", default="results/task_aligned_cost_audit")
    args = parser.parse_args()

    pre_pop, pre_summary = analyse_preselection(
        args.preselection_candidates, args.n_permutations, args.permutation_seed,
        args.n_bootstrap, args.bootstrap_seed,
    )
    branch_pop, branch_summary = analyse_branches(
        args.branch_candidates, args.n_bootstrap, args.bootstrap_seed + 1,
    )
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pre_pop.to_csv(prefix.with_name(prefix.name + "_preselection_populations.csv"), index=False)
    pre_summary.to_csv(prefix.with_name(prefix.name + "_preselection_summary.csv"), index=False)
    branch_pop.to_csv(prefix.with_name(prefix.name + "_branch_populations.csv"), index=False)
    branch_summary.to_csv(prefix.with_name(prefix.name + "_branch_summary.csv"), index=False)
    write_report(prefix.with_suffix(".md"), pre_summary, branch_summary, pre_pop)
    print(f"wrote task-aligned audit under {prefix.parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
