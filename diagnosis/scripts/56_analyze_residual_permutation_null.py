"""Matched-residual permutation null for state-probe candidate selection.

For every simulator-rolled candidate population, the state-probe proxy and the
simulator-state cost have the same units and functional form.  Write

    proxy_i = truth_i + residual_i.

The actual proxy may select a poor physical candidate either because generic
optimization over noisy scores produces a winner's curse, or because residuals
are structured with candidate geometry.  This audit preserves the exact
within-population residual distribution but randomly reassigns residuals to
candidates.  It then compares actual physical selection regret with the regret
under those matched-marginal shuffled proxies.

Large candidate dumps must be analyzed on a Slurm compute node; see the paired
``slurm_residual_permutation_null.sh`` wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["source", "task", "cost", "seed", "replan", "iter"]
REGRET_METRICS = ("argmin_regret_m", "elite_regret_m")


def _source(path: str) -> str:
    name = Path(path).name
    for suffix in ("_candidates.csv.gz", "_candidates.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _population_seed(base_seed: int, key: tuple[object, ...]) -> int:
    payload = "|".join(map(str, key)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return (base_seed + int.from_bytes(digest, "little")) % (2**63 - 1)


def _summarize_population(
    g: pd.DataFrame,
    *,
    topk_frac: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    g = g.sort_values("candidate")
    truth = g["true_shaped_cost"].to_numpy(float)
    proxy = g["proxy_cost"].to_numpy(float)
    if not (np.isfinite(truth).all() and np.isfinite(proxy).all()):
        raise ValueError("state-probe proxy and simulator truth must be finite")

    residual = proxy - truth
    if not np.allclose(truth + residual, proxy, rtol=1e-7, atol=1e-9):
        raise AssertionError("proxy != truth + residual")

    n = len(g)
    k = max(1, int(np.ceil(topk_frac * n)))
    truth_order = np.argsort(truth, kind="mergesort")
    proxy_order = np.argsort(proxy, kind="mergesort")
    truth_best = float(truth[truth_order[0]])
    truth_elite_mean = float(truth[truth_order[:k]].mean())

    actual_argmin = float(truth[proxy_order[0]] - truth_best)
    actual_elite = float(truth[proxy_order[:k]].mean() - truth_elite_mean)

    # Each row is one matched-marginal null proxy: simulator truth plus the
    # exact residual multiset randomly reassigned within this population.
    perm_indices = np.empty((n_permutations, n), dtype=np.int32)
    for b in range(n_permutations):
        perm_indices[b] = rng.permutation(n)
    shuffled_proxy = truth[None, :] + residual[perm_indices]

    null_argmin_idx = np.argmin(shuffled_proxy, axis=1)
    null_argmin = truth[null_argmin_idx] - truth_best
    null_elite_idx = np.argpartition(shuffled_proxy, kth=k - 1, axis=1)[:, :k]
    null_elite = truth[null_elite_idx].mean(axis=1) - truth_elite_mean

    out: dict[str, float] = {
        "n_candidates": float(n),
        "topk": float(k),
        "residual_mean_m": float(residual.mean()),
        "residual_std_m": float(residual.std()),
    }
    for metric, actual, null in (
        ("argmin_regret_m", actual_argmin, null_argmin),
        ("elite_regret_m", actual_elite, null_elite),
    ):
        lo, hi = np.percentile(null, [2.5, 97.5])
        null_mean = float(null.mean())
        out[f"actual_{metric}"] = actual
        out[f"null_mean_{metric}"] = null_mean
        out[f"actual_minus_null_{metric}"] = actual - null_mean
        out[f"null_p025_{metric}"] = float(lo)
        out[f"null_p975_{metric}"] = float(hi)
        out[f"null_tail_prob_ge_actual_{metric}"] = float(
            (1 + np.count_nonzero(null >= actual)) / (n_permutations + 1)
        )
    return out


def _bootstrap_seed_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, estimate, estimate
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def _aggregate(
    populations: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, object]] = []
    for key, g in populations.groupby(["source", "task", "iter"], sort=False):
        for metric in REGRET_METRICS:
            columns = {
                "actual": f"actual_{metric}",
                "matched_noise_null": f"null_mean_{metric}",
                "actual_minus_null": f"actual_minus_null_{metric}",
            }
            seed_means = g.groupby("seed", sort=False)[list(columns.values())].mean()
            for statistic, column in columns.items():
                est, lo, hi = _bootstrap_seed_mean(
                    seed_means[column].to_numpy(float),
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                )
                rows.append({
                    "source": key[0],
                    "task": key[1],
                    "iter": key[2],
                    "n_seed": len(seed_means),
                    "metric": metric,
                    "statistic": statistic,
                    "estimate": est,
                    "ci_lo": lo,
                    "ci_hi": hi,
                })
    return pd.DataFrame(rows)


def _csv_block(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_csv(index=False, float_format="%.5f") + "```"


def _write_report(summary: pd.DataFrame, out: Path) -> None:
    first = summary[
        summary["iter"]
        == summary.groupby(["source", "task"])["iter"].transform("min")
    ]
    final = summary[
        summary["iter"]
        == summary.groupby(["source", "task"])["iter"].transform("max")
    ]
    lines = [
        "# Matched-residual permutation null", "",
        "For each identical simulator-rolled population, the actual state-probe",
        "residual multiset is shuffled across candidates. This preserves residual",
        "magnitude and generic noisy-proxy winner's curse while breaking structured",
        "alignment between residuals and candidate geometry. `actual_minus_null > 0`",
        "means the real representation--readout cost causes more physical selection",
        "regret than its own matched-marginal shuffled-noise control.", "",
        "## First CEM population", "", _csv_block(first), "",
        "## Final CEM population", "", _csv_block(final), "",
        "## Interpretation gates", "",
        "- A CI-clean positive `actual_minus_null` supports structured residual",
        "  alignment beyond generic optimization over a noisy cost.",
        "- A CI containing zero means the current data do not distinguish the",
        "  observed misselection from a matched generic optimizer's curse.",
        "- This audit concerns the representation--readout--cost composition; it",
        "  does not assign the residual specifically to the encoder.",
        "- The null is a within-population selection test, not a closed-loop policy",
        "  comparison and not a claim that shuffled noise is deployment-realistic.",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--topk-frac", type=float, default=0.1)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--permutation-seed", type=int, default=56001)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=56002)
    ap.add_argument(
        "--out-prefix", default="results/cem_residual_permutation_null"
    )
    args = ap.parse_args()

    required = {
        "task", "cost", "seed", "replan", "iter", "candidate",
        "proxy_cost", "true_shaped_cost",
    }
    frames = []
    for path in args.candidates:
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        if set(df["cost"].unique()) != {"stateprobe"}:
            raise ValueError(
                f"{path}: matched-unit residual null is defined only for stateprobe"
            )
        df.insert(0, "source", _source(path))
        frames.append(df)
    candidates = pd.concat(frames, ignore_index=True)

    rows = []
    for key, g in candidates.groupby(KEYS, sort=False):
        rng = np.random.default_rng(_population_seed(args.permutation_seed, key))
        rows.append({
            **dict(zip(KEYS, key)),
            **_summarize_population(
                g,
                topk_frac=args.topk_frac,
                n_permutations=args.n_permutations,
                rng=rng,
            ),
        })
    populations = pd.DataFrame(rows)
    summary = _aggregate(
        populations,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    populations.to_csv(
        prefix.with_name(prefix.name + "_populations.csv"), index=False
    )
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    _write_report(summary, prefix.with_suffix(".md"))
    print(f"wrote matched-residual permutation audit under {prefix.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
