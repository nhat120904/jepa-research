"""Seed-clustered stateprobe validation on optimizer-induced candidates.

This analysis uses the persisted same-population candidate dumps.  It does not
load a model or simulator.  For the first and final CEM populations at every
replan, it reports object/end-effector Euclidean decode error, Spearman
agreement between stateprobe and simulator-state shaped costs, and recall of
the reference top-10% set by the stateprobe top-10% set.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


NAME_RE = re.compile(
    r"cem_preselection_(?P<model>dino|jepa)_(?P<task>push|pick)_"
    r"stateprobe_candidates\.csv(?:\.gz)?$"
)
METRICS = (
    "object_error_cm",
    "hand_error_cm",
    "cost_spearman",
    "reference_top10_recall",
)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy(float)
    br = pd.Series(b).rank(method="average").to_numpy(float)
    return float(np.corrcoef(ar, br)[0, 1])


def summarize_population(group: pd.DataFrame, topk_frac: float) -> dict[str, float]:
    group = group.sort_values("candidate")
    n = len(group)
    k = max(1, int(np.ceil(topk_frac * n)))
    proxy = group.proxy_cost.to_numpy(float)
    truth = group.true_shaped_cost.to_numpy(float)
    obj = group.obj_decode_error_cm.to_numpy(float)
    hand = group.ee_decode_error_cm.to_numpy(float)
    arrays = (proxy, truth, obj, hand)
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("candidate costs and decode errors must be finite")
    ptop = set(np.argsort(proxy, kind="mergesort")[:k].tolist())
    ttop = set(np.argsort(truth, kind="mergesort")[:k].tolist())
    return {
        "n_candidate": n,
        "object_error_cm": float(obj.mean()),
        "hand_error_cm": float(hand.mean()),
        "cost_spearman": spearman(proxy, truth),
        "reference_top10_recall": float(len(ptop & ttop) / k),
    }


def bootstrap_seed_mean(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    draws = rng.choice(
        values, size=(n_bootstrap, len(values)), replace=True
    ).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def load_populations(paths: list[str], topk_frac: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    required = {
        "seed",
        "replan",
        "iter",
        "candidate",
        "proxy_cost",
        "true_shaped_cost",
        "obj_decode_error_cm",
        "ee_decode_error_cm",
    }
    for raw_path in paths:
        path = Path(raw_path)
        match = NAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unrecognized candidate filename: {path.name}")
        frame = pd.read_csv(path, usecols=sorted(required))
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for key, group in frame.groupby(["seed", "replan", "iter"], sort=False):
            row: dict[str, float | int | str] = {
                "model": match.group("model").upper(),
                "task": "pick-place" if match.group("task") == "pick" else "push",
                "seed": int(key[0]),
                "replan": int(key[1]),
                "iter": int(key[2]),
            }
            row.update(summarize_population(group, topk_frac))
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(
    populations: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows = []
    for cell_index, ((model, task), cell) in enumerate(
        populations.groupby(["model", "task"], sort=True)
    ):
        first_iter = int(cell["iter"].min())
        final_iter = int(cell["iter"].max())
        for stage_index, (stage, iteration) in enumerate(
            (("initial", first_iter), ("final", final_iter))
        ):
            selected = cell[cell["iter"] == iteration]
            population_counts = selected.groupby("seed").size()
            row: dict[str, float | int | str] = {
                "model": model,
                "task": task,
                "stage": stage,
                "iteration": iteration,
                "n_seed": int(selected.seed.nunique()),
                "n_population": int(len(selected)),
                "n_candidate": int(selected.n_candidate.sum()),
                "replans_per_seed_min": int(population_counts.min()),
                "replans_per_seed_max": int(population_counts.max()),
            }
            seed_means = selected.groupby("seed", sort=True)[list(METRICS)].mean()
            for metric_index, metric in enumerate(METRICS):
                rng = np.random.default_rng(
                    bootstrap_seed
                    + 10_007 * cell_index
                    + 101 * stage_index
                    + metric_index
                )
                estimate, lo, hi = bootstrap_seed_mean(
                    seed_means[metric].to_numpy(float),
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                )
                row[metric] = estimate
                row[f"{metric}_ci_lo"] = lo
                row[f"{metric}_ci_hi"] = hi
            rows.append(row)
    return pd.DataFrame(rows)


def interval(row: pd.Series, metric: str, scale: float = 1.0) -> str:
    return (
        f"{scale * row[metric]:.2f} "
        f"[{scale * row[f'{metric}_ci_lo']:.2f}, "
        f"{scale * row[f'{metric}_ci_hi']:.2f}]"
    )


def load_probe_metadata(
    object_specs: list[str], hand_specs: list[str]
) -> pd.DataFrame:
    import torch

    rows: dict[str, dict[str, float | str]] = {}
    for kind, specs in (("object", object_specs), ("hand", hand_specs)):
        for spec in specs:
            model, raw_path = spec.split("=", 1)
            try:
                payload = torch.load(
                    raw_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                payload = torch.load(raw_path, map_location="cpu")
            val_mse = float(payload["val_mse"])
            median = float(payload["v1_median"])
            rows.setdefault(model.upper(), {"model": model.upper()})
            rows[model.upper()][f"{kind}_coord_rmse_cm"] = 100 * np.sqrt(val_mse)
            rows[model.upper()][f"{kind}_median_euclidean_cm"] = 100 * median
    frame = pd.DataFrame(rows.values()).sort_values("model").reset_index(drop=True)
    required = {
        "model",
        "object_coord_rmse_cm",
        "object_median_euclidean_cm",
        "hand_coord_rmse_cm",
        "hand_median_euclidean_cm",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"incomplete probe metadata: missing {missing}")
    return frame


def write_report(
    path: Path, summary: pd.DataFrame, probe_metadata: pd.DataFrame
) -> None:
    lines = [
        "# Stateprobe validation on optimizer-induced candidates",
        "",
        "## Held-out expert-trajectory validation",
        "",
        "The immutable trajectory split contains 1,368 validation transitions.",
        "Per-coordinate RMSE is derived from the saved validation MSE; median",
        "Euclidean error is the checkpoint's persisted endpoint metric.",
        "",
        "| Model | Object coord. RMSE cm | Object median cm | "
        "Hand coord. RMSE cm | Hand median cm |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in probe_metadata.iterrows():
        lines.append(
            f"| {row.model} | {row.object_coord_rmse_cm:.2f} | "
            f"{row.object_median_euclidean_cm:.2f} | "
            f"{row.hand_coord_rmse_cm:.2f} | "
            f"{row.hand_median_euclidean_cm:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Optimizer-induced candidates",
            "",
        "Each row aggregates populations within episode seed and then bootstraps",
        "the 16 episode seeds (5,000 resamples by default). Errors are mean",
        "Euclidean endpoint errors. Cost rank agreement compares the deployed",
        "stateprobe shaped cost with the simulator-state shaped reference on the",
        "same candidates. Reference top-10% recall is intersection-over-10.",
        "",
        "| Model | Task | Stage | Populations | Candidates | Object error cm | "
        "Hand error cm | Cost Spearman | Reference top-10% recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.model} | {row.task} | {row.stage} | "
            f"{int(row.n_population)} | {int(row.n_candidate)} | "
            f"{interval(row, 'object_error_cm')} | "
            f"{interval(row, 'hand_error_cm')} | "
            f"{interval(row, 'cost_spearman')} | "
            f"{interval(row, 'reference_top10_recall')} |"
        )
    lines.extend(
        [
            "",
            "All cells contain 16 independent environment seeds and "
            "7 replans per seed at each reported iteration.",
            "The analysis measures one fixed probe construction/training seed; it",
            "does not establish architecture- or training-seed invariance.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--topk-frac", type=float, default=0.1)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    parser.add_argument("--object-probes", nargs="+", required=True)
    parser.add_argument("--hand-probes", nargs="+", required=True)
    parser.add_argument(
        "--out-prefix", default="results/stateprobe_cem_validation"
    )
    args = parser.parse_args()

    populations = load_populations(args.candidates, args.topk_frac)
    probe_metadata = load_probe_metadata(args.object_probes, args.hand_probes)
    summary = aggregate(
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
    probe_metadata.to_csv(
        prefix.with_name(prefix.name + "_expert_validation.csv"), index=False
    )
    write_report(prefix.with_suffix(".md"), summary, probe_metadata)
    print(f"wrote {prefix}.md and CSV artifacts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
