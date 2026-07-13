"""Paired analysis for the eight ACID-style MetaWorld evaluation cells.

Expected cells are 2 checkpoints x 2 dynamics (learned/oracle) x 2 contact
tasks.  Each input CSV contains terminal and ACID rows on identical seeds.  The
script refuses duplicate, incomplete, or mismatched pairs and writes a compact
CSV plus a Markdown report.  Run through the CPU Slurm wrapper, not on login.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FILE_RE = re.compile(
    r"^acid_(?P<model>.+_metaworld)_(?P<dynamics>learned|oracle)_"
    r"(?P<task>mw-.+)_seed(?P<seed0>\d+)_n(?P<n>\d+)\.csv$"
)

DEFAULT_MODELS = ("dino_wm_metaworld", "jepa_wm_metaworld")
DEFAULT_DYNAMICS = ("learned", "oracle")
DEFAULT_TASKS = ("mw-push", "mw-pick-place")


def exact_mcnemar_p(n_terminal_only: int, n_acid_only: int) -> float:
    """Two-sided exact McNemar p-value (binomial conditional test)."""
    discordant = int(n_terminal_only) + int(n_acid_only)
    if discordant == 0:
        return 1.0
    k = min(int(n_terminal_only), int(n_acid_only))
    lower = sum(math.comb(discordant, i) for i in range(k + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * lower)


def paired_bootstrap_ci(
    difference: np.ndarray,
    *,
    n_resamples: int = 20000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean paired difference and percentile CI, resampling seed pairs."""
    diff = np.asarray(difference, dtype=float)
    if diff.ndim != 1 or len(diff) == 0 or not np.isfinite(diff).all():
        raise ValueError("paired differences must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    # Chunk to bound RAM when n_resamples or paired n is increased.
    chunk = 4096
    for lo in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - lo)
        idx = rng.integers(0, len(diff), size=(size, len(diff)))
        means[lo : lo + size] = diff[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(diff.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def parse_cell_path(path: Path) -> dict:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"unexpected ACID result filename: {path.name}")
    out = match.groupdict()
    out["seed0"] = int(out["seed0"])
    out["n"] = int(out["n"])
    return out


def load_paired_cell(path: Path) -> tuple[dict, pd.DataFrame]:
    meta = parse_cell_path(path)
    frame = pd.read_csv(path)
    required = {
        "task", "seed", "dynamics", "arm", "success", "success_end",
        "obj_goal_dist", "ee_dist", "final_state_dist", "mean_sigma_acid",
        "mean_acid_weight", "idm_architecture", "idm_split_sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if set(frame["task"].astype(str)) != {meta["task"]}:
        raise ValueError(f"{path.name}: task column disagrees with filename")
    if set(frame["dynamics"].astype(str)) != {meta["dynamics"]}:
        raise ValueError(f"{path.name}: dynamics column disagrees with filename")
    if frame.duplicated(["seed", "arm"]).any():
        raise ValueError(f"{path.name}: duplicate (seed, arm) rows")
    arms = set(frame["arm"].astype(str))
    if arms != {"terminal", "acid"}:
        raise ValueError(f"{path.name}: expected terminal+acid, found {sorted(arms)}")
    terminal = frame[frame.arm == "terminal"].set_index("seed").sort_index()
    acid = frame[frame.arm == "acid"].set_index("seed").sort_index()
    if not terminal.index.equals(acid.index):
        raise ValueError(f"{path.name}: terminal/ACID seed sets differ")
    if len(terminal) != meta["n"]:
        raise ValueError(f"{path.name}: filename says n={meta['n']}, paired rows={len(terminal)}")
    expected = np.arange(meta["seed0"], meta["seed0"] + meta["n"])
    if not np.array_equal(terminal.index.to_numpy(dtype=int), expected):
        raise ValueError(f"{path.name}: seeds are not the locked contiguous range")
    paired = terminal.add_suffix("_terminal").join(acid.add_suffix("_acid"), how="inner")
    return meta, paired


def checkpoint_validity(checkpoint_dir: Path, model: str) -> dict:
    path = checkpoint_dir / f"acid_idm_{model}_split0.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing IDM checkpoint metadata: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    test_mse = float(ckpt.get("test_action_mse", float("nan")))
    mean_mse = float(ckpt.get("test_constant_mean_mse", float("nan")))
    ratio = test_mse / mean_mse if np.isfinite(mean_mse) and mean_mse > 0 else float("nan")
    return {
        "idm_architecture_ckpt": ckpt.get("architecture", "unknown"),
        "idm_split_sha256_ckpt": ckpt.get("split_manifest_sha256", "unknown"),
        "idm_test_action_mse": test_mse,
        "idm_constant_mean_test_mse": mean_mse,
        "idm_test_mse_ratio": ratio,
        "idm_beats_constant_mean": bool(np.isfinite(ratio) and ratio < 1.0),
        "idm_official_architecture": False,
        "idm_approximation_note": ckpt.get("official_idm_difference", "missing"),
    }


def analyze_cell(meta, paired, validity, *, n_resamples, ci, seed):
    row = {k: meta[k] for k in ("model", "dynamics", "task", "seed0", "n")}
    for metric in ("success_end", "success"):
        terminal = paired[f"{metric}_terminal"].to_numpy(dtype=int)
        acid = paired[f"{metric}_acid"].to_numpy(dtype=int)
        diff, lo, hi = paired_bootstrap_ci(
            acid - terminal, n_resamples=n_resamples, ci=ci, seed=seed
        )
        terminal_only = int(np.sum((terminal == 1) & (acid == 0)))
        acid_only = int(np.sum((terminal == 0) & (acid == 1)))
        prefix = "end" if metric == "success_end" else "any"
        row.update({
            f"terminal_{prefix}_rate": float(terminal.mean()),
            f"acid_{prefix}_rate": float(acid.mean()),
            f"delta_{prefix}_acid_minus_terminal": diff,
            f"delta_{prefix}_lo": lo,
            f"delta_{prefix}_hi": hi,
            f"n_terminal_only_{prefix}": terminal_only,
            f"n_acid_only_{prefix}": acid_only,
            f"mcnemar_exact_p_{prefix}": exact_mcnemar_p(terminal_only, acid_only),
        })
    for offset, metric in enumerate(("obj_goal_dist", "ee_dist", "final_state_dist"), 1):
        delta = (
            paired[f"{metric}_acid"].to_numpy(dtype=float)
            - paired[f"{metric}_terminal"].to_numpy(dtype=float)
        )
        point, lo, hi = paired_bootstrap_ci(
            delta, n_resamples=n_resamples, ci=ci, seed=seed + offset
        )
        row.update({
            f"terminal_mean_{metric}": float(paired[f"{metric}_terminal"].mean()),
            f"acid_mean_{metric}": float(paired[f"{metric}_acid"].mean()),
            f"delta_{metric}_acid_minus_terminal": point,
            f"delta_{metric}_lo": lo,
            f"delta_{metric}_hi": hi,
        })
    acid_sigma = paired["mean_sigma_acid_acid"].to_numpy(dtype=float)
    acid_weight = paired["mean_acid_weight_acid"].to_numpy(dtype=float)
    architecture_values = set(paired["idm_architecture_acid"].astype(str))
    hash_values = set(paired["idm_split_sha256_acid"].astype(str))
    metadata_matches = (
        architecture_values == {str(validity["idm_architecture_ckpt"])}
        and hash_values == {str(validity["idm_split_sha256_ckpt"])}
    )
    row.update(validity)
    row.update({
        "mean_planning_sigma_acid": float(np.nanmean(acid_sigma)),
        "min_planning_sigma_acid": float(np.nanmin(acid_sigma)),
        "mean_adaptive_weight": float(np.nanmean(acid_weight)),
        "planning_consistency_has_spread": bool(np.isfinite(acid_sigma).all() and np.any(acid_sigma > 1e-8)),
        "checkpoint_csv_metadata_match": bool(metadata_matches),
    })
    row["approximation_valid_for_comparison"] = bool(
        row["idm_beats_constant_mean"]
        and row["planning_consistency_has_spread"]
        and row["checkpoint_csv_metadata_match"]
    )
    return row


def render_report(summary: pd.DataFrame, paths: list[Path]) -> str:
    lines = [
        "# ACID-style paired baseline analysis",
        "",
        "Primary endpoint is strict end-of-episode success. Deltas are ACID minus terminal; "
        "negative endpoint-distance deltas are improvements.",
        "",
        "## Paired results",
        "",
        "| Model | Dynamics | Task | n | Terminal end | ACID end | Paired Δ [CI] | "
        "Exact McNemar p | Δ object distance [CI] | Approx. valid? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.dynamics} | {row.task} | {row.n} | "
            f"{row.terminal_end_rate:.3f} | {row.acid_end_rate:.3f} | "
            f"{row.delta_end_acid_minus_terminal:+.3f} "
            f"[{row.delta_end_lo:+.3f}, {row.delta_end_hi:+.3f}] | "
            f"{row.mcnemar_exact_p_end:.4g} | "
            f"{row.delta_obj_goal_dist_acid_minus_terminal:+.4f} "
            f"[{row.delta_obj_goal_dist_lo:+.4f}, {row.delta_obj_goal_dist_hi:+.4f}] | "
            f"{'yes' if row.approximation_valid_for_comparison else 'NO'} |"
        )
    lines += [
        "",
        "## Approximation validity",
        "",
        "This is an ACID-style deterministic pooled-MLP verifier, not the paper's unreleased "
        "flow-matching transformer. `Approx. valid?` requires (i) held-out IDM MSE below the "
        "constant-mean predictor, (ii) nonzero consistency-cost spread in planning pools, and "
        "(iii) matching checkpoint/CSV architecture and split hashes.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.model}/{row.dynamics}/{row.task}`: IDM MSE ratio "
            f"{row.idm_test_mse_ratio:.3f}; mean sigma_a={row.mean_planning_sigma_acid:.6g}; "
            f"mean adaptive weight={row.mean_adaptive_weight:.6g}; "
            f"valid={row.approximation_valid_for_comparison}."
        )
    lines += ["", "## Inputs", ""] + [f"- `{p}`" for p in paths]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--seed0", type=int, default=22000)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--n-resamples", type=int, default=20000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--out-csv", default="results/acid_paired_summary.csv")
    ap.add_argument("--out-report", default="results/acid_paired_report.md")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    expected = {
        (model, dynamics, task)
        for model in DEFAULT_MODELS
        for dynamics in DEFAULT_DYNAMICS
        for task in DEFAULT_TASKS
    }
    paths = sorted(results_dir.glob(f"acid_*_seed{args.seed0}_n{args.episodes}.csv"))
    cells = {}
    for path in paths:
        meta = parse_cell_path(path)
        key = (meta["model"], meta["dynamics"], meta["task"])
        if key not in expected:
            continue
        if key in cells:
            raise ValueError(f"duplicate result file for cell {key}")
        cells[key] = path
    missing = expected - set(cells)
    if missing:
        raise FileNotFoundError(f"missing ACID evaluation cells: {sorted(missing)}")

    validity = {
        model: checkpoint_validity(Path(args.checkpoint_dir), model)
        for model in DEFAULT_MODELS
    }
    rows = []
    used = []
    for index, key in enumerate(sorted(expected)):
        path = cells[key]
        meta, paired = load_paired_cell(path)
        rows.append(analyze_cell(
            meta, paired, validity[meta["model"]],
            n_resamples=args.n_resamples, ci=args.ci, seed=100 + index,
        ))
        used.append(path)
    summary = pd.DataFrame(rows).sort_values(["model", "dynamics", "task"])
    out_csv, out_report = Path(args.out_csv), Path(args.out_report)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    out_report.write_text(render_report(summary, used))
    print(f"wrote {out_csv} and {out_report} ({len(summary)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
