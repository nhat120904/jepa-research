"""Analyze the locked n>=64 oracle-ladder confirmation.

This script is intentionally separate from experiment execution. Run it on a
Slurm compute node after every cell of ``slurm_confirmatory_locked.sh`` has
completed. It reports per-arm Wilson intervals and paired, seed-matched
contrasts for success and final object-to-goal distance.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


NAME_RE = re.compile(
    r"confirmatory_(?P<model>none|dino_wm_metaworld|jepa_wm_metaworld)_"
    r"(?P<cost>oracle|l2|stateprobe)_(?P<task>mw-push|mw-pick-place)_"
    r"seed(?P<seed0>\d+)_n(?P<n>\d+)\.csv$"
)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def paired_ci(delta: np.ndarray, *, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = delta[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_cells(results: Path, seed0: int, episodes: int) -> dict[tuple[str, str, str], pd.DataFrame]:
    cells: dict[tuple[str, str, str], pd.DataFrame] = {}
    for path in sorted(results.glob("confirmatory_*.csv")):
        match = NAME_RE.fullmatch(path.name)
        if not match:
            continue
        meta = match.groupdict()
        if int(meta["seed0"]) != seed0 or int(meta["n"]) != episodes:
            continue
        df = pd.read_csv(path)
        required = {"seed", "success_end", "obj_goal_dist"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)}")
        if len(df) != episodes:
            raise ValueError(f"{path} has {len(df)} rows; expected {episodes}")
        if df["seed"].duplicated().any():
            raise ValueError(f"{path} has duplicate seeds")
        key = (meta["model"], meta["cost"], meta["task"])
        cells[key] = df.sort_values("seed").reset_index(drop=True)
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--out-summary", type=Path, default=Path("results/confirmatory_summary.csv"))
    ap.add_argument("--out-contrasts", type=Path, default=Path("results/confirmatory_contrasts.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("results/confirmatory_report.md"))
    args = ap.parse_args()

    cells = load_cells(args.results, args.seed0, args.episodes)
    expected = {
        ("none", "oracle", task) for task in ("mw-push", "mw-pick-place")
    } | {
        (model, cost, task)
        for model in ("dino_wm_metaworld", "jepa_wm_metaworld")
        for cost in ("l2", "stateprobe")
        for task in ("mw-push", "mw-pick-place")
    }
    missing = expected - set(cells)
    if missing:
        raise SystemExit(f"confirmatory cells incomplete: {sorted(missing)}")

    summaries = []
    for (model, cost, task), df in sorted(cells.items()):
        k, n = int(df["success_end"].sum()), len(df)
        lo, hi = wilson(k, n)
        summaries.append({
            "model": model,
            "cost": cost,
            "task": task,
            "seed0": args.seed0,
            "n": n,
            "success_k": k,
            "success_rate": k / n,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "obj_goal_mean": float(df["obj_goal_dist"].mean()),
            "obj_goal_median": float(df["obj_goal_dist"].median()),
        })
    summary = pd.DataFrame(summaries)

    pairs = []
    for task in ("mw-push", "mw-pick-place"):
        oracle = ("none", "oracle", task)
        for model in ("dino_wm_metaworld", "jepa_wm_metaworld"):
            for cost in ("l2", "stateprobe"):
                pairs.append((oracle, (model, cost, task), "oracle_minus_latent"))
            pairs.append(((model, "stateprobe", task), (model, "l2", task),
                          "stateprobe_minus_l2"))

    contrasts = []
    for idx, (left_key, right_key, family) in enumerate(pairs):
        left = cells[left_key].set_index("seed")
        right = cells[right_key].set_index("seed")
        seeds = left.index.intersection(right.index)
        if len(seeds) != args.episodes:
            raise ValueError(f"seed mismatch: {left_key} vs {right_key}")
        ls = left.loc[seeds, "success_end"].to_numpy(float)
        rs = right.loc[seeds, "success_end"].to_numpy(float)
        success_delta = ls - rs
        suc_lo, suc_hi = paired_ci(success_delta, seed=100 + idx, n_boot=args.n_boot)
        discord_left = int(((ls == 1) & (rs == 0)).sum())
        discord_right = int(((ls == 0) & (rs == 1)).sum())
        discord_n = discord_left + discord_right
        mcnemar_p = (float(binomtest(discord_left, discord_n, 0.5).pvalue)
                     if discord_n else 1.0)

        ld = left.loc[seeds, "obj_goal_dist"].to_numpy(float)
        rd = right.loc[seeds, "obj_goal_dist"].to_numpy(float)
        dist_delta = ld - rd
        dist_lo, dist_hi = paired_ci(dist_delta, seed=1000 + idx, n_boot=args.n_boot)
        contrasts.append({
            "family": family,
            "task": left_key[2],
            "left_model": left_key[0],
            "left_cost": left_key[1],
            "right_model": right_key[0],
            "right_cost": right_key[1],
            "n_paired": len(seeds),
            "success_delta": float(success_delta.mean()),
            "success_delta_lo": suc_lo,
            "success_delta_hi": suc_hi,
            "discord_left_only": discord_left,
            "discord_right_only": discord_right,
            "mcnemar_exact_p": mcnemar_p,
            "obj_goal_delta_mean": float(dist_delta.mean()),
            "obj_goal_delta_lo": dist_lo,
            "obj_goal_delta_hi": dist_hi,
        })
    contrast = pd.DataFrame(contrasts)

    for path in (args.out_summary, args.out_contrasts, args.out_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)
    contrast.to_csv(args.out_contrasts, index=False)

    lines = [
        "# Locked confirmatory oracle ladder",
        "",
        f"Seeds `{args.seed0}..{args.seed0 + args.episodes - 1}`; "
        f"`n={args.episodes}` paired episodes per cell.",
        "",
        "## Per-arm success",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Paired contrasts",
        "",
        "Deltas are left minus right. Negative object-distance delta favors the left arm.",
        "",
        "```text",
        contrast.to_string(index=False),
        "```",
        "",
    ]
    args.out_md.write_text("\n".join(lines))
    print(f"wrote {args.out_summary}, {args.out_contrasts}, {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
