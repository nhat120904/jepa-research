"""Gate 1 analysis (goal-marginalization design, 2026-08-11) — task-SNR law.

Joins scripts/80's per-episode SNR against the already-measured plain-`l2`
oracle-arm outcome for each task (existing artifacts, not re-run here), and
reports whether mean SNR predicts final object-goal distance across tasks.

Label sources are explicit, not auto-discovered, so this script fails loudly
if an expected file/column is missing rather than silently mis-joining:

  - mw-push, mw-pick-place, mw-reach:            results/metaworld_latent_oracle.csv
  - mw-button-press/drawer-close/window-close:    results/task_breadth_ladder_{task}_l2_seed70000_n16.csv
  - mw-faucet-open/plate-slide/soccer:            results/task_breadth_ladder2_{task}_l2_seed71000_n16.csv

Excluded by design (weak/failed oracle positive control, per
docs/plans/2026-08-11-goal-marginalization-design.md): mw-door-open,
mw-assembly, mw-box-close, mw-shelf-place, mw-lever-pull.

    python scripts/81_analyze_task_snr.py --snr results/task_snr_pilot.csv \
        --out results/task_snr_law.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

LABEL_SOURCES = {
    "mw-push": ("results/metaworld_latent_oracle.csv", "task", "mw-push"),
    "mw-pick-place": ("results/metaworld_latent_oracle.csv", "task", "mw-pick-place"),
    "mw-reach": ("results/metaworld_latent_oracle.csv", "task", "mw-reach"),
    "mw-button-press": ("results/task_breadth_ladder_mw-button-press_l2_seed70000_n16.csv", None, None),
    "mw-drawer-close": ("results/task_breadth_ladder_mw-drawer-close_l2_seed70000_n16.csv", None, None),
    "mw-window-close": ("results/task_breadth_ladder_mw-window-close_l2_seed70000_n16.csv", None, None),
    "mw-faucet-open": ("results/task_breadth_ladder2_mw-faucet-open_l2_seed71000_n16.csv", None, None),
    "mw-plate-slide": ("results/task_breadth_ladder2_mw-plate-slide_l2_seed71000_n16.csv", None, None),
    "mw-soccer": ("results/task_breadth_ladder2_mw-soccer_l2_seed71000_n16.csv", None, None),
}

EXCLUDED = ["mw-door-open", "mw-assembly", "mw-box-close", "mw-shelf-place", "mw-lever-pull"]


def _load_label(task: str) -> dict:
    path, filter_col, filter_val = LABEL_SOURCES[task]
    full = ROOT / "diagnosis" / path
    if not full.exists():
        full = ROOT / path
    if not full.exists():
        raise FileNotFoundError(f"label source for {task} not found: {path}")
    df = pd.read_csv(full)
    if filter_col is not None:
        df = df[df[filter_col] == filter_val]
    if len(df) == 0:
        raise ValueError(f"label source for {task} matched zero rows: {path}")
    return {
        "n_label": len(df),
        "mean_obj_goal_dist": float(df["obj_goal_dist"].mean()),
        "mean_success_end": float(df["success_end"].mean()),
    }


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(x)
    point = float(stats.spearmanr(x, y).correlation)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xs, ys = x[idx], y[idx]
        if np.all(xs == xs[0]) or np.all(ys == ys[0]):
            draws[i] = np.nan
            continue
        draws[i] = stats.spearmanr(xs, ys).correlation
    draws = draws[~np.isnan(draws)]
    if len(draws) == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(draws, 2.5))
    hi = float(np.percentile(draws, 97.5))
    return point, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", default="results/task_snr_pilot.csv")
    ap.add_argument("--out", default="results/task_snr_law.md")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    snr_df = pd.read_csv(args.snr)
    tasks = sorted(t for t in snr_df["task"].unique() if t in LABEL_SOURCES)
    missing = sorted(set(snr_df["task"].unique()) - set(LABEL_SOURCES) - set(EXCLUDED))
    if missing:
        print(f"WARNING: no label source registered for tasks {missing}; skipping them")

    rows = []
    for task in tasks:
        sub = snr_df[snr_df["task"] == task]
        clean = sub[sub["no_contact_detected"] == 0]
        label = _load_label(task)
        rows.append({
            "task": task,
            "n_episodes": len(sub),
            "n_no_contact": int(sub["no_contact_detected"].sum()),
            "mean_snr": float(clean["snr"].mean()) if len(clean) else float("nan"),
            "mean_snr_all": float(sub["snr"].mean()),
            **label,
        })
    table = pd.DataFrame(rows).sort_values("mean_snr")

    x = table["mean_snr"].to_numpy()
    y = table["mean_obj_goal_dist"].to_numpy()
    rho, lo, hi = _bootstrap_spearman(x, y, args.bootstrap, args.seed)

    # Leave-one-task-out sign stability.
    loo_signs = []
    for i in range(len(table)):
        mask = np.arange(len(table)) != i
        if mask.sum() < 3:
            continue
        r = stats.spearmanr(x[mask], y[mask]).correlation
        loo_signs.append(np.sign(r))
    sign_stable = len(set(loo_signs)) == 1 if loo_signs else False

    verdict = "PASS" if (np.sign(rho) > 0 and abs(rho) >= 0.6 and sign_stable) else "FAIL"

    out = ROOT / "diagnosis" / args.out
    if not out.parent.exists():
        out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# Gate 1 — task-level SNR law\n\n")
        f.write(f"Bootstrap n={args.bootstrap}, seed={args.seed}. "
                f"Excluded (weak oracle positive control): {', '.join(EXCLUDED)}.\n\n")
        f.write("| task | n | no_contact | mean SNR (clean) | mean SNR (all) | "
                "l2 mean obj_goal_dist (m) | l2 success_end |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in table.iterrows():
            f.write(f"| {r['task']} | {int(r['n_episodes'])} | {int(r['n_no_contact'])} | "
                     f"{r['mean_snr']:.3f} | {r['mean_snr_all']:.3f} | "
                     f"{r['mean_obj_goal_dist']:.4f} | {r['mean_success_end']:.3f} |\n")
        f.write(f"\nSpearman(mean SNR, l2 mean obj_goal_dist) = **{rho:.3f}** "
                 f"[{lo:.3f}, {hi:.3f}] (task-level bootstrap, n_tasks={len(table)}).\n")
        stability_word = "stable" if sign_stable else "NOT stable"
        f.write(f"Leave-one-task-out sign stability: {stability_word} ({loo_signs}).\n\n")
        f.write(f"**Verdict: {verdict}** "
                f"(pass requires rho>0, |rho|>=0.6, sign-stable under leave-one-out).\n")
    print(table.to_string(index=False))
    print(f"\nSpearman rho={rho:.3f} [{lo:.3f}, {hi:.3f}]  verdict={verdict}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
