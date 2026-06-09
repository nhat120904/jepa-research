"""Correlate planning Action Error with CRA_eff — the evidence that action-grounding
failure (low CRA_eff) actually drives planning failure (high Action Error).

Headline = per-transition correlation (thousands of paired points) between
``action_error`` and ``cra_eff_correct`` from ``08_planning_probe``. Expectation:
a clear **negative** correlation (worse ranking ↔ worse plan). Per-regime means
are reported as a secondary, coarser view, cross-checked against the per-regime
CRA_eff from the main diagnostic CSV.

Numpy-only (Pearson + Spearman via rank-Pearson + a permutation p-value), plus a
scatter figure. Writes a markdown section that the decision report links.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    return _pearson(xr, yr)


def _perm_pvalue(x, y, observed, n_perm=5000, seed=0):
    """Two-sided permutation p-value for Spearman r (shuffle y, recompute)."""
    if np.isnan(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    count = 0
    for _ in range(n_perm):
        if abs(_spearman(x, rng.permutation(y))) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planning_csv", default="results/droid_planning.csv")
    ap.add_argument("--pertrans", default="results/droid_planning_pertrans.npz")
    ap.add_argument("--diagnostic_csv", default="results/droid_diagnostic.csv")
    ap.add_argument("--out_md", default="results/planning_correlation.md")
    ap.add_argument("--out_fig", default="results/figures/figure_c_planning_vs_cra.pdf")
    args = ap.parse_args()

    npz = np.load(args.pertrans, allow_pickle=True)
    err = np.asarray(npz["action_error"], float)
    cra = np.asarray(npz["cra_eff_correct"], float)
    regime = np.asarray(npz["regime"])
    horizon = np.asarray(npz["horizon"])

    lines = ["# Planning Action-Error vs CRA_eff correlation", ""]

    # ---- headline: per-transition correlation (overall + per horizon) ----
    lines.append("## Per-transition correlation (headline)")
    lines.append("")
    lines.append("| subset | n | Pearson r | Spearman r | perm p (Spearman) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label, mask in [("all", np.ones_like(err, bool))] + \
            [(f"H={h}", horizon == h) for h in sorted(set(horizon.tolist()))]:
        x, y = err[mask], cra[mask]
        sp = _spearman(x, y)
        p = _perm_pvalue(x, y, sp) if mask.sum() >= 3 else float("nan")
        lines.append(f"| {label} | {int(mask.sum())} | {_pearson(x, y):.3f} | {sp:.3f} | {p:.4f} |")
    lines.append("")
    lines.append("Expected sign: **negative** (higher Action Error ↔ lower CRA_eff).")
    lines.append(
        f"Observed CRA_eff positives: **{int(cra.sum())}/{len(cra)} "
        f"({cra.mean():.1%})**; this severe class imbalance limits correlation power."
    )
    lines.append("")

    # ---- per-regime view ----
    pl = pd.read_csv(args.planning_csv)
    run_cols = [
        "max_planning_transitions",
        "cem_num_samples",
        "cem_iterations",
        "cem_num_elites",
    ]
    if all(col in pl.columns for col in run_cols) and not pl.empty:
        run = pl.iloc[0]
        lines.append("## Run configuration")
        lines.append("")
        lines.append(
            f"- Maximum transitions per regime/horizon: "
            f"{int(run['max_planning_transitions'])}"
        )
        lines.append(
            f"- CEM: {int(run['cem_num_samples'])} samples, "
            f"{int(run['cem_iterations'])} iterations, "
            f"{int(run['cem_num_elites'])} elites"
        )
        lines.append("")
    lines.append("## Per-regime means")
    lines.append("")
    lines.append("| horizon | regime | n | Action Error | Action Score | CRA_eff (plan probe) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in pl.sort_values(["horizon", "regime"]).iterrows():
        score = r.get("action_score", float("nan"))
        lines.append(f"| {int(r['horizon'])} | {r['regime']} | {int(r['n_planned'])} | "
                     f"{r['action_error']:.4f} | {score:.3f} | {r['cra_eff']:.3f} |")
    lines.append("")

    # Regime-level correlation (coarse: few points) between CRA_eff and Action Error.
    for h in sorted(pl["horizon"].unique()):
        sub = pl[pl["horizon"] == h]
        if len(sub) >= 3:
            lines.append(f"- Regime-level (H={int(h)}, {len(sub)} regimes) Spearman(Action Error, "
                         f"CRA_eff) = {_spearman(sub['action_error'], sub['cra_eff']):.3f}")
    lines.append("")

    # Cross-check against the main diagnostic CRA_eff (hard_nn) if available.
    diag_path = Path(args.diagnostic_csv)
    if diag_path.exists():
        d = pd.read_csv(diag_path)
        d = d[(d["strategy"] == "hard_nn") & (d["status"] == "ok")]
        if not d.empty:
            lines.append("## Cross-check: main-diagnostic hard_nn CRA_eff per regime")
            lines.append("")
            lines.append("| regime | CRA_eff (05) |")
            lines.append("| --- | --- |")
            for reg, g in d.groupby("regime"):
                lines.append(f"| {reg} | {g['cra_top1_eff'].mean():.3f} |")
            lines.append("")

    # ---- scatter figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        regimes = sorted(set(regime.tolist()))
        for reg in regimes:
            m = regime == reg
            ax.scatter(err[m], cra[m], s=8, alpha=0.4, label=str(reg))
        ax.set_xlabel("Planning Action Error (lower = better plan)")
        ax.set_ylabel("CRA_eff correct (per transition)")
        ax.set_title("Action-grounding (CRA_eff) vs planning quality")
        ax.legend(fontsize=7)
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out_fig)
        fig_path = Path(args.out_fig)
        md_parent = Path(args.out_md).parent
        fig_ref = fig_path.relative_to(md_parent) if fig_path.is_relative_to(md_parent) else fig_path
        lines.append(f"![Figure C]({fig_ref.as_posix()})")
    except Exception as e:  # figure is optional
        lines.append(f"_(figure skipped: {e})_")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_md}")
    print("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
