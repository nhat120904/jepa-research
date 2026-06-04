"""Produce figures + decision report from the diagnostic CSVs.

Generates:
    results/figures/figure_a_cra_per_regime.pdf
    results/figures/figure_b_metaworld_per_task.pdf
    results/figures/figure_c_correlation_planning.pdf  (if planning SR available)
    results/decision_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


HARD_TASKS = ["mw-peg-insert-side", "mw-assembly", "mw-hammer", "mw-stick-pull"]
CONTACT_REGIMES = ["gripper_actuation", "contact_manipulation"]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_a_cra_per_regime(df: pd.DataFrame, out_path: Path) -> None:
    """Per-regime CRA top-1 across models, hard_nn strategy."""
    sub = df[(df.strategy == "hard_nn") & (df.status == "ok")].copy()
    if sub.empty:
        return
    agg = (sub.groupby(["dataset", "model", "regime"])
              .agg(cra_top1=("cra_top1", "mean"),
                    lo=("cra_top1_lo", "mean"),
                    hi=("cra_top1_hi", "mean"))
              .reset_index())
    datasets = sorted(agg.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5),
                              squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = agg[agg.dataset == ds]
        sns.barplot(data=d, x="regime", y="cra_top1", hue="model", ax=ax,
                     order=["free_space", "pre_grasp",
                            "gripper_actuation", "contact_manipulation"])
        ax.axhline(1.0 / (16 + 1), ls="--", color="grey", label="chance")
        ax.set_ylim(0, 1)
        ax.set_title(f"{ds} (hard_nn negatives)")
        ax.set_ylabel("CRA top-1")
        ax.set_xlabel("regime")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def figure_b_metaworld_per_task(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[(df.dataset == "metaworld") &
              (df.strategy == "hard_nn") &
              (df.regime == "contact_manipulation") &
              (df.status == "ok")].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="task", columns="model", values="cra_top1")
    pivot = pivot.sort_values(by=pivot.columns[0])
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(1.0 / (16 + 1), ls="--", color="grey", label="chance")
    ax.set_ylabel("CRA top-1 (contact_manipulation)")
    ax.set_ylim(0, 1)
    ax.set_title("Metaworld per-task action grounding (hard_nn negatives)")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def figure_c_correlation(df: pd.DataFrame, planning_sr_csv: Optional[Path],
                          out_path: Path) -> None:
    if planning_sr_csv is None or not planning_sr_csv.exists():
        print(f"  [skip] planning SR table not found at {planning_sr_csv}")
        return
    sr = pd.read_csv(planning_sr_csv)  # columns: task, model, success_rate
    cra = (df[(df.strategy == "hard_nn") &
               (df.regime == "contact_manipulation") &
               (df.status == "ok")]
            .groupby(["task", "model"])["cra_top1"].mean()
            .reset_index())
    merged = sr.merge(cra, on=["task", "model"])
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, group in merged.groupby("model"):
        ax.scatter(group.cra_top1, group.success_rate, label=model, s=80)
    r = np.corrcoef(merged.cra_top1, merged.success_rate)[0, 1]
    ax.set_xlabel("CRA top-1 (contact_manipulation, hard_nn)")
    ax.set_ylabel("Planning success rate")
    ax.set_title(f"CRA vs planning SR  (Pearson r = {r:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def make_decision(metaworld_df: pd.DataFrame, droid_df: pd.DataFrame) -> tuple[str, str]:
    """Returns (decision_tag, justification_string).

    Primary signal: **effect-conditioned** CRA (``cra_top1_eff``) — CRA computed
    only on transitions where the state actually changed (||Δz|| > τ). A low raw
    1-step CRA in contact regimes can just reflect tiny one-step latent deltas;
    the effect-conditioned CRA is what tells us the model fails to use actions
    *when they matter*. The decision is CI-aware: ABANDON requires the upper CI
    bound to be confidently high, so a single noisy 1-step number cannot trigger
    abandonment.
    """
    def critical(df: pd.DataFrame, model: str, tasks: Optional[list] = None):
        f = df[(df.model == model) & (df.strategy == "hard_nn") &
                (df.regime.isin(CONTACT_REGIMES)) & (df.status == "ok")]
        if tasks is not None:
            f = f[f.task.isin(tasks)]
        if f.empty:
            return float("nan"), float("nan")
        # Prefer effect-conditioned CRA; fall back to raw where eff is unavailable.
        eff = f["cra_top1_eff"].fillna(f["cra_top1"]) if "cra_top1_eff" in f else f["cra_top1"]
        hi = (f["cra_top1_eff_hi"].fillna(f["cra_top1_hi"]) if "cra_top1_eff_hi" in f
              else f["cra_top1_hi"])
        return float(eff.mean()), float(hi.mean())

    def _pick_model(df: pd.DataFrame, preferred: list) -> Optional[str]:
        """Pick the diagnostic's strongest available baseline for the decision.
        Falls back to whatever model is in the CSV (e.g. the 8GB box runs
        dino_wm_droid, not jepa_wm_droid)."""
        if df.empty:
            return None
        present = set(df.model.unique())
        for m in preferred:
            if m in present:
                return m
        return sorted(present)[0] if present else None

    mw_model = _pick_model(metaworld_df, ["jepa_wm_metaworld", "dino_wm_metaworld"])
    dr_model = _pick_model(droid_df, ["jepa_wm_droid", "dino_wm_droid"])
    mw, mw_hi = critical(metaworld_df, mw_model, HARD_TASKS) if mw_model else (float("nan"),) * 2
    dr, dr_hi = critical(droid_df, dr_model) if dr_model else (float("nan"),) * 2

    summary = (f"effect-conditioned CRA — MW(hard,contact)={mw:.3f} [hi {mw_hi:.3f}]; "
               f"DROID(contact)={dr:.3f} [hi {dr_hi:.3f}]")

    have = lambda x: not np.isnan(x)
    # GO: strong pathology on the point estimate of the best baseline.
    if have(mw) and have(dr) and mw < 0.60 and dr < 0.65:
        return "GO", f"Strong action-grounding pathology: {summary}"
    # ABANDON: only if even the UPPER CI bound is high in both datasets.
    if have(mw_hi) and have(dr_hi) and mw_hi >= 0.85 and dr_hi >= 0.85:
        return "ABANDON", f"No measurable pathology (CIs high): {summary}"
    # CONDITIONAL_GO: moderate pathology in at least one dataset.
    if (have(mw) and mw < 0.75) or (have(dr) and dr < 0.75):
        return "CONDITIONAL_GO", f"Moderate pathology in at least one dataset: {summary}"
    return "PIVOT", f"Mixed signal; consider CTD / harder regimes: {summary}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(metaworld_df, droid_df, decision, justification, out_path: Path) -> None:
    lines = ["# CAI-JEPA Diagnostic Decision Report", "",
             f"**Decision:** {decision}", "",
             f"**Justification:** {justification}", "",
             "## Critical cells", ""]

    for name, df in [("Metaworld", metaworld_df), ("DROID", droid_df)]:
        if df.empty:
            continue
        sub = df[(df.strategy == "hard_nn") &
                  (df.regime.isin(CONTACT_REGIMES)) &
                  (df.status == "ok")]
        if sub.empty:
            continue
        agg_kwargs = dict(cra=("cra_top1", "mean"), lo=("cra_top1_lo", "mean"),
                          hi=("cra_top1_hi", "mean"), aug=("aug", "mean"), ecs=("ecs", "mean"))
        if "cra_top1_eff" in sub:
            agg_kwargs["cra_eff"] = ("cra_top1_eff", "mean")
        agg = sub.groupby(["model", "regime"]).agg(**agg_kwargs).reset_index()
        lines += [f"### {name}", "",
                  "| model | regime | CRA top-1 [95% CI] | CRA (effect-cond.) | AUG | ECS |",
                  "|---|---|---|---|---|---|"]
        for _, r in agg.iterrows():
            eff = f"{r.cra_eff:.3f}" if "cra_eff" in agg.columns and not np.isnan(r.cra_eff) else "n/a"
            lines.append(
                f"| {r.model} | {r.regime} | "
                f"{r.cra:.3f} [{r.lo:.3f}, {r.hi:.3f}] | {eff} | "
                f"{r.aug:+.4f} | {r.ecs:+.4f} |"
            )
        lines.append("")

    lines += ["## Figures", "",
              "![Figure A](figures/figure_a_cra_per_regime.pdf)",
              "![Figure B](figures/figure_b_metaworld_per_task.pdf)",
              "![Figure C](figures/figure_c_correlation_planning.pdf)", ""]

    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def safe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    print(f"  [info] no {path}")
    return pd.DataFrame()


def main(metaworld_csv: str, droid_csv: str, planning_sr_csv: Optional[str]) -> int:
    results_dir = Path(metaworld_csv).parent
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    mw = safe_read(Path(metaworld_csv))
    dr = safe_read(Path(droid_csv))
    combined = pd.concat([mw, dr], ignore_index=True)

    print("Generating figures...")
    figure_a_cra_per_regime(combined, figures_dir / "figure_a_cra_per_regime.pdf")
    if not mw.empty:
        figure_b_metaworld_per_task(mw, figures_dir / "figure_b_metaworld_per_task.pdf")
    figure_c_correlation(combined,
                          Path(planning_sr_csv) if planning_sr_csv else None,
                          figures_dir / "figure_c_correlation_planning.pdf")

    decision, justification = make_decision(mw, dr)
    print(f"\nDECISION: {decision}\n  {justification}")
    render_report(mw, dr, decision, justification, results_dir / "decision_report.md")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metaworld_csv", default="results/metaworld_diagnostic.csv")
    parser.add_argument("--droid_csv", default="results/droid_diagnostic.csv")
    parser.add_argument("--planning_sr_csv", default=None,
                        help="Optional CSV (task,model,success_rate) for Figure C.")
    args = parser.parse_args()
    sys.exit(main(args.metaworld_csv, args.droid_csv, args.planning_sr_csv))
