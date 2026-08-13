"""Gate 2 analysis (goal-marginalization design, 2026-08-11).

Paired (within-episode) contrasts across the three arms scripts/82 produced,
with an episode-seed-clustered bootstrap (metrics/bootstrap.py). Applies the
pre-registered decision rule from
docs/plans/2026-08-11-goal-marginalization-design.md exactly. Does not touch
the rule based on what the numbers turn out to be.

    python scripts/83_analyze_goal_marginalization.py \
        --csv results/goal_marginalization_mw-push_seed90000_n16.csv \
        --out results/goal_marginalization_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.bootstrap import bootstrap_ci  # noqa: E402


def _paired(df: pd.DataFrame, task: str, arm_a: str, arm_b: str, col: str):
    a = df[(df.task == task) & (df.arm == arm_a)].set_index("seed")[col]
    b = df[(df.task == task) & (df.arm == arm_b)].set_index("seed")[col]
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(), b.loc[common].to_numpy(), common.to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--task", default="mw-push")
    ap.add_argument("--out", default="results/goal_marginalization_report.md")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    task = args.task

    summary = {}
    for arm in ["baseline", "arm_marginalized", "noise_matched_control"]:
        sub = df[(df.task == task) & (df.arm == arm)]
        summary[arm] = {
            "n": len(sub),
            "success_end_rate": float(sub["success_end"].mean()),
            "success_end_n": int(sub["success_end"].sum()),
            "mean_obj_goal_dist": float(sub["obj_goal_dist"].mean()),
        }

    clean_frac = float((df[(df.task == task) & (df.arm == "arm_marginalized")]
                        ["arm_marginalization_clean"]).mean())

    succ_am, succ_base, seeds1 = _paired(df, task, "arm_marginalized", "baseline", "success_end")
    diff_vs_base = succ_am.astype(float) - succ_base.astype(float)
    ci_vs_base = bootstrap_ci(diff_vs_base, statistic=np.mean,
                              n_resamples=args.bootstrap, seed=args.seed)

    succ_am2, succ_noise, seeds2 = _paired(df, task, "arm_marginalized",
                                           "noise_matched_control", "success_end")
    diff_vs_noise = succ_am2.astype(float) - succ_noise.astype(float)
    ci_vs_noise = bootstrap_ci(diff_vs_noise, statistic=np.mean,
                               n_resamples=args.bootstrap, seed=args.seed + 1)

    dist_am, dist_base, _ = _paired(df, task, "arm_marginalized", "baseline", "obj_goal_dist")
    dist_diff = dist_base.astype(float) - dist_am.astype(float)   # positive = arm_marg closer
    ci_dist = bootstrap_ci(dist_diff, statistic=np.mean,
                           n_resamples=args.bootstrap, seed=args.seed + 2)

    beats_baseline = ci_vs_base.low > 0.0
    beats_noise_control = ci_vs_noise.point > 0.0 and ci_vs_noise.low > 0.0
    noise_control_matches = abs(ci_vs_noise.point) < 1e-9 or (
        ci_vs_noise.low <= 0.0 <= ci_vs_noise.high)
    intervention_clean_enough = clean_frac >= 0.5

    if not intervention_clean_enough:
        verdict = "NO-GO (intervention not clean)"
    elif beats_baseline and beats_noise_control:
        verdict = "GO"
    elif beats_baseline and not beats_noise_control:
        verdict = "CONDITIONAL / mechanism-only"
    else:
        verdict = "NO-GO"

    out = ROOT / "diagnosis" / args.out
    if not out.parent.exists():
        out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# Gate 2 — goal-marginalization pilot ({task})\n\n")
        f.write("## Per-arm summary\n\n")
        f.write("| arm | n | success_end | mean obj_goal_dist (m) |\n|---|---:|---:|---:|\n")
        for arm, s in summary.items():
            f.write(f"| {arm} | {s['n']} | {s['success_end_n']}/{s['n']} "
                     f"({s['success_end_rate']:.3f}) | {s['mean_obj_goal_dist']:.4f} |\n")
        f.write(f"\n`arm_marginalization_clean` fraction (obj displacement <= 1cm): "
                f"**{clean_frac:.2f}**\n\n")
        f.write("## Paired contrasts (episode-clustered bootstrap, "
                f"n_resamples={args.bootstrap})\n\n")
        f.write(f"- arm_marginalized vs baseline, success_end diff: "
                f"{ci_vs_base.point:.3f} [{ci_vs_base.low:.3f}, {ci_vs_base.high:.3f}] "
                f"(n={len(seeds1)})\n")
        f.write(f"- arm_marginalized vs noise_matched_control, success_end diff: "
                f"{ci_vs_noise.point:.3f} [{ci_vs_noise.low:.3f}, {ci_vs_noise.high:.3f}] "
                f"(n={len(seeds2)})\n")
        f.write(f"- arm_marginalized vs baseline, obj_goal_dist improvement (m): "
                f"{ci_dist.point:.4f} [{ci_dist.low:.4f}, {ci_dist.high:.4f}]\n\n")
        f.write("## Decision (pre-registered rule, "
                "docs/plans/2026-08-11-goal-marginalization-design.md)\n\n")
        f.write(f"**{verdict}**\n\n")
        f.write(f"- beats_baseline (lower CI > 0): {beats_baseline}\n")
        f.write(f"- beats_noise_matched_control (point>0 and lower CI>0): {beats_noise_control}\n")
        f.write(f"- intervention_clean_enough (>=50% episodes clean): "
                f"{intervention_clean_enough}\n")

    print(f"verdict: {verdict}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
