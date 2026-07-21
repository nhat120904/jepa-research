"""Apply the pre-registered kill/continue gate to selection sprint outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("lora_tail", "last_blocks_regression", "last_blocks_pairwise", "last_blocks_tail")


def bootstrap_mean_ci(values, seed=20260721, draws=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-json", default="results/selection_sprint_report.json")
    ap.add_argument("--out-md", default="results/selection_sprint_report.md")
    args = ap.parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    data = pd.concat(frames, ignore_index=True)
    key_counts = data.groupby(["arm", "train_seed", "task"])["eval_seed"].nunique()
    if (key_counts != 16).any():
        raise SystemExit(f"incomplete cells (expected 16 unique eval seeds):\n{key_counts}")

    cells = (data.groupby(["arm", "train_seed", "task"], as_index=False)
             .agg(success_end=("success_end", "sum"), success_any=("success", "sum"),
                  val_selected_regret=("val_selected_regret", "first"),
                  val_mae=("val_mae", "first")))
    summary = {}
    for arm in ARMS:
        arm_cells = cells[cells.arm == arm]
        push = arm_cells[arm_cells.task == "mw-push"].sort_values("train_seed")
        reach = arm_cells[arm_cells.task == "mw-reach"].sort_values("train_seed")
        if len(push) != 3 or len(reach) != 3:
            raise SystemExit(f"arm {arm} is incomplete: push={len(push)} reach={len(reach)}")
        summary[arm] = {
            "train_seeds": push.train_seed.astype(int).tolist(),
            "push_success_end_per_seed": push.success_end.astype(int).tolist(),
            "push_success_end_mean": float(push.success_end.mean()),
            "push_success_end_seed_bootstrap_ci": bootstrap_mean_ci(push.success_end),
            "reach_success_end_per_seed": reach.success_end.astype(int).tolist(),
            "reach_success_end_mean": float(reach.success_end.mean()),
            "val_selected_regret_per_seed": push.val_selected_regret.tolist(),
            "val_selected_regret_mean": float(push.val_selected_regret.mean()),
            "val_mae_mean": float(push.val_mae.mean()),
        }

    tail = summary["last_blocks_tail"]
    regression = summary["last_blocks_regression"]
    checks = {
        "tail_beats_regression_push": (
            tail["push_success_end_mean"] > regression["push_success_end_mean"]),
        "tail_beats_regression_regret": (
            tail["val_selected_regret_mean"] < regression["val_selected_regret_mean"]),
        "all_tail_seeds_cross_prior_envelope": all(
            value >= 5 for value in tail["push_success_end_per_seed"]),
        "reach_preserved_every_tail_seed": all(
            value >= 13 for value in tail["reach_success_end_per_seed"]),
    }
    verdict = "CONTINUE" if all(checks.values()) else "STOP_METHOD_DIRECTION"
    report = {"verdict": verdict, "checks": checks, "summary": summary,
              "protocol": "docs/plans/2026-07-21-selection-aware-encoder-sprint.md"}
    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Selection-aware encoder sprint", "", f"**Verdict: {verdict}**", "",
             "## Locked checks", ""]
    lines.extend([f"- {name}: **{value}**" for name, value in checks.items()])
    lines.extend(["", "## Per-arm readout", ""])
    for arm in ARMS:
        value = summary[arm]
        lines.append(
            f"- `{arm}`: push={value['push_success_end_per_seed']} "
            f"(mean {value['push_success_end_mean']:.2f}/16), "
            f"reach={value['reach_success_end_per_seed']}, "
            f"val regret={value['val_selected_regret_mean']:.4f}")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
