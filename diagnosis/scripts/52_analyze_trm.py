"""Paired summary for the locked TRM oracle-dynamics comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


BASE_RE = re.compile(
    r"trm_heldout_(?P<model>.+)_(?P<arm>l2|stateprobe)_"
    r"(?P<task>mw-push|mw-pick-place)_seed(?P<seed0>\d+)_n(?P<n>\d+)\.csv$"
)
TRM_RE = re.compile(
    r"trm_heldout_(?P<model>.+)_(?P<arm>replacement|hybrid)_h(?P<head>\d+)_"
    r"(?P<task>mw-push|mw-pick-place)_seed(?P<seed0>\d+)_n(?P<n>\d+)\.csv$"
)
ORACLE_RE = re.compile(
    r"trm_heldout_oracle_(?P<task>mw-push|mw-pick-place)_"
    r"seed(?P<seed0>\d+)_n(?P<n>\d+)\.csv$"
)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    improve = int(np.sum((a == 1) & (b == 0)))
    regress = int(np.sum((a == 0) & (b == 1)))
    discordant = improve + regress
    if not discordant:
        return improve, regress, 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(improve, regress) + 1))
    p = min(1.0, 2.0 * tail / (2 ** discordant))
    return improve, regress, p


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng, n_boot=10000) -> tuple[float, float, float]:
    delta = a - b
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    draws = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load_cell(path: Path, *, seed0: int, n: int) -> dict[int, dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_seed = {int(row["seed"]): row for row in rows}
    expected = set(range(seed0, seed0 + n))
    if set(by_seed) != expected or len(rows) != n:
        raise ValueError(f"{path}: expected exactly seeds {seed0}..{seed0+n-1}")
    return by_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--out-json", default="results/trm_heldout_summary.json")
    ap.add_argument("--out-report", default="results/trm_heldout_report.md")
    args = ap.parse_args()

    cells: dict[tuple[str, str, str, int | None], dict[int, dict]] = {}
    provenance = {}
    suffix = f"_seed{args.seed0}_n{args.episodes}.csv"
    for path in sorted(Path(args.results_dir).glob(f"trm_heldout_*{suffix}")):
        name = path.name
        match = ORACLE_RE.fullmatch(name)
        if match:
            key = ("oracle", "oracle", match["task"], None)
        else:
            match = BASE_RE.fullmatch(name)
            if match:
                key = (match["model"], match["arm"], match["task"], None)
            else:
                match = TRM_RE.fullmatch(name)
                if not match:
                    continue
                key = (match["model"], match["arm"], match["task"], int(match["head"]))
        if int(match["seed0"]) != args.seed0 or int(match["n"]) != args.episodes:
            continue
        if key in cells:
            raise ValueError(f"duplicate TRM result cell {key}")
        cells[key] = load_cell(path, seed0=args.seed0, n=args.episodes)
        provenance[str(key)] = str(path)

    models = ("dino_wm_metaworld", "jepa_wm_metaworld")
    tasks = ("mw-push", "mw-pick-place")
    expected = {("oracle", "oracle", task, None) for task in tasks}
    expected |= {(model, arm, task, None)
                 for model in models for arm in ("l2", "stateprobe") for task in tasks}
    expected |= {(model, arm, task, head)
                 for model in models for arm in ("replacement", "hybrid")
                 for task in tasks for head in range(3)}
    missing = expected - set(cells)
    extra = set(cells) - expected
    if missing or extra:
        raise SystemExit(f"incomplete TRM matrix: missing={sorted(missing)} extra={sorted(extra)}")

    summary = []
    for key in sorted(cells, key=str):
        rows = cells[key]
        success = np.asarray([int(rows[s]["success"]) for s in sorted(rows)])
        distance = np.asarray([float(rows[s]["obj_goal_dist"]) for s in sorted(rows)])
        lo, hi = wilson(int(success.sum()), len(success))
        summary.append({
            "model": key[0], "arm": key[1], "task": key[2], "head_seed": key[3],
            "successes": int(success.sum()), "episodes": len(success),
            "success_rate": float(success.mean()), "wilson95": [lo, hi],
            "obj_goal_dist_mean": float(distance.mean()),
            "obj_goal_dist_median": float(np.median(distance)),
        })

    contrasts = []
    rng = np.random.default_rng(51073)
    for model in models:
        for task in tasks:
            for arm in ("replacement", "hybrid"):
                for head in range(3):
                    trm_rows = cells[(model, arm, task, head)]
                    seeds = sorted(trm_rows)
                    trm_success = np.asarray([int(trm_rows[s]["success"]) for s in seeds])
                    trm_dist = np.asarray([float(trm_rows[s]["obj_goal_dist"]) for s in seeds])
                    for control in ("l2", "stateprobe", "oracle"):
                        control_key = (("oracle", "oracle", task, None) if control == "oracle"
                                       else (model, control, task, None))
                        control_rows = cells[control_key]
                        base_success = np.asarray([int(control_rows[s]["success"]) for s in seeds])
                        base_dist = np.asarray([float(control_rows[s]["obj_goal_dist"]) for s in seeds])
                        delta, dlo, dhi = paired_bootstrap(trm_success, base_success, rng)
                        improve, regress, pvalue = exact_mcnemar(trm_success, base_success)
                        dist_delta, dist_lo, dist_hi = paired_bootstrap(trm_dist, base_dist, rng)
                        contrasts.append({
                            "model": model, "task": task, "trm_arm": arm,
                            "head_seed": head, "control": control,
                            "success_rate_delta": delta,
                            "success_delta_bootstrap95": [dlo, dhi],
                            "discordant_improve": improve,
                            "discordant_regress": regress,
                            "exact_mcnemar_p": pvalue,
                            "obj_goal_dist_delta": dist_delta,
                            "obj_goal_dist_delta_bootstrap95": [dist_lo, dist_hi],
                        })

    payload = {
        "protocol": {"seed0": args.seed0, "episodes": args.episodes,
                     "head_seeds": [0, 1, 2], "paired_by_simulator_seed": True},
        "summary": summary, "contrasts": contrasts, "files": provenance,
    }
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    lines = [
        "# TRM-style held-out oracle-dynamics comparison", "",
        f"Fresh simulator seeds `{args.seed0}`--`{args.seed0+args.episodes-1}`; "
        f"n={args.episodes} paired episodes per cell. Three independently trained "
        "heads use the same immutable trajectory split. Test trajectories were not used "
        "for checkpoint selection.", "",
        "## Success summary", "",
        "| Model | Task | Arm | Head seed | Success | Wilson 95% | Mean object distance |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        ci = row["wilson95"]
        lines.append(
            f"| {row['model']} | {row['task']} | {row['arm']} | "
            f"{('-' if row['head_seed'] is None else row['head_seed'])} | "
            f"{row['successes']}/{row['episodes']} ({100*row['success_rate']:.1f}%) | "
            f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}] | {row['obj_goal_dist_mean']:.4f} |"
        )
    lines += [
        "", "## Interpretation guardrails", "",
        "- Replacement and hybrid heads are TRM-style adaptations: large JEPA token grids "
        "are mean/max pooled before the symmetric pair MLP.",
        "- Each head seed is reported separately. Repeated evaluation episodes across head "
        "seeds are paired replications, not independent samples to pool into 3n.",
        "- The true-state oracle is a privileged upper bound. TRM uses only cached latent "
        "trajectory order at training and frozen visual latents at planning.",
        "- Success contrasts and exact paired tests are in the JSON artifact.", "",
    ]
    out_report = Path(args.out_report); out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines))
    print(f"wrote {out_json} and {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
