"""Analyze paired factorized-cost cells after the Slurm array completes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

MODELS = ("dino_wm_metaworld", "jepa_wm_metaworld")
TASKS = ("mw-push", "mw-pick-place")
ARMS = ("decoded_both", "true_object", "true_hand", "true_both")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denominator = 1.0 + z * z / n
    midpoint = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return midpoint - half, midpoint + half


def paired_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_cells(results: Path, seed0: int, episodes: int):
    cells = {}
    pattern = f"factorized_cost_*_seed{seed0}_n{episodes}.csv"
    for path in sorted(results.glob(pattern)):
        frame = pd.read_csv(path)
        required = {"model", "task", "arm", "seed", "success_end", "obj_goal_dist", "true_shaped_cost"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        keys = frame[["model", "task", "arm"]].drop_duplicates()
        if len(keys) != 1 or len(frame) != episodes or frame.seed.duplicated().any():
            raise ValueError(f"invalid cell shape or identity in {path}")
        key = tuple(keys.iloc[0].tolist())
        cells[key] = frame.sort_values("seed").reset_index(drop=True)
    expected = {(model, task, arm) for model in MODELS for task in TASKS for arm in ARMS}
    missing = expected - set(cells)
    if missing:
        raise SystemExit(f"factorized cells incomplete: {sorted(missing)}")
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--seed0", type=int, default=61000)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--out-prefix", type=Path, default=Path("results/factorized_cost_ladder_pilot"))
    args = parser.parse_args()
    cells = load_cells(args.results, args.seed0, args.episodes)

    # With both channels privileged the encoder is computationally exercised but
    # cannot affect the score. The two model-labelled cells must therefore be
    # numerically identical; treat disagreement as a protocol failure.
    for task in TASKS:
        dino = cells[(MODELS[0], task, "true_both")].set_index("seed")
        jepa = cells[(MODELS[1], task, "true_both")].set_index("seed")
        if not dino.index.equals(jepa.index):
            raise ValueError(f"true_both seed mismatch across models for {task}")
        for column in ("success_end", "obj_goal_dist", "true_shaped_cost"):
            if not np.allclose(dino[column], jepa[column], atol=1e-7, rtol=0.0):
                raise ValueError(
                    f"true_both model-invariance check failed for {task}, {column}"
                )

    summary_rows = []
    for (model, task, arm), frame in sorted(cells.items()):
        successes = frame.success_end.to_numpy(int)
        lo, hi = wilson(int(successes.sum()), len(successes))
        summary_rows.append(
            {
                "model": model,
                "task": task,
                "arm": arm,
                "seed0": args.seed0,
                "n": len(frame),
                "success_k": int(successes.sum()),
                "success_rate": float(successes.mean()),
                "wilson_lo": lo,
                "wilson_hi": hi,
                "obj_goal_mean": float(frame.obj_goal_dist.mean()),
                "true_shaped_cost_mean": float(frame.true_shaped_cost.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    comparisons = (
        ("true_object", "decoded_both", "object_channel_correction"),
        ("true_hand", "decoded_both", "hand_channel_correction"),
        ("true_both", "decoded_both", "full_correction"),
        ("true_both", "true_object", "hand_given_true_object"),
        ("true_both", "true_hand", "object_given_true_hand"),
    )
    contrast_rows = []
    contrast_index = 0
    for model in MODELS:
        for task in TASKS:
            for left_arm, right_arm, family in comparisons:
                left = cells[(model, task, left_arm)].set_index("seed")
                right = cells[(model, task, right_arm)].set_index("seed")
                if not left.index.equals(right.index):
                    raise ValueError(f"seed mismatch for {model}, {task}, {family}")
                left_success = left.success_end.to_numpy(float)
                right_success = right.success_end.to_numpy(float)
                success_delta = left_success - right_success
                success_ci = paired_ci(success_delta, 1000 + contrast_index, args.n_boot)
                left_only = int(((left_success == 1) & (right_success == 0)).sum())
                right_only = int(((left_success == 0) & (right_success == 1)).sum())
                discordant = left_only + right_only
                exact_p = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
                shaped_delta = (
                    left.true_shaped_cost.to_numpy(float) - right.true_shaped_cost.to_numpy(float)
                )
                shaped_ci = paired_ci(shaped_delta, 2000 + contrast_index, args.n_boot)
                contrast_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "family": family,
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        "n_paired": len(left),
                        "success_delta": float(success_delta.mean()),
                        "success_delta_lo": success_ci[0],
                        "success_delta_hi": success_ci[1],
                        "discord_left_only": left_only,
                        "discord_right_only": right_only,
                        "mcnemar_exact_p": exact_p,
                        "shaped_cost_delta": float(shaped_delta.mean()),
                        "shaped_cost_delta_lo": shaped_ci[0],
                        "shaped_cost_delta_hi": shaped_ci[1],
                    }
                )
                contrast_index += 1
    contrasts = pd.DataFrame(contrast_rows)

    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    contrast_path = args.out_prefix.with_name(args.out_prefix.name + "_contrasts.csv")
    report_path = args.out_prefix.with_suffix(".md")
    for path in (summary_path, contrast_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrast_path, index=False)
    report = [
        "# Factorized cost ladder pilot",
        "",
        f"Fresh seeds `{args.seed0}..{args.seed0 + args.episodes - 1}`; paired `n={args.episodes}` per cell.",
        "Primary endpoint is strict MetaWorld `success_end`. Hand means end-effector xyz, not gripper aperture.",
        "",
        "## Per-arm outcomes",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Paired causal contrasts",
        "",
        "Positive success delta favors the corrected channel; negative shaped-cost delta favors it.",
        "",
        "```text",
        contrasts.to_string(index=False),
        "```",
        "",
        "## Interpretation guardrails",
        "",
        "- This localizes channels in the representation--probe--cost composition; it does not prove information is absent from the encoder.",
        "- The true-both arm passed an exact cross-model invariance check; otherwise analysis aborts.",
        "- The n=16 run is a directional pilot. Any promoted claim requires a fresh locked n=64 confirmation.",
        "- The intervention uses privileged simulator state and is diagnostic, not a deployable method.",
    ]
    report_path.write_text("\n".join(report) + "\n")
    print(f"wrote {summary_path}, {contrast_path}, {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
