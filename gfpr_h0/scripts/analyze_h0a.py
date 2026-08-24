#!/usr/bin/env python3
"""Aggregate the four leakage-safe outer folds and apply the locked H0-A gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PRIMARY = "latent_context_gated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folds-root", type=Path, default=REPO / "gfpr_h0/outputs/folds"
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def bootstrap_ci(
    values: np.ndarray, draws: int, rng: np.random.Generator
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def fmt_ci(value: float, ci: list[float], digits: int = 3) -> str:
    return f"{value:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 1000:
        raise ValueError("bootstrap-draws is too small")
    records: list[dict[str, object]] = []
    for fold in range(4):
        records.extend(
            json.loads((args.folds_root / str(fold) / "state_records.json").read_text())
        )
    records.sort(key=lambda row: int(row["snapshot_index"]))
    indices = [int(row["snapshot_index"]) for row in records]
    if indices != list(range(32)) or len(set(indices)) != 32:
        raise RuntimeError("outer-fold predictions do not cover each snapshot exactly once")

    arm_names = sorted(records[0]["arms"])
    if any(sorted(row["arms"]) != arm_names for row in records):
        raise RuntimeError("fold arm sets differ")
    rng = np.random.default_rng(args.seed)
    native_success = np.asarray(
        [row["arms"]["native"]["success"] for row in records], dtype=float
    )
    summary_arms: dict[str, dict[str, object]] = {}
    for arm in arm_names:
        success = np.asarray([row["arms"][arm]["success"] for row in records], dtype=float)
        distance = np.asarray(
            [row["arms"][arm]["physical_distance_m"] for row in records], dtype=float
        )
        gain = np.asarray([row["arms"][arm]["gain_cm"] for row in records], dtype=float)
        corrective = np.asarray(
            [row["arms"][arm]["corrective"] for row in records], dtype=float
        )
        switch = np.asarray([row["arms"][arm]["switch"] for row in records], dtype=float)
        harmful = np.asarray(
            [row["arms"][arm]["harmful_switch"] for row in records], dtype=float
        )
        success_gain = success - native_success
        switches = int(switch.sum())
        summary_arms[arm] = {
            "success_rate": float(success.mean()),
            "success_rate_ci": bootstrap_ci(success, args.bootstrap_draws, rng),
            "success_gain_vs_native": float(success_gain.mean()),
            "success_gain_vs_native_ci": bootstrap_ci(
                success_gain, args.bootstrap_draws, rng
            ),
            "physical_distance_m": float(distance.mean()),
            "physical_gain_cm": float(gain.mean()),
            "physical_gain_cm_ci": bootstrap_ci(gain, args.bootstrap_draws, rng),
            "corrective_rate": float(corrective.mean()),
            "corrective_rate_ci": bootstrap_ci(
                corrective, args.bootstrap_draws, rng
            ),
            "switches": switches,
            "harmful_switches": int(harmful.sum()),
            "harmful_switch_rate_among_switches": float(
                harmful.sum() / max(switches, 1)
            ),
        }

    primary = summary_arms[PRIMARY]
    gain = float(primary["physical_gain_cm"])
    gain_lo = float(primary["physical_gain_cm_ci"][0])
    success_gain = float(primary["success_gain_vs_native"])
    success_lo = float(primary["success_gain_vs_native_ci"][0])
    harmful_rate = float(primary["harmful_switch_rate_among_switches"])
    switches = int(primary["switches"])
    if gain_lo > 0 and success_lo > 0 and harmful_rate <= 0.05 and switches >= 4:
        verdict = "STRONG_GO_FRESH_H0"
    elif gain_lo > 0 and success_gain > 0 and harmful_rate <= 0.10 and switches >= 4:
        verdict = "GO_FRESH_H0"
    elif gain <= 0 or success_gain < 0 or harmful_rate > 0.20:
        verdict = "STOP_GFPR_FORMULATION"
    else:
        verdict = "HOLD_DIAGNOSE"

    summary = {
        "scope": "Four-fold episode-held-out GFPR H0-A on 32 Phase-0d snapshots.",
        "primary_arm": PRIMARY,
        "n_snapshots": 32,
        "bootstrap_draws": args.bootstrap_draws,
        "verdict": verdict,
        "arms": summary_arms,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# GFPR H0-A report",
        "",
        f"**Locked verdict: `{verdict}`**",
        "",
        "Four-fold outer predictions cover 32 distinct snapshots/episodes. "
        "No candidate from a held-out snapshot was used to fit its scorer.",
        "",
        "| arm | success | success gain vs native | physical gain (cm) | corrective | switches | harmful / switches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arm_names:
        values = summary_arms[arm]
        lines.append(
            f"| {arm} | "
            f"{fmt_ci(values['success_rate'], values['success_rate_ci'])} | "
            f"{fmt_ci(values['success_gain_vs_native'], values['success_gain_vs_native_ci'])} | "
            f"{fmt_ci(values['physical_gain_cm'], values['physical_gain_cm_ci'])} | "
            f"{fmt_ci(values['corrective_rate'], values['corrective_rate_ci'])} | "
            f"{values['switches']} | {values['harmful_switches']} / {values['switches']} |"
        )
    lines.extend(
        [
            "",
            "`action_diverse_oracle8` and `physical_oracle_full` inspect physical "
            "outcomes and are upper bounds, not deployable zero-query selectors.",
            "",
            "The primary gate is defined in `PROTOCOL.md`. A GO verdict licenses "
            "a fresh frozen-model endpoint; it is not final paper evidence.",
        ]
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

