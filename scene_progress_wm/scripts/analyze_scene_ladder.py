"""Aggregate the goal-offset ladder and contrast arms against the baseline.

Episodes are paired: every arm at a given offset and plan seed sees the same (start,
goal) pairs, so the contrast is computed per episode and bootstrapped over episodes as
clusters, with model seeds averaged inside a cluster before resampling.

Each offset also carries a ceiling measured by the replay gate -- the rate at which the
dataset's own actions reach the goal. Reporting a planning number without it would
overstate the headroom, since at offset 200 even a perfect planner replaying the recorded
solution reaches the goal only 92% of the time.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PROTOCOL = "scene_progress_wm_ladder_analysis_v1"
BOOTSTRAP = 10000
BASELINE_ARM = "latent_l2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True, help="harness_gate.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def cluster_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap over episodes, which are the independent unit here."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(BOOTSTRAP, values.size))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def load_results(root: Path) -> list[dict]:
    files = sorted(root.rglob("*.json"))
    payloads = []
    for path in files:
        if path.name in {"summary.json", "harness_gate.json"}:
            continue
        payload = json.loads(path.read_text())
        if payload.get("protocol") != "scene_progress_wm_eval_v1":
            continue
        payloads.append(payload)
    return payloads


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("analysis must run inside a Slurm compute job")

    payloads = load_results(args.results_root)
    if len(payloads) != args.expected_shards:
        raise RuntimeError(
            f"expected {args.expected_shards} result files, found {len(payloads)}"
        )

    gate = json.loads(args.gate.read_text())
    ceilings = {
        offset: block["final_success_rate"]
        for offset, block in gate["per_offset"].items()
    }

    # arm -> offset -> episode -> list of successes (one per plan seed)
    table: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    checkpoints: dict = defaultdict(set)
    for payload in payloads:
        arm = payload["objective"]
        if arm != BASELINE_ARM:
            arm = f"{payload['progress_arm']}_w{payload['mixture_weight']:.2f}"
        offset = str(payload["goal_offset"])
        checkpoints[arm].add(payload["checkpoint_sha256"])
        for record in payload["records"]:
            table[arm][offset][record["spec"]["episode"]].append(bool(record["success"]))

    for arm, digests in checkpoints.items():
        if len(digests) != 1:
            raise RuntimeError(f"arm {arm} mixes world-model checkpoints: {digests}")

    arms = sorted(table)
    offsets = sorted({o for arm in arms for o in table[arm]}, key=int)

    per_arm = {}
    for arm in arms:
        per_arm[arm] = {}
        for offset in offsets:
            episodes = table[arm].get(offset, {})
            if not episodes:
                continue
            values = np.array(
                [np.mean(v) for _, v in sorted(episodes.items())], dtype=np.float64
            )
            mean, low, high = cluster_ci(values, args.bootstrap_seed + int(offset))
            per_arm[arm][offset] = {
                "num_episodes": int(values.size),
                "success_rate": mean,
                "ci_low": low,
                "ci_high": high,
                "ceiling": ceilings.get(offset),
                "headroom": (
                    None if ceilings.get(offset) is None else ceilings[offset] - mean
                ),
            }

    contrasts = {}
    if BASELINE_ARM in table:
        for arm in arms:
            if arm == BASELINE_ARM:
                continue
            contrasts[arm] = {}
            for offset in offsets:
                base = table[BASELINE_ARM].get(offset, {})
                other = table[arm].get(offset, {})
                shared = sorted(set(base) & set(other))
                if not shared:
                    continue
                diff = np.array(
                    [np.mean(other[e]) - np.mean(base[e]) for e in shared],
                    dtype=np.float64,
                )
                mean, low, high = cluster_ci(diff, args.bootstrap_seed + 7 * int(offset))
                contrasts[arm][offset] = {
                    "paired_episodes": len(shared),
                    "difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "clean": bool(low > 0.0 or high < 0.0),
                }

    summary = {
        "protocol": PROTOCOL,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "results_root": str(args.results_root),
        "num_result_files": len(payloads),
        "gate": str(args.gate),
        "ceilings": ceilings,
        "arms": arms,
        "offsets": offsets,
        "per_arm": per_arm,
        "contrasts_vs_baseline": contrasts,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = ["# Scene progress-WM: goal-offset ladder", ""]
    header = "| Arm | " + " | ".join(f"Δ={o}" for o in offsets) + " |"
    lines += [header, "|" + "---|" * (len(offsets) + 1)]
    lines.append(
        "| _ceiling (dataset replay)_ | "
        + " | ".join(
            f"{100 * ceilings[o]:.0f}%" if o in ceilings else "--" for o in offsets
        )
        + " |"
    )
    for arm in arms:
        cells = []
        for offset in offsets:
            cell = per_arm[arm].get(offset)
            cells.append(
                "--"
                if cell is None
                else f"{100 * cell['success_rate']:.1f}% [{100 * cell['ci_low']:.1f}, {100 * cell['ci_high']:.1f}]"
            )
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    if contrasts:
        lines += ["", "## Paired difference vs `latent_l2` (points)", ""]
        lines += [header, "|" + "---|" * (len(offsets) + 1)]
        for arm, block in contrasts.items():
            cells = []
            for offset in offsets:
                cell = block.get(offset)
                if cell is None:
                    cells.append("--")
                else:
                    mark = "**" if cell["clean"] else ""
                    cells.append(
                        f"{mark}{100 * cell['difference']:+.1f}{mark} "
                        f"[{100 * cell['ci_low']:+.1f}, {100 * cell['ci_high']:+.1f}]"
                    )
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
        lines += ["", "Bold marks an interval excluding zero."]

    (args.out_dir / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"arms": arms, "offsets": offsets, "ceilings": ceilings}, sort_keys=True))


if __name__ == "__main__":
    main()
