#!/usr/bin/env python3
"""Read the elite-optimism artifacts and emit the locked verdict.

The verdict conditions live in ``docs/SCENE_ELITE_OPTIMISM_PROTOCOL.md`` and are
transcribed here once.  They are read at ``delta = 25`` -- one committed chunk --
because that is the margin at which an ordering mistake costs the planner a whole
replan rather than a rounding error.

Nothing here re-derives a threshold from the data it is judging.  If a threshold
turns out to be mis-specified, amend the protocol in the open and change this
file to match; do not relax a bar to pass a run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PROTOCOL = "scene_progress_wm_elite_optimism_analysis_v1"

VERDICT_DELTA = 25.0
AMPLIFICATION_BAR = 1.5
ABSENT_UPPER_BAR = 1.25
MAX_CENSORING = 0.30
# Amended 2026-09-04 before any result was read; see the protocol's amendment
# note.  The original bar (20 decisive pairs per replan) is unreachable: with
# ``--n-per-stratum 6`` a replan holds at most C(6,2) = 15 pairs.  The quantities
# that actually carry the estimate are the total pair count and the number of
# replans contributing a defined rate, so the bar moved to those.
MIN_DECISIVE_PAIRS_TOTAL = 200
MIN_CONTRIBUTING_REPLANS = 30
MAX_PROBE_SPREAD = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--primary-arm", default="latent_l2")
    return parser.parse_args()


def verdict_for(payload: dict) -> dict:
    """Apply the locked conditions to one arm's artifact."""
    key = f"delta_{VERDICT_DELTA:g}"
    summary = payload["summary"]
    block = summary["per_delta"].get(key)
    if block is None:
        return {"verdict": "INSTRUMENT_UNUSABLE", "reason": f"no {key} in summary"}

    amp = block["elite_amplification"]
    paired = amp["paired_difference"]
    censoring = summary["censoring"]["rate"]
    probe = payload["instrument"].get("repeat_probe_max_spread")
    strata = [s for s in ("random_init", "gen_mid", "gen_last", "elite") if s in block]
    total_pairs = min(block[s]["decisive_pairs_total"] for s in strata)
    contributing = paired["n"]

    unusable = []
    if censoring is not None and censoring > MAX_CENSORING:
        unusable.append(f"censoring {censoring:.3f} > {MAX_CENSORING}")
    if total_pairs < MIN_DECISIVE_PAIRS_TOTAL:
        unusable.append(
            f"decisive pairs {total_pairs} < {MIN_DECISIVE_PAIRS_TOTAL} in some stratum"
        )
    if contributing < MIN_CONTRIBUTING_REPLANS:
        unusable.append(
            f"only {contributing} replans contribute a paired rate "
            f"(< {MIN_CONTRIBUTING_REPLANS})"
        )
    if probe is not None and probe > MAX_PROBE_SPREAD:
        unusable.append(f"repeat-probe spread {probe} > {MAX_PROBE_SPREAD}")
    if unusable:
        return {
            "verdict": "INSTRUMENT_UNUSABLE",
            "reason": "; ".join(unusable),
            "amplification": amp,
            "censoring": censoring,
            "decisive_pairs_total_min_stratum": total_pairs,
            "contributing_replans": contributing,
            "repeat_probe_max_spread": probe,
        }

    ratio = amp["ratio"]
    lo, hi = paired["lo"], paired["hi"]
    confirmed = (
        ratio is not None and ratio >= AMPLIFICATION_BAR and lo is not None and lo > 0
    )
    absent = (
        lo is not None
        and hi is not None
        and lo <= 0 <= hi
        and (ratio is None or ratio < ABSENT_UPPER_BAR)
    )
    if confirmed:
        verdict = "EXPLOITATION_CONFIRMED"
    elif absent:
        verdict = "EXPLOITATION_ABSENT"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "amplification": amp,
        "censoring": censoring,
        "decisive_pairs_total_min_stratum": total_pairs,
        "contributing_replans": contributing,
        "repeat_probe_max_spread": probe,
        "replans": paired["n"],
        # power is a property of the budget, not of the result: an ABSENT verdict
        # at n = 36 rules out a 2x effect, not a 1.5x one
        "power_note": (
            "n>=72 rules out 1.5x; n>=36 rules out 2.0x; below 36 rules out neither"
        ),
    }


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("analysis must run inside a Slurm compute job")

    files = sorted(args.result_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"no artifacts under {args.result_dir}")

    arms: dict[str, dict] = {}
    for path in files:
        payload = json.loads(path.read_text())
        arms[payload["arm"]] = {"path": str(path), "payload": payload}

    report = {"protocol": PROTOCOL, "verdict_delta": VERDICT_DELTA, "arms": {}}
    for arm, entry in arms.items():
        report["arms"][arm] = verdict_for(entry["payload"])
        report["arms"][arm]["artifact"] = entry["path"]

    primary = report["arms"].get(args.primary_arm)
    report["primary_arm"] = args.primary_arm
    report["verdict"] = primary["verdict"] if primary else "MISSING_PRIMARY_ARM"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "elite_optimism_verdict.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    print(f"verdict delta = {VERDICT_DELTA:g} (one committed chunk)\n")
    header = f"{'arm':<18}{'base':>8}{'elite':>8}{'ratio':>8}{'paired diff [95% CI]':>28}{'n':>5}  verdict"
    print(header)
    print("-" * len(header))
    for arm, block in sorted(report["arms"].items()):
        amp = block.get("amplification")
        if amp is None:
            print(f"{arm:<18}{'-':>8}{'-':>8}{'-':>8}{'-':>28}{'-':>5}  {block['verdict']}")
            continue
        d = amp["paired_difference"]
        ratio = amp["ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.2f}"
        diff_text = f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
        print(
            f"{arm:<18}"
            f"{amp['random_init_rate']:>8.3f}"
            f"{amp['elite_rate']:>8.3f}"
            f"{ratio_text:>8}"
            f"{diff_text:>28}"
            f"{d['n']:>5}  {block['verdict']}"
        )
        print(
            f"{'':<18}censoring={block['censoring']:.3f}  "
            f"decisive pairs (min stratum)={block['decisive_pairs_total_min_stratum']}  "
            f"probe spread={block['repeat_probe_max_spread']}"
        )
    print(f"\nPRIMARY ({args.primary_arm}): {report['verdict']}")
    print(f"wrote {args.out_dir / 'elite_optimism_verdict.json'}")


if __name__ == "__main__":
    main()
