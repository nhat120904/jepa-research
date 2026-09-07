#!/usr/bin/env python3
"""Parse the operator ablation for one plan seed.

Both arms use the same checkpoint, plan seed, episodes and budget; only the
executed action differs. The episode lists are checked equal, and a run whose
substituted plan never differed from the elite mean is flagged -- that would
mean the two arms are the same experiment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ARMS = ("elite_mean", "best_seen")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    return p.parse_args()


def parse(log: Path) -> dict:
    text = log.read_text(errors="ignore")
    rate = re.findall(r"'success_rate':\s*([0-9.]+)", text)
    succ = re.findall(r"'episode_successes':\s*array\((\[[^)]*\])", text, re.S)
    eps = re.findall(r"valid starting points found for evaluation\.\s*(\[[^\]]*\])",
                     text, re.S)
    out: dict = {"success_rate": float(rate[-1]) if rate else None}
    if succ:
        out["episode_successes"] = [t == "True" for t in
                                    re.findall(r"True|False", succ[-1])]
    if eps:
        out["episodes"] = [int(x) for x in re.findall(r"\d+", eps[-1])]
    return out


def main() -> None:
    args = parse_args()
    arms = {a: parse(args.run_dir / f"{a}.log") for a in ARMS}
    paired = (arms["elite_mean"].get("episodes")
              == arms["best_seen"].get("episodes")
              and bool(arms["elite_mean"].get("episodes")))

    report = {
        "plan_seed": args.seed, "paired": paired,
        "success_rate": {a: arms[a]["success_rate"] for a in ARMS},
    }
    if arms["elite_mean"]["success_rate"] is not None and arms["best_seen"]["success_rate"] is not None:
        report["difference_points"] = (arms["best_seen"]["success_rate"]
                                       - arms["elite_mean"]["success_rate"])
    both = [arms[a].get("episode_successes") for a in ARMS]
    if paired and all(both):
        report["episodes_changed"] = sum(x != y for x, y in zip(*both))
        report["flipped_to_success"] = sum(
            (not x) and y for x, y in zip(*both))
        report["flipped_to_failure"] = sum(
            x and (not y) for x, y in zip(*both))
        if report["episodes_changed"] == 0:
            report["warning"] = ("identical per-episode outcomes; the operator "
                                 "swap may not have taken effect")
    if not paired:
        report["error"] = "arms did not evaluate the same episodes"

    (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
