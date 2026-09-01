#!/usr/bin/env python3
"""Parse the two operator arms for one plan seed into a paired record.

Both arms run the same checkpoint, seed and search, so they must have evaluated
the same episodes; that is checked rather than assumed.  A swap that produced
an identical action sequence would silently reproduce the control, so the
recorded max |best - mean| is required to be non-zero.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ARMS = ("exec_mean", "exec_best")


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
    same_eps = arms["exec_mean"].get("episodes") == arms["exec_best"].get("episodes")
    report = {
        "plan_seed": args.seed,
        "paired": bool(same_eps and arms["exec_mean"].get("episodes")),
        "success_rate": {a: arms[a]["success_rate"] for a in ARMS},
    }
    if report["paired"]:
        report["delta_points"] = (arms["exec_best"]["success_rate"]
                                  - arms["exec_mean"]["success_rate"])
        f_m, f_b = (arms[a].get("episode_successes") for a in ARMS)
        if f_m and f_b:
            report["episodes_flipped"] = {
                "mean_only": sum(m and not b for m, b in zip(f_m, f_b)),
                "best_only": sum(b and not m for m, b in zip(f_m, f_b)),
                "identical_outcome": sum(m == b for m, b in zip(f_m, f_b)),
            }
    else:
        report["error"] = "arms did not evaluate the same episodes"
    (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
