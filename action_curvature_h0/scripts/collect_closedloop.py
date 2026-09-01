#!/usr/bin/env python3
"""Parse one plan seed's four arm logs into a paired record.

The comparison is only paired if every arm evaluated the SAME episodes. The
evaluator prints the episode indices it selected, so they are captured and
checked equal across arms; a mismatch means the arms were scored on different
tasks and the seed is marked invalid rather than reported.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ARMS = ("acm_original", "acm_lam0_seed0", "acm_lam0_seed1", "acm_lam0_seed2")


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
        flags = [t.strip() == "True" for t in
                 re.findall(r"True|False", succ[-1])]
        out["n_success"], out["n_episodes"] = sum(flags), len(flags)
        out["episode_successes"] = flags
    if eps:
        out["episodes"] = [int(x) for x in re.findall(r"\d+", eps[-1])]
    return out


def main() -> None:
    args = parse_args()
    arms = {a: parse(args.run_dir / f"{a}.log") for a in ARMS}
    episode_sets = {a: v.get("episodes") for a, v in arms.items()}
    reference = episode_sets[ARMS[0]]
    paired = all(episode_sets[a] == reference for a in ARMS) and bool(reference)

    report = {
        "plan_seed": args.seed, "paired": paired,
        "n_episodes": len(reference) if reference else None,
        "success_rate": {a: arms[a]["success_rate"] for a in ARMS},
    }
    if paired and all(arms[a].get("episode_successes") for a in ARMS):
        base = arms[ARMS[0]]["episode_successes"]
        report["per_episode_agreement_with_original"] = {
            a: sum(x == y for x, y in zip(base, arms[a]["episode_successes"]))
            for a in ARMS[1:]
        }
    if not paired:
        report["error"] = ("arms did not evaluate the same episodes; "
                           "this seed is not a paired comparison")

    (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
