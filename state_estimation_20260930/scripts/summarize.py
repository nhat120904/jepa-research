#!/usr/bin/env python3
"""Summarise official_eval JSON outputs: success per arm and paired differences.

Arms are file stems without the seed suffix (e.g. official_h1, m10_h3p). Pairs are
the same (seed, episode) across two arms of one task; CIs bootstrap episodes.
Small (tens of JSON files); fine on the login node.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def boot(d, draws=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--pairs", nargs="*", default=[], help="armA:armB (reports A - B)")
    a = ap.parse_args()
    for task_dir in sorted(p for p in a.run_dir.iterdir() if p.is_dir()):
        arms = defaultdict(dict)
        for f in sorted(task_dir.glob("*.json")):
            m = re.match(r"(.+)_s(\d+)$", f.stem)
            if not m:
                continue
            d = json.loads(f.read_text())
            arms[m.group(1)][int(m.group(2))] = d
        if not arms:
            continue
        print(f"== {task_dir.name}")
        for arm, by_seed in arms.items():
            rates = [by_seed[s]["success_rate"] for s in sorted(by_seed)]
            print(f"  {arm:22s} mean {np.mean(rates):5.1f}  seeds {rates}")
        for pair in a.pairs:
            x, y = pair.split(":")
            if x not in arms or y not in arms:
                continue
            diffs = []
            for s in sorted(set(arms[x]) & set(arms[y])):
                A, B = arms[x][s], arms[y][s]
                assert A["episodes"] == B["episodes"] and A["start_steps"] == B["start_steps"]
                diffs += list(np.array(A["episode_successes"]) - np.array(B["episode_successes"]))
            m, lo, hi = boot(diffs)
            d = np.array(diffs)
            print(f"  {x} - {y}: {100 * m:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]  "
                  f"({(d > 0).sum()} vs {(d < 0).sum()} discordant, n={len(d)})")


if __name__ == "__main__":
    main()
