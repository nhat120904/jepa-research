"""Compact view of OGBench offline ladders (v1 offline_ladder.json and v2 / play offline_ladder.json), side by side.

Usage: ogb_ladder_brief.py <label>=<run dir> ...   (prints held-out retained gap [CI] per tier and goal set, plus code
usage where reported). Small JSON reads only.
"""
import json
import sys
from pathlib import Path

TIERS = ("full", "code", "direct", "cta", "endpoint", "frame")


def fmt(m):
    g = m["retained_gap"]
    return f"{g['ratio']:.3f}" + (f" [{g['lo']:.2f},{g['hi']:.2f}]" if "lo" in g else "")


for spec in sys.argv[1:]:
    label, path = spec.split("=", 1)
    rep = json.loads((Path(path) / "offline_ladder.json").read_text())
    print(f"== {label}  ({path})")
    if "ladder" not in rep:                                   # v1: {tier: metrics} on the own goal
        for t in TIERS:
            if t in rep:
                print(f"  {t:9s} own {fmt(rep[t])}")
        continue
    lad = rep["ladder"]
    goals = [g for g in lad["cta"] if g not in ("own_on_heldout_tasks",)]
    for t in TIERS:
        row = "  ".join(f"{g}:{fmt(lad[t][g])}" if g == "own" else f"{g}:{lad[t][g]['retained_gap']['ratio']:.3f}"
                        for g in goals)
        extra = f"  heldout-task-own:{fmt(lad[t]['own_on_heldout_tasks'])}" if "own_on_heldout_tasks" in lad[t] else ""
        print(f"  {t:9s} {row}{extra}")
    print("  summary", json.dumps(rep.get("summary_retained_gap", {}).get("cta", {})))
    print("  codes  ", json.dumps(rep.get("codes", {})))
    sel = rep.get("config", {})
    print("  selected", json.dumps({k: sel.get(k) for k in ("selected_stage1", "selected_stage2")}))
