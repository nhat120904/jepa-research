#!/usr/bin/env python3
"""Inspect a u_closed_loop.json: per episode, the completed events (acted identity, target, final reader state,
error to target, steps, other identities changed) and timeouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=Path, required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--max-events", type=int, default=8)
    a = ap.parse_args()
    d = json.loads((a.loop / "u_closed_loop.json").read_text())
    errs, steps = [], []
    for ep in d["episodes"]:
        if ep["task"] not in a.tasks:
            continue
        print(f"task {ep['task']} ep {ep['episode']} success {ep['success']} steps {ep['steps']} replans {ep['replans']} timeouts {ep['timeouts']} "
              f"first_plan {ep['first_plan']}")
        for e in ep["events"][: a.max_events]:
            x, f = np.array(e["x"][:2]), np.array(e["final"][:2])
            err = float(np.linalg.norm(x - f)); errs.append(err); steps.append(e["steps"])
            print(f"    e {e['e']} target {np.round(x, 1)} final {np.round(f, 1)} err {err:.1f} px  steps {e['steps']}  others {e['others_changed']}")
    if errs:
        errs = np.array(errs)
        print({"events": len(errs), "err_px_p25_50_75": np.round(np.percentile(errs, [25, 50, 75]), 1).tolist(),
               "frac_err_le_2px": round(float((errs <= 2).mean()), 3), "frac_err_le_4px": round(float((errs <= 4).mean()), 3),
               "steps_p50": float(np.median(steps))})


if __name__ == "__main__":
    main()
