#!/usr/bin/env python3
"""Assert that the cached iteration-0 population is pre-refit.

The fifteenth amendment makes iteration 0 the primary arena precisely because
it is drawn from the initial proposal before any model has refit anything.  If
the cache were written after an original-model refit it would already carry the
control arm's bias and the primary would be invalid.  Three conditions together
establish the trace ``initial mu, sigma -> sample N -> cache P0 -> score/refit``:

1. ``proposal_mean[0]`` is exactly zero -- the initial mean with no warm start;
2. ``proposal_std[0]`` is exactly ``var_scale`` (1.0) -- the untouched initial
   variance, not a refit standard deviation;
3. candidate 0 equals ``proposal_mean[0]`` exactly -- CEM forces the first
   sample to be the current mean, so this pins which distribution the
   population was drawn from.

The final iterate must fail 1 and 2, otherwise the two arenas are not distinct.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--populations", required=True)
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--var-scale", type=float, default=1.0)
    a = p.parse_args()

    d = np.load(a.populations)
    step, act = d["step"], d["actions_normalized"]
    pm, ps = d["proposal_mean"], d["proposal_std"]
    checks: list[tuple[str, bool, str]] = []

    checks.append(("step[0] == 0", int(step[0]) == 0, f"step={step.tolist()}"))
    checks.append((
        "shape is (2, N, H, D)",
        act.ndim == 4 and act.shape[0] == 2 and act.shape[1] == a.num_samples,
        f"actions_normalized {act.shape}",
    ))
    checks.append((
        "iter0 proposal_mean is exactly zero",
        bool(np.all(pm[0] == 0.0)), f"max|mean| = {np.abs(pm[0]).max():.6g}",
    ))
    checks.append((
        "iter0 proposal_std is exactly var_scale",
        bool(np.all(ps[0] == a.var_scale)), f"unique = {np.unique(ps[0])[:4]}",
    ))
    checks.append((
        "iter0 candidate 0 equals the proposal mean",
        bool(np.array_equal(act[0, 0], pm[0])),
        f"max|diff| = {np.abs(act[0, 0] - pm[0]).max():.6g}",
    ))
    checks.append((
        "final iterate IS refit (arenas are distinct)",
        bool(np.abs(pm[1]).max() > 0.0), f"max|mean| = {np.abs(pm[1]).max():.6g}",
    ))
    checks.append((
        "topk fits the population",
        1 < a.topk <= a.num_samples, f"topk={a.topk} N={a.num_samples}",
    ))
    checks.append((
        "elite indices recorded for both iterations",
        d["native_elite"].shape == (2, a.topk), f"native_elite {d['native_elite'].shape}",
    ))

    ok = True
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
        ok &= passed
    print("index-0 pre-refit gate:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
