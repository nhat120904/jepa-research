"""Apply the CI-aware Boundary Blindness gate to a {dataset}_boundary.csv.

Per model x regime: n_boundary-weighted pooled bb_boundary, plus per-task paired
comparisons (contact-rich regime vs free_space within the same task, CI-aware:
elevated iff bb_boundary_lo(regime) > bb_boundary_hi(free_space))."""
import sys

import numpy as np
import pandas as pd

CONTACT = ["pre_grasp", "gripper_actuation", "contact_manipulation"]

df = pd.read_csv(sys.argv[1])
print(f"rows={len(df)} models={sorted(df.model.unique())}\n")

for model, g in df.groupby("model"):
    print(f"=== {model} ===")
    print(f"{'regime':22s} {'cells':>5s} {'n_trans':>8s} {'n_bound':>8s} "
          f"{'bb(w)':>7s} {'bb_b(w)':>8s} {'bb_b range over tasks':>24s}")
    for regime, rg in g.groupby("regime"):
        ok = rg[np.isfinite(rg.bb_boundary) & (rg.n_boundary > 0)]
        w = ok.n_boundary
        bbw = float(np.average(rg.bb, weights=rg.n_transitions))
        bbbw = float(np.average(ok.bb_boundary, weights=w)) if len(ok) else float("nan")
        rng = (f"[{ok.bb_boundary.min():.3f}, {ok.bb_boundary.max():.3f}]"
               if len(ok) else "n/a")
        print(f"{regime:22s} {len(rg):5d} {rg.n_transitions.sum():8d} "
              f"{int(rg.n_boundary.sum()):8d} {bbw:7.3f} {bbbw:8.3f} {rng:>24s}")

    # Paired per-task CI-aware comparison vs free_space.
    fs = g[g.regime == "free_space"].set_index("task")
    for regime in CONTACT:
        rg = g[g.regime == regime].set_index("task")
        shared = [t for t in rg.index if t in fs.index]
        elevated = higher = lower = conf_lower = 0
        details = []
        for t in shared:
            r, f = rg.loc[t], fs.loc[t]
            if not (np.isfinite(r.bb_boundary) and np.isfinite(f.bb_boundary)):
                continue
            if r.bb_boundary > f.bb_boundary:
                higher += 1
            else:
                lower += 1
            if r.bb_boundary_lo > f.bb_boundary_hi:
                elevated += 1
                details.append(f"{t}^")
            elif r.bb_boundary_hi < f.bb_boundary_lo:
                conf_lower += 1
                details.append(f"{t}v")
        n = higher + lower
        if n:
            print(f"  {regime:20s} vs free_space (paired, {n} tasks): "
                  f"{higher} higher / {lower} lower; CI-confident: "
                  f"{elevated} elevated, {conf_lower} lower  {details}")
    print()
