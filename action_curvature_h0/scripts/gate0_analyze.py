#!/usr/bin/env python3
"""Gate 0: does the frozen latent admit a descent certificate at all?

A control-Lyapunov cost must fall exactly when the true distance to the goal
falls, so the statistic is a SIGN agreement on transitions of off-policy
(planner-induced) rollouts, not a regression error:

    agreement = P[ sign(Delta cost) == sign(Delta true_distance) ]   (chance 0.5)

**The test is WITHIN a snapshot.**  V(., g) only has to certify progress for the
goal at hand, and a goal-conditioned readout fit across snapshots cannot be
learned from a few dozen goals -- an earlier across-snapshot version of this
script produced a high in-sample and a chance-level held-out number, which
measured goal generalisation rather than the latent's content.  Here the goal is
fixed inside a snapshot, the readout is fit on some of that snapshot's
trajectories and evaluated on its held-out trajectories, and the per-snapshot
scores are then pooled.

Arms:
  deployed   ``||z - z_goal||^2`` -- what the planner uses today
  latent_ub  best readout fit FROM THE LATENT -- an upper bound on any
             latent-based cost, including every published one
  proprio    same readout from proprioception, which the world model never sees
  oracle     the true distance -- 1.0 by construction, a wiring check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--test-frac", type=float, default=0.34,
                   help="fraction of each snapshot's trajectories held out")
    p.add_argument("--min-abs-delta-m", type=float, default=1e-4)
    p.add_argument("--min-test-transitions", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260901)
    return p.parse_args()


def agreement(delta_cost: np.ndarray, delta_true: np.ndarray) -> float:
    m = np.abs(delta_true) > 0
    if not m.any():
        return float("nan")
    return float((np.sign(delta_cost[m]) == np.sign(delta_true[m])).mean())


def fit_readout(x_tr: np.ndarray, y_tr: np.ndarray, seed: int,
                n_components: int | None = None):
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(x_tr)
    xs = sc.transform(x_tr)
    cap = 64 if n_components is None else n_components
    k = int(min(cap, xs.shape[1], max(2, xs.shape[0] - 1)))
    pca = PCA(n_components=k, random_state=seed).fit(xs)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800,
                       random_state=seed, alpha=1e-3)
    mlp.fit(pca.transform(xs), y_tr)
    return lambda x: mlp.predict(pca.transform(sc.transform(x)))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for f in sorted(args.root.glob("snapshot_*/gate0.npz")):
        d = {k: np.load(f)[k] for k in np.load(f).files}
        z, prop, dist, tid = d["z"], d["proprio"], d["true_distance_m"], d["traj_id"]
        goal = d["z_goal"][0]
        pair = tid[1:] == tid[:-1]                    # no cross-trajectory pairs
        big = (np.abs(dist[1:] - dist[:-1]) >= args.min_abs_delta_m) & pair
        tr_id = tid[:-1][big].astype(int)
        z0, z1 = z[:-1][big], z[1:][big]
        p0, p1 = prop[:-1][big], prop[1:][big]
        d0, d1 = dist[:-1][big], dist[1:][big]
        dep = ((z1 - goal) ** 2).sum(1) - ((z0 - goal) ** 2).sum(1)

        trajs = np.unique(tr_id)
        if len(trajs) < 3:
            continue
        perm = rng.permutation(trajs)
        n_te = max(1, int(round(args.test_frac * len(trajs))))
        te_set = set(perm[:n_te].tolist())
        te = np.array([t in te_set for t in tr_id])
        if te.sum() < args.min_test_transitions or (~te).sum() < args.min_test_transitions:
            continue

        dtrue = (d1 - d0)[te]
        row = {"snapshot": int(d["snapshot"]), "n_test": int(te.sum()),
               "oracle": agreement(dtrue, dtrue),
               "deployed": agreement(dep[te], dtrue)}
        # latent_matched uses the SAME number of PCA components as proprio, so
        # the comparison is about information content rather than about a
        # 3-dim input generalising better than a 192-dim one from small data.
        k_prop = int(p0.shape[1])
        arms = (("latent", z0, z1, None), ("proprio", p0, p1, None),
                ("latent_matched", z0, z1, k_prop))
        for name, a0, a1, k in arms:
            h = fit_readout(np.vstack([a0[~te], a1[~te]]),
                            np.concatenate([d0[~te], d1[~te]]), args.seed, k)
            row[name] = agreement(h(a1[te]) - h(a0[te]), dtrue)
            h_in = fit_readout(np.vstack([a0[te], a1[te]]),
                               np.concatenate([d0[te], d1[te]]), args.seed, k)
            row[f"{name}_insample"] = agreement(h_in(a1[te]) - h_in(a0[te]), dtrue)
        rows.append(row)

    if not rows:
        raise RuntimeError("no snapshot had enough transitions")

    def pooled(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        boot = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(10000)]
        return {"mean": float(v.mean()),
                "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "n_snapshots": int(len(v))}

    report = {"n_snapshots": len(rows),
              "n_test_transitions": int(sum(r["n_test"] for r in rows)),
              "chance": 0.5,
              **{k: pooled(k) for k in ("oracle", "deployed", "latent",
                                        "latent_insample", "latent_matched",
                                        "latent_matched_insample", "proprio",
                                        "proprio_insample")},
              "per_snapshot": rows}
    ub = report["latent"]["mean"]
    hi = report["latent"]["ci"][1]
    report["verdict"] = (
        "LATENT_CANNOT_CERTIFY_PROGRESS" if hi < 0.6 else
        "LATENT_ADMITS_A_DESCENT_CERTIFICATE" if ub > 0.75 else
        "AMBIGUOUS")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_snapshot"}, indent=2))


if __name__ == "__main__":
    main()
