#!/usr/bin/env python3
"""Gate 1: can the latent ORDER different candidates by outcome?

Gate 0 asked whether a function of the latent tracks progress along a single
rollout.  That is not the planner's question.  CEM never compares consecutive
states of one trajectory -- it compares the OUTCOMES OF DIFFERENT CANDIDATES and
keeps the ones it judges best.  This measures the upper bound on exactly that
comparison.

Within a snapshot (goal fixed), states from DIFFERENT candidate rollouts are
paired and the task is to say which of the two is truly closer to the goal:

    label = 1 if true_distance(i) < true_distance(j)

Three arms, all trained on pairs from training trajectories and evaluated on
pairs from held-out trajectories of the same snapshot:

  deployed   ``||z - z_goal||^2`` -- no training, what the planner uses
  latent     a learned pairwise classifier on (z_i, z_j) -- the upper bound on
             any latent-based ranker, which is what a cost really is
  proprio    the same classifier on proprioception

Chance is 0.5.  Pairs are drawn across trajectories, never within one, so the
statistic is about candidate comparison rather than about smoothness in time.
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
    p.add_argument("--label", default="")
    p.add_argument("--test-frac", type=float, default=0.34)
    p.add_argument("--max-pairs", type=int, default=20000)
    p.add_argument("--min-gap-m", type=float, default=5e-3,
                   help="ignore pairs whose true distances are within this; the "
                        "ordering of near-ties is not what a planner needs")
    p.add_argument("--seed", type=int, default=20260901)
    return p.parse_args()


def make_pairs(idx: np.ndarray, tid: np.ndarray, dist: np.ndarray,
               rng: np.random.Generator, n: int, min_gap: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random cross-trajectory pairs with a meaningful outcome gap."""
    a = rng.choice(idx, size=n * 4)
    b = rng.choice(idx, size=n * 4)
    ok = (tid[a] != tid[b]) & (np.abs(dist[a] - dist[b]) >= min_gap)
    a, b = a[ok][:n], b[ok][:n]
    return a, b, (dist[a] < dist[b]).astype(int)


def main() -> None:
    args = parse_args()
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(args.seed)
    rows = []
    for f in sorted(args.root.glob("snapshot_*/gate0.npz")):
        d = {k: np.load(f)[k] for k in np.load(f).files}
        z, pr, dist, tid = d["z"], d["proprio"], d["true_distance_m"], d["traj_id"]
        goal = d["z_goal"][0]
        trajs = np.unique(tid)
        if len(trajs) < 4:
            continue
        perm = rng.permutation(trajs)
        n_te = max(2, int(round(args.test_frac * len(trajs))))
        te_tr, tr_tr = set(perm[:n_te].tolist()), set(perm[n_te:].tolist())
        i_te = np.where(np.isin(tid, list(te_tr)))[0]
        i_tr = np.where(np.isin(tid, list(tr_tr)))[0]
        if len(i_te) < 40 or len(i_tr) < 40:
            continue

        a_tr, b_tr, y_tr = make_pairs(i_tr, tid, dist, rng, args.max_pairs, args.min_gap_m)
        a_te, b_te, y_te = make_pairs(i_te, tid, dist, rng, args.max_pairs // 4, args.min_gap_m)
        if len(y_tr) < 200 or len(y_te) < 100:
            continue

        row = {"snapshot": int(d["snapshot"]), "n_test_pairs": int(len(y_te))}
        # deployed: no training at all
        cost = ((z - goal) ** 2).sum(1)
        row["deployed"] = float(((cost[a_te] < cost[b_te]).astype(int) == y_te).mean())

        for name, feat in (("latent", z), ("proprio", pr)):
            def pair_feat(a, b):
                return np.hstack([feat[a], feat[b], feat[a] - feat[b]])
            X_tr, X_te = pair_feat(a_tr, b_tr), pair_feat(a_te, b_te)
            sc = StandardScaler().fit(X_tr)
            k = int(min(64, X_tr.shape[1], max(2, X_tr.shape[0] - 1)))
            pca = PCA(n_components=k, random_state=args.seed).fit(sc.transform(X_tr))
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=600,
                                random_state=args.seed, alpha=1e-3,
                                early_stopping=True)
            clf.fit(pca.transform(sc.transform(X_tr)), y_tr)
            row[name] = float((clf.predict(pca.transform(sc.transform(X_te))) == y_te).mean())
        rows.append(row)

    if not rows:
        raise RuntimeError("no snapshot yielded enough pairs")

    def pooled(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        bo = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(20000)]
        return {"mean": float(v.mean()),
                "ci": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                "n_snapshots": int(len(v))}

    rep = {"label": args.label, "n_snapshots": len(rows),
           "n_test_pairs": int(sum(r["n_test_pairs"] for r in rows)),
           "chance": 0.5,
           **{k: pooled(k) for k in ("deployed", "latent", "proprio")},
           "per_snapshot": rows}
    v = np.array([r["latent"] - r["deployed"] for r in rows])
    bo = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(20000)]
    rep["learned_minus_deployed"] = {
        "mean": float(v.mean()),
        "ci": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]}
    hi = rep["latent"]["ci"][1]
    rep["verdict"] = ("LATENT_CANNOT_RANK_CANDIDATES" if hi < 0.6 else
                      "LATENT_CAN_RANK_CANDIDATES" if rep["latent"]["mean"] > 0.75
                      else "AMBIGUOUS")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_snapshot"}, indent=2))


if __name__ == "__main__":
    main()
