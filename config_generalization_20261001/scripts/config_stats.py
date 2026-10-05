#!/usr/bin/env python3
"""How novel are held-out configurations in OGBench play data? (CPU, no model)

Puzzle: a configuration is the button-state vector. For every validation step we
compute the Hamming distance to the nearest configuration present anywhere in the
training split (0 = seen). Events are steps where the button states change.
Cube: a configuration is the three block positions. We report stack structure
(how many blocks rest on another block) and, for a subsample of validation
steps, the nearest-neighbour distance to training configurations (max over
blocks of the xyz distance, blocks are distinguishable by colour).
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np


def load(path):
    z = np.load(path)
    out = {k: z[k] for k in z.files if k != "observations"}
    out["n"] = len(z["terminals"])
    return out


def pack(bits):
    return (bits.astype(np.int64) << np.arange(bits.shape[1], dtype=np.int64)).sum(1)


def hamming_nn(queries, train_set, n_bits, cap=6):
    """Distance to nearest member of train_set by enumerating flips up to cap."""
    flips = {d: [sum(1 << i for i in c) for c in combinations(range(n_bits), d)] for d in range(1, cap + 1)}
    out = np.full(len(queries), cap + 1, np.int64)
    for qi, q in enumerate(queries):
        q = int(q)
        if q in train_set:
            out[qi] = 0
            continue
        for d in range(1, cap + 1):
            if any((q ^ f) in train_set for f in flips[d]):
                out[qi] = d
                break
    return out


def puzzle_stats(tr, va):
    bt, bv = tr["button_states"].astype(np.int64), va["button_states"].astype(np.int64)
    n = bt.shape[1]
    ct, cv = pack(bt), pack(bv)
    train_set = set(np.unique(ct).tolist())
    uq_v, inv = np.unique(cv, return_inverse=True)
    d_uq = hamming_nn(uq_v, train_set, n, cap=min(4, n))   # >4 reported as 5
    d_val = d_uq[inv]
    ev_t = np.nonzero((bt[1:] != bt[:-1]).any(1) & (tr["terminals"][:-1] == 0))[0]
    ev_v = np.nonzero((bv[1:] != bv[:-1]).any(1) & (va["terminals"][:-1] == 0))[0]
    toggles = (bt[ev_t + 1] != bt[ev_t]).sum(1)
    d_ev = d_val[ev_v]  # novelty of the pre-event configuration
    hist = lambda d: {str(k): int((d == k).sum()) for k in range(int(d.max()) + 1)}
    return {
        "buttons": int(n), "configs_total": int(2 ** n),
        "train_steps": int(tr["n"]), "train_unique_configs": len(train_set),
        "train_coverage": len(train_set) / 2 ** n,
        "train_events": int(len(ev_t)), "train_episodes": int(tr["terminals"].sum()),
        "toggles_per_event": {str(k): int((toggles == k).sum()) for k in range(1, 6)},
        "val_steps": int(va["n"]), "val_unique_configs": int(len(uq_v)),
        "val_step_novelty_hist": hist(d_val), "val_event_novelty_hist": hist(d_ev),
        "val_events": int(len(ev_v)),
        "frac_val_events_unseen": float((d_ev > 0).mean()),
    }


def cube_stats(tr, va, qpos_slices, rng):
    def blocks(d):
        return np.stack([d["qpos"][:, s:s + 3] for s in qpos_slices], 1)  # (N, k, 3)

    pt, pv = blocks(tr), blocks(va)

    def stacked(p):  # block i rests on block j: xy within 2 cm and z higher by ~4 cm
        k = p.shape[1]
        on = np.zeros(len(p), np.int64)
        for i in range(k):
            for j in range(k):
                if i != j:
                    dxy = np.linalg.norm(p[:, i, :2] - p[:, j, :2], axis=-1)
                    dz = p[:, i, 2] - p[:, j, 2]
                    on += ((dxy < 0.02) & (np.abs(dz - 0.04) < 0.01)).astype(np.int64)
        return on

    st, sv = stacked(pt), stacked(pv)
    sub_v = rng.choice(len(pv), size=min(4000, len(pv)), replace=False)
    flat_t = pt.reshape(len(pt), -1)
    nn = np.empty(len(sub_v))
    for a in range(0, len(sub_v), 50):
        q = pv[sub_v[a:a + 50]].reshape(-1, 1, pt.shape[1], 3)
        best = np.full(len(q), np.inf)
        for b in range(0, len(pt), 100000):
            c = pt[b:b + 100000][None]
            dist = np.linalg.norm(q - c, axis=-1).max(-1)   # (q, chunk)
            best = np.minimum(best, dist.min(1))
        nn[a:a + 50] = best
    hist = lambda s: {str(k): int((s == k).sum()) for k in range(int(s.max()) + 1)}
    return {
        "train_steps": int(tr["n"]), "val_steps": int(va["n"]),
        "train_stack_hist": hist(st), "val_stack_hist": hist(sv),
        "val_nn_dist_quantiles_m": {q: float(np.quantile(nn, float(q))) for q in ("0.1", "0.5", "0.9", "0.99")},
        "val_nn_dist_by_stack": {str(k): float(np.median(nn[sv[sub_v] == k])) for k in np.unique(sv[sub_v])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--envs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("bulk data analysis runs under sbatch")
    rng = np.random.default_rng(0)
    rep = {}
    for env in a.envs:
        tr = load(a.data / f"{env}.npz")
        va = load(a.data / f"{env}-val.npz")
        if "puzzle" in env:
            rep[env] = puzzle_stats(tr, va)
        else:
            import gymnasium
            import ogbench  # noqa: F401  (registers the manipspace envs)

            kind = env.split("-")[2]                        # visual-cube-triple-play-v0 -> triple
            e = gymnasium.make(f"cube-{kind}-v0").unwrapped
            e.reset(seed=0)
            slices = [int(e._model.joint(f"object_joint_{i}").qposadr[0]) for i in range(e._num_cubes)]
            rep[env] = cube_stats(tr, va, slices, rng) | {"qpos_block_slices": slices}
        print(env, json.dumps(rep[env]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
