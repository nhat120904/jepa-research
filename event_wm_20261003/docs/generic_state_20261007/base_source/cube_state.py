#!/usr/bin/env python3
"""Cube domain, step 1: does the label-free slow-feature recipe recover object state?

Puzzle state is binary (lights); cube state is continuous (positions). This analysis runs the same
front end as sfa_code.py (PCA-whitened patch tokens -> SFA) and then:
  1. ICA inside the N slowest directions -> independent slow sources;
  2. groups sources into objects by co-change: sources whose changes happen at the same time belong
     to the same object (a carried cube changes its coordinates together);
  3. scores everything against the privileged cube positions (qpos), which are not inputs:
     linear R^2 of each true coordinate from the slow subspace, best |corr| per coordinate among the
     sources, and the purity of the co-change groups w.r.t. true cube identity.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import Split, load_encoder, save_json, tokens
from sfa_code import fastica

SLICES = {"single": [14], "double": [14, 21], "triple": [14, 21, 28], "quadruple": [14, 21, 28, 35]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--kind", default="triple")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--fit-frames", type=int, default=300_000)
    ap.add_argument("--val-frames", type=int, default=100_000)
    ap.add_argument("--pcs", type=int, default=1024)
    ap.add_argument("--ica-dims", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    dev = "cuda"
    t0 = time.time()
    train, val = Split(a.cache, "train", a.fit_frames), Split(a.cache, "val", a.val_frames)
    enc, _ = load_encoder(a.base, dev)
    D, bs = 64 * 192, 2002
    rng = np.random.default_rng(0)

    def flat(obs, i0, i1):
        return tokens(enc, obs[i0:i1], dev).reshape(i1 - i0, D)

    mu0 = tokens(enc, train.obs[np.sort(rng.integers(0, train.n, 4096))], dev).reshape(-1, D).mean(0)
    s1 = torch.zeros(D, device=dev, dtype=torch.float64)
    xtx = torch.zeros(D, D, device=dev)
    for i in range(0, train.n, bs):
        X = flat(train.obs, i, min(i + bs, train.n)) - mu0
        s1 += X.double().sum(0)
        xtx += X.T @ X
    delta = (s1 / train.n).float()
    mu = mu0 + delta
    cov = xtx / train.n - torch.outer(delta, delta)
    del xtx
    ev, evec = torch.linalg.eigh(cov)
    del cov
    ev, evec = ev.flip(0), evec.flip(1)
    Wp = evec[:, : a.pcs] / ev[: a.pcs].clamp_min(1e-8).sqrt()
    del evec
    dd = torch.zeros(a.pcs, a.pcs, device=dev, dtype=torch.float64)
    nd = 0
    for i in range(0, train.n, bs):
        j = min(i + bs, train.n)
        Z = (flat(train.obs, i, j) - mu) @ Wp
        same = torch.as_tensor(train.ep[i + 1:j] == train.ep[i:j - 1], device=dev)
        dZ = (Z[1:] - Z[:-1])[same]
        dd += (dZ.T @ dZ).double()
        nd += len(dZ)
    slowness, V = torch.linalg.eigh((dd / nd).float())
    print({"slowness_head": np.round(slowness[:40].cpu().numpy(), 5).tolist(), "min": round((time.time() - t0) / 60, 1)}, flush=True)

    def whitened(obs, n):
        return torch.cat([(tokens(enc, obs[i:min(i + 4096, n)], dev).reshape(-1, D) - mu) @ Wp for i in range(0, n, 4096)])

    nfit = min(train.n, 200_000)
    Zf, Zv = whitened(train.obs, nfit), whitened(val.obs, val.n)
    sl = SLICES[a.kind]
    Pf = np.concatenate([train.qpos[:nfit, s:s + 3] for s in sl], 1)            # (n, 3K) privileged
    Pv = np.concatenate([val.qpos[:, s:s + 3] for s in sl], 1)
    K = len(sl)
    res = {"slowness_head": np.round(slowness[:64].cpu().numpy(), 5).tolist()}
    for N in a.ica_dims:
        Yf, Yv = (Zf @ V[:, :N]).cpu().numpy(), (Zv @ V[:, :N]).cpu().numpy()
        # linear decodability of true coordinates from the N slowest directions (fit on train, score val)
        Xf = np.c_[Yf, np.ones(len(Yf))]
        W, *_ = np.linalg.lstsq(Xf, Pf, rcond=None)
        pred = np.c_[Yv, np.ones(len(Yv))] @ W
        r2 = 1 - ((pred - Pv) ** 2).mean(0) / Pv.var(0)
        # ICA sources and co-change grouping
        ym, ys = Yf.mean(0), Yf.std(0) + 1e-9
        Wi, it = fastica(torch.as_tensor((Yf - ym) / ys, device=dev, dtype=torch.float32), contrast="logcosh")
        S = ((Yv - ym) / ys) @ Wi.cpu().numpy().T                                # (nv, N) sources
        corr = np.abs(np.corrcoef(np.c_[S, Pv].T)[:N, N:])                         # (N, 3K)
        same_v = val.ep[1:] == val.ep[:-1]
        ch = np.abs(np.diff(S, axis=0))[same_v]
        ch = (ch > np.percentile(ch, 99, axis=0)[None]).astype(float)              # top-1% changes per source
        # smooth over a short window so changes within one carry coincide
        k_ = np.ones(15)
        chs = np.stack([np.convolve(ch[:, j], k_, mode="same") > 0 for j in range(N)], 1).astype(float)
        cc = np.corrcoef(chs.T)
        # sources that track a cube coordinate (best |corr| > .5) and their co-change groups
        best_coord = corr.argmax(1)
        tracked = np.nonzero(corr.max(1) > 0.5)[0]
        res[f"N{N}"] = {
            "r2_per_coordinate": np.round(r2, 3).tolist(),
            "best_abs_corr_per_coordinate": np.round(corr.max(0), 3).tolist(),
            "sources_tracking_a_coordinate": int(len(tracked)),
            "tracked_source_cube": (best_coord[tracked] // 3).tolist(),
            "cochange_corr_within_true_cube": float(np.nanmean([cc[i, j] for i in tracked for j in tracked
                                                                if i < j and best_coord[i] // 3 == best_coord[j] // 3])) if len(tracked) > 1 else None,
            "cochange_corr_across_cubes": float(np.nanmean([cc[i, j] for i in tracked for j in tracked
                                                            if i < j and best_coord[i] // 3 != best_coord[j] // 3])) if len(tracked) > 1 else None,
            "ica_iters": int(it)}
        print(N, json.dumps(res[f"N{N}"]), flush=True)
    save_json(a.out / "cube_state.json", res)


if __name__ == "__main__":
    main()
