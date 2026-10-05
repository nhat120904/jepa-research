#!/usr/bin/env python3
"""Linear event code: slow feature analysis + ICA on frozen patch tokens (no state labels).

1. PCA-whiten the flattened patch tokens (64 x 192) of contiguous play frames.
2. SFA: the directions whose frame-to-frame change has the least variance. Object state
   (button lights) changes only at presses; the arm moves every step.
3. The number of slow components m is the largest ratio gap in the sorted slowness spectrum.
4. ICA (symmetric FastICA, logcosh) inside the slow subspace separates independent binary
   sources (individual lights); each component is binarised at the midpoint of a 1-D 2-means.

The result is saved in the same code.pt format as train_code.py (kind = "linear") and scored
with train_code.evaluate (privileged button states score the code; they are not inputs).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import Split, load_encoder, save_json, size_of, tokens
from train_code import evaluate


def fastica(Y, iters=2000, seed=0, contrast="cube", tol=1e-5):
    """Symmetric FastICA on whitened Y (n, m) -> unmixing W (m, m).
    contrast "cube" (kurtosis) suits binary, sub-Gaussian sources; "logcosh" is the default of FastICA."""
    import torch

    m = Y.shape[1]
    g = torch.Generator(device=Y.device).manual_seed(seed)
    W = torch.linalg.qr(torch.randn(m, m, device=Y.device, generator=g))[0]

    def sym(W):
        s, U = torch.linalg.eigh(W @ W.T)
        return U @ torch.diag(s.clamp_min(1e-12).rsqrt()) @ U.T @ W

    W = sym(W)
    for it in range(iters):
        WX = Y @ W.T
        if contrast == "cube":
            G, dG = WX ** 3, 3 * WX ** 2
        else:
            G = torch.tanh(WX)
            dG = 1 - G ** 2
        Wn = (G.T @ Y) / len(Y) - torch.diag(dG.mean(0)) @ W
        Wn = sym(Wn)
        lim = (torch.abs(torch.diag(Wn @ W.T)) - 1).abs().max().item()
        W = Wn
        if lim < tol:
            break
    return W, it


def two_means_threshold(x, iters=50):
    c = np.percentile(x, [10, 90]).astype(np.float64)
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None]).argmin(1)
        c = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(2)])
    thr = c.mean()
    lo, hi = x[x <= thr], x[x > thr]
    sep = abs(c[1] - c[0]) / (np.sqrt(0.5 * (lo.var() + hi.var())) + 1e-9)
    return float(thr), float(sep), float((x > thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--fit-frames", type=int, default=300_000, help="contiguous train frames for PCA/SFA")
    ap.add_argument("--val-frames", type=int, default=100_000)
    ap.add_argument("--pcs", type=int, default=1024)
    ap.add_argument("--max-slow", type=int, default=48)
    ap.add_argument("--ica-dims", type=int, nargs="*", default=[32, 64, 128, 256],
                    help="ICA on the N slowest SFA directions (empty = eigen-gap rule)")
    ap.add_argument("--contrast", choices=["cube", "logcosh"], default="cube")
    ap.add_argument("--bands", action="store_true", help="ICA separately below and above the eigen-gap")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    dev = "cuda"
    rows, cols = size_of(a.env)
    t0 = time.time()
    train = Split(a.cache, "train", a.fit_frames)
    val = Split(a.cache, "val", a.val_frames)
    enc, _ = load_encoder(a.base, dev)
    D = 64 * 192
    bs = 2002                                                     # two whole episodes per batch

    def tok_flat(i0, i1, obs):
        return tokens(enc, obs[i0:i1], dev).reshape(i1 - i0, D)

    # Pass 1: mean and covariance of flattened tokens (accumulated around a provisional mean).
    rng = np.random.default_rng(0)
    mu0 = tokens(enc, train.obs[np.sort(rng.integers(0, train.n, 4096))], dev).reshape(-1, D).mean(0)
    s1 = torch.zeros(D, device=dev, dtype=torch.float64)
    xtx = torch.zeros(D, D, device=dev, dtype=torch.float32)
    for i in range(0, train.n, bs):
        X = tok_flat(i, min(i + bs, train.n), train.obs) - mu0
        s1 += X.double().sum(0)
        xtx += X.T @ X
    delta = (s1 / train.n).float()
    mu = mu0 + delta
    cov = xtx / train.n - torch.outer(delta, delta)
    del xtx
    evals, evecs = torch.linalg.eigh(cov)
    del cov
    evals, evecs = evals.flip(0), evecs.flip(1)
    P = a.pcs
    Wp = evecs[:, :P] / evals[:P].clamp_min(1e-8).sqrt()        # (D, P) whitening
    var_kept = float(evals[:P].sum() / evals.clamp_min(0).sum())
    del evecs
    print({"pca_var_kept": var_kept, "min": round((time.time() - t0) / 60, 1)}, flush=True)

    # Pass 2: covariance of whitened frame-to-frame differences inside episodes.
    dd = torch.zeros(P, P, device=dev, dtype=torch.float64)
    nd = 0
    for i in range(0, train.n, bs):
        j = min(i + bs, train.n)
        Z = (tok_flat(i, j, train.obs) - mu) @ Wp
        same = torch.as_tensor(train.ep[i + 1:j] == train.ep[i:j - 1], device=dev)
        dZ = (Z[1:] - Z[:-1])[same]
        dd += (dZ.T @ dZ).double()
        nd += len(dZ)
    dcov = (dd / nd).float()
    slowness, V = torch.linalg.eigh(dcov)                         # ascending: slowest first
    sl = slowness[: a.max_slow].cpu().numpy()
    ratios = sl[1:] / np.maximum(sl[:-1], 1e-12)
    m_gap = int(np.argmax(ratios[3:]) + 4)                        # eigen-gap rule (at least 4 components)
    print({"slowness_head": np.round(sl[:40], 6).tolist(), "m_gap": m_gap, "min": round((time.time() - t0) / 60, 1)},
          flush=True)

    # Whitened tokens of the fit frames (contiguous episodes) and of the validation frames.
    def whitened(obs, n):
        out = []
        for i in range(0, n, 4096):
            out.append((tokens(enc, obs[i:min(i + 4096, n)], dev).reshape(-1, D) - mu) @ Wp)
        return torch.cat(out)

    nfit = min(train.n, 200_000)
    Zfit, Zval = whitened(train.obs, nfit), whitened(val.obs, val.n)
    same_fit = train.ep[1:nfit] == train.ep[:nfit - 1]
    scale = 20.0                                                  # sharp sigmoid: probability ~ bit
    summary = {}
    for N in a.ica_dims or [m_gap]:
        # ICA inside the N slowest SFA directions; keep components that are binary and flip rarely.
        Ws = V[:, :N]
        Y = Zfit @ Ws
        ym, ys = Y.mean(0), Y.std(0)
        if a.bands and m_gap < N:
            # ICA separately in each slowness band split at the eigen-gap: [0, m_gap) and [m_gap, N).
            W = torch.zeros(N, N, device=dev)
            W1, it1 = fastica(((Y - ym) / ys)[:, :m_gap], contrast=a.contrast)
            W2, it2 = fastica(((Y - ym) / ys)[:, m_gap:], contrast=a.contrast)
            W[:m_gap, :m_gap], W[m_gap:, m_gap:] = W1, W2
            iters = max(it1, it2)
        else:
            W, iters = fastica((Y - ym) / ys, contrast=a.contrast)
        ic = (((Y - ym) / ys) @ W.T).cpu().numpy()
        del Y
        thr, sep, frac = (np.array(x) for x in zip(*[two_means_threshold(ic[:, k]) for k in range(N)]))
        bits = ic > thr[None]
        flip = (bits[1:] != bits[:-1])[same_fit].mean(0)          # binarised flip rate per step
        valid = (frac > 0.05) & (frac < 0.95)
        lf = np.log(flip[valid] + 1e-6)
        c2 = np.array([lf.min(), lf.max()])
        for _ in range(50):
            lab = np.abs(lf[:, None] - c2[None]).argmin(1)
            c2 = np.array([lf[lab == j].mean() if (lab == j).any() else c2[j] for j in range(2)])
        slow_c = np.zeros(N, bool)
        slow_c[np.nonzero(valid)[0][lab == 0]] = True
        # Binary components: 2-means on log separation (a slowly drifting continuous component also
        # flips rarely once binarised at its median, but it is not bimodal).
        ls_ = np.log(sep[valid])
        c3 = np.array([ls_.min(), ls_.max()])
        for _ in range(50):
            lab3 = np.abs(ls_[:, None] - c3[None]).argmin(1)
            c3 = np.array([ls_[lab3 == j].mean() if (lab3 == j).any() else c3[j] for j in range(2)])
        binary = np.zeros(N, bool)
        binary[np.nonzero(valid)[0][lab3 == 1]] = True
        sel = slow_c & binary
        if sel.sum() > 63:                                       # keep the 63 slowest (pack() limit)
            keep = np.argsort(np.where(sel, flip, np.inf))[:63]
            sel = np.zeros(N, bool)
            sel[keep] = True
        Wsel = W.T[:, torch.as_tensor(np.nonzero(sel)[0], device=dev)]           # (N, m)
        thr_t = torch.as_tensor(thr[sel], device=dev, dtype=torch.float32)
        A = (Wp @ Ws / ys) @ Wsel                                 # (D, m): code logits = (tok - mu) @ A + c
        c = -(ym / ys) @ Wsel - thr_t
        pv = torch.sigmoid(scale * (((Zval @ Ws - ym) / ys) @ Wsel - thr_t)).cpu().numpy()
        res, _ = evaluate(pv, val, rows, cols)
        res.update({"method": "sfa+ica, binarised-flip + bimodality selection", "ica_dims": N, "selected": int(sel.sum()),
                    "contrast": a.contrast, "bands": bool(a.bands), "slow_components": int(slow_c.sum()), "binary_components": int(binary.sum()),
                    "binary_separation_threshold": float(np.exp(c3.mean())),
                    "pcs": P, "pca_var_kept": var_kept, "ica_iters": int(iters), "m_gap": m_gap,
                    "slowness_head": np.round(sl, 6).tolist(),
                    "selected_flip_rate": np.round(np.sort(flip[sel]), 5).tolist(),
                    "selected_separation": np.round(sep[sel], 2).tolist(),
                    "rejected_flip_rate_min": float(flip[valid & ~sel].min()) if (valid & ~sel).any() else None,
                    "minutes": round((time.time() - t0) / 60, 1)})
        out = a.out / f"N{N}"
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "linear", "bits": int(sel.sum()), "mu": mu.cpu(), "A": (A * scale).cpu(),
                    "c": (c * scale).cpu(), "slow_mask": torch.ones(int(sel.sum()), dtype=torch.bool),
                    "base": str(a.base), "env": a.env, "ica_dims": N}, out / "code.pt")
        save_json(out / "code_eval.json", res)
        summary[N] = {"selected": int(sel.sum()), "lights_ge_0.99": res["lights_with_bit_acc_ge_0.99"],
                      "all": res.get("all"), "per_light": res["per_light_best_bit_acc"]}
        print(N, json.dumps(summary[N], default=float), flush=True)
    save_json(a.out / "summary.json", summary)

if __name__ == "__main__":
    main()
