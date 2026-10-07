#!/usr/bin/env python3
"""Self-trained refinement of the event code (no state labels).

The SFA+ICA code reads each light at .99-.998 per frame; a single wrong bit changes a Lights Out
solution completely, so plans built from one goal frame fail. The debounced code sequence from
build_events.py (per-bit hysteresis over time) is more reliable than single frames. This script
uses it as pseudo-labels and fits a per-bit logistic readout on the full flattened patch tokens
(64 x 192), skipping frames within `--margin` of a detected event, where the label is ambiguous.
The bit order is unchanged, so event types, the event WM and the cost-to-go stay valid.
Privileged button states only score the result (train_code.evaluate).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from common import Split, load_code, save_json, size_of, tokens
from train_code import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--code", type=Path, required=True, help="code.pt to refine (bit order is kept)")
    ap.add_argument("--events", type=Path, required=True, help="build_events.py output for that code")
    ap.add_argument("--train-frames", type=int, default=600_000)
    ap.add_argument("--val-frames", type=int, default=100_000)
    ap.add_argument("--margin", type=int, default=12)
    ap.add_argument("--labels", choices=["debounced", "segment"], default="segment")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    rows, cols = size_of(a.env)
    t0 = time.time()
    enc, old_logits, ck = load_code(a.code, dev)
    mask = np.load(a.events / "bit_mask.npy")
    tr = np.load(a.events / "events_train.npz")
    train, val = Split(a.cache, "train", a.train_frames), Split(a.cache, "val", a.val_frames)
    y = tr["codes"][: train.n].astype(np.float32)                    # debounced code (pseudo-labels)
    if a.labels == "segment":
        # Object state is constant between consecutive events: label every frame of a segment with the
        # per-bit majority of the code over that segment (frames under arm occlusion get the right label).
        ep = train.ep
        bounds = np.r_[0, np.nonzero(ep[1:] != ep[:-1])[0] + 1]
        cuts = np.unique(np.r_[bounds, tr["t"][tr["t"] < train.n - 1] + 1, train.n])
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            y[s0:s1] = (y[s0:s1].mean(0) > 0.5)
    # Frames near an event span are ambiguous (lights revealed over several frames).
    near = np.zeros(train.n + 1, np.int64)
    for s, e in zip(tr["t_start"], tr["t"]):
        if s < train.n:
            near[max(0, s - a.margin)] += 1
            near[min(train.n, e + a.margin + 1)] -= 1
    ok = np.nonzero(np.cumsum(near)[: train.n] == 0)[0]
    K = y.shape[1]
    D = 64 * 192
    sample = tokens(enc, train.obs[rng.choice(ok, 4096)], dev).reshape(-1, D)
    tmu, tsd = sample.mean(0), sample.std(0) + 1e-4
    lin = torch.nn.Linear(D, K).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    yt = torch.as_tensor(y, device=dev)
    log = []
    for step in range(a.steps):
        i = rng.choice(ok, a.batch)
        x = (tokens(enc, train.obs[i], dev).reshape(-1, D) - tmu) / tsd
        loss = F.binary_cross_entropy_with_logits(lin(x), yt[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            log.append({"step": step, "loss": loss.item(), "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    # Fold the standardisation into an affine map on raw flattened tokens: logits = tok @ A + c.
    W, b = lin.weight.detach(), lin.bias.detach()
    A = (W / tsd).T                                                  # (D, K)
    c = b - (tmu / tsd) @ W.T
    with torch.no_grad():
        pv, pold = [], []
        for s in range(0, val.n, 4096):
            tok = tokens(enc, val.obs[s:s + 4096], dev)
            pv.append(torch.sigmoid(tok.reshape(len(tok), -1) @ A + c).cpu())
            pold.append(torch.sigmoid(old_logits(tok)).cpu()[:, torch.as_tensor(mask)])
        pv, pold = torch.cat(pv).numpy(), torch.cat(pold).numpy()
    res_new, _ = evaluate(pv, val, rows, cols)
    res_old, _ = evaluate(pold, val, rows, cols)
    # Frames where every light is read correctly by its best-matching bit (privileged scoring).
    def exact(p):
        b_ = p > 0.5
        st = val.button_states.astype(bool)
        agree = (b_[:, :, None] == st[:, None, :]).mean(0)                 # (K, L)
        k = np.argmax(np.maximum(agree, 1 - agree), 0)                     # best bit per light
        inv = agree[k, np.arange(st.shape[1])] < 0.5
        return float(((b_[:, k] ^ inv[None]) == st).all(1).mean())
    out = {"refined": {k: res_new[k] for k in ("per_light_best_bit_acc", "lights_with_bit_acc_ge_0.99", "all")},
           "original": {k: res_old[k] for k in ("per_light_best_bit_acc", "lights_with_bit_acc_ge_0.99", "all")},
           "frame_exact_refined": exact(pv), "frame_exact_original": exact(pold),
           "pseudo_label_frames": int(len(ok)), "train_frames": int(train.n), "log": log}
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "linear", "bits": K, "mu": torch.zeros(D), "A": A.cpu(), "c": c.cpu(),
                "slow_mask": torch.ones(K, dtype=torch.bool), "base": ck["base"], "env": a.env,
                "refined_from": str(a.code)}, a.out / "code.pt")
    np.save(a.out / "bit_mask.npy", np.ones(K, bool))
    save_json(a.out / "refine_eval.json", out)
    print({k: out[k] for k in ("frame_exact_refined", "frame_exact_original")}, flush=True)
    print("refined", out["refined"]["per_light_best_bit_acc"], flush=True)
    print("original", out["original"]["per_light_best_bit_acc"], flush=True)


if __name__ == "__main__":
    main()
