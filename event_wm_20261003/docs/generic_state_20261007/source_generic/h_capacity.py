#!/usr/bin/env python3
"""Why does the imagined cost-to-go saturate at ~5 presses? Capacity check with exact labels.

The cost-to-go learned by value iteration in the event WM (train_planner.py) tracks the true
distance up to d* = 4 and is flat beyond (job 56939), like the GCIVL value. This script trains the
same network family by direct regression on exact distances (privileged labels; a diagnostic for
choosing the heuristic's input and size, not part of the method) on unlimited random pairs:
  - "concat":     [s, g]           (the current heuristic input)
  - "concat_xor": [s, g, s xor g]  (bitwise goal-difference features; domain-agnostic for binary codes)
  - widths 512 and 2048.
If exact-label regression also saturates, the limit is the function class/optimisation (the
Lights Out distance is a parity-like function of the state difference); if it does not, the
bootstrapped value iteration is the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from lightsout import solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn as nn

    dev = "cuda"
    A, M, piv, span = solver(a.rows, a.cols)
    n = a.rows * a.cols
    assert len(piv) == n, "full-rank boards only (4x5, 4x6)"
    E = torch.as_tensor(M[:, n:], device=dev, dtype=torch.float32)        # x = E t (mod 2) solves A x = t
    At = torch.as_tensor(A, device=dev, dtype=torch.float32)

    def batch(B, gen):
        s = torch.randint(0, 2, (B, n), device=dev, generator=gen).float()
        k = torch.randint(0, n + 1, (B, 1), device=dev, generator=gen)
        x = (torch.rand(B, n, device=dev, generator=gen).argsort(1) < k).float()   # k distinct presses
        g = (s + x @ At.T) % 2
        d = ((g - s).abs() @ E.T % 2).sum(1)                                     # exact minimal presses
        return s, g, d

    def make(kind, width):
        inp = 3 * n if kind == "concat_xor" else 2 * n
        net = nn.Sequential(nn.Linear(inp, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)).to(dev)

        def f(s, g):
            z = torch.cat([s, g, (s - g).abs()], -1) if kind == "concat_xor" else torch.cat([s, g], -1)
            return nn.functional.softplus(net(z)).squeeze(-1)
        return net, f

    test_gen = torch.Generator(device=dev).manual_seed(123)
    st, gt, dt = batch(50000, test_gen)
    res = {}
    for kind in ("concat", "concat_xor"):
        for width in (512, 2048):
            torch.manual_seed(0)
            gen = torch.Generator(device=dev).manual_seed(0)
            net, f = make(kind, width)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
            t0 = time.time()
            for step in range(a.steps):
                s, g, d = batch(a.batch, gen)
                loss = (f(s, g) - d).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                p = torch.cat([f(st[i:i + 10000], gt[i:i + 10000]) for i in range(0, len(st), 10000)])
                # single-press sign test: press a random button, does h move with d*?
                i = torch.randint(0, n, (len(st),), device=dev, generator=test_gen)
                s2 = (st + At[:, i].T) % 2
                d2 = ((gt - s2).abs() @ E.T % 2).sum(1)
                p2 = torch.cat([f(s2[j:j + 10000], gt[j:j + 10000]) for j in range(0, len(st), 10000)])
            d_np, p_np = dt.cpu().numpy(), p.cpu().numpy()
            sign = (np.sign((p2 - p).cpu().numpy()) == np.sign((d2 - dt).cpu().numpy()))
            key = f"{kind}_w{width}"
            res[key] = {
                "mse": float(((p_np - d_np) ** 2).mean()),
                "mean_by_dstar": {int(x): round(float(p_np[d_np == x].mean()), 2) for x in np.unique(d_np)},
                "sign_by_dstar": {f"{lo}-{hi}": round(float(sign[(d_np >= lo) & (d_np <= hi)].mean()), 3)
                                  for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 99))
                                  if ((d_np >= lo) & (d_np <= hi)).any()},
                "minutes": round((time.time() - t0) / 60, 1)}
            print(key, json.dumps(res[key]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "h_capacity.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
