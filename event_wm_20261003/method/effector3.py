#!/usr/bin/env python3
"""Component 5c v3: the EFFECTOR as the place where the commanded translation is VISIBLE, learned from frames + actions.

Why v3. A keypoint whose position feeds a learned camera map (effector2.py) is identified only up to an affine image of the
effector position: runs on held-out 3x3 put it off the arm (33 px), on an arm link with collapsed workspace bounds (no
vertical tracking), or let a running floor drift (q_z blew up). Here position never enters the prediction, so there is
no gauge freedom: a fully convolutional net with a small receptive field (~15 px) looks at the frame pair (x_t, x_{t+d})
and gives, at every cell of a 32 x 32 grid, an estimate of the mean commanded translation a_{t..t+d-1}[:3] and a
confidence logit; the prediction is the softmax(confidence)-weighted mean of the local estimates, trained with the squared
error. The command is an effector displacement, so the cells whose local motion predicts it are the effector's (an arm
link moves by configuration-dependent amounts, a static region not at all); the effector point is the confidence-weighted
mean cell position. One setting for every environment; frames and actions only.

Outputs (--out): effector.pt; track_{split}.npy (n, 4) float32 = u, v (px, from the pair (t, t+1); the last frame of an
episode copies the previous row), peak confidence, local residual; effector_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from utils import Frames, save_json

GAPS = (1, 2, 4)


def make_effector3(w=48):
    import torch
    import torch.nn as nn

    def cb(cin, cout, stride=1):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.GELU())

    class Effector3(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(cb(6, w), cb(w, w), cb(w, 2 * w, 2), cb(2 * w, 2 * w), cb(2 * w, 2 * w))   # RF ~15 px, 32 x 32
            self.est = nn.Conv2d(2 * w + 1, 3, 1)
            self.conf = nn.Conv2d(2 * w + 1, 1, 1)
            g = torch.arange(32, dtype=torch.float32) * 2 + 0.5
            self.register_buffer("gu", g.repeat(32)); self.register_buffer("gv", g.repeat_interleave(32))

        def forward(self, x0, x1, d):
            """x0, x1 (B, 3, 64, 64) in [0, 1], d (B,) gap -> prediction (B, 3), weights (B, 1024), point (B, 2) px."""
            h = self.f(torch.cat([x0 * 2 - 1, x1 * 2 - 1], 1))
            h = torch.cat([h, (d.float() / 4)[:, None, None, None].expand(-1, 1, *h.shape[2:]).to(h.dtype)], 1)
            est = self.est(h).flatten(2).float()                                  # (B, 3, 1024)
            p = torch.softmax(self.conf(h).flatten(1).float(), -1)               # (B, 1024)
            pred = (est * p[:, None]).sum(-1)
            return pred, p, torch.stack([(p * self.gu).sum(-1), (p * self.gv).sum(-1)], -1)

    return Effector3()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch / local/run_stage.ps1")
    import torch

    t0 = time.time()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    dev = a.device
    a.out.mkdir(parents=True, exist_ok=True)
    S = {}
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        term = np.load(a.cache / f"{split}_terminals.npy")
        ends = np.flatnonzero(term)[:n_ep]; n = int(ends[-1] + 1)
        ep_of = np.concatenate([[0], np.cumsum(term[:n - 1])]).astype(np.int64)
        act = np.asarray(np.load(a.cache / f"{split}_actions.npy", mmap_mode="r")[:n, :3], np.float32)
        S[split] = (Frames(a.cache / f"{split}_observations.npy", n, ram=(split == "val")), act, ends[ep_of], n)
    mu, sd = S["train"][1].mean(0), S["train"][1].std(0) + 1e-6

    def sample(split, B, r):
        obs, act, last, n = S[split]
        d = np.array(GAPS)[r.integers(0, len(GAPS), B)]
        t = r.integers(0, n, B)
        t = np.maximum(np.minimum(t, last[t] - d), 0)
        d = np.maximum(np.minimum(d, last[t] - t), 1)
        y = np.stack([act[np.minimum(t + i, n - 1)] for i in range(max(GAPS))], 1)
        m = (np.arange(max(GAPS))[None] < d[:, None]).astype(np.float32)
        y = ((y * m[..., None]).sum(1) / d[:, None] - mu) / sd
        x0 = torch.as_tensor(obs[t], device=dev).permute(0, 3, 1, 2).float().div_(255)
        x1 = torch.as_tensor(obs[np.minimum(t + d, n - 1)], device=dev).permute(0, 3, 1, 2).float().div_(255)
        return x0, x1, torch.as_tensor(d, device=dev), torch.as_tensor(y, device=dev).float()

    net = make_effector3().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    vr = np.random.default_rng(123)
    vb = [sample("val", 256, vr) for _ in range(4)]
    log = []
    for step in range(a.steps):
        x0, x1, d, y = sample("train", a.batch, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred, p, pt = net(x0, x1, d)
        loss = ((pred.float() - y) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                P, Y = [], []
                for b in vb:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                        P.append(net(*b[:3])[0].float().cpu().numpy())
                    Y.append(b[3].cpu().numpy())
                P, Y = np.concatenate(P), np.concatenate(Y)
            net.train()
            log.append({"step": step, "loss": round(loss.item(), 4), "val_r2_xyz": (1 - ((P - Y) ** 2).mean(0) / Y.var(0)).round(3).tolist(),
                        "peak": round(float(p.max(-1).values.mean()), 4), "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    net.eval()
    torch.save({"net": net.state_dict(), "gaps": GAPS}, a.out / "effector.pt")
    rep = {"cache": str(a.cache), "steps": a.steps, "log": log}
    for split in ("train", "val"):
        obs, act, last, n = S[split]
        T = np.lib.format.open_memmap(a.out / f"track_{split}.npy", "w+", np.float32, (n, 4))
        with torch.no_grad():
            for s0 in range(0, n, 512):
                t = np.arange(s0, min(s0 + 512, n)); t1 = np.minimum(t + 1, last[t])
                same = t1 == t; t1 = np.where(same, t, t1); t0_ = np.where(same, np.maximum(t - 1, 0), t)
                x0 = torch.as_tensor(obs[t0_], device=dev).permute(0, 3, 1, 2).float().div_(255)
                x1 = torch.as_tensor(obs[t1], device=dev).permute(0, 3, 1, 2).float().div_(255)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    pred, p, pt = net(x0, x1, torch.ones(len(t), device=dev))
                y = torch.as_tensor((act[t0_] - mu) / sd, device=dev)
                T[t] = torch.cat([pt.float(), p.max(-1).values.float()[:, None], ((pred.float() - y) ** 2).mean(-1, keepdim=True)], -1).cpu().numpy()
        T.flush()
        v = np.asarray(T[:min(n, 200000)])
        rep[split] = {"frames": n, "u_p5_p95": np.percentile(v[:, 0], [5, 95]).round(1).tolist(), "v_p5_p95": np.percentile(v[:, 1], [5, 95]).round(1).tolist(),
                      "peak_p50": float(np.median(v[:, 2])), "step_px_p50": float(np.median(np.linalg.norm(np.diff(v[:, :2], axis=0), axis=-1)))}
        print(split, json.dumps(rep[split]), flush=True)
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    save_json(a.out / "effector_report.json", rep)


if __name__ == "__main__":
    main()
