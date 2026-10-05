#!/usr/bin/env python3
"""Unified backend, step 4: one event-conditioned pixel skill for every task family.

pi(a | two frames, event) with event = (identity e, its current state, its target state): FiLM conditioning
on [identity embedding, MLP(current pos, target pos, current app, target app)] -- a cube is moved from one
place to another, a light or a button is pressed in place (target app = the other appearance), a drawer is
slid. Segments from u_events.py: from the end of the previous event to the end of this one plus `release`
frames, hindsight labels (e, before[e], after[e]); events where several entities moved (knocks) are
excluded. CNN on two stacked frames, 8-action chunks, shift augmentation (the skill-v3 recipe).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from common import Split, save_json


def make_skill(K: int, chunk: int = 8, act_dim: int = 5, width: int = 32, hidden: int = 512, cond: str = "full"):
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(), nn.MaxPool2d(2))

    class Skill(nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk, self.act_dim = chunk, act_dim
            self.cnn = nn.Sequential(block(6, width), block(width, 2 * width), block(2 * width, 2 * width), block(2 * width, 4 * width))
            self.emb = nn.Embedding(K, 64)
            self.cond = nn.Sequential(nn.Linear(10 if cond == "full" else 5, 64), nn.GELU(), nn.Linear(64, 64))
            self.film = nn.Linear(128, 4 * width * 2)
            self.head = nn.Sequential(nn.Linear(4 * width * 16 + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, chunk * act_dim))

        def forward(self, px, e, cur, tgt):
            """px (B, 6, 64, 64) [0,1]; e (B,); cur, tgt (B, 6) raw states (px, rgb) -> (B, chunk, act_dim)."""
            if cond == "full":
                z = torch.cat([cur[:, :2] / 32 - 1, tgt[:, :2] / 32 - 1, cur[:, 2:5] * 2 - 1, tgt[:, 2:5] * 2 - 1], -1)
            else:                                   # target only: the current state is in the image
                z = torch.cat([tgt[:, :2] / 32 - 1, tgt[:, 2:5] * 2 - 1], -1)
            c = torch.cat([self.emb(e.long()), self.cond(z)], -1)
            f = self.cnn(px * 2 - 1)
            g, b = self.film(c).chunk(2, -1)
            f = f * (1 + g[:, :, None, None]) + b[:, :, None, None]
            return self.head(torch.cat([f.flatten(1), c], -1)).view(len(px), self.chunk, self.act_dim)

    return Skill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--train-frames", type=int, default=1_500_000)
    ap.add_argument("--val-frames", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hist-gap", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--release", type=int, default=10)
    ap.add_argument("--cond", choices=("full", "target"), default="full", help="condition on (start, target) or the target only")
    ap.add_argument("--device", default="cuda", help="cpu for smoke tests")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = a.device
    t0 = time.time()
    train, val = Split(a.cache, "train", a.train_frames), Split(a.cache, "val", a.val_frames)
    tr, va = np.load(a.events / "events_train.npz"), np.load(a.events / "events_val.npz")
    K = tr["before"].shape[1]

    def samples(ev, n):
        idx, ee, cc, tt = [], [], [], []
        for s0, te, e, b, t_, kn in zip(ev["seg_start"], ev["t"], ev["e"], ev["before"], ev["target"], ev["knock"]):
            if kn or te + a.release >= n:
                continue
            f = np.arange(s0, te + a.release + 1)
            idx.append(f); ee.append(np.full(len(f), e)); cc.append(np.repeat(b[e][None], len(f), 0)); tt.append(np.repeat(t_[None], len(f), 0))
        return np.concatenate(idx), np.concatenate(ee), np.concatenate(cc).astype(np.float32), np.concatenate(tt).astype(np.float32)

    xi, xe, xc, xt = samples(tr, train.n)
    vi, ve, vc, vt = samples(va, val.n)
    mu, sd = train.actions.mean(0), train.actions.std(0) + 1e-6
    act = torch.as_tensor((train.actions - mu) / sd, device=dev)
    vact = torch.as_tensor((val.actions - mu) / sd, device=dev)
    H = a.chunk
    bounds = {}
    for name, sp in (("train", train), ("val", val)):
        bounds[name] = (np.r_[0, np.nonzero(sp.terminals)[0] + 1][sp.ep], np.r_[np.nonzero(sp.terminals)[0], sp.n - 1][sp.ep])

    def batch(split, idx, actions, augment=False):
        first_s, last_s = bounds["train" if split is train else "val"]
        prev = np.maximum(idx - a.hist_gap, first_s[idx])
        px = torch.as_tensor(np.concatenate([split.obs[prev], split.obs[idx]], -1), device=dev).permute(0, 3, 1, 2).float().div_(255.0)
        if augment:
            pad = torch.nn.functional.pad(px, (2, 2, 2, 2), mode="replicate")
            dx, dy = rng.integers(0, 5, 2)
            px = pad[:, :, dy:dy + 64, dx:dx + 64]
        steps = idx[:, None] + np.arange(H)[None]
        valid = steps <= last_s[idx][:, None]
        steps = np.minimum(steps, last_s[idx][:, None])
        return px, actions[torch.as_tensor(steps, device=dev)], torch.as_tensor(valid, device=dev).float()

    pi = make_skill(K, chunk=H, cond=a.cond).to(dev)
    opt = torch.optim.AdamW(pi.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    print({"train_samples": len(xi), "val_samples": len(vi), "identities": K}, flush=True)
    log, best = [], float("inf")
    for step in range(a.steps):
        s = rng.integers(0, len(xi), a.batch)
        px, tgt, m = batch(train, xi[s], act, augment=True)
        with torch.autocast(dev, dtype=torch.bfloat16):
            pred = pi(px, torch.as_tensor(xe[s], device=dev), torch.as_tensor(xc[s], device=dev), torch.as_tensor(xt[s], device=dev)).float()
        loss = (((pred - tgt) ** 2).mean(-1) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt.step(); sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            pi.eval()
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                sv = np.random.default_rng(1).integers(0, len(vi), 2048)
                vpx, vtg, vm = batch(val, vi[sv], vact)
                vp = pi(vpx, torch.as_tensor(ve[sv], device=dev), torch.as_tensor(vc[sv], device=dev), torch.as_tensor(vt[sv], device=dev)).float()
                vl = ((((vp - vtg) ** 2).mean(-1) * vm).sum() / vm.sum()).item()
            pi.train()
            log.append({"step": step, "loss": loss.item(), "val_chunk": vl, "min": round((time.time() - t0) / 60, 1)}); print(log[-1], flush=True)
            if vl < best:                                   # the skill overfits after ~10-20k steps: keep the best
                best = vl
                a.out.mkdir(parents=True, exist_ok=True)
                torch.save({"skill": pi.state_dict(), "K": K, "chunk": H, "hist_gap": a.hist_gap, "release": a.release,
                            "action_mean": mu, "action_std": sd, "step": step, "cond": a.cond}, a.out / "u_skill_best.pt")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"skill": pi.state_dict(), "K": K, "chunk": H, "hist_gap": a.hist_gap, "release": a.release, "action_mean": mu, "action_std": sd,
                "cond": a.cond},
               a.out / "u_skill.pt")
    save_json(a.out / "skill_log.json", log)


if __name__ == "__main__":
    main()
