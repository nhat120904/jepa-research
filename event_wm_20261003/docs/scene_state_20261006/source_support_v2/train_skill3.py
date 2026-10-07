#!/usr/bin/env python3
"""Event-conditioned skill policy v3: a CNN on raw pixels trained end to end (no frozen encoder).

The v1/v2 skills read frozen LeWM patch tokens. Pressing a button needs the end effector within
about 1-2 pixels of the button in a 64x64 image, and a predictive latent may not keep that detail.
v3 follows the usual pixel behaviour-cloning recipe (e.g. the IMPALA-style encoder used by HIQL on
OGBench): a small CNN on two stacked frames (t - gap, t), trained jointly with the policy head;
event conditioning = type embedding + the spatial target map of skill v2; output = action chunk.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from common import Split, load_encoder, save_json, tokens
from train_skill import samples


def make_skill3(events: int, tmaps, chunk: int = 8, act_dim: int = 5, width: int = 32, hidden: int = 512):
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(), nn.MaxPool2d(2))

    class Skill3(nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk, self.act_dim = chunk, act_dim
            self.cnn = nn.Sequential(block(6, width), block(width, 2 * width), block(2 * width, 2 * width),
                                     block(2 * width, 4 * width))              # 64 -> 4x4
            self.ev = nn.Embedding(events, 64)
            self.register_buffer("tmap", torch.as_tensor(tmaps, dtype=torch.float32))
            self.film = nn.Linear(128, 4 * width * 2)                          # event-conditioned FiLM on features
            self.head = nn.Sequential(nn.Linear(4 * width * 16 + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                      nn.GELU(), nn.Linear(hidden, chunk * act_dim))

        def forward(self, px, e):
            """px (B, 6, 64, 64) float in [0, 1] (two frames), e (B,) -> (B, chunk, act_dim)."""
            c = torch.cat([self.ev(e.long()), self.tmap[e.long()]], -1)
            f = self.cnn(px * 2 - 1)
            g, b = self.film(c).chunk(2, -1)
            f = f * (1 + g[:, :, None, None]) + b[:, :, None, None]
            return self.head(torch.cat([f.flatten(1), c], -1)).view(len(px), self.chunk, self.act_dim)

    return Skill3()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--skill2", type=Path, required=True, help="skill v2 checkpoint (reuses its target maps)")
    ap.add_argument("--train-frames", type=int, default=1_500_000)
    ap.add_argument("--val-frames", type=int, default=50_000)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hist-gap", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--max-tau", type=int, default=0)
    ap.add_argument("--release", type=int, default=-1, help=">= 0: segments run through the visible change + N frames (lift-off)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    t0 = time.time()
    s2 = torch.load(a.skill2, map_location="cpu", weights_only=False)
    tm = s2["tmaps"]
    train, val = Split(a.cache, "train", a.train_frames), Split(a.cache, "val", a.val_frames)
    tr, va = np.load(a.events / "events_train.npz"), np.load(a.events / "events_val.npz")
    E = len(np.load(a.events / "vocab.npy"))
    xi, xe, _ = samples(tr, train.n, a.max_tau, a.release)
    vi, ve, _ = samples(va, val.n, a.max_tau, a.release)
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
        px = torch.as_tensor(np.concatenate([split.obs[prev], split.obs[idx]], -1), device=dev)  # (B, 64, 64, 6)
        px = px.permute(0, 3, 1, 2).float().div_(255.0)
        if augment:                                                   # small random shifts (pixel BC standard)
            pad = torch.nn.functional.pad(px, (2, 2, 2, 2), mode="replicate")
            dx, dy = rng.integers(0, 5, 2)
            px = pad[:, :, dy:dy + 64, dx:dx + 64]
        steps = idx[:, None] + np.arange(H)[None]
        valid = steps <= last_s[idx][:, None]
        steps = np.minimum(steps, last_s[idx][:, None])
        return px, actions[torch.as_tensor(steps, device=dev)], torch.as_tensor(valid, device=dev).float()

    pi = make_skill3(E, tm, chunk=H).to(dev)
    opt = torch.optim.AdamW(pi.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    print({"train_samples": len(xi), "val_samples": len(vi), "events": E, "load_min": round((time.time() - t0) / 60, 1)},
          flush=True)
    log = []
    for step in range(a.steps):
        s = rng.integers(0, len(xi), a.batch)
        px, tgt, m = batch(train, xi[s], act, augment=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = pi(px, torch.as_tensor(xe[s], device=dev)).float()
        loss = (((pred - tgt) ** 2).mean(-1) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 5000 == 0 or step == a.steps - 1:
            pi.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                sv = np.random.default_rng(1).integers(0, len(vi), 2048)
                vpx, vtg, vm = batch(val, vi[sv], vact)
                vp = pi(vpx, torch.as_tensor(ve[sv], device=dev)).float()
                vl = ((((vp - vtg) ** 2).mean(-1) * vm).sum() / vm.sum()).item()
                v1 = (((vp[:, 0] - vtg[:, 0]) ** 2).mean()).item()
            pi.train()
            log.append({"step": step, "loss": loss.item(), "val_chunk": vl, "val_first": v1,
                        "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"skill": pi.state_dict(), "version": 3, "events": E, "tmaps": tm, "chunk": H, "hist_gap": a.hist_gap,
                "max_tau": a.max_tau, "release": a.release, "action_mean": mu, "action_std": sd}, a.out / "skill.pt")
    save_json(a.out / "skill_log.json", log)


if __name__ == "__main__":
    main()
