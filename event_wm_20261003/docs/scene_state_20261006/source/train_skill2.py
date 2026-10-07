#!/usr/bin/env python3
"""Event-conditioned skill policy v2: frame history, action chunks, spatial event target.

Round-2 closed loops showed that the v1 skill (single frame, one action, event embedding) rarely
completes a press within 80 steps, and about half its presses hit the wrong button. v2 changes:
  - history: patch tokens of frames t - gap and t (velocity information);
  - action chunks: predict the next `chunk` actions; the closed loop executes the first few and
    re-queries (smoother, less compounding);
  - spatial target: each event type's mean patch-change map (where its lights change, from detected
    events; no labels) is an input next to the type embedding, so "where to go" is grounded in the image.
Training frames: the segment that leads to each detected event (or its last max_tau + 1 frames).
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


def make_skill2(events: int, tmaps, hist: int = 2, chunk: int = 8, dim: int = 192, proj: int = 32,
                hidden: int = 1024, act_dim: int = 5):
    import torch
    import torch.nn as nn

    class Skill2(nn.Module):
        def __init__(self):
            super().__init__()
            self.hist, self.chunk, self.act_dim = hist, chunk, act_dim
            self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proj), nn.GELU())
            self.ev = nn.Embedding(events, 64)
            self.register_buffer("tmap", torch.as_tensor(tmaps, dtype=torch.float32))
            self.mlp = nn.Sequential(nn.Linear(hist * 64 * proj + 64 + 64, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, chunk * act_dim))

        def forward(self, toks, e):
            """toks (B, hist, 64, dim), e (B,) -> (B, chunk, act_dim) normalised actions."""
            z = self.proj(toks).flatten(1)
            c = torch.cat([z, self.ev(e.long()), self.tmap[e.long()]], -1)
            return self.mlp(c).view(len(toks), self.chunk, self.act_dim)

    return Skill2()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--train-frames", type=int, default=900_000)
    ap.add_argument("--val-frames", type=int, default=50_000)
    ap.add_argument("--steps", type=int, default=30000)
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
    ck = torch.load(a.code, map_location="cpu", weights_only=False)
    enc, _ = load_encoder(Path(ck["base"]), dev)
    train, val = Split(a.cache, "train", a.train_frames), Split(a.cache, "val", a.val_frames)
    tr, va = np.load(a.events / "events_train.npz"), np.load(a.events / "events_val.npz")
    E = len(np.load(a.events / "vocab.npy"))
    sample = tokens(enc, train.obs[rng.integers(0, train.n, 4096)], dev)
    tmu, tsd = sample.mean((0, 1)), sample.std((0, 1)) + 1e-6

    # Spatial target per event type: mean per-patch token change across the event (label-free).
    first = np.r_[0, np.nonzero(train.terminals)[0] + 1]
    ep_first, ep_last = first[train.ep], np.r_[np.nonzero(train.terminals)[0], train.n - 1][train.ep]
    tm = np.zeros((E, 64))
    for e in range(E):
        sel = np.nonzero((tr["e"] == e) & (tr["t"] < train.n - 8))[0][:200]
        b0 = np.maximum(tr["t_start"][sel] - 6, ep_first[tr["t_start"][sel]])
        b1 = np.minimum(tr["t"][sel] + 6, ep_last[tr["t"][sel]])
        with torch.no_grad():
            d = (tokens(enc, train.obs[b1], dev) - tokens(enc, train.obs[b0], dev)).norm(dim=-1).mean(0)
        tm[e] = (d / d.max()).cpu().numpy()

    xi, xe, _ = samples(tr, train.n, a.max_tau, a.release)
    vi, ve, _ = samples(va, val.n, a.max_tau, a.release)
    mu, sd = train.actions.mean(0), train.actions.std(0) + 1e-6
    act = torch.as_tensor((train.actions - mu) / sd, device=dev)
    vact = torch.as_tensor((val.actions - mu) / sd, device=dev)
    H = a.chunk

    bounds = {}
    for name, sp in (("train", train), ("val", val)):
        bounds[name] = (np.r_[0, np.nonzero(sp.terminals)[0] + 1][sp.ep],
                        np.r_[np.nonzero(sp.terminals)[0], sp.n - 1][sp.ep])

    def batch(split, idx, actions):
        """History tokens (B, 2, 64, 192), target chunk (B, H, 5) and validity mask (B, H)."""
        first_s, last_s = bounds["train" if split is train else "val"]
        prev = np.maximum(idx - a.hist_gap, first_s[idx])
        frames = np.stack([split.obs[prev], split.obs[idx]], 1).reshape(-1, 64, 64, 3)
        tok = ((tokens(enc, frames, dev) - tmu) / tsd).view(len(idx), 2, 64, -1)
        steps = idx[:, None] + np.arange(H)[None]
        valid = steps <= last_s[idx][:, None]
        steps = np.minimum(steps, last_s[idx][:, None])
        return tok, actions[torch.as_tensor(steps, device=dev)], torch.as_tensor(valid, device=dev).float()

    pi = make_skill2(E, tm, chunk=H).to(dev)
    opt = torch.optim.AdamW(pi.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = min(500, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    print({"train_samples": len(xi), "val_samples": len(vi), "events": E, "load_min": round((time.time() - t0) / 60, 1)},
          flush=True)
    log = []
    for step in range(a.steps):
        s = rng.integers(0, len(xi), a.batch)
        tok, tgt, m = batch(train, xi[s], act)
        pred = pi(tok, torch.as_tensor(xe[s], device=dev))
        loss = (((pred - tgt) ** 2).mean(-1) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            pi.eval()
            with torch.no_grad():
                sv = np.random.default_rng(1).integers(0, len(vi), 2048)
                vt, vtg, vm = batch(val, vi[sv], vact)
                vp = pi(vt, torch.as_tensor(ve[sv], device=dev))
                vl = ((((vp - vtg) ** 2).mean(-1) * vm).sum() / vm.sum()).item()
                v1 = (((vp[:, 0] - vtg[:, 0]) ** 2).mean()).item()
            pi.train()
            log.append({"step": step, "loss": loss.item(), "val_chunk": vl, "val_first": v1,
                        "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"skill": pi.state_dict(), "version": 2, "events": E, "tmaps": tm, "chunk": H, "hist_gap": a.hist_gap,
                "max_tau": a.max_tau, "release": a.release, "action_mean": mu, "action_std": sd, "token_mean": tmu.cpu(), "token_std": tsd.cpu(),
                "code": str(a.code)}, a.out / "skill.pt")
    save_json(a.out / "skill_log.json", log)


if __name__ == "__main__":
    main()
