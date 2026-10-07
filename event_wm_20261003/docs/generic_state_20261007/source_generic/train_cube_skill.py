#!/usr/bin/env python3
"""Cube skill: pi(a | two raw frames, cube k, target xy) -- the skill-v3 recipe for pick-and-place.

Segments come from cube_events.py: from the end of the previous move to the end of this move plus
`release` frames (release + lift-off), with hindsight labels (k = moved cube, q = its final xy).
CNN on two stacked frames trained end to end; FiLM conditioning on [cube embedding, target MLP];
8-action chunks; shift augmentation.
--support (pixel events with the coverage bit): also condition on the support object -- the object
that the move covers (hindsight: its coverage bit goes 0 -> 1), or none. In the image a cube stacked
on B and a cube on the ground just in front of B project to nearly the same pixel, so the target
pixel alone does not say whether to stack (job 57138: the skill put the cube in front of B).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from common import Split, save_json
from cube_planner import HI, LO


def make_cube_skill(K: int, chunk: int = 8, act_dim: int = 5, width: int = 32, hidden: int = 512, lo=LO[:2], hi=HI[:2],
                    support: bool = False):
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(), nn.MaxPool2d(2))

    class CubeSkill(nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk, self.act_dim = chunk, act_dim
            self.cnn = nn.Sequential(block(6, width), block(width, 2 * width), block(2 * width, 2 * width),
                                     block(2 * width, 4 * width))
            self.emb = nn.Embedding(K, 64)
            self.tgt = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64))
            C = 128 + (32 if support else 0)
            if support:
                self.sup = nn.Embedding(K + 1, 32)                                      # K = no support (ground)
            self.film = nn.Linear(C, 4 * width * 2)
            self.head = nn.Sequential(nn.Linear(4 * width * 16 + C, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                      nn.GELU(), nn.Linear(hidden, chunk * act_dim))

        def forward(self, px, k, q, sup=None):
            """px (B, 6, 64, 64) in [0, 1]; k (B,); q (B, 2) target (metres in D, pixels in A) -> (B, chunk, act_dim)."""
            l, h = torch.as_tensor(lo, device=q.device), torch.as_tensor(hi, device=q.device)
            c = torch.cat([self.emb(k.long()), self.tgt((q - l) / (h - l) * 2 - 1)], -1)
            if support:
                c = torch.cat([c, self.sup(sup.long())], -1)
            f = self.cnn(px * 2 - 1)
            g, b = self.film(c).chunk(2, -1)
            f = f * (1 + g[:, :, None, None]) + b[:, :, None, None]
            return self.head(torch.cat([f.flatten(1), c], -1)).view(len(px), self.chunk, self.act_dim)

    return CubeSkill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--train-frames", type=int, default=1_500_000)
    ap.add_argument("--val-frames", type=int, default=50_000)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hist-gap", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--release", type=int, default=10)
    ap.add_argument("--support", action="store_true", help="condition on the covered (support) object")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    t0 = time.time()
    train, val = Split(a.cache, "train", a.train_frames), Split(a.cache, "val", a.val_frames)
    tr, va = np.load(a.events / "cube_events_train.npz"), np.load(a.events / "cube_events_val.npz")
    K = tr["before"].shape[1]
    lo, hi = (tr["lo"][:2], tr["hi"][:2]) if "lo" in tr.files else (LO[:2], HI[:2])   # target xy: pixels (A) or metres (D)

    def support_of(ev):
        """Per move: the object it covers (coverage 0 -> 1), nearest to the target if several; else K."""
        sup = np.full(len(ev["k"]), K)
        if not a.support:
            return sup
        newly = (ev["after"][..., 2] > 0.5) & (ev["before"][..., 2] < 0.5)
        newly[np.arange(len(ev["k"])), ev["k"]] = False
        d = np.linalg.norm(ev["after"][..., :2] - ev["target_xy"][:, None], axis=-1)
        d[~newly] = np.inf
        has = newly.any(1)
        sup[has] = d[has].argmin(1)
        return sup

    def samples(ev, n):
        idx, kk, qq, ss = [], [], [], []
        for s0, te, k, q, kn, sp in zip(ev["seg_start"], ev["t"], ev["k"], ev["target_xy"], ev["knock"], support_of(ev)):
            if kn or te + a.release >= n:
                continue
            f = np.arange(s0, te + a.release + 1)
            idx.append(f); kk.append(np.full(len(f), k)); qq.append(np.repeat(q[None], len(f), 0)); ss.append(np.full(len(f), sp))
        return np.concatenate(idx), np.concatenate(kk), np.concatenate(qq).astype(np.float32), np.concatenate(ss)

    xi, xk, xq, xs = samples(tr, train.n)
    vi, vk, vq, vs = samples(va, val.n)
    if a.support:
        print({"moves_with_support_train": float((support_of(tr) < K).mean())}, flush=True)
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

    pi = make_cube_skill(K, chunk=H, lo=lo, hi=hi, support=a.support).to(dev)
    opt = torch.optim.AdamW(pi.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    print({"train_samples": len(xi), "val_samples": len(vi), "cubes": K, "load_min": round((time.time() - t0) / 60, 1)}, flush=True)
    log, best = [], float("inf")
    for step in range(a.steps):
        s = rng.integers(0, len(xi), a.batch)
        px, tgt, m = batch(train, xi[s], act, augment=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = pi(px, torch.as_tensor(xk[s], device=dev), torch.as_tensor(xq[s], device=dev),
                      torch.as_tensor(xs[s], device=dev)).float()
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
                vp = pi(vpx, torch.as_tensor(vk[sv], device=dev), torch.as_tensor(vq[sv], device=dev),
                        torch.as_tensor(vs[sv], device=dev)).float()
                vl = ((((vp - vtg) ** 2).mean(-1) * vm).sum() / vm.sum()).item()
            pi.train()
            log.append({"step": step, "loss": loss.item(), "val_chunk": vl, "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
            if vl < best:                                   # val loss rises after ~10k steps (overfit): keep the best
                best = vl
                a.out.mkdir(parents=True, exist_ok=True)
                torch.save({"skill": pi.state_dict(), "K": K, "chunk": H, "hist_gap": a.hist_gap, "release": a.release,
                            "action_mean": mu, "action_std": sd, "step": step, "lo": lo, "hi": hi, "support": a.support}, a.out / "cube_skill_best.pt")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"skill": pi.state_dict(), "K": K, "chunk": H, "hist_gap": a.hist_gap, "release": a.release,
                "action_mean": mu, "action_std": sd, "lo": lo, "hi": hi, "support": a.support}, a.out / "cube_skill.pt")
    save_json(a.out / "skill_log.json", log)


if __name__ == "__main__":
    main()
