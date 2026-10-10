#!/usr/bin/env python3
"""Skill v3 (train_skill3.make_skill3: CNN on two raw frames, FiLM conditioning on the event type and its 8 x 8 target
map, action chunks) trained on the scene-memory-v2 reader events (sm2_reader.py output: events_{split}.npz, tmaps.npy).
Same recipe as train_skill3.py; only the target maps come from the reader events instead of a skill v2 checkpoint.
Segments: from the frame after the previous event to the first visible change, plus --release frames after the change
(lift-off), train_skill.samples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from sm2_train import Frames
from train_skill import samples
from train_skill3 import make_skill3


class MSplit:
    """common.Split without copying the frames into RAM: frames memory-mapped (sm2_train.Frames), the rest in RAM."""

    def __init__(self, cache: Path, split: str, n_frames: int):
        term = np.load(cache / f"{split}_terminals.npy")
        ends = np.nonzero(term[:n_frames])[0]
        self.n = int(ends[-1] + 1)
        self.obs = Frames(cache / f"{split}_observations.npy", self.n, ram=(split == "val"))
        self.actions = np.asarray(np.load(cache / f"{split}_actions.npy", mmap_mode="r")[:self.n], np.float32)
        self.terminals = np.asarray(term[:self.n])
        self.ep = np.concatenate([[0], np.cumsum(self.terminals[:-1])]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True, help="sm2_reader.py output dir")
    ap.add_argument("--train-frames", type=int, default=1_001_000)
    ap.add_argument("--val-frames", type=int, default=100_100)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hist-gap", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--max-tau", type=int, default=0)
    ap.add_argument("--release", type=int, default=3, help="segments run through the visible change + N frames (lift-off)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    dev = a.device
    torch.manual_seed(0); rng = np.random.default_rng(0)
    t0 = time.time()
    tm = np.load(a.events / "tmaps.npy")
    train, val = MSplit(a.cache, "train", a.train_frames), MSplit(a.cache, "val", a.val_frames)
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
        px = torch.as_tensor(np.concatenate([split.obs[prev], split.obs[idx]], -1), device=dev).permute(0, 3, 1, 2).float().div_(255.0)
        if augment:
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
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    print({"train_samples": len(xi), "val_samples": len(vi), "events": E, "load_min": round((time.time() - t0) / 60, 1)}, flush=True)
    log = []
    best = (np.inf, None)
    for step in range(a.steps):
        s = rng.integers(0, len(xi), a.batch)
        px, tgt, m = batch(train, xi[s], act, augment=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred = pi(px, torch.as_tensor(xe[s], device=dev)).float()
        loss = (((pred - tgt) ** 2).mean(-1) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt.step(); sched.step()
        if step % 5000 == 0 or step == a.steps - 1:
            pi.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                sv = np.random.default_rng(1).integers(0, len(vi), 2048)
                vpx, vtg, vm = batch(val, vi[sv], vact)
                vp = pi(vpx, torch.as_tensor(ve[sv], device=dev)).float()
                vl = ((((vp - vtg) ** 2).mean(-1) * vm).sum() / vm.sum()).item()
            pi.train()
            log.append({"step": step, "loss": loss.item(), "val_chunk": vl, "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
            if vl < best[0]:                                                     # the skill overfits late (train_cube_skill)
                best = (vl, {k: v.detach().clone() for k, v in pi.state_dict().items()})
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"version": 3, "events": E, "tmaps": tm, "chunk": H, "hist_gap": a.hist_gap, "max_tau": a.max_tau, "release": a.release,
            "action_mean": mu, "action_std": sd}
    torch.save({"skill": pi.state_dict(), **meta}, a.out / "skill.pt")
    torch.save({"skill": best[1], "best_val_chunk": best[0], **meta}, a.out / "skill_best.pt")
    (a.out / "skill_log.json").write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
