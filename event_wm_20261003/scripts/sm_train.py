#!/usr/bin/env python3
"""Train the scene-memory model (sm_model.py) on pixel play data. Same hyperparameters for every family.

Data: the first --episodes train episodes of <cache>/train_observations.npy (64x64 RGB uint8) and actions, read into
RAM once (random clip reads over network storage were the bottleneck in earlier jobs). Clips of --clip frames at
stride 1 inside one episode; the action that led into frame t is fed with frame t (zero at episode start).

Loss (all means over pixels / tokens / frames, so a token-frame of fast-layer coverage costs lam_alpha and a
token-frame gate opening costs lam_gate in the same units):
  reconstruction   w * (x_hat - x)^2, w = 1 + kappa * (per-pixel temporal std over the clip) / its max
  fast layer       lam_alpha * mean(alpha)
  memory changes   lam_gate * mean(opened), frames 1..T-1 (frame 0 is the free initial read)
  occluders        half of the clips also get a synthetic occluder (random box, colour from the clip, a random
                   interval). Both versions run; the occluded run must keep the clean run's memory values
                   (invariance, clean run detached), the fast layer must cover the occluder (BCE alpha -> 1),
                   and the visibility head learns 0 on occluded tokens and 1 on tokens whose clean frame the scene
                   layer already explains (|x - scene| small); other tokens give it no target. Alpha is not
                   used as a visibility label.
No auxiliary FSQ loss (FSQ has none). Stability / distinctness of codes are measured by sm_diag.py, not assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sm_model import SceneMemory


def load_split(cache: Path, split: str, episodes: int):
    term = np.load(cache / f"{split}_terminals.npy")
    ends = np.flatnonzero(term)
    n = int(ends[min(episodes, len(ends)) - 1] + 1)
    obs = np.empty((n, 64, 64, 3), np.uint8)
    src = np.load(cache / f"{split}_observations.npy", mmap_mode="r")
    for s in range(0, n, 50_000):                                   # sequential read into RAM
        obs[s:min(s + 50_000, n)] = src[s:min(s + 50_000, n)]
    act = np.asarray(np.load(cache / f"{split}_actions.npy", mmap_mode="r")[:n], np.float32)
    starts = np.r_[0, ends[:min(episodes, len(ends)) - 1] + 1]
    return obs, act, starts, ends[:len(starts)]


def sample_clips(rng, starts, ends, B, T):
    e = rng.integers(len(starts), size=B)
    first = starts[e] + rng.integers(0, ends[e] - starts[e] + 2 - T, size=B)
    return first, first == starts[e]


def batch(obs, act, first, at_start, T, dev):
    idx = first[:, None] + np.arange(T)
    x = torch.as_tensor(obs[idx], device=dev).permute(0, 1, 4, 2, 3).float() / 255           # (B, T, 3, 64, 64)
    a_prev = np.zeros((len(first), T, act.shape[1]), np.float32)
    a_prev[:, 1:] = act[idx[:, :-1]]
    a_prev[:, 0] = np.where(at_start[:, None], 0, act[np.maximum(first - 1, 0)])
    return x, torch.as_tensor(a_prev, device=dev)


def add_occluders(rng, x):
    """x (B, T, 3, 64, 64) -> occluded copy and pixel mask (B, T, 1, 64, 64). Box 6-20 px, frames [t0, t0 + n),
    t0 >= 1 so the initial read sees the scene; colour = a random pixel of the clip plus small noise."""
    B, T = x.shape[:2]
    xo, m = x.clone(), torch.zeros_like(x[:, :, :1])
    for b in range(B):
        h, w = rng.integers(6, 21, size=2)
        r, c = rng.integers(0, 64 - h + 1), rng.integers(0, 64 - w + 1)
        n = int(rng.integers(3, min(16, T - 1) + 1))
        t0 = int(rng.integers(1, T - n + 1))
        col = x[b, rng.integers(T), :, rng.integers(64), rng.integers(64)]
        patch = (col[:, None, None] + 0.05 * torch.randn(3, h, w, device=x.device)).clamp(0, 1)
        xo[b, t0:t0 + n, :, r:r + h, c:c + w] = patch
        m[b, t0:t0 + n, :, r:r + h, c:c + w] = 1
    return xo, m


def losses(out, x, cfg):
    std = x.std(1, keepdim=True).mean(2, keepdim=True)                                       # (B,1,1,64,64)
    w = 1 + cfg["kappa"] * std / std.amax((3, 4), keepdim=True).clamp_min(1e-6)
    rec = (w * (out["x_hat"] - x).pow(2).mean(2, keepdim=True)).mean()
    fast = out["alpha"].mean()
    gate = out["open"][:, 1:].mean()
    return rec, fast, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--clip", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--lam-alpha", type=float, default=0.01)
    ap.add_argument("--lam-gate", type=float, default=0.04)
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--occ-frac", type=float, default=0.5)
    ap.add_argument("--w-inv", type=float, default=1.0)
    ap.add_argument("--w-occ-alpha", type=float, default=0.1)
    ap.add_argument("--w-vis", type=float, default=0.1)
    ap.add_argument("--vis-explained", type=float, default=0.01, help="per-token MSE below which the scene layer explains the clean frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("GPU training runs under sbatch")
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = vars(a).copy(); cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = a.device

    t0 = time.time()
    obs, act, st, en = load_split(a.cache, "train", a.episodes)
    vobs, vact, vst, ven = load_split(a.cache, "val", a.val_episodes)
    print(f"DATA train {len(obs)} frames / {len(st)} episodes, val {len(vobs)} / {len(vst)}, {time.time() - t0:.0f}s", flush=True)
    vrng = np.random.default_rng(12345)
    vfirst, vstart = sample_clips(vrng, vst, ven, 64, a.clip)

    model = SceneMemory(a.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else dict(device_type="cpu", enabled=False)
    log, t_last = [], time.time()
    for step in range(a.steps):
        model.train()
        first, at_start = sample_clips(rng, st, en, a.batch, a.clip)
        x, ap_ = batch(obs, act, first, at_start, a.clip, dev)
        n_occ = int(round(a.occ_frac * a.batch))
        with torch.autocast(**amp):
            out = model(x, ap_, train=True)
            rec, fast, gate = losses(out, x, cfg)
            loss = rec + a.lam_alpha * fast + a.lam_gate * gate
            inv = occ_a = vis_l = torch.zeros((), device=dev)
            if n_occ:
                xo, m = add_occluders(rng, x[:n_occ])
                oo = model(xo, ap_[:n_occ], train=True)
                r2, f2, g2 = losses(oo, xo, cfg)
                inv = (oo["q"].float() - out["q"][:n_occ].detach().float()).pow(2).mean()
                occ_a = F.binary_cross_entropy(oo["alpha"].float().clamp(1e-4, 1 - 1e-4), torch.ones_like(m), weight=m, reduction="sum") / m.sum().clamp_min(1)
                tok_occ = F.avg_pool2d(m.flatten(0, 1), 4).unflatten(0, m.shape[:2]) >= 0.5                     # (b,T,1,16,16)
                expl = F.avg_pool2d((out["scene"][:n_occ] - x[:n_occ]).pow(2).mean(2, keepdim=True).flatten(0, 1).float(), 4).unflatten(0, m.shape[:2]) < a.vis_explained
                tgt = torch.where(tok_occ, 0.0, 1.0)
                wv = (tok_occ | expl.detach()).float()
                vis_l = F.binary_cross_entropy_with_logits(oo["vis_logit"].float(), tgt, weight=wv, reduction="sum") / wv.sum().clamp_min(1)
                loss = loss + (r2 + a.lam_alpha * f2 + a.lam_gate * g2) * n_occ / a.batch + a.w_inv * inv + a.w_occ_alpha * occ_a + a.w_vis * vis_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            row = {"step": step, "loss": float(loss), "rec": float(rec), "alpha": float(fast), "open": float(gate),
                   "inv": float(inv), "occ_alpha": float(occ_a), "vis": float(vis_l),
                   "codes_used": int(torch.unique(model.fsq.index(out["q"].detach().permute(0, 1, 3, 4, 2).float())).numel()),
                   "sec_per_step": (time.time() - t_last) / (1 if step == 0 else 500)}
            if dev == "cuda":
                row["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            if step % 2000 == 0 or step == a.steps - 1:
                model.eval()
                vr = []
                with torch.no_grad(), torch.autocast(**amp):
                    for i in range(0, 64, 16):
                        xv, av = batch(vobs, vact, vfirst[i:i + 16], vstart[i:i + 16], a.clip, dev)
                        vr.append(losses(model(xv, av), xv, cfg))
                row.update(val_rec=float(np.mean([float(r[0]) for r in vr])), val_alpha=float(np.mean([float(r[1]) for r in vr])),
                           val_open=float(np.mean([float(r[2]) for r in vr])))
            t_last = time.time()
            log.append(row); print(json.dumps(row), flush=True)
        if (step + 1) % 5000 == 0 or step == a.steps - 1:
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step + 1}, a.out / "sm.pt")
            (a.out / "train_log.json").write_text(json.dumps({"cfg": cfg, "log": log}, indent=1) + "\n")
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
