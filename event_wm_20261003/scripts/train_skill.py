#!/usr/bin/env python3
"""Event-conditioned skill policy pi(a | o_t, e, tau) by behaviour cloning on play data.

Training samples are the frames of the segment that leads to each detected event (from the
frame after the previous event to the event itself), labelled with the event type e and the
number of steps tau until the event happens. Play segments start with the lift-and-wander part
of the previous press; conditioning on tau separates that part (large tau) from the approach
and press (small tau). At test time tau follows a short countdown (scripts/closed_loop.py).
Inputs are the frozen encoder's patch tokens of the current frame; output is the 5-d action.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from common import Split, load_encoder, save_json, tokens

TAU_MAX = 48


def make_skill(events: int, dim: int = 192, proj: int = 48, hidden: int = 1024, act_dim: int = 5):
    import torch
    import torch.nn as nn

    class Skill(nn.Module):
        def __init__(self):
            super().__init__()
            self.events = events
            self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proj), nn.GELU())
            self.ev = nn.Embedding(events, 64)
            self.tau = nn.Sequential(nn.Linear(1, 32), nn.GELU())
            self.mlp = nn.Sequential(nn.Linear(64 * proj + 64 + 32, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, act_dim))

        def forward(self, tok, e, tau):
            t = (tau.float().clamp(max=TAU_MAX) / TAU_MAX).unsqueeze(-1)
            return self.mlp(torch.cat([self.proj(tok).flatten(1), self.ev(e.long()), self.tau(t)], -1))

    return Skill()


def samples(ev, n_frames, max_tau=0, release=-1):
    """Frame index, event type and tau for every frame of every in-vocabulary segment
    (only the last max_tau + 1 frames before the event if max_tau > 0).
    release >= 0: the segment runs through the end of the visible change plus `release` frames, so the
    skill also learns to lift off the button (the change is only visible once the gripper has left it)."""
    idx, typ, tau = [], [], []
    for s, t, te, e in zip(ev["seg_start"], ev["t_start"], ev["t"], ev["e"]):   # tau counts to the first visible change
        if e < 0 or t >= n_frames:
            continue
        stop = min(te + release, n_frames - 1) if release >= 0 else t
        f = np.arange(max(s, t - max_tau) if max_tau else s, stop + 1)
        idx.append(f)
        typ.append(np.full(len(f), e))
        tau.append(np.maximum(t - f, 0))
    return np.concatenate(idx), np.concatenate(typ), np.concatenate(tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--code", type=Path, required=True, help="code.pt (gives the frozen encoder)")
    ap.add_argument("--train-frames", type=int, default=600_000)
    ap.add_argument("--val-frames", type=int, default=50_000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-tau", type=int, default=0, help="train only on the last T+1 frames before each event")
    ap.add_argument("--no-tau", action="store_true", help="constant tau input (closed loop then uses tau = 0)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    t0 = time.time()
    ck = torch.load(a.code, map_location="cpu", weights_only=False)
    enc, base_ck = load_encoder(Path(ck["base"]), dev)
    train, val = Split(a.cache, "train", a.train_frames), Split(a.cache, "val", a.val_frames)
    tr, va = np.load(a.events / "events_train.npz"), np.load(a.events / "events_val.npz")
    E = len(np.load(a.events / "vocab.npy"))
    xi, xe, xt = samples(tr, train.n, a.max_tau)
    vi, ve, vt = samples(va, val.n, a.max_tau)
    xt_in = np.zeros_like(xt) if a.no_tau else xt     # --no-tau: the policy never sees tau
    vt_in = np.zeros_like(vt) if a.no_tau else vt
    mu, sd = train.actions.mean(0), train.actions.std(0) + 1e-6
    act = torch.as_tensor((train.actions - mu) / sd, device=dev)
    vact = torch.as_tensor((val.actions - mu) / sd, device=dev)
    sample = tokens(enc, train.obs[rng.integers(0, train.n, 4096)], dev)
    tmu, tsd = sample.mean((0, 1)), sample.std((0, 1)) + 1e-6
    pi = make_skill(E).to(dev)
    opt = torch.optim.AdamW(pi.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = min(500, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    print({"train_samples": len(xi), "val_samples": len(vi), "events": E, "load_min": round((time.time() - t0) / 60, 1)}, flush=True)
    log = []

    def val_loss():
        out = {}
        with torch.no_grad():
            sel = np.random.default_rng(1).integers(0, len(vi), 8192)
            for lo, hi in ((0, 5), (6, 15), (16, 30), (31, 999)):
                s = sel[(vt[sel] >= lo) & (vt[sel] <= hi)][:2048]
                if len(s) == 0:
                    continue
                tok = (tokens(enc, val.obs[vi[s]], dev) - tmu) / tsd
                pred = pi(tok, torch.as_tensor(ve[s], device=dev), torch.as_tensor(vt_in[s], device=dev))
                out[f"tau{lo}-{hi}"] = F.mse_loss(pred, vact[vi[s]]).item()
        return out

    for step in range(a.steps):
        s = rng.integers(0, len(xi), a.batch)
        tok = (tokens(enc, train.obs[xi[s]], dev) - tmu) / tsd
        pred = pi(tok, torch.as_tensor(xe[s], device=dev), torch.as_tensor(xt_in[s], device=dev))
        loss = F.mse_loss(pred, act[xi[s]])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            pi.eval()
            rec = {"step": step, "loss": loss.item(), "val": val_loss(), "min": round((time.time() - t0) / 60, 1)}
            pi.train()
            log.append(rec)
            print(rec, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"skill": pi.state_dict(), "events": E, "no_tau": bool(a.no_tau), "max_tau": a.max_tau, "action_mean": mu, "action_std": sd,
                "token_mean": tmu.cpu(), "token_std": tsd.cpu(), "code": str(a.code)}, a.out / "skill.pt")
    save_json(a.out / "skill_log.json", log)


if __name__ == "__main__":
    main()
