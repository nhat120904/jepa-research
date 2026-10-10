#!/usr/bin/env python3
"""Component 5c (method/README.md): the agent's EFFECTOR as a keypoint learned by inverse dynamics. One setting for every
environment.

Why. The acted entity of an event is the one the effector touches when it changes. The arm mask (segmenter) holds the
whole transparent arm and its shadow, so "touched by the mask" picks neighbours (held-out puzzle-3x3 board edges: the
inner neighbour of the pressed light), and the AgentNet action sensitivity spreads over every moving link (held-out 4x4:
p50 5-9 px from the pressed light). With the effector at the true press frame the nearest light is the pressed one in
.958 of 3x3 presses (PRIVILEGED check), so what is missing is a reliable effector position.

Principle. The action is an end-effector command (OGBench: dx, dy, dz, yaw, gripper). The image point whose displacement
best predicts the commanded translation is the effector: a link nearer the base moves less and along configuration-
dependent directions, the shadow does not follow dz. Model: a CNN maps one frame to a probability map over a 32 x 32 grid;
the keypoint k_t is its expectation (soft-argmax, px). An inverse model g(k_t, k_{t+d} - k_t, d) predicts the mean
translation command a_{t..t+d-1}[:3] (standardised); the loss is its squared error. A constant keypoint cannot predict
the command, so the keypoint must move with the effector. d in {1, 2, 4} (small displacements are below a pixel).

Outputs (--out): effector.pt, eff_{split}.npy (n, 3) float32 = (u, v, map peak mass) per frame, effector_report.json
(VAL R^2 of the command per axis; keypoint statistics).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from utils import Frames, save_json

G = 32


def make_effector(w=64):
    import torch
    import torch.nn as nn
    from frontend import Res, conv_block

    class Effector(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(conv_block(3, w // 2), conv_block(w // 2, w, 2), Res(w), conv_block(w, w, 2), Res(w), Res(w))
            self.up = nn.Sequential(nn.ConvTranspose2d(w, w // 2, 4, 2, 1), nn.GroupNorm(8, w // 2), nn.GELU(), nn.Conv2d(w // 2, 1, 3, 1, 1))
            self.inv = nn.Sequential(nn.Linear(5, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3))
            c = torch.arange(G, dtype=torch.float32) * (64 / G) + (64 / G - 1) / 2
            self.register_buffer("cu", c.repeat(G))                                # column centre of each cell (px)
            self.register_buffer("cv", c.repeat_interleave(G))                     # row centre

        def keypoint(self, x):
            """frames (B, 3, 64, 64) in [0, 1] -> keypoint (B, 2) px (u, v), peak mass (B,)."""
            p = torch.softmax(self.up(self.f(x * 2 - 1)).flatten(1).float(), -1)
            return torch.stack([(p * self.cu).sum(-1), (p * self.cv).sum(-1)], -1), p.max(-1).values

        def forward(self, x0, x1, d):
            k0, _ = self.keypoint(x0); k1, _ = self.keypoint(x1)
            z = torch.cat([k0 / 32 - 1, (k1 - k0) / 4, d[:, None].float() / 4], -1)
            return self.inv(z), k0, k1

    return Effector()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gaps", default="1,2,4")
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
    gaps = np.array([int(g) for g in a.gaps.split(",")])
    S = {}
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        term = np.load(a.cache / f"{split}_terminals.npy")
        ends = np.flatnonzero(term)[:n_ep]; n = int(ends[-1] + 1)
        ep_of = np.concatenate([[0], np.cumsum(term[:n - 1])]).astype(np.int64)
        last = ends[ep_of]
        act = np.asarray(np.load(a.cache / f"{split}_actions.npy", mmap_mode="r")[:n, :3], np.float32)
        S[split] = (Frames(a.cache / f"{split}_observations.npy", n, ram=(split == "val")), act, last, n)
    mu, sd = S["train"][1].mean(0), S["train"][1].std(0) + 1e-6

    def sample(split, B, r):
        obs, act, last, n = S[split]
        d = gaps[r.integers(0, len(gaps), B)]
        t = r.integers(0, n, B)
        t = np.minimum(t, last[t] - d)                                            # t + d inside the episode
        ok = t >= 0
        t, d = t[ok], d[ok]
        y = np.stack([act[np.minimum(t + i, n - 1)] for i in range(gaps.max())], 1)   # (B, max gap, 3)
        m = (np.arange(gaps.max())[None] < d[:, None]).astype(np.float32)
        y = ((y * m[..., None]).sum(1) / d[:, None] - mu) / sd                    # mean command over the gap, standardised
        x0 = torch.as_tensor(obs[t], device=dev).permute(0, 3, 1, 2).float().div_(255)
        x1 = torch.as_tensor(obs[t + d], device=dev).permute(0, 3, 1, 2).float().div_(255)
        return x0, x1, torch.as_tensor(d, device=dev), torch.as_tensor(y, device=dev).float()

    net = make_effector().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    vr = np.random.default_rng(123)
    vb = [sample("val", 256, vr) for _ in range(4)]
    log = []
    for step in range(a.steps):
        x0, x1, d, y = sample("train", a.batch, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred, k0, k1 = net(x0, x1, d)
        loss = ((pred.float() - y) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                P, Y = [], []
                for x0v, x1v, dv, yv in vb:
                    P.append(net(x0v, x1v, dv)[0].float().cpu().numpy()); Y.append(yv.cpu().numpy())
                P, Y = np.concatenate(P), np.concatenate(Y)
                r2 = (1 - ((P - Y) ** 2).mean(0) / Y.var(0)).round(3).tolist()
            net.train()
            log.append({"step": step, "loss": round(loss.item(), 4), "val_r2_xyz": r2, "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    net.eval()
    torch.save({"net": net.state_dict(), "gaps": a.gaps, "action_mean": mu, "action_std": sd}, a.out / "effector.pt")
    rep = {"cache": str(a.cache), "steps": a.steps, "log": log}
    for split in ("train", "val"):
        obs, act, last, n = S[split]
        E = np.lib.format.open_memmap(a.out / f"eff_{split}.npy", "w+", np.float32, (n, 3))
        with torch.no_grad():
            for s0 in range(0, n, 1024):
                x = torch.as_tensor(obs[np.arange(s0, min(s0 + 1024, n))], device=dev).permute(0, 3, 1, 2).float().div_(255)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    k, pk = net.keypoint(x)
                E[s0:s0 + len(k)] = torch.cat([k.float(), pk.float()[:, None]], -1).cpu().numpy()
        E.flush()
        v = np.asarray(E[:min(n, 200000)])
        rep[split] = {"frames": n, "u_p5_p95": np.percentile(v[:, 0], [5, 95]).round(1).tolist(), "v_p5_p95": np.percentile(v[:, 1], [5, 95]).round(1).tolist(),
                      "peak_mass_p50": float(np.median(v[:, 2])), "step_px_p50": float(np.median(np.linalg.norm(np.diff(v[:, :2], axis=0), axis=-1)))}
        print(split, json.dumps(rep[split]), flush=True)
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    save_json(a.out / "effector_report.json", rep)


if __name__ == "__main__":
    main()
