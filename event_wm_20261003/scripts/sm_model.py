#!/usr/bin/env python3
"""Scene-memory model (v1): a persistent, sparsely updated token memory of the scene learned from pixel play.

Frame x_t (64x64 RGB) -> encoder tokens f_t (16x16, one token per 4x4 px). Two pathways explain the frame:
  fast layer   (alpha_t, A_t) from the current frame only, no state carried between frames. It explains whatever
               moves frame to frame (arm, shadow, a carried object). Every covered pixel costs lam_alpha per frame.
  memory       m_t (16x16 tokens, FSQ-quantised). A token changes only through its gate g_t in [0, 1); every open
               gate costs lam_gate (an L0 count, GateL0RD style). Closed gate = the code is copied exactly.
  composite    x_hat = alpha * A + (1 - alpha) * decode(m_t).
Hypothesis (to be measured, not assumed): static content is cheaper in memory (written once) than in the fast
layer (paid every frame), continuously moving content is cheaper in the fast layer (a gate per frame otherwise),
and content hidden by an occluder survives in memory because a closed gate costs nothing.
Visibility head v_t (per token): learned only from synthetic occluders pasted during training; alpha is not
treated as visibility. The first frame of a sequence is written into memory without gate cost (initial read).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = (5, 5, 5, 5, 5, 5)                    # FSQ levels per memory channel -> 15625 codes per token
GRID = 16                                      # tokens per side; token = 4 x 4 px at 64 x 64


class FSQ(nn.Module):
    """Finite scalar quantisation (Mentzer et al. 2023): bound each channel, round to a fixed grid, straight-through.
    Odd levels only, so the grid is {-1, ..., 1} in steps of 2 / (L - 1) and rounding an on-grid value is exact."""

    def __init__(self, levels=LEVELS):
        super().__init__()
        assert all(l % 2 == 1 for l in levels)
        self.register_buffer("hl", torch.tensor([(l - 1) / 2 for l in levels], dtype=torch.float32))
        self.register_buffer("radix", torch.tensor(np.cumprod((1,) + tuple(levels[:-1])), dtype=torch.long))
        self.dim = len(levels)

    def bound(self, z):
        return torch.tanh(z)

    def quantize(self, y):
        """y (B, dim, H, W) in [-1, 1] (bounded, or a convex mix of grid points) -> nearest grid point,
        straight-through."""
        y = y.float()
        hl = self.hl.view(1, -1, 1, 1)
        q = torch.round(y * hl) / hl
        return y + (q - y).detach()

    def index(self, q):
        """grid values (..., dim), channel last -> integer code (...)."""
        k = torch.round(q * self.hl + self.hl).long()
        return (k * self.radix).sum(-1)

    def values(self, idx):
        """integer code (...) -> grid values (..., dim)."""
        k = (idx[..., None] // self.radix) % (2 * self.hl.long() + 1)
        return (k.float() - self.hl) / self.hl


def conv_block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 4 if stride == 2 else 3, stride, 1), nn.GroupNorm(8, cout), nn.GELU())


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(8, c))

    def forward(self, x):
        return F.gelu(x + self.f(x))


class Up(nn.Module):
    """16x16 x c -> 64x64 x cout."""

    def __init__(self, c, cout):
        super().__init__()
        self.f = nn.Sequential(Res(c), nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.GroupNorm(8, c // 2), nn.GELU(),
                               nn.ConvTranspose2d(c // 2, c // 4, 4, 2, 1), nn.GroupNorm(8, c // 4), nn.GELU(),
                               nn.Conv2d(c // 4, cout, 3, 1, 1))

    def forward(self, x):
        return self.f(x)


class SceneMemory(nn.Module):
    def __init__(self, width=128, act_dim=5, gate_noise=0.1):
        super().__init__()
        self.fsq = FSQ()
        d = self.fsq.dim
        self.enc = nn.Sequential(conv_block(3, width // 2), conv_block(width // 2, width, 2), Res(width),
                                 conv_block(width, width, 2), Res(width), Res(width))
        self.act = nn.Linear(act_dim, width)
        self.fast = Up(width, 4)                                       # alpha logit + RGB of the fast layer
        self.vis = nn.Conv2d(width, 1, 1)                              # token visibility logit
        self.mem_in = nn.Conv2d(d, width // 2, 1)
        self.mem_nb = nn.Conv2d(d, width // 2, 3, 1, 1)                 # neighbourhood: side effects spread locally
        self.cell = nn.Sequential(nn.Conv2d(2 * width, width, 1), nn.GELU(), Res(width), nn.Conv2d(width, d + 1, 1))
        self.write = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, d, 1))   # initial read
        self.dec = nn.Sequential(nn.Conv2d(d, width, 1), Res(width), Up(width, 3))
        self.gate_noise = gate_noise

    # -- per-frame pieces -------------------------------------------------------------------------------------
    def encode(self, x, a_prev):
        """x (B, 3, 64, 64) in [0, 1]; a_prev (B, act_dim) -> tokens (B, C, 16, 16)."""
        return self.enc(x * 2 - 1) + self.act(a_prev)[:, :, None, None]

    def fast_layer(self, f):
        o = self.fast(f)
        return torch.sigmoid(o[:, :1]), torch.sigmoid(o[:, 1:])     # alpha (B,1,64,64), A (B,3,64,64)

    def initial(self, f):
        return self.fsq.quantize(self.fsq.bound(self.write(f).float()))

    def step(self, q_prev, f, train=False):
        """One memory update. Returns q (B, d, 16, 16), gate g (B, 1, 16, 16), open (B, 1, 16, 16) with
        straight-through gradient (forward = 1[s > 0], backward = d sigmoid(4 s))."""
        h = torch.cat([f, self.mem_in(q_prev), self.mem_nb(q_prev)], 1)
        o = self.cell(h).float()
        u, s = self.fsq.bound(o[:, :-1]), o[:, -1:]
        q_prev = q_prev.float()
        if train and self.gate_noise > 0:
            s = s + self.gate_noise * torch.randn_like(s)
        g = torch.relu(torch.tanh(s))
        hard = (s > 0).float()
        soft = torch.sigmoid(4 * s)
        opened = soft + (hard - soft).detach()
        q = self.fsq.quantize(q_prev + g * (u - q_prev))
        return q, g, opened

    def decode(self, q):
        return torch.sigmoid(self.dec(q))

    # -- sequence -----------------------------------------------------------------------------------------------
    def forward(self, x, a_prev, train=False):
        """x (B, T, 3, 64, 64); a_prev (B, T, act_dim) (action that led into frame t, zero at episode start).
        Returns dict of per-step tensors stacked over T."""
        B, T = x.shape[:2]
        f = self.encode(x.flatten(0, 1), a_prev.flatten(0, 1)).unflatten(0, (B, T))
        alpha, A = self.fast_layer(f.flatten(0, 1))
        vis = self.vis(f.flatten(0, 1))
        qs, opens, gs = [], [], []
        q = self.initial(f[:, 0])
        qs.append(q); opens.append(torch.ones_like(q[:, :1])); gs.append(torch.ones_like(q[:, :1]))
        for t in range(1, T):
            q, g, op = self.step(q, f[:, t], train)
            qs.append(q); opens.append(op); gs.append(g)
        q = torch.stack(qs, 1)
        scene = self.decode(q.flatten(0, 1))
        out = {"q": q, "open": torch.stack(opens, 1), "g": torch.stack(gs, 1),
               "alpha": alpha.unflatten(0, (B, T)), "A": A.unflatten(0, (B, T)),
               "scene": scene.unflatten(0, (B, T)), "vis_logit": vis.unflatten(0, (B, T))}
        out["x_hat"] = out["alpha"] * out["A"] + (1 - out["alpha"]) * out["scene"]
        return out


@torch.no_grad()
def run_batch(model, obs, act, device, tchunk=64):
    """Causal pass over E episodes of equal length in parallel. obs (E, T, 64, 64, 3) uint8 (array-like),
    act (E, T, act_dim) with act[t] taken at obs[t]. Returns per-frame arrays (E, T, 256):
    codes int32, opened bool, gate / alpha_tok / vis float32 (vis = sigmoid of the visibility head)."""
    E, T = obs.shape[:2]
    a_prev = np.zeros((E, T, act.shape[-1]), np.float32); a_prev[:, 1:] = act[:, :-1]
    out = {k: [] for k in ("codes", "opened", "gate", "alpha_tok", "vis")}
    q = None
    for s in range(0, T, tchunk):
        x = torch.as_tensor(np.asarray(obs[:, s:s + tchunk]), device=device)
        n = x.shape[1]
        x = x.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        ap = torch.as_tensor(a_prev[:, s:s + n], device=device).flatten(0, 1)
        f = model.encode(x, ap)
        alpha, _ = model.fast_layer(f)
        out["alpha_tok"].append(F.avg_pool2d(alpha, 4).flatten(1).view(E, n, -1).float().cpu())
        out["vis"].append(torch.sigmoid(model.vis(f)).flatten(1).view(E, n, -1).float().cpu())
        f = f.view(E, n, *f.shape[1:])
        cs, ops, gs = [], [], []
        for t in range(n):
            if q is None:
                q = model.initial(f[:, t]); g = torch.ones_like(q[:, :1]); op = g
            else:
                q, g, op = model.step(q, f[:, t])
            cs.append(model.fsq.index(q.permute(0, 2, 3, 1)).flatten(1)); ops.append((op > 0.5).flatten(1)); gs.append(g.flatten(1))
        out["codes"].append(torch.stack(cs, 1).int().cpu()); out["opened"].append(torch.stack(ops, 1).cpu())
        out["gate"].append(torch.stack(gs, 1).float().cpu())
    return {k: torch.cat(v, 1).numpy() for k, v in out.items()}


def run_episode(model, obs, act, device, chunk=64):
    """One episode: obs (T, 64, 64, 3), act (T, act_dim) -> per-frame arrays (T, 256)."""
    return {k: v[0] for k, v in run_batch(model, np.asarray(obs)[None], np.asarray(act)[None], device, chunk).items()}
