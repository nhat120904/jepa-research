#!/usr/bin/env python3
"""Method front end, components 1-6 of method/README.md. Code identical to scripts/sm2_model.py (sha256 0e121d4914d5fa05...)
plus LEVELS / GRID / FSQ / conv_block / Res / Up from scripts/sm_model.py (sha256 770e1aa18860dbfc...); the
trained front-end checkpoints (E:\jepa-data\event_wm\scene_memory_v2\<family>_grow_*) come from that code.

Scene memory v2: the agent is separated by action contingency, the scene by learned per-patch discrete codes, and
the memory / event boundaries are rules.

Why v2. v1 (sm_model.py) let a per-frame layer with an area cost compete with the scene memory. That is the layered
robust-PCA structure stopped on 2026-10-04 (jobs 57195 / 57199: "the mask cost scales with area, so small objects always
go to the sparse part"). Four local v1 configurations failed either way (memory = empty room, or memory rewritten every
frame including the arm). v2 has no such competition: what is masked is decided by the action, not by a cost.

Components, learned from offline play (observations + actions, what every OGBench method has), the same hyper-parameters
for every family:
  AgentNet    next-frame predictor x_{t+1} = x_t + D(x_t, a_t, a_{t-1}, flag), trained with action dropout, so flag 0
              is the action-free prediction. Contingency gain per pixel = e_free - e_action: where the action explains
              the change (the arm, its shadow, a carried object). Pixel analogue of g_entities.py's agent rule ("the agent
              responds directly to the action"); cf. contingency-aware exploration (Choi et al. 2019), Iso-Dream (2022).
  teacher     a token (4 x 4 px) is contingent in the pair (t, t+1) when its pixels change (rendering is deterministic, so
              a static token's change is exactly 0) and the action explains most of the change: R = clip(1 - e_action /
              e_free, 0, 1) above the 2-means split of R over changing TRAIN tokens, 0.5 if not bimodal -- the same form as
              g_entities.py's split_or(score, 0.5). Then grown through changing tokens (8-neighbourhood): change
              connected to action-explained change is the agent's footprint (the faint transparent arm, the shadow's
              edge); a change elsewhere (a side effect) is not. Label of frame t = pair (t-1, t) or pair (t, t+1).
  Segmenter   frame -> per-token agent logit, trained on the teacher labels, so a single frame (the goal image, a
              pausing arm) gets an agent mask without actions.
  SceneCodes  per-patch encoder (receptive field = its own 4 x 4 patch, no normalisation across positions: unchanged
              pixels give the same code exactly) -> FSQ 5^6 -> decoder with context; the reconstruction counts only
              pixels outside the agent mask.
  memory      rule: a token's memory takes its current code after k consecutive frames outside the (dilated) agent mask
              with that same code; known = taken at least once. A memory change of a known token is an event frame.
  SeeThrough  (stage 2) frame -> per-token code of the scene BEHIND the agent, with a probability. OGBench draws the arm
              ~90% transparent and its shadow only darkens the scene, so a binary agent mask throws visible state away
              (stale memory) or lets the tint in (false changes). Targets come from the data itself (see_targets):
              agent-free frames give their own confirmed code; frames inside an agent visit whose memory before the visit
              equals the code confirmed right after it (the scene behind did not change) give that code. The memory then
              takes a token's predicted code when its probability > .5 for k consecutive frames (no agent mask).
  events      memory changes grouped into interactions (assemble_events): a change joins an open event when its tokens
              lie within one token of that event's tokens and it comes at most `gap` frames after the event's last change
              (late-observed effects attach to their interaction); `gap` = 2-means split of log gaps between such changes
              on TRAIN.
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



GRID = 16
NTOK = GRID * GRID


def two_means_threshold(x, iters=50):
    """Same rule as sfa_code.two_means_threshold (copied: that module imports the token pipeline).
    Returns (threshold, separation = Ashman's D, fraction above)."""
    x = np.asarray(x, np.float64)
    c = np.percentile(x, [10, 90]).astype(np.float64)
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None]).argmin(1)
        c = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(2)])
    thr = c.mean()
    lo, hi = x[x <= thr], x[x > thr]
    sep = abs(c[1] - c[0]) / (np.sqrt(0.5 * (lo.var() + hi.var())) + 1e-9)
    return float(thr), float(sep), float((x > thr).mean())


def up_block(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.GroupNorm(8, cout), nn.GELU())


class FiLM(nn.Module):
    def __init__(self, cond, c):
        super().__init__()
        self.f = nn.Linear(cond, 2 * c)

    def forward(self, h, z):
        g, b = self.f(z).chunk(2, -1)
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]


class AgentNet(nn.Module):
    """x_t (B, 3, 64, 64) in [0, 1], a_t, a_{t-1} (B, act_dim), flag (B,) in {0, 1} -> predicted x_{t+1}."""

    def __init__(self, act_dim=5, w=32):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(2 * act_dim + 1, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.e1 = conv_block(3, w)                   # 64
        self.e2 = conv_block(w, 2 * w, 2)            # 32
        self.e3 = conv_block(2 * w, 4 * w, 2)        # 16
        self.e4 = conv_block(4 * w, 4 * w, 2)        # 8
        self.mid = Res(4 * w)
        self.film = nn.ModuleList([FiLM(128, 4 * w), FiLM(128, 4 * w), FiLM(128, 2 * w)])
        self.u3 = up_block(4 * w, 4 * w)             # 8 -> 16
        self.u2 = up_block(8 * w, 2 * w)             # 16 -> 32
        self.u1 = up_block(4 * w, w)                 # 32 -> 64
        self.out = nn.Conv2d(2 * w, 3, 3, 1, 1)

    def forward(self, x, a, a_prev, flag):
        f = flag.float()[:, None]
        z = self.cond(torch.cat([a * f, a_prev * f, f], 1))
        h1 = self.e1(x * 2 - 1)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        h4 = self.mid(self.film[0](self.e4(h3), z))
        u3 = self.film[1](self.u3(h4), z)
        u2 = self.film[2](self.u2(torch.cat([u3, h3], 1)), z)
        u1 = self.u1(torch.cat([u2, h2], 1))
        return x + self.out(torch.cat([u1, h1], 1))


@torch.no_grad()
def token_errors(net, x, x_next, a, a_prev):
    """Per-token (B, 256), means over the token's pixels and channels: action-free error, action error, and the actual
    change |x_{t+1} - x_t|^2."""
    B = len(x)
    one = torch.ones(B, device=x.device)
    ef = (net(x, a, a_prev, 0 * one).float() - x_next).pow(2).mean(1, keepdim=True)
    ea = (net(x, a, a_prev, one).float() - x_next).pow(2).mean(1, keepdim=True)
    ch = (x_next - x).pow(2).mean(1, keepdim=True)
    return F.avg_pool2d(ef, 4).flatten(1), F.avg_pool2d(ea, 4).flatten(1), F.avg_pool2d(ch, 4).flatten(1)


def action_share(ef, ea):
    """R = share of the action-free error that the action removes, clipped to [0, 1]."""
    return np.clip(1 - ea / np.maximum(ef, 1e-12), 0, 1)


def contingent(ef, ea, ch, thr_r):
    """teacher rule: the token's pixels change and the action explains most of the change."""
    return (ch > 1e-9) & (action_share(ef, ea) > thr_r)


class Segmenter(nn.Module):
    """frame (B, 3, 64, 64) in [0, 1] -> per-token agent logit (B, 256)."""

    def __init__(self, w=64):
        super().__init__()
        self.f = nn.Sequential(conv_block(3, w // 2), conv_block(w // 2, w, 2), Res(w), conv_block(w, w, 2), Res(w), Res(w),
                               nn.Conv2d(w, 1, 1))

    def forward(self, x):
        return self.f(x * 2 - 1).flatten(1)


class SceneCodes(nn.Module):
    """Per-patch discrete scene code. encode: (B, 3, 64, 64) in [0, 1] -> grid values (B, 6, 16, 16), straight-through."""

    def __init__(self, width=128):
        super().__init__()
        self.fsq = FSQ()
        d = self.fsq.dim
        self.enc = nn.Sequential(nn.Conv2d(3, width, 4, 4), nn.GELU(), nn.Conv2d(width, width, 1), nn.GELU(),
                                 nn.Conv2d(width, d, 1))
        self.dec = nn.Sequential(nn.Conv2d(d, width, 1), Res(width), Res(width), Up(width, 3))

    def encode(self, x):
        return self.fsq.quantize(self.fsq.bound(self.enc(x * 2 - 1).float()))

    def codes(self, x):
        """integer code per token (B, 256)."""
        return self.fsq.index(self.encode(x).permute(0, 2, 3, 1)).flatten(1)

    def decode(self, q):
        return torch.sigmoid(self.dec(q))


def dilate_tokens(mask, r=1):
    """mask (..., 256) bool -> dilated by a (2r+1)^2 square on the 16 x 16 grid (numpy)."""
    if r <= 0:
        return mask
    m = mask.reshape(-1, GRID, GRID)
    p = np.pad(m, ((0, 0), (r, r), (r, r)))
    out = np.zeros_like(m)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[:, dy:dy + GRID, dx:dx + GRID]
    return out.reshape(mask.shape)


def grow_tokens(seed, region, max_iter=16):
    """Geodesic growth on the 16 x 16 grid: tokens of `region` 8-connected to `seed` through `region` (seed included)."""
    m = seed.copy()
    for _ in range(max_iter):
        nm = m | (dilate_tokens(m, 1) & region)
        if (nm == m).all():
            break
        m = nm
    return m


def agent_masks(idx, pair_packed, seg_prob, is_start, r=1, teacher=True):
    """Agent token mask (B, 256) bool of frames idx = teacher pair (t-1, t) | teacher pair (t, t+1) | segmenter p > .5,
    dilated by r tokens. pair_packed (N, 32) uint8 = np.packbits of the pair (t, t+1) bits (0 at an episode's last
    frame); seg_prob (N, 256) uint8 (p x 255) or None; is_start (N,) bool. teacher=False: segmenter only (one frame)."""
    idx = np.asarray(idx)
    m = np.zeros((len(idx), NTOK), bool)
    if teacher:
        m |= np.unpackbits(pair_packed[idx], axis=1).astype(bool)
        prv = np.unpackbits(pair_packed[np.maximum(idx - 1, 0)], axis=1).astype(bool)
        prv[is_start[idx]] = False                                                # no pair across an episode start
        m |= prv
    if seg_prob is not None:
        m |= np.asarray(seg_prob[idx]) > 127
    return dilate_tokens(m, r)


def memory_rule(codes, agent, k=3, return_confirmed=False):
    """One episode. codes (T, N) int, agent (T, N) bool -> memory (T, N) int64 (-1 = unknown) and changed (T, N) bool
    (a known token's memory replaced by a different code; a token's first acceptance is not a change); with
    return_confirmed also confirmed (T, N) bool = the token's current code is confirmed at t (k agent-free frames)."""
    T, N = codes.shape
    mem = np.full((T, N), -1, np.int64)
    changed = np.zeros((T, N), bool)
    confirmed = np.zeros((T, N), bool)
    cur = np.full(N, -1, np.int64)
    cand = np.full(N, -1, np.int64)
    run = np.zeros(N, np.int64)
    for t in range(T):
        free = ~agent[t]
        c = codes[t].astype(np.int64)
        run = np.where(free & (c == cand), run + 1, np.where(free, 1, 0))
        cand = np.where(free, c, -1)
        ok = free & (run >= k)
        acc = ok & (cand != cur)
        changed[t] = acc & (cur >= 0)
        confirmed[t] = ok
        cur = np.where(acc, cand, cur)
        mem[t] = cur
    return (mem, changed, confirmed) if return_confirmed else (mem, changed)


def see_targets(codes, agent, mem, confirmed, partial=False):
    """One episode -> targets (T, N) int64, -1 = none: agent-free frames whose code is the memory (clean anchors), and
    agent-covered frames whose memory (frozen during the visit = the value before it) equals the code confirmed next after
    the visit (the scene behind the agent did not change).
    partial: also -> second targets y2 (T, N): agent-covered frames of a visit during which the token changed (memory before
    != code confirmed after) take the SET {before, after} (y = before, y2 = after; partial-label learning: the frame decides
    which). Without it these frames had no target (held-out puzzle-3x3: 58% of agent-covered token frames), and the
    moments that matter most, an object changing under the agent, were never trained."""
    T, N = codes.shape
    nxt = np.full((T, N), -1, np.int64)
    run = np.full(N, -1, np.int64)
    for t in range(T - 1, -1, -1):
        nxt[t] = run                                                              # next confirmation strictly after t
        run = np.where(confirmed[t], codes[t], run)
    y = np.full((T, N), -1, np.int64)
    clean = ~agent & (mem >= 0) & (codes == mem)
    y[clean] = codes[clean]
    behind = agent & (mem >= 0) & (nxt == mem)
    y[behind] = mem[behind]
    if not partial:
        return y
    y2 = np.full((T, N), -1, np.int64)
    during = agent & (mem >= 0) & (nxt >= 0) & (nxt != mem)
    y[during] = mem[during]; y2[during] = nxt[during]
    return y, y2


def pixel_codes(x, levels=8):
    """raw-pixel control code per token: (B, 64, 64, 3) uint8 -> (B, 256) int = quantised 4 x 4 patch mean RGB,
    r * 64 + g * 8 + b (digits little-endian b, g, r)."""
    m = np.asarray(x, np.float32).reshape(-1, GRID, 4, GRID, 4, 3).mean((2, 4))
    q = np.minimum((m / 256 * levels).astype(np.int64), levels - 1)
    return (q[..., 0] * levels * levels + q[..., 1] * levels + q[..., 2]).reshape(len(m), NTOK)


def code_digits(codes, n_digits, n_levels):
    """int codes (...) -> digits (..., n_digits), little-endian base n_levels (FSQ.index order for 5^6 codes)."""
    codes = np.asarray(codes, np.int64)
    return np.stack([(codes // n_levels ** j) % n_levels for j in range(n_digits)], -1)


class SeeThrough(nn.Module):
    """frame (B, 3, 64, 64) in [0, 1] -> per-token digit logits (B, 256, n_digits, n_levels); with context, so that it
    can undo the transparent arm's tint and the shadow."""

    def __init__(self, n_digits, n_levels, w=128):
        super().__init__()
        self.nd, self.nl = n_digits, n_levels
        self.f = nn.Sequential(conv_block(3, w // 2), conv_block(w // 2, w, 2), Res(w), conv_block(w, w, 2), Res(w), Res(w), Res(w),
                               nn.Conv2d(w, n_digits * n_levels, 1))

    def forward(self, x):
        o = self.f(x * 2 - 1)
        return o.view(len(x), self.nd, self.nl, NTOK).permute(0, 3, 1, 2)


@torch.no_grad()
def see_codes(model, x):
    """-> int codes (B, 256) and probability of that code (product over digits of the max softmax), numpy."""
    lp = torch.log_softmax(model(x).float(), -1)
    best, arg = lp.max(-1)                                                        # (B, 256, nd)
    w = torch.tensor([model.nl ** j for j in range(model.nd)], device=x.device)
    return (arg * w).sum(-1).cpu().numpy(), best.sum(-1).exp().cpu().numpy()


def backdate(changed_conf, mem_conf, confirmed, mem_fresh, starts, T):
    """Confirmed changes (memory over agent-free frames only) moved back to the first frame at which the fresh
    (see-through) memory showed the confirmed value, searched after the last agent-free confirmation of the old value
    (the change cannot be earlier). The change is known to be real (it outlived the agent's visit); the fresh memory
    tells when it became visible. Offline use (looks ahead)."""
    out = np.zeros_like(changed_conf)
    for s0 in starts:
        mf, ch, cf = mem_fresh[s0:s0 + T], changed_conf[s0:s0 + T], confirmed[s0:s0 + T]
        for t in np.flatnonzero(ch.any(1)):
            for i in np.flatnonzero(ch[t]):
                old = np.flatnonzero(cf[:t, i])
                lo = old[-1] + 1 if len(old) else 0
                seen = np.flatnonzero(mf[lo:t + 1, i] == mem_conf[s0 + t, i])
                out[s0 + (lo + seen[0] if len(seen) else t), i] = True
    return out


def adjacent_gaps(changed, starts, T, r=1, merge=5):
    """Gaps (frames) from each memory change to the latest earlier change within r tokens, per episode; gaps <= merge
    (one run) dropped."""
    gaps = []
    for s0 in starts:
        last = np.full(NTOK, -10 ** 9)
        for t in np.flatnonzero(changed[s0:s0 + T].any(1)) + s0:
            c = changed[t]
            prev = last[dilate_tokens(c[None], r)[0]].max()
            if prev > -10 ** 9 and t - prev > merge:
                gaps.append(t - prev)
            last[c] = t
    return np.array(gaps, np.int64)


def assemble_events(changed, starts, T, gap, r=1):
    """Memory changes -> interactions [(start, end, n_tokens)]: a change joins every open event whose tokens lie within r
    tokens of it and whose last change is at most `gap` frames earlier (they merge); otherwise it opens a new event."""
    out = []
    for s0 in starts:
        evs = []                                                                  # [t0, t_last, token mask]
        for t in np.flatnonzero(changed[s0:s0 + T].any(1)) + s0:
            c = changed[t]
            near = dilate_tokens(c[None], r)[0]
            hit = [e for e in evs if t - e[1] <= gap and (e[2] & near).any()]
            if hit:
                base = hit[0]
                for e in hit[1:]:
                    base[0] = min(base[0], e[0]); base[2] |= e[2]; evs.remove(e)
                base[1] = t; base[2] |= c
            else:
                evs.append([t, t, c.copy()])
        out += [(e[0], e[1], int(e[2].sum())) for e in evs]
    return np.array(sorted(out), np.int64).reshape(-1, 3)


def merge_onsets(on, starts, T, merge=5):
    """Event frames (bool, N) -> events [(start, end)] = runs of on-frames merged when the gap is <= merge frames,
    never across episodes (same rule as slowmap.py)."""
    ev = []
    for s0 in starts:
        idx = np.flatnonzero(on[s0:s0 + T]) + s0
        if not len(idx):
            continue
        a = b = idx[0]
        for t in idx[1:]:
            if t - b <= merge:
                b = t
            else:
                ev.append((a, b)); a = b = t
        ev.append((a, b))
    return np.array(ev, np.int64).reshape(-1, 2)
