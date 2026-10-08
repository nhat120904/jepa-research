#!/usr/bin/env python3
"""Event alignment of an exported scene memory (sm_diag.py export/val) against PRIVILEGED object state (scoring only).

Memory event frames: frames where at least --min-tokens memory codes change. Privileged change segments: maximal runs
of frames where an object state changes (cube moves > tol, drawer / window joint moves > tol, a button toggles), per
episode, per object type. Reported:
  precision   fraction of memory event frames within +-W frames of any privileged change frame
  recall      fraction of privileged change segments (per type) with a memory event frame inside [start-W, end+W]
  rate        memory event frames per 1000 frames
Matched control: raw-pixel frame difference (mean |x_t - x_{t-1}|), thresholded to fire on the SAME number of frames
as the memory, scored the same way. The arm moves almost every frame, so pixel change alone is not an event detector;
the memory has to beat it at an equal firing budget.
With --debounce K (> 0) also a rule-based boundary on the exported codes, independent of the learned gate: a frame is
an event frame when some token starts a new value that is held for >= K frames and differs from that token's previous
value held for >= K frames. Control: the same rule on raw-pixel tokens (4x4-px patch mean RGB, 8 levels per channel),
so the rule's own contribution is separated from the representation's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sm_diag import episode_bounds, privileged_state


def change_frames(st, tol_pos=0.002, tol_joint=0.002):
    """dict type -> bool (n,) frame t changed relative to t-1 (frame 0 of the split False)."""
    out = {}
    if st["pos"] is not None:
        d = np.linalg.norm(np.diff(st["pos"], axis=0), axis=-1)
        for k in range(d.shape[1]):
            out[f"cube{k}" if d.shape[1] > 1 else "cube"] = np.r_[False, d[:, k] > tol_pos]
    if st["joint"] is not None:
        for k, name in enumerate(("drawer", "window")[:st["joint"].shape[1]]):
            out[name] = np.r_[False, np.abs(np.diff(st["joint"][:, k])) > tol_joint]
    if st["disc"] is not None:
        out["buttons"] = np.r_[False, (np.diff(st["disc"], axis=0) != 0).any(-1)]
    return out


def segments(mask, starts, T):
    seg = []
    for s0 in starts:
        m = mask[s0:s0 + T].copy(); m[0] = False                                 # episode boundary is not a change
        d = np.diff(np.r_[0, m.astype(int), 0])
        seg += [(s0 + a, s0 + b - 1) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return seg


def stable_changes(codes, starts, T, K):
    """(n,) number of tokens that start a new value held >= K frames, differing from their previous such value."""
    cnt = np.zeros(len(codes), np.int32)
    for s0 in starts:
        c = np.asarray(codes[s0:s0 + T]).astype(np.int64)
        stable = np.zeros(c.shape, bool)
        same = np.ones((T - K + 1, c.shape[1]), bool)
        for j in range(1, K):
            same &= c[j:T - K + 1 + j] == c[:T - K + 1]
        stable[:T - K + 1] = same
        last = np.full(c.shape[1], -1, np.int64)
        for t in range(T):
            m = stable[t]
            cnt[s0 + t] = (m & (last >= 0) & (c[t] != last)).sum()
            last = np.where(m, c[t], last)
    return cnt


def pixel_tokens(obs, n, levels=8):
    out = np.zeros((n, 256), np.int64)
    for s in range(0, n, 5000):
        x = np.asarray(obs[s:min(s + 5000, n)], np.float32).reshape(-1, 16, 4, 16, 4, 3).mean((2, 4))
        q = np.minimum((x / 256 * levels).astype(np.int64), levels - 1)
        out[s:s + len(q)] = (q[..., 0] * levels * levels + q[..., 1] * levels + q[..., 2]).reshape(len(q), 256)
    return out


def score(fire, chg, starts, T, W):
    n = len(fire)
    anychg = np.zeros(n, bool)
    for m in chg.values():
        anychg |= m
    near = anychg.copy()
    for k in range(1, W + 1):
        near[k:] |= anychg[:-k]; near[:-k] |= anychg[k:]
    res = {"fires": int(fire.sum()), "rate_per_1000": float(1000 * fire.mean()),
           "precision": float(near[fire].mean()) if fire.any() else None, "recall": {}}
    cs = np.r_[0, np.cumsum(fire)]
    for name, m in chg.items():
        seg = segments(m, starts, T)
        hit = [cs[min(b + W, n - 1) + 1] - cs[max(a - W, 0)] > 0 for a, b in seg]
        res["recall"][name] = {"segments": len(seg), "recall": float(np.mean(hit)) if seg else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", type=Path, required=True, help="sm_diag output dir (export/val)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--family", choices=("cube", "puzzle", "scene"), required=True)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--min-tokens", type=int, default=1)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--debounce", type=int, default=0, help="K > 0: also score the stable-change rule on codes and pixel tokens")
    a = ap.parse_args()
    starts, ends = episode_bounds(a.cache, "val", a.val_episodes)
    T = int(ends[0] - starts[0] + 1); n = len(starts) * T
    codes = np.load(a.diag / "export" / "val" / "codes.npy", mmap_mode="r")[:n]
    st = privileged_state(a.cache, a.family, n)
    chg = change_frames(st)
    nch = np.r_[0, (np.asarray(codes[1:]) != np.asarray(codes[:-1])).sum(1)]
    nch[starts] = 0                                                              # initial read is not an event
    fire_mem = nch >= a.min_tokens
    obs = np.load(a.cache / "val_observations.npy", mmap_mode="r")
    pix = np.zeros(n, np.float32)
    for s in range(0, n, 5000):
        e = min(s + 5000, n)
        x = np.asarray(obs[max(s - 1, 0):e], np.int16)
        pix[max(s, 1):e] = np.abs(np.diff(x, axis=0)).mean((1, 2, 3))
    pix[starts] = 0
    k = int(fire_mem.sum())
    fire_pix = np.zeros(n, bool)
    if k:
        fire_pix[np.argsort(-pix, kind="stable")[:k]] = True
    res = {"diag": str(a.diag), "family": a.family, "frames": n, "min_tokens": a.min_tokens, "window": a.window,
           "changed_tokens_per_event_frame_median": float(np.median(nch[fire_mem])) if k else None,
           "memory": score(fire_mem, chg, starts, T, a.window),
           "pixel_diff_rate_matched": score(fire_pix, chg, starts, T, a.window),
           "privileged_change_frame_fraction": {name: float(m.mean()) for name, m in chg.items()}}
    if a.debounce > 0:
        cm = stable_changes(codes, starts, T, a.debounce)
        cp = stable_changes(pixel_tokens(obs, n), starts, T, a.debounce)
        res["debounce"] = {"K": a.debounce,
                           "memory": score(cm >= a.min_tokens, chg, starts, T, a.window),
                           "pixel_tokens": score(cp >= a.min_tokens, chg, starts, T, a.window),
                           "memory_tokens_per_event_frame_median": float(np.median(cm[cm >= a.min_tokens])) if (cm >= a.min_tokens).any() else None,
                           "pixel_tokens_per_event_frame_median": float(np.median(cp[cp >= a.min_tokens])) if (cp >= a.min_tokens).any() else None}
    (a.diag / "event_align.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
