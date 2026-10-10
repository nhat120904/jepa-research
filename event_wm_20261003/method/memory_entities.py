#!/usr/bin/env python3
"""Component 7 (method/README.md): scene memory -> entity table, the input format of events.py. One rule for every family.

Each token of the 16 x 16 grid whose agent-free memory (component 6) changes at least once in TRAIN is a fixed-place
entity (no identity extraction: an entity's identity is its grid position; a cube that moves shows up as code changes at
the tokens it leaves and reaches). Per frame and entity:
  app      the SeeThrough code of the token (component 5) as its 6 FSQ digits 0..4 (uint8; appearance = digit / 4 in [0, 1])
  area     1 where the SeeThrough probability > .5 (readable, also through the transparent arm), else 0
  free     the token lies outside the agent mask (segmenter p > .5, one frame, dilated by one token: the agent-free test of
           component 6); events.py counts rest observations only where readable and free
  agent    the undilated segmenter mask, 256 bits packed (used to find the entity closest to the agent at a change)
Constant: pos (K, 2) token centres (u = column, v = row, px), the selected token ids, thr_pos = one token (4 px).
The segmenter (single frame) is used rather than the action-contingency teacher (needs the next frame), so the closed
loop can compute the same masks online.
SeeThrough codes are read from --see-cache (see_codes_{split}.npy / see_prob_{split}.npy of an earlier run of the same
SeeThrough model) when given, otherwise computed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from utils import episode_bounds, save_json

TOK = 4                                                                            # px per token side
W5 = 5 ** np.arange(6)


def digits(codes):
    """int codes (...,) -> FSQ digits (..., 6) uint8 (digit j = (code // 5^j) % 5, the SeeThrough / FSQ order)."""
    return ((np.asarray(codes, np.int64)[..., None] // W5) % 5).astype(np.uint8)


def token_centres():
    r, c = np.divmod(np.arange(256), 16)
    return np.stack([TOK * c + (TOK - 1) / 2, TOK * r + (TOK - 1) / 2], 1).astype(np.float32)   # (u, v)


def see_cache(run: Path, cache: Path, split: str, n: int, out: Path, see_cache_dir: Path | None, dev: str):
    if see_cache_dir is not None and (see_cache_dir / f"see_codes_{split}.npy").exists():
        C = np.load(see_cache_dir / f"see_codes_{split}.npy", mmap_mode="r"); P = np.load(see_cache_dir / f"see_prob_{split}.npy", mmap_mode="r")
        assert len(C) >= n, (len(C), n)
        return C, P
    import torch
    from frontend import SeeThrough, see_codes
    from utils import Frames
    fc, fp = out / f"see_codes_{split}.npy", out / f"see_prob_{split}.npy"
    if not (fc.exists() and fp.exists()):
        sk = torch.load(run / "seethru_learned.pt", map_location=dev, weights_only=False)
        m = SeeThrough(sk["nd"], sk["nl"], sk["width"]).to(dev).eval(); m.load_state_dict(sk["model"])
        obs = Frames(cache / f"{split}_observations.npy", n)
        C = np.lib.format.open_memmap(fc, "w+", np.uint16, (n, 256)); P = np.lib.format.open_memmap(fp, "w+", np.uint8, (n, 256))
        with torch.no_grad():
            for s in range(0, n, 1024):
                x = torch.as_tensor(obs[np.arange(s, min(s + 1024, n))], device=dev).permute(0, 3, 1, 2).float().div_(255)
                c, p = see_codes(m, x)
                C[s:s + len(c)] = c; P[s:s + len(c)] = np.round(p * 255)
        C.flush(); P.flush()
    return np.load(fc, mmap_mode="r"), np.load(fp, mmap_mode="r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="front-end training dir (frontend_train.py --out)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="memory rule frames (component 6)")
    ap.add_argument("--dilate", type=int, default=1, help="agent mask dilation in tokens (component 6)")
    ap.add_argument("--see-cache", type=Path, default=None, help="dir with see_codes_{split}.npy / see_prob_{split}.npy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    from frontend import agent_masks, memory_rule
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    data = {}
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        starts, ends = episode_bounds(a.cache, split, n_ep)
        n = int(ends[-1] + 1)
        C, P = see_cache(a.run, a.cache, split, n, a.out, a.see_cache, a.device)
        seg = np.load(a.run / f"seg_{split}.npy", mmap_mode="r")
        data[split] = (starts, ends, n, C, P, seg)
    # entity selection on TRAIN: tokens whose agent-free memory changes at least once
    starts, ends, n, C, P, seg = data["train"]
    is_start = np.zeros(n, bool); is_start[starts] = True
    changes = np.zeros(256, np.int64)
    for s0, e0 in zip(starts, ends):
        idx = np.arange(s0, e0 + 1)
        blocked = agent_masks(idx, None, seg, is_start, r=a.dilate, teacher=False) | (np.asarray(P[s0:e0 + 1]) <= 127)
        _, ch = memory_rule(np.asarray(C[s0:e0 + 1], np.int64), blocked, a.k)
        changes += ch.sum(0)
    tokens = np.flatnonzero(changes > 0)
    K = len(tokens)
    pos = token_centres()[tokens]
    rep = {"run": str(a.run), "cache": str(a.cache), "k": a.k, "dilate": a.dilate, "entities": int(K), "tokens": tokens.tolist(),
           "train_confirmed_changes_per_entity": changes[tokens].tolist()}
    print({"entities": K, "min": round((time.time() - t0) / 60, 1)}, flush=True)
    for split, (starts, ends, n, C, P, seg) in data.items():
        is_start = np.zeros(n, bool); is_start[starts] = True
        app = np.lib.format.open_memmap(a.out / f"app_{split}.npy", "w+", np.uint8, (n, K, 6))
        area = np.zeros((n, K), np.uint8); free = np.zeros((n, K), bool); agent = np.zeros((n, 32), np.uint8)
        for s in range(0, n, 20000):
            idx = np.arange(s, min(s + 20000, n))
            c = np.asarray(C[idx], np.int64)
            app[idx] = digits(c[:, tokens])
            area[idx] = (np.asarray(P[idx])[:, tokens] > 127).astype(np.uint8)
            free[idx] = ~agent_masks(idx, None, seg, is_start, r=a.dilate, teacher=False)[:, tokens]
            agent[idx] = np.packbits(np.asarray(seg[idx]) > 127, axis=1)
        app.flush(); del app
        np.savez(a.out / f"entities_{split}.npz", area=area, free=free, agent=agent, pos=pos, tokens=tokens, app_scale=np.float32(0.25),
                 thr_pos=np.float32(TOK), n=np.int64(n), episodes=np.int64(len(starts)))
        rep[split] = {"frames": int(n), "episodes": int(len(starts)), "readable": float(area.mean()), "free": float(free.mean()),
                      "readable_and_free": float((area.astype(bool) & free).mean())}
        print(split, rep[split], flush=True)
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    save_json(a.out / "entities_report.json", rep)


if __name__ == "__main__":
    main()
