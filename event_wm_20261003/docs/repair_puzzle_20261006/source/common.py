"""Shared pieces: cached data access, frozen encoder, event code head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIZES = {"3x3": (3, 3), "4x4": (4, 4), "4x5": (4, 5), "4x6": (4, 6)}


def size_of(env: str):
    return SIZES[env.split("-")[2]]


class Split:
    """Memory-mapped cached split (scripts/cache_data.py); `load(n)` copies the first n frames to RAM."""

    def __init__(self, cache: Path, split: str, n_frames: int = 0):
        self.dir, self.split = cache, split
        mm = {k: np.load(cache / f"{split}_{k}.npy", mmap_mode="r")
              for k in ("observations", "actions", "terminals", "button_states", "qpos")
              if (cache / f"{split}_{k}.npy").exists()}
        n = len(mm["terminals"]) if not n_frames else min(n_frames, len(mm["terminals"]))
        # Cut at an episode boundary.
        ends = np.nonzero(np.asarray(mm["terminals"][:n]))[0]
        n = int(ends[-1] + 1) if len(ends) else n
        self.obs = np.ascontiguousarray(mm["observations"][:n])
        self.actions = np.asarray(mm["actions"][:n], np.float32)
        self.terminals = np.asarray(mm["terminals"][:n])
        # privileged arrays, evaluation only
        self.button_states = np.asarray(mm["button_states"][:n], np.uint8) if "button_states" in mm else None
        self.qpos = np.asarray(mm["qpos"][:n], np.float32) if "qpos" in mm else None
        self.ep = np.concatenate([[0], np.cumsum(self.terminals[:-1])]).astype(np.int64)
        self.n = n

    def pairs(self, rng, batch, max_gap):
        """Random (t, t + gap) frame pairs inside one episode, gap in 1..max_gap."""
        while True:
            t = rng.integers(0, self.n - max_gap, batch * 2)
            g = rng.integers(1, max_gap + 1, batch * 2)
            ok = self.ep[t] == self.ep[t + g]
            if ok.sum() >= batch:
                return t[ok][:batch], (t + g)[ok][:batch]

    def true_events(self):
        """Indices t with a button toggle between frames t and t + 1 (evaluation only)."""
        ch = (self.button_states[1:] != self.button_states[:-1]).any(1) & (self.ep[1:] == self.ep[:-1])
        return np.nonzero(ch)[0]


def load_encoder(base: Path, device):
    import torch

    from train_wm import build_model

    ck = torch.load(base, map_location="cpu", weights_only=False)
    m = build_model(ck["action_dim"])
    m.load_state_dict(ck["state_dict"])
    enc = m.encoder.to(device).eval().requires_grad_(False)
    return enc, ck


def tokens(enc, frames_uint8, device):
    """uint8 (B, 64, 64, 3) -> (B, 64, 192) last-layer patch tokens (bf16 autocast)."""
    import torch

    from patch_wm import patch_tokens
    from train_wm import to_pixels

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return patch_tokens(enc, to_pixels(frames_uint8, device)).float()


def make_code_head(bits: int, dim: int = 192, proj: int = 32, hidden: int = 512):
    import torch.nn as nn

    class CodeHead(nn.Module):
        """Patch tokens (B, 64, dim) -> K logits of the binary event code."""

        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proj), nn.GELU())
            self.mlp = nn.Sequential(nn.Linear(64 * proj, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, bits))

        def forward(self, tok):
            return self.mlp(self.proj(tok).flatten(1))

    return CodeHead()


def load_code(code_path: Path, device):
    """code.pt (train_code.py MLP head or sfa_code.py linear map) -> (encoder, f(tokens) -> logits, ck)."""
    import torch

    ck = torch.load(code_path, map_location="cpu", weights_only=False)
    enc, _ = load_encoder(Path(ck["base"]), device)
    if ck.get("kind") == "linear":
        mu, A, c = ck["mu"].to(device), ck["A"].to(device), ck["c"].to(device)

        def logits(tok):
            return (tok.reshape(len(tok), -1) - mu) @ A + c
    else:
        head = make_code_head(ck["bits"]).to(device).eval()
        head.load_state_dict(ck["head"])
        tmu, tsd = ck["token_mean"].to(device), ck["token_std"].to(device)

        def logits(tok):
            return head((tok - tmu) / tsd)
    return enc, logits, ck


def debounce(b: np.ndarray, ep: np.ndarray, m: int) -> np.ndarray:
    """Per-bit hysteresis: a bit takes a new value only after holding it for m frames (within an episode)."""
    out = b.copy()
    starts = np.r_[0, np.nonzero(ep[1:] != ep[:-1])[0] + 1, len(b)]
    for s, e in zip(starts[:-1], starts[1:]):
        for k in range(b.shape[1]):
            x = b[s:e, k]
            ch = np.nonzero(x[1:] != x[:-1])[0] + 1                # run starts
            if len(ch) == 0:
                continue
            run_s = np.r_[0, ch]
            run_e = np.r_[ch, len(x)]
            cur = x[0]
            y = out[s:e, k]
            for rs, re_ in zip(run_s, run_e):
                if x[rs] != cur and re_ - rs >= m:
                    cur = x[rs]
                y[rs:re_] = cur
    return out


def detect_events(b: np.ndarray, ep: np.ndarray, m: int = 3, window: int = 10):
    """Events from a binary code sequence: per-bit debounce, then changes closer than `window`
    frames (one press reveals its 3-5 lights over several frames) merge into one event.
    Returns (t_start, t_end, cleaned code); before = c[t_start], after = c[t_end + 1]."""
    c = debounce(b, ep, m)
    ch = np.nonzero((c[1:] != c[:-1]).any(1) & (ep[1:] == ep[:-1]))[0]
    if len(ch) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), c
    new = np.r_[True, (np.diff(ch) > window) | (ep[ch[1:]] != ep[ch[:-1]])]
    gid = np.cumsum(new) - 1
    t_start = ch[new]
    t_end = np.zeros(len(t_start), np.int64)
    np.maximum.at(t_end, gid, ch)
    keep = (c[t_end + 1] != c[t_start]).any(1)                     # changes that cancel are not events
    return t_start[keep], t_end[keep], c


def pack(bits: np.ndarray) -> np.ndarray:
    """(N, K <= 63) {0,1} -> int64 keys."""
    return (bits.astype(np.int64) << np.arange(bits.shape[1], dtype=np.int64)).sum(1)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
