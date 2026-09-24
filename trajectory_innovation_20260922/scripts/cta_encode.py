"""Encode a CTA collection once and cache it (docs/CTA_E2E_PROTOCOL.md).

Frozen DINOv2 ViT-S/14 patch tokens -> PCA-128, with the basis fitted on training decision frames only. Per split:
cur (N, 256, D), prev (N, 64, D), end (N, K, 256, D), seg (N, K, 3, 64, D) as fp16 .npy, plus meta.npz.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ti_wm.codec import pool_grid
from ti_wm.contract import require_compute
from ti_wm.cta import N_SEG, SMALL
from ti_wm.pusht_runtime import VisualScorer
from ti_wm.sibling import PCA_DIM, TOKENS, fit_pca, project

K = 8
ENC_BATCH, PCA_FRAMES = 512, 2000
META = ("root", "decision", "t", "ctx_pos", "chunk", "end_pos", "end_prev_pos", "cov8", "done8")
SPLITS = {"train": (30250, 31049), "dev": (2000, 2099)}


def shards(collect, lo, hi):
    return sorted(p for p in Path(collect).glob("shard_*.npz") if lo <= int(p.stem.split("_")[1]) <= hi)


class Tokens:
    def __init__(self, visual):
        self.visual = visual
        self.mean = self.basis = None

    @torch.inference_mode()
    def fit(self, frames):
        raw = torch.cat([self.visual.features(frames[s:s + ENC_BATCH]) for s in range(0, len(frames), ENC_BATCH)])
        self.mean, self.basis = fit_pca(raw.reshape(-1, raw.shape[-1]))

    @torch.inference_mode()
    def __call__(self, frames, side=None):
        """(..., H, W, 3) uint8 -> (..., tokens, PCA_DIM) fp16 numpy; optionally pooled to side x side."""
        lead = frames.shape[:-3]
        flat = frames.reshape(-1, *frames.shape[-3:])
        out = []
        for s in range(0, len(flat), ENC_BATCH):
            x = project(self.visual.features(flat[s:s + ENC_BATCH]), self.mean, self.basis)
            out.append((pool_grid(x, side) if side else x).half().cpu())
        out = torch.cat(out).numpy()
        return out.reshape(*lead, *out.shape[1:])


def encode_split(tok, files, out):
    out.mkdir(parents=True, exist_ok=True)
    sizes = [len(np.load(f)["root"]) for f in files]
    n = sum(sizes)
    shapes = {"cur": (n, TOKENS, PCA_DIM), "prev": (n, SMALL, PCA_DIM), "end": (n, K, TOKENS, PCA_DIM),
              "seg": (n, K, N_SEG, SMALL, PCA_DIM)}
    arrays = {k: open_memmap(out / f"{k}.npy", mode="w+", dtype=np.float16, shape=s) for k, s in shapes.items()}
    meta = {k: [] for k in META}
    o = 0
    for f, size in zip(files, sizes):
        d = np.load(f)
        arrays["cur"][o:o + size] = tok(d["ctx"])
        arrays["prev"][o:o + size] = tok(d["ctx_prev"], 8)
        arrays["end"][o:o + size] = tok(d["end"])
        arrays["seg"][o:o + size] = tok(d["seg"], 8)
        for k in META:
            meta[k].append(d[k])
        o += size
        print(f"{out.name}: {f.name} ({size} decisions)", flush=True)
    for a in arrays.values():
        a.flush()
    np.savez(out / "meta.npz", **{k: np.concatenate(v) for k, v in meta.items()})
    return n


def main(collect, out, smoke, device, smoke_mode):
    require_compute()
    t0 = time.perf_counter()
    files = ({s: shards(collect, 0, 99999) for s in SPLITS} if smoke_mode
             else {s: shards(collect, *r) for s, r in SPLITS.items()})
    if not smoke_mode:
        assert len(files["train"]) == 16 and len(files["dev"]) == 2, {k: [f.name for f in v] for k, v in files.items()}
    tok = Tokens(VisualScorer(device))
    ctx = np.concatenate([np.load(f)["ctx"] for f in files["train"]])
    rng = np.random.default_rng(0)
    tok.fit(ctx[rng.choice(len(ctx), min(PCA_FRAMES, len(ctx)), replace=False)])
    del ctx
    torch.save({"pca_mean": tok.mean.cpu(), "pca_basis": tok.basis.cpu()}, out / "pca.pt")
    np.save(out / "goals.npy", tok(np.load(smoke / "goal_frames.npz")["frames"]))
    counts = {s: encode_split(tok, f, out / s) for s, f in files.items()}
    report = {"decisions": counts, "shards": {s: [f.name for f in v] for s, v in files.items()},
              "smoke_mode": smoke_mode, "seconds": time.perf_counter() - t0}
    (out / "features.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("collect", "out", "smoke"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-mode", action="store_true", help="all shards as both splits (code-path check only)")
    a = parser.parse_args()
    main(a.collect, a.out, a.smoke, a.device, a.smoke_mode)
