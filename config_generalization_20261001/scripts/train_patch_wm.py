#!/usr/bin/env python3
"""Train the patch-token predictor (scripts/patch_wm.py) on a frozen encoder taken
from a trained LeWM-style checkpoint (scripts/train_wm.py). Same data, frameskip and
action blocks as train_wm.py; loss = MSE between predicted and actual next-frame
patch tokens (standardised per channel), teacher-forced on all three context frames.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_wm import PatchPredictor, patch_tokens  # noqa: E402
from train_wm import SKIP, batch_from, build_model, drop_windows, episode_starts, held_out  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--base", type=Path, required=True, help="train_wm.py checkpoint (encoder source)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--exclude", type=int, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("training runs under sbatch")
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    ck = torch.load(a.base, map_location="cpu", weights_only=False)
    base = build_model(ck["action_dim"]).to(dev).eval()
    base.load_state_dict(ck["state_dict"])
    enc = base.encoder.requires_grad_(False)
    mu, sd = ck["action_mean"], ck["action_std"]
    z = np.load(a.data / f"{a.env}.npz")
    obs, terminals = z["observations"], z["terminals"]
    act_norm = ((z["actions"].astype(np.float32) - mu) / sd).astype(np.float32)
    starts = episode_starts(terminals)
    if a.exclude:
        starts = drop_windows(starts, held_out(z["button_states"], a.exclude))
    zv = np.load(a.data / f"{a.env}-val.npz")
    vobs, vstarts = zv["observations"], episode_starts(zv["terminals"])
    vact = ((zv["actions"].astype(np.float32) - mu) / sd).astype(np.float32)

    def tokens(px):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            B, F = px.shape[:2]
            return patch_tokens(enc, px.flatten(0, 1)).float().view(B, F, 64, -1)

    # Per-channel token statistics from a sample of training frames.
    px, _ = batch_from(obs, act_norm, starts, rng.integers(0, len(starts), 512), dev)
    t = tokens(px)
    tmu, tsd = t.mean((0, 1, 2)), t.std((0, 1, 2)) + 1e-6
    pred = PatchPredictor(dim=t.shape[-1], action_dim=SKIP * ck["action_dim"]).to(dev)
    opt = torch.optim.AdamW(pred.parameters(), lr=a.lr, weight_decay=1e-3)
    warm = min(1000, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    a.out.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    for step in range(a.steps):
        px, blocks = batch_from(obs, act_norm, starts, rng.integers(0, len(starts), a.batch), dev)
        tk = (tokens(px) - tmu) / tsd
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pred(tk[:, :3], blocks[:, :3])
        loss = (out.float() - tk[:, 1:]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            with torch.no_grad():
                vi = np.random.default_rng(1).integers(0, len(vstarts), 256)
                vpx, vb = batch_from(vobs, vact, vstarts, vi, dev)
                vt = (tokens(vpx) - tmu) / tsd
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vo = pred(vt[:, :3], vb[:, :3]).float()
                vl = (vo - vt[:, 1:]).pow(2).mean().item()
                vc = (vt[:, :3] - vt[:, 1:]).pow(2).mean().item()
            rec = {"step": step, "loss": loss.item(), "val": vl, "val_copy": vc, "min": (time.time() - t0) / 60}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    torch.save({"predictor": pred.state_dict(), "token_mean": tmu.cpu(), "token_std": tsd.cpu(),
                "base": str(a.base), "env": a.env, "steps": a.steps}, a.out / "patch_wm.pt")
    (a.out / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
