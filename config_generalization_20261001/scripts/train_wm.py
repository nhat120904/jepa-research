#!/usr/bin/env python3
"""Train a LeWM-style JEPA world model on an OGBench visual play dataset.

LeWM recipe (le-wm/config/train): 4 frames at frameskip 5 (history 3, one-step
prediction), action blocks of 5 steps, loss = MSE(pred, next emb) + 0.09 SIGReg,
predictor depth 6 / 192-d, AdamW, bf16. Changes for 64x64 OGBench images: ViT-tiny
with patch 8 at 64 px, and a shorter schedule (--epochs, warmup + cosine).
The module classes are the released stable-worldmodel LeWM classes, so the saved
model plugs into the same rollout / planning code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

FRAMES = 4      # history 3 + 1 prediction
SKIP = 5


def episode_starts(terminals):
    """Valid sequence starts: t .. t + SKIP * FRAMES - 1 inside one episode."""
    ep = np.concatenate([[0], np.cumsum(terminals[:-1])])
    span = SKIP * FRAMES - 1
    t = np.arange(len(terminals) - span)
    return t[ep[t] == ep[t + span]]


def held_out(button_states, exclude):
    """Frames whose configuration lies in the held-out region: all `exclude` buttons ON."""
    if not exclude:
        return np.zeros(len(button_states), bool)
    return (np.asarray(button_states)[:, exclude] == 1).all(1)


def drop_windows(starts, bad):
    """Remove sequence starts whose window (SKIP * FRAMES steps) touches a held-out frame."""
    if not bad.any():
        return starts
    c = np.concatenate([[0], np.cumsum(bad)])
    span = SKIP * FRAMES
    return starts[(c[starts + span] - c[starts]) == 0]


def build_model(action_dim):
    import torch.nn as nn
    from stable_pretraining.backbone.utils import vit_hf
    from stable_worldmodel.wm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor

    enc = vit_hf(size="tiny", patch_size=8, image_size=64, pretrained=False, use_mask_token=False)
    proj = MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)
    pred = Predictor(num_frames=3, depth=6, heads=16, mlp_dim=2048, input_dim=192, hidden_dim=192,
                     output_dim=192, dim_head=64, dropout=0.1, emb_dropout=0.0)
    act = Embedder(input_dim=SKIP * action_dim, emb_dim=192)
    pproj = MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)
    return LeWM(enc, pred, act, proj, pproj)


def to_pixels(frames, device):
    """uint8 (..., 64, 64, 3) -> ImageNet-normalised float (..., 3, 64, 64)."""
    import torch

    x = torch.as_tensor(frames, device=device).float().div_(255.0)
    x = x.movedim(-1, -3)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return (x - mean) / std


def batch_from(obs, act_norm, starts, idx, device):
    import torch

    t = starts[idx][:, None] + SKIP * np.arange(FRAMES)[None]          # (B, 4) frame indices
    a = starts[idx][:, None, None] + SKIP * np.arange(FRAMES)[None, :, None] + np.arange(SKIP)[None, None]
    blocks = act_norm[a].reshape(len(idx), FRAMES, -1)                # (B, 4, 5 * A)
    return to_pixels(obs[t], device), torch.as_tensor(blocks, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigreg", type=float, default=0.09)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=None, help="event_wm cache root (<cache>/<env>/train_*.npy)")
    ap.add_argument("--cache-frames", type=int, default=0, help="with --cache: first N frames only")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="puzzle: drop every training window touching a configuration with all these buttons ON")
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("training runs under sbatch")
    import torch
    from stable_worldmodel.wm.loss import SIGReg

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    if a.cache:                                    # event_wm cache: first episodes as uncompressed .npy
        c = a.cache / a.env
        terminals = np.load(c / "train_terminals.npy")
        n = len(terminals)
        if a.cache_frames:
            ends = np.nonzero(terminals[:a.cache_frames])[0]
            n = int(ends[-1] + 1)
        obs = np.ascontiguousarray(np.load(c / "train_observations.npy", mmap_mode="r")[:n])
        actions = np.load(c / "train_actions.npy")[:n].astype(np.float32)
        terminals = terminals[:n]
        bs = c / "train_button_states.npy"                  # puzzle only (used by --exclude)
        z = {"button_states": np.load(bs, mmap_mode="r")[:n]} if bs.exists() else {}
    else:
        z = np.load(a.data / f"{a.env}.npz")
        obs = z["observations"]                    # uint8 (N, 64, 64, 3)
        actions = z["actions"].astype(np.float32)
        terminals = z["terminals"]
    mu, sd = actions.mean(0), actions.std(0) + 1e-6
    act_norm = ((actions - mu) / sd).astype(np.float32)
    starts = episode_starts(terminals)
    if a.exclude:
        n0 = len(starts)
        starts = drop_windows(starts, held_out(z["button_states"], a.exclude))
        print(json.dumps({"exclude": a.exclude, "starts_kept": len(starts), "starts_total": n0}), flush=True)
    if a.cache:
        c = a.cache / a.env
        zv = {"observations": np.load(c / "val_observations.npy", mmap_mode="r"),
              "actions": np.load(c / "val_actions.npy"), "terminals": np.load(c / "val_terminals.npy")}
    else:
        zv = np.load(a.data / f"{a.env}-val.npz")
    vobs, vact = zv["observations"], ((zv["actions"].astype(np.float32) - mu) / sd).astype(np.float32)
    vstarts = episode_starts(zv["terminals"])
    model = build_model(actions.shape[1]).to(dev)
    sig = SIGReg(knots=17, num_proj=1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    steps = a.max_steps or int(a.epochs * len(starts) / a.batch)
    warm = min(1000, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    a.out.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(starts), a.batch)
        px, blocks = batch_from(obs, act_norm, starts, idx, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            info = model.encode({"pixels": px, "action": blocks})
            emb, aemb = info["emb"], info["act_emb"]
            pred = model.predict(emb[:, :3], aemb[:, :3])
            lp = (pred.float() - emb[:, 1:].float()).pow(2).mean()
            ls = sig(emb.float().transpose(0, 1))
            loss = lp + a.sigreg * ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vi = np.random.default_rng(1).integers(0, len(vstarts), 512)
                vpx, vb = batch_from(vobs, vact, vstarts, vi, dev)
                inf = model.encode({"pixels": vpx, "action": vb})
                vp = model.predict(inf["emb"][:, :3], inf["act_emb"][:, :3])
                vloss = (vp.float() - inf["emb"][:, 1:].float()).pow(2).mean().item()
                copy = (inf["emb"][:, :3].float() - inf["emb"][:, 1:].float()).pow(2).mean().item()
            model.train()
            rec = {"step": step, "pred": lp.item(), "sigreg": ls.item(), "val_pred": vloss,
                   "val_copy": copy, "lr": sched.get_last_lr()[0], "min": (time.time() - t0) / 60}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    model.eval()
    torch.save({"state_dict": model.state_dict(), "action_mean": mu, "action_std": sd,
                "action_dim": int(actions.shape[1]), "env": a.env, "steps": steps, "args": vars(a) | {"data": str(a.data), "out": str(a.out)}},
               a.out / "wm.pt")
    (a.out / "train_log.json").write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
