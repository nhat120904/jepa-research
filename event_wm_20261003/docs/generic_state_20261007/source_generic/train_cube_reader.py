#!/usr/bin/env python3
"""Cube direction A, step 3: CNN object reader trained on the rest-state pseudo-labels.

The colour tracks lose a cube when the arm's shadow or the gripper covers it (~36% of rest frames).
The state is constant inside a rest, so cube_events_px.py labels those frames too; a CNN trained on
these labels reads every object's position from one frame, also under shadow -- as the puzzle's CNN
reader does for the button lights. Per object: a 64x64 heatmap and a soft-argmax -> (u, v) in px.
Shift augmentation moves the labels with the image. If the labels carry the coverage bit, a second
head reads it: features pooled with each object's heatmap -> MLP -> logit (BCE on labelled frames).
Diagnostics: error w.r.t. the pseudo-labels on val, split by whether the colour track sees the object;
coverage accuracy; PRIVILEGED: held-out quadratic fit to qpos xy.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from common import save_json

SLICES = {"single": [14], "double": [14, 21], "triple": [14, 21, 28], "quadruple": [14, 21, 28, 35]}


def make_cube_reader(K: int, width: int = 32, coverage: bool = False):
    import torch
    import torch.nn as nn

    def conv(cin, cout, dil=1):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil), nn.GroupNorm(8, cout), nn.GELU())

    class CubeReader(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = K
            self.full = nn.Sequential(conv(3, width), conv(width, width))
            self.low = nn.Sequential(nn.MaxPool2d(2), conv(width, 2 * width), conv(2 * width, 2 * width, 2),
                                     conv(2 * width, 2 * width, 4), conv(2 * width, 2 * width, 8))
            self.head = nn.Sequential(conv(3 * width, width), nn.Conv2d(width, K, 1))
            self.log_temp = nn.Parameter(torch.zeros(()))
            self.coverage = coverage
            if coverage:
                self.cov_head = nn.Sequential(nn.Linear(3 * width, 128), nn.GELU(), nn.Linear(128, 1))
            v, u = torch.meshgrid(torch.arange(64).float(), torch.arange(64).float(), indexing="ij")
            self.register_buffer("grid", torch.stack([u.flatten(), v.flatten()], -1))       # (4096, 2): (u, v)

        def forward(self, x, with_cov=False):
            """x (B, 3, 64, 64) in [0, 1] -> (B, K, 2) pixel positions (u = column, v = row)
            [, (B, K) coverage logits]."""
            f = self.full(x * 2 - 1)
            g = nn.functional.interpolate(self.low(f), scale_factor=2, mode="bilinear", align_corners=False)
            fg = torch.cat([f, g], 1)
            h = self.head(fg).flatten(2) * self.log_temp.exp()                          # (B, K, 4096)
            with torch.autocast(x.device.type, enabled=False):                         # bf16 would round px to 0.25
                att = torch.softmax(h.float(), -1)
                pos = att @ self.grid
            if not with_cov:
                return pos
            pooled = att.to(fg.dtype) @ fg.flatten(2).transpose(1, 2)                  # (B, K, 3w)
            return pos, self.cov_head(pooled).squeeze(-1).float()

    return CubeReader()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--discover", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True, help="cube_events_px.py output (labels)")
    ap.add_argument("--kind", default="triple")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    t0 = time.time()
    obs = {s: np.load(a.cache / f"{s}_observations.npy", mmap_mode="r") for s in ("train", "val")}
    lab = {s: np.load(a.events / f"labels_{s}.npz") for s in ("train", "val")}
    pos = {s: lab[s]["pos"] for s in lab}; valid = {s: lab[s]["valid"] for s in lab}
    use_cov = "cov" in lab["train"].files
    if use_cov:
        cov = {s: lab[s]["cov"] for s in lab}; cvalid = {s: lab[s]["cov_valid"] for s in lab}
    K = pos["train"].shape[1]
    idx_tr = np.nonzero(valid["train"].any(1))[0]
    print({"objects": K, "train_frames": len(idx_tr), "load_min": round((time.time() - t0) / 60, 1)}, flush=True)
    net = make_cube_reader(K, coverage=use_cov).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))

    def batch(split, idx, augment):
        x = torch.as_tensor(np.asarray(obs[split][np.sort(idx)]), device=dev).permute(0, 3, 1, 2).float().div_(255.0)
        y = torch.as_tensor(pos[split][np.sort(idx)], device=dev)
        m = torch.as_tensor(valid[split][np.sort(idx)], device=dev).float()
        if augment:
            dx, dy = rng.integers(0, 5, 2)
            x = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="replicate")[:, :, dy:dy + 64, dx:dx + 64]
            y = y + torch.tensor([2.0 - dx, 2.0 - dy], device=dev)
        if not use_cov:
            return x, y, m, None, None
        c = torch.as_tensor(cov[split][np.sort(idx)], device=dev).float()
        cm = torch.as_tensor(cvalid[split][np.sort(idx)], device=dev).float()
        return x, y, m, c, cm

    log = []
    for step in range(a.steps):
        x, y, m, c, cm = batch("train", rng.choice(idx_tr, a.batch, replace=False), True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x, with_cov=use_cov)
        p = out[0] if use_cov else out
        loss = (torch.nn.functional.smooth_l1_loss(p, y, reduction="none").sum(-1) * m).sum() / m.sum()
        if use_cov:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(out[1], c, reduction="none")
            loss = loss + (bce * cm).sum() / cm.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            log.append({"step": step, "loss": loss.item(), "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    net.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"reader": net.state_dict(), "K": K, "coverage": use_cov}, a.out / "cube_reader.pt")

    # val: error w.r.t. pseudo-labels, split by colour-track visibility
    nv = len(obs["val"])
    pred = np.zeros((nv, K, 2), np.float32); pcov = np.zeros((nv, K), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, nv, 4096):
            x = torch.as_tensor(np.asarray(obs["val"][s:s + 4096]), device=dev).permute(0, 3, 1, 2).float().div_(255.0)
            if use_cov:
                p_, c_ = net(x, with_cov=True)
                pcov[s:s + len(x)] = torch.sigmoid(c_).cpu().numpy()
            else:
                p_ = net(x)
            pred[s:s + len(x)] = p_.float().cpu().numpy()
    np.save(a.out / "reader_val_pred.npy", pred)
    if use_cov:
        np.save(a.out / "reader_val_cov.npy", pcov)
    mass = np.load(a.discover / "tracks_val.npz")["mass"]
    err = np.linalg.norm(pred - pos["val"], axis=-1)
    res = {"steps": a.steps}
    for name, sel in (("all", valid["val"]), ("track_visible", valid["val"] & (mass >= 3)),
                      ("track_invisible", valid["val"] & (mass < 3))):
        e = err[sel]
        res[name] = {"n": int(sel.sum()), "median_px": float(np.median(e)), "p95_px": float(np.percentile(e, 95)),
                     "within_1px": float((e < 1).mean())}
    # PRIVILEGED diagnostic: held-out quadratic map reader (u, v) -> qpos xy, frames where the label is valid
    import json
    disc = json.loads((a.discover / "discover.json").read_text())
    q = np.load(a.cache / "val_qpos.npy", mmap_mode="r")
    xyz = np.stack([np.asarray(q[:, s:s + 3]) for s in SLICES[a.kind]], 1)
    half = nv // 2
    diag = []
    for d in disc["privileged_diagnostic"]:
        if not d:
            continue
        k, j = d["object"], d["cube"]
        f = np.c_[np.ones(nv), pred[:, k], pred[:, k] ** 2, pred[:, k, :1] * pred[:, k, 1:]]
        tr, te = valid["val"][:, k] & (np.arange(nv) < half), valid["val"][:, k] & (np.arange(nv) >= half)
        W = np.linalg.lstsq(f[tr], xyz[tr, j, :2], rcond=None)[0]
        e = np.linalg.norm(f[te] @ W - xyz[te, j, :2], axis=-1)
        diag.append({"object": k, "cube": j, "median_err_cm": float(np.median(e) * 100), "p95_err_cm": float(np.percentile(e, 95) * 100),
                     "within_2cm": float((e < 0.02).mean())})
    res["privileged_diagnostic"] = diag
    if use_cov:
        cv = cvalid["val"]
        res["coverage"] = {"acc": float(((pcov > 0.5) == cov["val"])[cv].mean()),
                           "recall_covered": float((pcov > 0.5)[cv & cov["val"]].mean()),
                           "false_cov_on_free": float((pcov > 0.5)[cv & ~cov["val"]].mean()), "n": int(cv.sum())}
    res["minutes"] = round((time.time() - t0) / 60, 1)
    save_json(a.out / "reader_eval.json", {"eval": res, "log": log})
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
