#!/usr/bin/env python3
"""Unified backend, step 3: fast entity reader trained on the rest-state pseudo-labels of u_events.py.

Per identity k from one frame: position (64x64 heatmap + soft-argmax), appearance (features pooled with the
heatmap -> RGB), covered (logit) and at-rest (logit; frames between rests of k -- carried, in transit --
are 0). The at-rest head lets the closed loop end an event only when the acted entity is at rest again
(a cube paused in the gripper is in transit). Shift augmentation moves the labels with the image.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from common import save_json


def make_reader(K: int, width: int = 32, agent: bool = False):
    import torch
    import torch.nn as nn

    def conv(cin, cout, dil=1):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil), nn.GroupNorm(8, cout), nn.GELU())

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = K
            self.full = nn.Sequential(conv(3, width), conv(width, width))
            self.low = nn.Sequential(nn.MaxPool2d(2), conv(width, 2 * width), conv(2 * width, 2 * width, 2),
                                     conv(2 * width, 2 * width, 4), conv(2 * width, 2 * width, 8))
            self.head = nn.Sequential(conv(3 * width, width), nn.Conv2d(width, K, 1))
            self.log_temp = nn.Parameter(torch.zeros(()))
            self.attr = nn.Sequential(nn.Linear(3 * width, 128), nn.GELU(), nn.Linear(128, 5))     # app(3), covered, rest
            self.agent = nn.Sequential(conv(3 * width, width), nn.Conv2d(width, 1, 1)) if agent else None   # agent-mask logit
            v, u = torch.meshgrid(torch.arange(64).float(), torch.arange(64).float(), indexing="ij")
            self.register_buffer("grid", torch.stack([u.flatten(), v.flatten()], -1))

        def forward(self, x, return_agent=False):
            """x (B, 3, 64, 64) in [0, 1] -> state (B, K, 6) = (u, v, r, g, b, covered LOGIT), rest logit (B, K)
            [, agent-mask logit (B, 64, 64) if built with agent=True and return_agent]."""
            f = self.full(x * 2 - 1)
            g = nn.functional.interpolate(self.low(f), scale_factor=2, mode="bilinear", align_corners=False)
            fg = torch.cat([f, g], 1)
            h = self.head(fg).flatten(2) * self.log_temp.exp()
            with torch.autocast(x.device.type, enabled=False):
                att = torch.softmax(h.float(), -1)
                pos = att @ self.grid
            pooled = att.to(fg.dtype) @ fg.flatten(2).transpose(1, 2)
            o = self.attr(pooled).float()
            out = torch.cat([pos, torch.sigmoid(o[..., :3]), o[..., 3:4]], -1), o[..., 4]
            if self.agent is not None and return_agent:
                return out + (self.agent(fg)[:, 0].float(),)
            return out

    return Reader()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True, help="u_events.py output (labels)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--agent-head", action="store_true", help="also predict the agent mask (front-end agent segments)")
    ap.add_argument("--entities", type=Path, default=None, help="front-end dir with entities_{split}.npz (agent masks)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda", help="cpu for smoke tests")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = a.device
    t0 = time.time()
    L = {s: dict(np.load(a.events / f"labels_{s}.npz")) for s in ("train", "val")}   # decompress once (NpzFile reloads per access)
    K = L["train"]["pos"].shape[1]
    # frames of processed episodes: any identity labelled or covered-labelled
    idx = {s: np.nonzero(L[s]["valid"].any(1) | L[s]["cov_valid"].any(1))[0] for s in L}
    # the labelled frames lie in the first episodes: read that prefix once into RAM (random access on the
    # memory-mapped file over network storage is slow)
    obs = {s: np.ascontiguousarray(np.load(a.cache / f"{s}_observations.npy", mmap_mode="r")[: idx[s].max() + 1]) for s in L}
    net = make_reader(K, agent=a.agent_head).to(dev)
    if a.agent_head:
        AG = {s_: dict(np.load(a.entities / f"entities_{s_}.npz")) for s_ in ("train", "val")}
        AGm = {s_: AG[s_]["agent"] for s_ in AG}; AGp = {s_: AG[s_]["processed"] for s_ in AG}
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    def batch(split, ii, augment):
        ii = np.sort(ii)
        x = torch.as_tensor(np.asarray(obs[split][ii]), device=dev).permute(0, 3, 1, 2).float().div_(255.0)
        lab = {k: torch.as_tensor(L[split][k][ii], device=dev) for k in ("pos", "app", "valid", "cov", "cov_valid")}
        if a.agent_head:
            lab["agent"] = torch.as_tensor(np.unpackbits(AGm[split][ii], axis=-1)[:, :, :64].astype(np.float32), device=dev)
            lab["agent_ok"] = torch.as_tensor(AGp[split][ii], device=dev).float()
        if augment:
            dx, dy = rng.integers(0, 5, 2)
            x = F.pad(x, (2, 2, 2, 2), mode="replicate")[:, :, dy:dy + 64, dx:dx + 64]
            lab["pos"] = lab["pos"] + torch.tensor([2.0 - dx, 2.0 - dy], device=dev)
            if a.agent_head:
                lab["agent"] = F.pad(lab["agent"][:, None], (2, 2, 2, 2), mode="replicate")[:, 0, dy:dy + 64, dx:dx + 64]
        return x, lab

    for step in range(a.steps):
        x, lb = batch("train", rng.choice(idx["train"], a.batch, replace=False), True)
        with torch.autocast(dev, dtype=torch.bfloat16):
            outs = net(x, return_agent=a.agent_head)
        st, rest = outs[0], outs[1]
        m = lb["valid"].float(); cm = lb["cov_valid"].float()
        loss = (F.smooth_l1_loss(st[..., :2], lb["pos"], reduction="none").sum(-1) * m).sum() / m.sum().clamp(min=1)
        loss = loss + 10 * (((st[..., 2:5] - lb["app"]) ** 2).sum(-1) * m).sum() / m.sum().clamp(min=1)
        loss = loss + (F.binary_cross_entropy_with_logits(st[..., 5], lb["cov"].float(), reduction="none") * cm).sum() / cm.sum().clamp(min=1)
        loss = loss + F.binary_cross_entropy_with_logits(rest, m)
        if a.agent_head and lb["agent_ok"].sum() > 0:                  # agent masks exist on processed frames only
            la = F.binary_cross_entropy_with_logits(outs[2], lb["agent"], reduction="none").mean((1, 2))
            loss = loss + (la * lb["agent_ok"]).sum() / lb["agent_ok"].sum()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            print({"step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)}, flush=True)
    net.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"reader": net.state_dict(), "K": K, "agent": a.agent_head}, a.out / "u_reader.pt")
    # val evaluation on processed frames
    vi = idx["val"]
    P, R = [], []
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for s in range(0, len(vi), 4096):
            x, _ = batch("val", vi[s:s + 4096], False)
            st, rest = net(x)
            P.append(st.float().cpu().numpy()); R.append(rest.float().cpu().numpy())
    P, R = np.concatenate(P), np.concatenate(R)
    lv = {k: L["val"][k][vi] for k in ("pos", "app", "valid", "cov", "cov_valid")}
    pe = np.linalg.norm(P[..., :2] - lv["pos"], axis=-1)[lv["valid"]]
    ae = np.abs(P[..., 2:5] - lv["app"]).max(-1)[lv["valid"]]
    res = {"K": K, "val_frames": int(len(vi)), "pos_err_median_px": float(np.median(pe)), "pos_within_1px": float((pe < 1).mean()),
           "app_err_median": float(np.median(ae)), "covered_acc": float(((P[..., 5] > 0) == lv["cov"])[lv["cov_valid"]].mean()) if lv["cov_valid"].any() else None,
           "rest_acc": float(((R > 0) == lv["valid"]).mean()), "rest_recall_in_transit": float((R <= 0)[~lv["valid"]].mean()),
           "minutes": round((time.time() - t0) / 60, 1)}
    save_json(a.out / "u_reader_eval.json", res)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
