#!/usr/bin/env python3
"""CNN code reader on raw pixels, trained on the pipeline's own pseudo-labels (no state labels).

All failures of the 4x5 official-protocol run (57014) are one misread light in the goal frame, which
is read once (arm at a random, often low pose). The linear token readout plateaus at .985-.988 frame
exactness after two self-training rounds. Like the skill (v3), a CNN trained end to end on pixels
may localise occluded lights better. Pseudo-labels: per-bit majority of the current code over each
inter-event segment (refine_code.py), frames > margin from event spans; shift augmentation.
Output keeps the bit order, so event types, the event WM, the cost-to-go and the skill stay valid.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from common import Split, save_json, size_of
from train_code import evaluate


def make_reader(bits: int, width: int = 32, hidden: int = 512):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(), nn.MaxPool2d(2))

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(block(3, width), block(width, 2 * width), block(2 * width, 2 * width),
                                     block(2 * width, 4 * width))
            self.head = nn.Sequential(nn.Linear(4 * width * 16, hidden), nn.GELU(), nn.Linear(hidden, bits))

        def forward(self, px):
            """px (B, 3, 64, 64) in [0, 1] -> (B, bits) logits."""
            return self.head(self.cnn(px * 2 - 1).flatten(1))

    return Reader()


def to_px(frames, dev):
    import torch
    return torch.as_tensor(np.asarray(frames), device=dev).permute(0, 3, 1, 2).float().div_(255.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--events", type=Path, required=True, help="build_events.py output of the current code")
    ap.add_argument("--train-frames", type=int, default=1_500_000)
    ap.add_argument("--val-frames", type=int, default=100_000)
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    rows, cols = size_of(a.env)
    t0 = time.time()
    tr = np.load(a.events / "events_train.npz")
    train, val = Split(a.cache, "train", a.train_frames), Split(a.cache, "val", a.val_frames)
    y = tr["codes"][: train.n].astype(np.float32)
    bounds = np.r_[0, np.nonzero(train.ep[1:] != train.ep[:-1])[0] + 1]
    cuts = np.unique(np.r_[bounds, tr["t"][tr["t"] < train.n - 1] + 1, train.n])
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        y[s0:s1] = (y[s0:s1].mean(0) > 0.5)
    near = np.zeros(train.n + 1, np.int64)
    for s, e in zip(tr["t_start"], tr["t"]):
        if s < train.n:
            near[max(0, s - a.margin)] += 1
            near[min(train.n, e + a.margin + 1)] -= 1
    ok = np.nonzero(np.cumsum(near)[: train.n] == 0)[0]
    K = y.shape[1]
    yt = torch.as_tensor(y, device=dev)
    net = make_reader(K).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    log = []
    for step in range(a.steps):
        i = rng.choice(ok, a.batch)
        px = to_px(train.obs[i], dev)
        pad = F.pad(px, (2, 2, 2, 2), mode="replicate")
        dx, dy = rng.integers(0, 5, 2)
        px = pad[:, :, dy:dy + 64, dx:dx + 64]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.binary_cross_entropy_with_logits(net(px).float(), yt[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 5000 == 0 or step == a.steps - 1:
            log.append({"step": step, "loss": loss.item(), "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pv = torch.cat([torch.sigmoid(net(to_px(val.obs[s:s + 4096], dev)).float()).cpu()
                        for s in range(0, val.n, 4096)]).numpy()
    res, _ = evaluate(pv, val, rows, cols)
    b_ = pv > 0.5
    st = val.button_states.astype(bool)
    agree = (b_[:, :, None] == st[:, None, :]).mean(0)
    k = np.argmax(np.maximum(agree, 1 - agree), 0)
    inv = agree[k, np.arange(st.shape[1])] < 0.5
    res["frame_exact"] = float(((b_[:, k] ^ inv[None]) == st).all(1).mean())
    res["log"] = log
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "cnn", "bits": K, "reader": net.state_dict(), "env": a.env,
                "slow_mask": torch.ones(K, dtype=torch.bool)}, a.out / "reader.pt")
    save_json(a.out / "reader_eval.json", res)
    print({"frame_exact": res["frame_exact"], "per_light": res["per_light_best_bit_acc"]}, flush=True)


if __name__ == "__main__":
    main()
