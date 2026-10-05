#!/usr/bin/env python3
"""Where is button-state information lost? Probe several feature levels of a trained
world model (and raw pixels) for the puzzle button states, train vs validation.

Features: raw pixels (16x16 average-pooled), encoder patch tokens (last layer,
spatial grid), encoder CLS before the projector, CLS after the projector (the
world-model latent). Probes: linear and a 2-layer MLP, trained on training frames,
scored on held-out validation frames (bit accuracy and exact configuration match).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_puzzle import episodes  # noqa: E402
from train_wm import build_model, to_pixels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-val", type=int, default=20000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda"
    ck = torch.load(a.wm, map_location="cpu", weights_only=False)
    model = build_model(ck["action_dim"]).to(dev).eval()
    model.load_state_dict(ck["state_dict"])
    rng = np.random.default_rng(0)
    out = {"env": a.env}
    feats = {}
    for split, n in (("train", a.n_train), ("val", a.n_val)):
        z = np.load(a.data / (f"{a.env}.npz" if split == "train" else f"{a.env}-val.npz"))
        idx = np.sort(rng.choice(len(z["terminals"]), n, replace=False))
        obs = z["observations"][idx]
        y = z["button_states"][idx].astype(np.float32)
        f = {"pixels16": [], "patch": [], "cls_pre": [], "latent": []}
        with torch.no_grad():
            for i in range(0, n, 2048):
                px = to_pixels(obs[i:i + 2048], dev)                       # (B, 3, 64, 64)
                f["pixels16"].append(F.avg_pool2d(px, 4).flatten(1))
                h = model.encoder(px, interpolate_pos_encoding=True).last_hidden_state
                f["patch"].append(h[:, 1:].flatten(1))                     # (B, 64 * 192)
                f["cls_pre"].append(h[:, 0])
                f["latent"].append(model.projector(h[:, 0]))
        feats[split] = ({k: torch.cat(v).float() for k, v in f.items()}, torch.as_tensor(y, device=dev))
    for name in ("pixels16", "patch", "cls_pre", "latent"):
        Xtr, ytr = feats["train"][0][name], feats["train"][1]
        Xva, yva = feats["val"][0][name], feats["val"][1]
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xva = (Xtr - m) / s, (Xva - m) / s
        res = {}
        for kind in ("linear", "mlp"):
            d = Xtr.shape[1]
            net = (nn.Linear(d, ytr.shape[1]) if kind == "linear" else
                   nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, ytr.shape[1]))).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            for step in range(6000):
                i = torch.randint(0, len(Xtr), (1024,), device=dev)
                loss = F.binary_cross_entropy_with_logits(net(Xtr[i]), ytr[i])
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                ptr = (net(Xtr[:20000]) > 0).float()
                pva = (net(Xva) > 0).float()
            res[kind] = {"train_bit": round(float((ptr == ytr[:20000]).float().mean()), 4),
                         "val_bit": round(float((pva == yva).float().mean()), 4),
                         "val_exact": round(float((pva == yva).all(1).float().mean()), 4)}
        out[name] = res
        print(name, json.dumps(res), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
