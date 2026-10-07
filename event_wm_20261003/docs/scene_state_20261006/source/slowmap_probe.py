#!/usr/bin/env python3
"""PRIVILEGED probes of a slow-feature map (slowmap.py): is the scene in it, is the robot out of it?

Cube positions: keypoint probe -- a 1x1 conv (C -> n objects) + spatial softmax gives an expected grid
position per object, then a learned affine map to metres; trained on the first half of the val frames,
scored (R^2, median error) on the second half. Arm: MLP on the 8x8-pooled map -> arm joints (qpos[:6]).
Both probes are also run on raw pixels (same architecture; RGB in place of Phi) as the reference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from slowmap import make_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--slowmap", type=Path, required=True, help="slowmap.pt")
    ap.add_argument("--frames", type=int, default=60_000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    dev = "cuda"
    ck = torch.load(a.slowmap, map_location=dev)
    enc = make_encoder(ck["C"]).to(dev).eval(); enc.load_state_dict(ck["enc"])
    vo = np.load(a.cache / "val_observations.npy", mmap_mode="r")
    m = min(a.frames, len(vo))
    q = np.asarray(np.load(a.cache / "val_qpos.npy", mmap_mode="r")[:m]).astype(np.float32)
    slices = [s for s in (14, 21, 28, 35) if s + 3 <= q.shape[1]]
    tgt = torch.as_tensor(np.concatenate([q[:, s:s + 2] for s in slices], 1), device=dev)     # (m, 2K)
    arm = torch.as_tensor(q[:, :6], device=dev)
    X = torch.as_tensor(np.array(vo[:m]), device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        Z = torch.cat([enc(X[s:s + 2048].permute(0, 3, 1, 2).float().div(127.5).sub(1)).float() for s in range(0, m, 2048)])
    half = m // 2
    res = {"env": a.cache.name, "frames": m}

    def r2(p, y):
        return (1 - ((p - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).cpu().numpy().round(3).tolist()

    def keypoint_probe(F, K):
        """F (m, C, H, W) -> per-object expected position -> affine to metres."""
        C, H, W = F.shape[1:]
        conv = nn.Conv2d(C, K, 1).to(dev)
        aff = nn.Linear(2 * K, 2 * K).to(dev)
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
        opt = torch.optim.Adam(list(conv.parameters()) + list(aff.parameters()), lr=3e-3)
        mu, sd = tgt[:half].mean(0), tgt[:half].std(0)

        def fwd(f):
            att = torch.softmax(conv(f).flatten(2), -1).view(len(f), K, H, W)
            kp = torch.stack([(att * gx).sum((2, 3)), (att * gy).sum((2, 3))], -1).flatten(1)
            return aff(kp) * sd + mu

        for _ in range(a.steps):
            i = torch.randint(0, half, (512,), device=dev)
            loss = ((fwd(F[i]) - tgt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p = torch.cat([fwd(F[s:s + 4096]) for s in range(half, m, 4096)])
        err = (p - tgt[half:]).view(-1, K, 2).norm(dim=-1)
        return {"r2": r2(p, tgt[half:]), "median_err_cm": (err.median(0).values * 100).cpu().numpy().round(2).tolist(),
                "within_2cm": (err < 0.02).float().mean(0).cpu().numpy().round(3).tolist()}

    def mlp_probe(F, Y):
        Fp = nn.functional.adaptive_avg_pool2d(F, 8).flatten(1)
        mu, sd = Fp[:half].mean(0), Fp[:half].std(0) + 1e-3
        net = nn.Sequential(nn.Linear(Fp.shape[1], 256), nn.GELU(), nn.Linear(256, Y.shape[1])).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        ym, ys = Y[:half].mean(0), Y[:half].std(0) + 1e-6
        for _ in range(a.steps):
            i = torch.randint(0, half, (512,), device=dev)
            loss = ((net((Fp[i] - mu) / sd) - (Y[i] - ym) / ys) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p = net((Fp[half:] - mu) / sd) * ys + ym
        return r2(p, Y[half:])

    K = len(slices)
    res["phi_cube_keypoint"] = keypoint_probe(Z, K)
    res["phi_arm_mlp_r2"] = mlp_probe(Z, arm)
    Xp = X.permute(0, 3, 1, 2).float().div(127.5).sub(1)
    res["pixels_cube_keypoint"] = keypoint_probe(Xp, K)
    res["pixels_arm_mlp_r2"] = mlp_probe(Xp, arm)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"probe_{a.cache.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
