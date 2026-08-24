"""
Phase 0 / Gate 2 -- does the FROZEN ContactWorld world-model latent carry the object state
that its own success metric thresholds?

Script 01 measures how much plug-pose information is present in the raw observations
(an upper bound). This script measures how much survives into the latent `z` that the CEM
planner actually scores. The gap between the two is a readout/representation failure rather
than an observability failure.

Uses ContactWorld's own `build_model` + `load_lightning_ckpt`, so the encoder is exactly the
released one. No IsaacGym import is involved on this path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zarr

CW_ROOT = Path("/mnt/data/nhatnc129/contactworld/ContactWorld")
sys.path.insert(0, str(CW_ROOT))

from planner_utils import PlannerConfig, build_model, load_lightning_ckpt  # noqa: E402


def load_arrays(task_root: Path, vision_key: str, tactile_key: str, target_key: str):
    z = zarr.open(str(task_root), mode="r")
    g = z["data"]
    ends = np.asarray(z["meta"]["episode_ends"]).astype(int)
    starts = np.concatenate([[0], ends[:-1]])
    vision = np.asarray(g[vision_key]).astype(np.float32)
    tactile = np.asarray(g[tactile_key]).astype(np.float32)
    # ContactWorld's ZarrDataset converts [T,H,W,C] -> [T,C,H,W] when C in (1,3)
    # (dataset/zarr_dataset.py:179-181). Reproduce it exactly or the encoder asserts.
    if vision.ndim == 4 and vision.shape[-1] in (1, 3):
        vision = np.ascontiguousarray(vision.transpose(0, 3, 1, 2))
    if tactile.ndim == 4 and tactile.shape[-1] in (1, 3):
        tactile = np.ascontiguousarray(tactile.transpose(0, 3, 1, 2))
    return {
        "vision": vision,
        "tactile": tactile,
        "target": np.asarray(g[target_key]).astype(np.float32),
        "action_dim": int(np.asarray(g["action"]).shape[-1]),
        "episodes": [(int(s), int(e)) for s, e in zip(starts, ends)],
    }


@torch.no_grad()
def encode_all(model, data, vision_key, tactile_key, use_tactile, device, bs=64):
    zs = []
    n = len(data["vision"])
    for i in range(0, n, bs):
        v = torch.from_numpy(data["vision"][i:i + bs]).unsqueeze(1).to(device)   # [B,1,...]
        batch = {vision_key: v}
        if use_tactile:
            batch[tactile_key] = torch.from_numpy(data["tactile"][i:i + bs]).unsqueeze(1).to(device)
        emb = model.encode(batch)          # [B,T,D]
        zs.append(emb[:, -1].float().cpu().numpy())
    return np.concatenate(zs, 0)


def episode_split(n_ep, seed, test_frac):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ep)
    n_te = max(4, int(round(test_frac * n_ep)))
    return perm[n_te:], perm[:n_te]


def frame_episode_ids(episodes, n):
    ids = np.zeros(n, dtype=int)
    for i, (s, e) in enumerate(episodes):
        ids[s:e] = i
    return ids


def fit_linear(Xtr, ytr, Xte, lam=1e-3):
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1), np.float32)], 1)
    A = Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ W


def fit_mlp(Xtr, ytr, Xte, device, epochs=60, hid=512, lr=1e-3, bs=256, seed=0):
    torch.manual_seed(seed)
    xm, xs = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = ytr.mean(0, keepdims=True), ytr.std(0, keepdims=True) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hid), nn.ReLU(),
                        nn.Linear(hid, hid), nn.ReLU(),
                        nn.Linear(hid, ytr.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy((Xtr - xm) / xs).to(device)
    Yt = torch.from_numpy((ytr - ym) / ys).to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = nn.functional.mse_loss(net(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy((Xte - xm) / xs).to(device)).cpu().numpy()
    return pred * ys + ym


def cluster_bootstrap(err, ep_ids, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    eps = np.unique(ep_ids)
    by = {e: err[ep_ids == e] for e in eps}
    vals = [np.concatenate([by[e] for e in rng.choice(eps, len(eps), replace=True)]).mean()
            for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vision-key", default="pointcloud")
    p.add_argument("--vision-type", default="pc")
    p.add_argument("--pc-in-channels", type=int, default=6)
    p.add_argument("--use-tactile", action="store_true")
    p.add_argument("--tactile-key", default="tactile_force_field_right")
    p.add_argument("--tactile-dim", type=int, default=64)
    p.add_argument("--vision-dim", type=int, default=512)
    p.add_argument("--target-key", default="plug_pos")
    p.add_argument("--success-thresh", type=float, default=0.01)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_root = CW_ROOT / "data" / "demo_data" / args.task
    data = load_arrays(task_root, args.vision_key, args.tactile_key, args.target_key)

    cfg = PlannerConfig(
        vision_key=args.vision_key, vision_type=args.vision_type,
        pc_in_channels=args.pc_in_channels, vision_dim=args.vision_dim,
        use_tactile=args.use_tactile, tactile_key=args.tactile_key,
        tactile_dim=args.tactile_dim, tactile_in_channels=3,
        tactile_height=10, tactile_width=14, fusion_type="concat",
        reg_loss_type="vc", reg_on_vision_only=True,
        action_dim=data["action_dim"], history_size=1,
    )
    model = build_model(cfg)
    model = load_lightning_ckpt(model, args.ckpt).to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)

    Z = encode_all(model, data, args.vision_key, args.tactile_key, args.use_tactile, device)
    Y = data["target"]
    ep_ids = frame_episode_ids(data["episodes"], len(Y))
    print(f"task={args.task} latent_dim={Z.shape[1]} frames={len(Z)} episodes={len(data['episodes'])}",
          flush=True)

    res = {"task": args.task, "ckpt": args.ckpt, "latent_dim": int(Z.shape[1]),
           "target": args.target_key, "use_tactile": args.use_tactile,
           "success_thresh_m": args.success_thresh, "readouts": {}}

    acc = {"linear": [], "mlp": []}
    for seed in args.seeds:
        tr_ep, te_ep = episode_split(len(data["episodes"]), seed, args.test_frac)
        tr = np.isin(ep_ids, tr_ep); te = np.isin(ep_ids, te_ep)
        for name, fn in [("linear", lambda: fit_linear(Z[tr], Y[tr], Z[te])),
                         ("mlp", lambda: fit_mlp(Z[tr], Y[tr], Z[te], device, seed=seed))]:
            pred = fn()
            err = np.linalg.norm(pred - Y[te], axis=-1)
            acc[name].append((err, ep_ids[te] + 1000 * seed))
            print(f"  seed{seed} {name:7s} mean={err.mean()*100:6.2f}cm "
                  f"median={np.median(err)*100:6.2f}cm hit@1cm={(err < args.success_thresh).mean():.3f}",
                  flush=True)

    for name, runs in acc.items():
        e = np.concatenate([r[0] for r in runs]); ep = np.concatenate([r[1] for r in runs])
        lo, hi = cluster_bootstrap(e, ep)
        res["readouts"][name] = {
            "mean_err_cm": float(e.mean() * 100), "mean_err_cm_ci": [lo * 100, hi * 100],
            "median_err_cm": float(np.median(e) * 100),
            "hit_at_thresh": float((e < args.success_thresh).mean()),
            "per_seed_mean_cm": [float(r[0].mean() * 100) for r in runs],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
