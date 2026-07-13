"""Train the learned latent metric d_θ(z, z_goal) — Track B (label-free cost fix).

The oracle ladder localised the contact wall to the **L2-in-latent cost**: the plain
‖z − z_goal‖² has no minimum at task success because the object is a few patch tokens
in a ~98k-dim latent dominated by arm pose / background. ``scripts/22``+``gobj`` fix
this with a sim-state object label (Metaworld-only). This script learns a cost from
the **temporal order of frames alone** — no object GT — so the same recipe runs on
DROID. d_θ is an MRN-style quasimetric (``models/heads/latent_metric.py``) trained so
an in-trajectory pair ``(z_t, z_{t'})``, ``t<t'``, has ``d_θ(z_t, z_{t'}) ≈ (t'−t)``
(temporal distance), with cross-trajectory goals pushed large. The frozen encoder
already produced the cached latents; only d_θ trains.

Gates (printed; do not spend closed-loop budget unless they pass): held-out Spearman
of d_θ vs the true step gap > ~0.7, and monotone-to-goal Spearman (d_θ(z_t, z_T) vs
T−t) > ~0.7 — i.e. the goal frame is a usable minimum.

    python scripts/33_train_latent_metric.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --out checkpoints/latent_metric_dino_wm_metaworld.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path  # noqa: E402
from models.heads.latent_metric import LatentMetric  # noqa: E402


def load_traj_latents(cache, tids, max_frames_per_traj=None):
    """Return a list of (T, *frame) float tensors, one per trajectory (CPU)."""
    out = []
    for tid in tids:
        z = cache.read_trajectory(tid)["z"]                      # (T, *frame)
        if max_frames_per_traj is not None and len(z) > max_frames_per_traj:
            z = z[:max_frames_per_traj]
        if len(z) >= 2:
            out.append(torch.tensor(np.asarray(z)).float())
    return out


def sample_pairs(trajs, batch, K, rng, device):
    """Sample (z_i, z_j, gap) with i<=j<=i+K from random trajectories."""
    zi, zj, gaps = [], [], []
    n = len(trajs)
    for _ in range(batch):
        tr = trajs[rng.integers(n)]
        T = tr.shape[0]
        i = int(rng.integers(T - 1))
        g = int(rng.integers(1, min(K, T - 1 - i) + 1))         # gap in [1, K]
        zi.append(tr[i]); zj.append(tr[i + g]); gaps.append(float(g))
    return (torch.stack(zi).to(device), torch.stack(zj).to(device),
            torch.tensor(gaps, device=device))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (rank → Pearson), no scipy dependency."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


@torch.no_grad()
def evaluate(metric, trajs, K, device, rng, n_pairs=2000):
    """Gate metrics: (1) Spearman(d, gap) on random within-traj pairs;
    (2) monotone-to-goal Spearman of d(z_t, z_T) vs (T−t), averaged over trajs."""
    metric.eval()
    # (1) random-pair rank correlation
    preds, tgts = [], []
    done = 0
    while done < n_pairs:
        b = min(256, n_pairs - done)
        zi, zj, gaps = sample_pairs(trajs, b, K, rng, device)
        preds.append(metric(zi, zj).cpu().numpy()); tgts.append(gaps.cpu().numpy())
        done += b
    pair_sp = _spearman(np.concatenate(preds), np.concatenate(tgts))
    # (2) monotone-to-goal: each traj's final frame is the goal
    mono = []
    for tr in trajs:
        T = tr.shape[0]
        zt = tr.to(device)
        zg = tr[-1:].to(device).expand(T, *([-1] * (tr.dim() - 1)))
        d = metric(zt, zg).cpu().numpy()                         # d(z_t -> z_T)
        steps_to_go = np.arange(T)[::-1].astype(np.float64)      # T-1-t
        mono.append(_spearman(d, steps_to_go))
    mono_sp = float(np.nanmean(mono))
    return pair_sp, mono_sp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--K", type=int, default=12, help="max temporal gap (model-steps)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-neg", type=float, default=1.0,
                    help="cross-trajectory margin weight (push other goals far)")
    ap.add_argument("--neg-margin", type=float, default=4.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-trajs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/latent_metric.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, cfg["dataset"]["name"])
    rng = np.random.default_rng(args.seed)

    with LatentCache(cache_path, mode="r") as cache:
        tids = sorted(cache.trajectory_ids())
        rng.shuffle(tids)
        if args.max_trajs is not None:
            tids = tids[: args.max_trajs]
        n_val = max(1, int(len(tids) * args.val_frac))
        val_tids, train_tids = tids[:n_val], tids[n_val:]
        train = load_traj_latents(cache, train_tids)
        val = load_traj_latents(cache, val_tids)

    if not train:
        raise SystemExit("no usable training trajectories in cache")
    latent_dim = train[0].shape[-1]
    print(f"latent_dim={latent_dim} train_trajs={len(train)} val_trajs={len(val)} "
          f"K={args.K} device={device}", flush=True)

    metric = LatentMetric(latent_dim, embed_dim=args.embed_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(metric.parameters(), lr=args.lr)
    huber = torch.nn.SmoothL1Loss()
    far = float(args.K + args.neg_margin)

    metric.train()
    for step in range(1, args.steps + 1):
        zi, zj, gaps = sample_pairs(train, args.batch, args.K, rng, device)
        d_pos = metric(zi, zj)
        loss_reg = huber(d_pos, gaps)
        d_neg = metric(zi, torch.roll(zj, 1, dims=0))           # goal from another sample
        loss_neg = torch.relu(far - d_neg).mean()
        loss = loss_reg + args.lambda_neg * loss_neg
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps // 20) == 0 or step == 1:
            print(f"  step {step:5d}/{args.steps}  reg={loss_reg.item():.4f} "
                  f"neg={loss_neg.item():.4f}", flush=True)

    pair_sp, mono_sp = evaluate(metric, val, args.K, device, rng)
    print(f"\n=== latent metric gates (held-out) ===")
    print(f"  Spearman(d, gap)        = {pair_sp:.3f}   (target > 0.7)")
    print(f"  Spearman(d→goal, steps) = {mono_sp:.3f}   (target > 0.7)")
    gate = (pair_sp > 0.7) and (mono_sp > 0.7)
    print(f"  GATE: {'PASS' if gate else 'FAIL'} — "
          f"{'metric usable as a planning cost' if gate else 'do NOT spend closed-loop budget'}",
          flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": metric.state_dict(), "latent_dim": latent_dim,
                "embed_dim": args.embed_dim, "hidden": args.hidden, "K": args.K,
                "val_spearman": pair_sp, "val_mono_spearman": mono_sp,
                "model": args.model}, out)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
