"""Train a temporal-straightening projector on the frozen MetaWorld latent cache.

Arms (`--gate`):
  none    global straightening -- straighten every triple (the ICML-2026 objective)
  switch  mode-gated straightening -- skip triples where the contact regime changes
          (the original HyS-JEPA proposal; kept as an ABLATION because the pre-gate
          refuted its premise: physics is *smoother* at mode switches, not kinkier)
  off     no curvature term (anti-collapse only) -- control for "does straightening
          do anything at all, or is any reshaping enough"

Encoder and predictor stay frozen; only the projector trains.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent / "diagnosis"
sys.path.insert(0, str(REPO))

from data.latent_cache import LatentCache  # noqa: E402
import torch.nn as nn  # noqa: E402

from models.heads.straightening_projector import (  # noqa: E402
    StraighteningProjector, curvature_loss, effective_rank, save_projector, vicreg_terms,
)


class PSpacePredictor(nn.Module):
    """Action-conditioned one-step predictor in projector space.

    The ICML-2026 method co-trains encoder AND predictor; the prediction loss is what
    stops the straightening objective from discarding information. Training a projector
    with curvature + VICReg alone collapses to ~3 effective dimensions
    (sweep, job 44207), so this term is not optional.
    """

    def __init__(self, d: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + action_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, p_t, a_t):
        return p_t + self.net(torch.cat([p_t, a_t], dim=-1))


class BoundaryHead(nn.Module):
    """q(P(z_t), P(z_{t+1})) -> did a contact-mode switch happen at t?

    The curvature loss pushes the representation toward smoothness. Pushed far enough it
    erases the very discontinuities that mark contact events -- the boundary between
    "about to touch" and "touching" gets blurred. This head is the counterweight: it
    demands the representation keep enough information to answer that question, so
    straightness is bought everywhere it is free but not at the cost of the event.

    Its held-out accuracy is also the honest read on whether the projector still carries
    contact-boundary information at all.
    """

    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, p_t, p_t1):
        return self.net(torch.cat([p_t, p_t1], dim=-1))


class PDecoder(nn.Module):
    """Decode P(z) back to the frozen pooled latent.

    Neither VICReg nor a same-space predictor stops the curvature objective from
    collapsing the projector to ~3 effective dimensions (jobs 44207, 44291 -- in the
    latter the prediction loss sat at ~0.003 precisely BECAUSE the representation was
    rank-3, so it was satisfied by collapse rather than opposing it). Reconstruction
    back to the 2D-dimensional pooled encoder feature is the one constraint that
    collapse cannot satisfy: a rank-3 code cannot reproduce the pooled latent.
    """

    def __init__(self, d: int, pooled_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, pooled_dim),
        )

    def forward(self, p):
        return self.net(p)


def load_windows(cache, regimes, tasks, win, max_traj_per_task, holdout_frac, seed):
    """Return (store, train_idx, val_idx).

    Windows are stride-1 and therefore overlap ~`win`-fold. Materialising each window
    separately multiplied memory by ~8x (17 GB for one task) and made the job
    unschedulable on a shared node, so trajectories are stored ONCE and windows are
    (traj_index, start) pairs into them.
    """
    rng = np.random.default_rng(seed)
    per_task = {}
    store, train_idx, val_idx = [], [], []
    with LatentCache(cache, "r") as c:
        for tid in c.trajectory_ids():
            task = str(tid).split("/")[0]
            if tasks and task not in tasks:
                continue
            per_task.setdefault(task, []).append(tid)

        for task, tids in per_task.items():
            tids = sorted(tids)
            if max_traj_per_task:
                tids = tids[:max_traj_per_task]
            rng.shuffle(tids)
            n_val = max(1, int(round(holdout_frac * len(tids))))
            split = {t: ("val" if i < n_val else "train") for i, t in enumerate(tids)}
            for tid in tids:
                traj = c.read_trajectory(tid)
                z = traj["z"].astype(np.float32)
                act = traj["action"].astype(np.float32)
                st = traj.get("state")
                obj = (np.asarray(st)[:, 4:7].astype(np.float32)
                       if st is not None and np.asarray(st).shape[-1] >= 7 else None)
                reg = np.asarray(regimes.get(str(tid), []), dtype=int)
                T = len(z)
                if T < win or len(reg) < T - 1 or len(act) < T - 1:
                    continue
                sw_full = (reg[:-1] != reg[1:]).astype(np.float32)   # (T-2,)
                k = len(store)
                store.append({"z": z, "a": act, "sw": sw_full, "obj": obj, "tid": tid})
                tgt = val_idx if split[tid] == "val" else train_idx
                for st in range(0, T - win + 1):
                    tgt.append((k, st))
    return store, train_idx, val_idx


def batches(store, index, win, bs, rng, shuffle=True):
    order = np.arange(len(index))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        sel = [index[j] for j in order[i:i + bs]]
        z = torch.from_numpy(np.stack([store[k]["z"][st:st + win] for k, st in sel]))
        a = torch.from_numpy(np.stack([store[k]["a"][st:st + win - 1] for k, st in sel]))
        sw = torch.from_numpy(np.stack([store[k]["sw"][st:st + win - 2] for k, st in sel]))
        yield z, a, sw


@torch.no_grad()
def evaluate(model, store, index, win, bs, device, latent_dim):
    rng = np.random.default_rng(0)
    cur_w, cur_s, feats = [], [], []
    for z, _a, sw in batches(store, index, win, bs, rng, shuffle=False):
        z = z.to(device)
        B, T = z.shape[:2]
        p = model(z.reshape(B * T, *z.shape[2:])).reshape(B, T, -1)
        _, c = curvature_loss(p)
        c = c.cpu().numpy()
        sw = sw.numpy()
        cur_w.append(c[sw == 0]); cur_s.append(c[sw == 1])
        feats.append(p.reshape(-1, p.shape[-1])[:64].cpu())
    w = np.concatenate([x for x in cur_w if len(x)]) if any(len(x) for x in cur_w) else np.array([np.nan])
    s = np.concatenate([x for x in cur_s if len(x)]) if any(len(x) for x in cur_s) else np.array([np.nan])
    F = torch.cat(feats)[:4096]
    return {"curv_within": float(np.nanmean(w)), "curv_switch": float(np.nanmean(s)),
            "curv_all": float(np.nanmean(np.concatenate([w, s]))),
            "eff_rank": effective_rank(F)}


@torch.no_grad()
def object_decodability(model, store, train_idx, val_idx, win, device, bs=64):
    """Ridge readout of object xyz from P(z), held out by episode.

    Effective rank is the wrong guard: reconstruction MSE over the pooled latent is
    dominated by high-variance directions, while the object occupies only a few patch
    tokens (~9% of the L2 residual, see models/heads/latent_metric.py). A projector can
    reconstruct to 1% and still have discarded the object. This measures the quantity
    planning actually needs.
    """
    def feats(index):
        P, Y = [], []
        seen = set()
        for k, st in index:
            if store[k]["obj"] is None:
                continue
            for t in range(st, st + win):
                if (k, t) in seen:
                    continue
                seen.add((k, t))
                P.append((k, t))
        out_p, out_y = [], []
        for i in range(0, len(P), bs):
            chunk = P[i:i + bs]
            z = torch.from_numpy(np.stack([store[k]["z"][t] for k, t in chunk])).to(device)
            out_p.append(model(z).cpu().numpy())
            out_y.append(np.stack([store[k]["obj"][t] for k, t in chunk]))
        if not out_p:
            return None, None
        return np.concatenate(out_p), np.concatenate(out_y)

    Xtr, Ytr = feats(train_idx)
    Xte, Yte = feats(val_idx)
    if Xtr is None or Xte is None:
        return {}
    Xtr1 = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    Xte1 = np.concatenate([Xte, np.ones((len(Xte), 1), np.float32)], 1)
    A = Xtr1.T @ Xtr1 + 1e-3 * len(Xtr1) * np.eye(Xtr1.shape[1], dtype=np.float64)
    W = np.linalg.solve(A, Xtr1.T @ Ytr)
    err = np.linalg.norm(Xte1 @ W - Yte, axis=-1)
    return {"obj_decode_mean_cm": float(err.mean() * 100),
            "obj_decode_median_cm": float(np.median(err) * 100),
            "obj_decode_within5cm": float((err < 0.05).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--regimes", default=None)
    ap.add_argument("--tasks", nargs="*", default=["mw-push", "mw-pick-place", "mw-reach"])
    ap.add_argument("--gate", choices=["none", "switch", "random", "off"], default="none",
                    help="none=straighten everything; switch=skip contact-mode switches "
                         "(the HyS-JEPA proposal); random=skip a MATCHED random fraction "
                         "(control that isolates contact semantics from the mere act of "
                         "dropping the highest-curvature tail); off=no curvature term")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=384)
    ap.add_argument("--lambda-curve", type=float, default=1.0)
    ap.add_argument("--lambda-pred", type=float, default=10.0)
    ap.add_argument("--lambda-recon", type=float, default=50.0)
    ap.add_argument("--lambda-head", type=float, default=1.0)
    ap.add_argument("--lambda-var", type=float, default=25.0)
    ap.add_argument("--lambda-cov", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-traj-per-task", type=int, default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    regimes = json.loads(Path(args.regimes or (args.cache + ".regimes.json")).read_text())

    store, train, val = load_windows(args.cache, regimes, args.tasks, args.window,
                                     args.max_traj_per_task, args.holdout_frac, args.seed)
    gb = sum(v["z"].nbytes for v in store) / 1e9
    print(f"gate={args.gate} train_windows={len(train)} val_windows={len(val)} "
          f"trajs={len(store)} store={gb:.1f}GB device={device}", flush=True)

    action_dim = int(store[0]["a"].shape[-1])
    model = StraighteningProjector(args.latent_dim, args.out_dim).to(device)
    pred = PSpacePredictor(args.out_dim, action_dim).to(device)
    dec = PDecoder(args.out_dim, 2 * args.latent_dim).to(device)
    head = BoundaryHead(args.out_dim).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(pred.parameters())
                            + list(dec.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    switch_rate = float(np.mean([v["sw"].mean() for v in store]))
    pos_w = torch.tensor([1.0, max(1.0, (1 - switch_rate) / max(switch_rate, 1e-6))],
                         device=device)
    print(f"  switch_rate={switch_rate:.4f} (gate=random drops this fraction at random) "
          f"lambda_head={args.lambda_head}", flush=True)
    print(f"  action_dim={action_dim} lambda_curve={args.lambda_curve} "
          f"lambda_pred={args.lambda_pred} lambda_var={args.lambda_var}", flush=True)
    rng = np.random.default_rng(args.seed)

    pre = evaluate(model, store, val, args.window, args.batch_size, device, args.latent_dim)
    print(f"  init  curv_all={pre['curv_all']:.4f} within={pre['curv_within']:.4f} "
          f"switch={pre['curv_switch']:.4f} rank={pre['eff_rank']:.1f}", flush=True)

    history = []
    for ep in range(args.epochs):
        model.train()
        agg = {"curve": 0.0, "var": 0.0, "cov": 0.0, "pred": 0.0, "recon": 0.0, "head": 0.0, "n": 0}
        for z, a, sw in batches(store, train, args.window, args.batch_size, rng):
            z, a, sw = z.to(device), a.to(device), sw.to(device)
            B, T = z.shape[:2]
            p = model(z.reshape(B * T, *z.shape[2:])).reshape(B, T, -1)

            if args.gate == "off":
                lc = torch.zeros((), device=device)
            else:
                if args.gate == "switch":
                    keep = 1.0 - sw
                elif args.gate == "random":
                    # drop the same FRACTION as the switch rate, but at random, so any
                    # gain that survives here is not about contact at all
                    keep = (torch.rand_like(sw) >= switch_rate).float()
                else:
                    keep = None
                lc, _ = curvature_loss(p, keep)

            # one-step prediction in P-space, normalised so it cannot be won by shrinking P
            p_hat = pred(p[:, :-1], a)
            tgt = p[:, 1:]
            lp = ((p_hat - tgt) ** 2).mean() / (tgt.detach().var() + 1e-6)

            # invertibility: P must retain enough to reproduce the frozen pooled latent
            with torch.no_grad():
                pooled = model._pool(z.reshape(B * T, *z.shape[2:]))
            rec = dec(p.reshape(B * T, -1))
            lr_ = ((rec - pooled) ** 2).mean() / (pooled.var() + 1e-6)

            # boundary preservation: keep the contact event decodable from P
            logits = head(p[:, :-2], p[:, 1:-1])                # (B, T-2, 2)
            lh = nn.functional.cross_entropy(
                logits.reshape(-1, 2), sw.reshape(-1).long(), weight=pos_w)

            lv, lcov = vicreg_terms(p.reshape(-1, p.shape[-1]))
            loss = (args.lambda_curve * lc + args.lambda_pred * lp
                    + args.lambda_recon * lr_ + args.lambda_head * lh
                    + args.lambda_var * lv + args.lambda_cov * lcov)
            opt.zero_grad(); loss.backward(); opt.step()
            agg["curve"] += float(lc); agg["var"] += float(lv); agg["cov"] += float(lcov)
            agg["pred"] += float(lp); agg["recon"] += float(lr_)
            agg["head"] += float(lh); agg["n"] += 1

        ev = evaluate(model, store, val, args.window, args.batch_size, device, args.latent_dim)
        history.append({"epoch": ep, **ev,
                        "train_curve": agg["curve"] / max(agg["n"], 1),
                        "train_pred": agg["pred"] / max(agg["n"], 1),
                        "train_recon": agg["recon"] / max(agg["n"], 1),
                        "train_head": agg["head"] / max(agg["n"], 1),
                        "train_var": agg["var"] / max(agg["n"], 1)})
        print(f"  ep{ep:02d} curve={agg['curve']/max(agg['n'],1):.4f} "
              f"pred={agg['pred']/max(agg['n'],1):.4f} "
              f"recon={agg['recon']/max(agg['n'],1):.4f} "
              f"head={agg['head']/max(agg['n'],1):.4f} "
              f"| val curv_all={ev['curv_all']:.4f} within={ev['curv_within']:.4f} "
              f"switch={ev['curv_switch']:.4f} rank={ev['eff_rank']:.1f}", flush=True)

    # held-out boundary AUC: can a contact-mode switch still be told from P(z_t), P(z_t1)?
    with torch.no_grad():
        sc, lb = [], []
        for z, _a, sw in batches(store, val, args.window, args.batch_size,
                                 np.random.default_rng(0), shuffle=False):
            z = z.to(device); B, T = z.shape[:2]
            p_ = model(z.reshape(B * T, *z.shape[2:])).reshape(B, T, -1)
            lg = head(p_[:, :-2], p_[:, 1:-1])
            sc.append(torch.softmax(lg, -1)[..., 1].reshape(-1).cpu().numpy())
            lb.append(sw.reshape(-1).numpy())
        sc = np.concatenate(sc); lb = np.concatenate(lb)
    if lb.sum() > 0 and lb.sum() < len(lb):
        order = np.argsort(sc)
        ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
        n1, n0 = lb.sum(), (1 - lb).sum()
        auc = float((ranks[lb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    else:
        auc = float("nan")
    print(f"  BOUNDARY AUC on held-out (0.5 = switch info destroyed): {auc:.3f}", flush=True)

    objdec = object_decodability(model, store, train, val, args.window, device)
    print(f"  OBJECT DECODABILITY from P(z): "
          f"median={objdec.get('obj_decode_median_cm', float('nan')):.2f}cm "
          f"within5cm={objdec.get('obj_decode_within5cm', float('nan')):.3f}", flush=True)

    meta = {"gate": args.gate, "tasks": args.tasks, "window": args.window,
            "object_decodability": objdec,
            "lambda_pred": args.lambda_pred, "lambda_recon": args.lambda_recon,
            "lambda_head": args.lambda_head, "switch_rate": switch_rate,
            "boundary_auc": auc,
            "cache": args.cache, "seed": args.seed, "epochs": args.epochs,
            "lambda_curve": args.lambda_curve, "lambda_var": args.lambda_var,
            "lambda_cov": args.lambda_cov, "init_eval": pre, "final_eval": history[-1],
            "history": history}
    save_projector(model, args.out, meta)
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
