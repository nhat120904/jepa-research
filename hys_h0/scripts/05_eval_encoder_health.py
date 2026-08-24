"""Proper health evaluation of a trained encoder-LoRA straightener.

The in-training `health()` in scripts/04 caps at 48 windows and fits a ridge on ~34
samples, which is far too few: object decode for the same arm swung 2.72 / 4.69 / 10.78 cm
across seeds. That noise swamps the question this stage exists to answer --

    does a trainable encoder straighten WHILE KEEPING object precision,
    or does it buy straightness with the object the way the projector did?

This re-evaluates saved checkpoints on a proper held-out split with trajectory-clustered
CIs. No retraining.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util as _ilu
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent / "diagnosis"
sys.path.insert(0, str(REPO))

from models.adapters import build_adapter  # noqa: E402
from models.heads.lora_encoder import load_encoder_lora  # noqa: E402
from models.heads.straightening_projector import curvature_loss, effective_rank  # noqa: E402


def _s38():
    spec = _ilu.spec_from_file_location("_s38", REPO / "scripts" / "38_train_encoder_lora.py")
    m = _ilu.module_from_spec(spec); sys.modules["_s38"] = m
    spec.loader.exec_module(m)
    return m


def pool(z):
    D = z.shape[-1]
    t = z.reshape(z.shape[0], -1, D)
    return torch.cat([t.mean(1), t.amax(1)], dim=-1)


def cluster_ci(v, g, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    v, g = np.asarray(v, float), np.asarray(g)
    u = np.unique(g); by = {k: v[g == k] for k in u}
    b = [np.concatenate([by[k] for k in rng.choice(u, len(u), replace=True)]).mean()
         for _ in range(n)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


@torch.no_grad()
def evaluate(adapter, s38, trajs, val_ids, window):
    P, Y, G, C, CG = [], [], [], [], []
    for ti in val_ids:
        tr = trajs[ti]
        T = tr["frames"].shape[0]
        for s in range(0, T - window + 1, max(1, window // 2)):
            z = s38.encode_grad(adapter, tr["frames"][s:s + window].float(),
                                tr["prop"][s:s + window])
            p = pool(z)
            P.append(p.cpu()); Y.append(tr["obj"][s:s + window])
            G.extend([ti] * window)
            c, _ = curvature_loss(p.unsqueeze(0))
            C.append(float(c)); CG.append(ti)
    X = torch.cat(P).numpy(); Yv = torch.cat(Y).numpy(); G = np.asarray(G)
    return X, Yv, G, np.asarray(C), np.asarray(CG)


def object_readout(Xtr, Ytr, Xte, Yte):
    X1 = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    T1 = np.concatenate([Xte, np.ones((len(Xte), 1), np.float32)], 1)
    A = X1.T @ X1 + 1e-3 * len(X1) * np.eye(X1.shape[1])
    W = np.linalg.solve(A, X1.T @ Ytr)
    return np.linalg.norm(T1 @ W - Yte, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="dino_wm_metaworld")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="*", default=["mw-push"])
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--max-trajs-per-task", type=int, default=60)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["dataset"]["tasks"] = {"easy": args.tasks, "medium": [], "hard": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    s38 = _s38()
    trajs = s38.load_expert_trajs(cfg, args.max_trajs_per_task)

    paths = []
    for c in args.ckpts:
        paths.extend(sorted(glob.glob(c)))
    rows = []

    for path in [None] + paths:            # None == the frozen encoder reference
        tag = "FROZEN" if path is None else Path(path).stem
        # A fresh adapter per checkpoint. Reusing one and toggling `enabled` silently
        # left every arm after the first reading the FROZEN encoder (job 44932: all
        # arms returned their seed's frozen numbers exactly), because load_encoder_lora
        # re-injects onto already-wrapped layers.
        adapter = build_adapter(args.model, device=str(device)).eval()
        if path is not None:
            load_encoder_lora(adapter, path, device)
            mp = Path(str(path) + ".json")
            meta = json.loads(mp.read_text()) if mp.exists() else {}
            seed = meta.get("seed", 0)
        else:
            seed = 0
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(trajs))
        n_val = max(2, int(round(args.holdout_frac * len(trajs))))
        val_ids = order[:n_val].tolist()
        fit_ids = order[n_val:n_val + max(4, n_val)].tolist()

        Xf, Yf, Gf, _, _ = evaluate(adapter, s38, trajs, fit_ids, args.window)
        Xv, Yv, Gv, C, CG = evaluate(adapter, s38, trajs, val_ids, args.window)
        err = object_readout(Xf, Yf, Xv, Yv)

        cm, clo, chi = cluster_ci(C, CG)
        em, elo, ehi = cluster_ci(err * 100, Gv)
        rows.append({"arm": tag, "curv": cm, "curv_ci": [clo, chi],
                     "obj_cm": em, "obj_ci": [elo, ehi],
                     "obj_median_cm": float(np.median(err) * 100),
                     "eff_rank": effective_rank(torch.from_numpy(Xv).float()),
                     "n_val_traj": len(val_ids), "n_windows": len(C)})
        print(f"{tag:28s} curv={cm:.4f}[{clo:.4f},{chi:.4f}]  "
              f"obj={em:6.2f}cm[{elo:.2f},{ehi:.2f}] med={np.median(err)*100:6.2f}  "
              f"rank={rows[-1]['eff_rank']:6.1f}  n={len(C)}", flush=True)
        del adapter
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
