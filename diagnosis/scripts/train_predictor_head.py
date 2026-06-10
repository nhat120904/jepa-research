"""C1 fix training — mixture-density + boundary head on cached Metaworld latents.

Design 2026-06-09 §4 / HANDOFF_BOUNDARY_FIX §4 step 1. The encoder and the base
predictor stay **frozen**; only the small head trains (the "force-free hero").
Trains the hero head (K components) and, in the same pass over the same batches,
a K=1 unimodal baseline head — the ablation that isolates "distributional" as the
active ingredient (identical trunk, data, schedule; only K differs).

    python scripts/train_predictor_head.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --K 3 --epochs 3

Outputs ``checkpoints/mdn_<model>_K<K>.pt`` and ``checkpoints/mdn_<model>_K1.pt``
(head state + config + boundary threshold + per-epoch metrics). Evaluate with
``scripts/13_eval_fix_boundary.py`` (BB before/after — the success criterion).

Boundary label: ``g_{t+1} = ‖obj_{t+1} − obj_t‖ > τ`` from the 39-dim Metaworld
state (object-displacement proxy — no MuJoCo contact GT; stated in every result),
with τ = the --boundary-quantile quantile of train-set displacements.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.heads import (  # noqa: E402
    MixtureDensityHead,
    total_loss,
    flatten_tokens,
    metaworld_boundary_state_slice,
    MW_STATE_SLICE_DIM,
)
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def object_displacement(state_t: torch.Tensor, state_t1: torch.Tensor) -> torch.Tensor:
    return (state_t1[:, OBJECT_SLICE] - state_t[:, OBJECT_SLICE]).norm(dim=-1)


def head_action(adapter, a_t: torch.Tensor) -> torch.Tensor:
    """Normalize the raw stacked action exactly as the base model does, flat."""
    B = a_t.shape[0]
    a = a_t.reshape(B, -1, adapter.action_dim())
    return adapter.normalize_action(a).reshape(B, -1)


def split_by_trajectory(records, val_frac: float, seed: int):
    tids = sorted({r["tid"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(tids)
    n_val = max(1, int(len(tids) * val_frac))
    val_tids = set(tids[:n_val])
    train = [r for r in records if r["tid"] not in val_tids]
    val = [r for r in records if r["tid"] in val_tids]
    return train, val


def iter_chunks(records, chunk: int, rng: np.random.Generator):
    """Yield shuffled record chunks, keeping each chunk's records grouped by
    trajectory for HDF5 read locality (shuffle order across and within chunks)."""
    order = np.arange(len(records))
    rng.shuffle(order)
    for lo in range(0, len(order), chunk):
        idx = order[lo: lo + chunk]
        sel = [records[int(i)] for i in idx]
        sel.sort(key=lambda r: r["tid"])  # read locality; batches reshuffled below
        yield sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2048, help="records materialized at once (RAM bound)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-b", type=float, default=0.1)
    ap.add_argument("--objective", choices=["nll", "wta", "boundary"], default="nll",
                    help="wta = winner-take-all/hard-EM; boundary = supervised "
                         "mode assignment by the boundary label (K=2, needs "
                         "Metaworld state; the K1 control trains with wta)")
    ap.add_argument("--use-state", action="store_true",
                    help="C1+D: condition the head on the boundary-relevant "
                         "Metaworld state slice (ee/gripper/object geometry)")
    ap.add_argument("--boundary-quantile", type=float, default=0.75)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--ctx-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-records", type=int, default=0,
                    help="debug/smoke: subsample this many transitions (0 = all)")
    ap.add_argument("--out-dir", default="checkpoints")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    dataset_name = cfg["dataset"]["name"]
    if dataset_name != "metaworld":
        print("[warn] boundary label needs Metaworld object state; "
              "other datasets train with NLL only (no boundary supervision).")
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, dataset_name)
    if not cache_path.exists():
        print(f"[error] cache missing: {cache_path}")
        return 1
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    want_state = dataset_name == "metaworld"

    regime_by_traj = read_regimes(cache_path)
    rng = np.random.default_rng(args.seed)

    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(
            cache, regime_by_traj, step, per_task=dataset_name == "metaworld")
        if args.max_records and len(records) > args.max_records:
            sub = np.random.default_rng(args.seed).choice(
                len(records), size=args.max_records, replace=False)
            records = [records[int(i)] for i in sub]
        train_recs, val_recs = split_by_trajectory(records, args.val_frac, args.seed)
        print(f"transitions: train={len(train_recs)} val={len(val_recs)} "
              f"(trajectory-split, val_frac={args.val_frac})", flush=True)

        # ---- pass 0: boundary threshold τ from train-set object displacement ----
        tau = float("nan")
        if want_state:
            disps = []
            for sel in iter_chunks(train_recs, args.chunk, np.random.default_rng(0)):
                d = helpers.materialize_records(cache, sel, step,
                                                want_proprio=False, want_state=True)
                disps.append(object_displacement(d["state_t"], d["state_t1"]).numpy())
                del d
                gc.collect()
            tau = float(np.quantile(np.concatenate(disps), args.boundary_quantile))
            pos = float((np.concatenate(disps) > tau).mean())
            print(f"boundary τ = {tau:.5f} (q={args.boundary_quantile}, "
                  f"{pos:.1%} positives)", flush=True)

        # ---- heads: hero (K) + unimodal baseline (K=1), same trunk inputs ----
        probe = helpers.materialize_records(cache, train_recs[:2], step,
                                            want_proprio=adapter.uses_proprio(),
                                            want_state=False)
        latent_dim = int(probe["z_t"].shape[-1])
        flat_action_dim = int(probe["a_t"].shape[-1])
        del probe

        if args.objective == "boundary" and not want_state:
            print("[error] objective='boundary' needs the Metaworld state label")
            return 1
        state_dim = MW_STATE_SLICE_DIM if (args.use_state and want_state) else 0

        def make_head(K):
            return MixtureDensityHead(latent_dim=latent_dim, action_dim=flat_action_dim,
                                      K=K, hidden=args.hidden, ctx_dim=args.ctx_dim,
                                      state_dim=state_dim).to(device).train()

        heads = {f"K{args.K}": make_head(args.K), "K1": make_head(1)}
        opts = {k: torch.optim.Adam(h.parameters(), lr=args.lr) for k, h in heads.items()}
        n_params = sum(p.numel() for p in heads[f"K{args.K}"].parameters())
        print(f"head params: {n_params/1e6:.2f}M (latent_dim={latent_dim}, "
              f"action_dim={flat_action_dim}, K={args.K})", flush=True)

        def batches(chunk_data, batch_size, shuffle=True):
            n = chunk_data["z_t"].shape[0]
            order = np.arange(n)
            if shuffle:
                rng.shuffle(order)
            for lo in range(0, n, batch_size):
                idx = torch.as_tensor(order[lo: lo + batch_size], dtype=torch.long)
                yield idx

        @torch.no_grad()
        def base_forward(d, idx):
            z_t = d["z_t"][idx].to(device)
            a_t = d["a_t"][idx].to(device)
            prop = d["proprio_t"][idx].to(device) if d.get("proprio_t") is not None else None
            base_pred = adapter.predict(z_t, a_t, proprio_t=prop)
            return (flatten_tokens(z_t, latent_dim),
                    flatten_tokens(base_pred, latent_dim),
                    head_action(adapter, a_t),
                    flatten_tokens(d["z_t1"][idx].to(device), latent_dim))

        def epoch_pass(recs, train: bool, tag: str):
            stats = {k: {"nll": 0.0, "bce": 0.0, "n": 0} for k in heads}
            ent_sum, ent_n = {k: 0.0 for k in heads}, {k: 0 for k in heads}
            for sel in iter_chunks(recs, args.chunk, rng if train else np.random.default_rng(1)):
                d = helpers.materialize_records(cache, sel, step,
                                                want_proprio=adapter.uses_proprio(),
                                                want_state=want_state)
                g_all = (object_displacement(d["state_t"], d["state_t1"]) > tau).float() \
                    if want_state else None
                s_all = (metaworld_boundary_state_slice(d["state_t"]).float()
                         if state_dim else None)
                for idx in batches(d, args.batch_size, shuffle=train):
                    zt, bt, act, zt1 = base_forward(d, idx)
                    g = g_all[idx].to(device) if g_all is not None else None
                    s = s_all[idx].to(device) if s_all is not None else None
                    for k, h in heads.items():
                        out = h(zt, bt, act, state=s)
                        # K=1 cannot take the supervised-assignment objective;
                        # its control trains with wta in a 'boundary' run.
                        obj = ("wta" if (args.objective == "boundary" and h.K < 2)
                               else args.objective)
                        losses = total_loss(out, zt1, g, lambda_b=args.lambda_b,
                                            objective=obj)
                        if train:
                            opts[k].zero_grad()
                            losses["loss"].backward()
                            opts[k].step()
                        b = len(idx)
                        stats[k]["nll"] += losses["nll"].item() * b
                        stats[k]["bce"] += losses["bce"].item() * b
                        stats[k]["n"] += b
                        with torch.no_grad():
                            p = torch.softmax(out["pi_logits"], dim=-1)
                            ent = -(p * (p + 1e-12).log()).sum(-1).mean().item()
                        ent_sum[k] += ent * b
                        ent_n[k] += b
                del d
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            out = {}
            for k, s in stats.items():
                out[k] = {"nll": s["nll"] / max(s["n"], 1),
                          "bce": s["bce"] / max(s["n"], 1),
                          "pi_entropy": ent_sum[k] / max(ent_n[k], 1)}
            msg = " | ".join(f"{k}: nll={v['nll']:.1f} bce={v['bce']:.3f} "
                             f"πH={v['pi_entropy']:.3f}" for k, v in out.items())
            print(f"  [{tag}] {msg}", flush=True)
            return out

        history = []
        for ep in range(args.epochs):
            t0 = time.time()
            print(f"epoch {ep + 1}/{args.epochs}", flush=True)
            for h in heads.values():
                h.train()
            tr = epoch_pass(train_recs, train=True, tag="train")
            for h in heads.values():
                h.eval()
            with torch.no_grad():
                va = epoch_pass(val_recs, train=False, tag="val")
            history.append({"epoch": ep + 1, "train": tr, "val": va,
                            "minutes": round((time.time() - t0) / 60, 1)})

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "model": args.model, "dataset": dataset_name, "latent_dim": latent_dim,
        "action_dim": flat_action_dim, "hidden": args.hidden, "ctx_dim": args.ctx_dim,
        "boundary_tau": tau, "boundary_quantile": args.boundary_quantile,
        "lambda_b": args.lambda_b, "lr": args.lr, "epochs": args.epochs,
        "objective": args.objective, "state_dim": state_dim,
        "seed": args.seed, "history": history,
    }
    suffix = "" if args.objective == "nll" else f"_{args.objective}"
    if state_dim:
        suffix += "_state"
    for k, h in heads.items():
        K = h.K
        path = out_dir / f"mdn_{args.model}_{k}{suffix}.pt"
        torch.save({**common, "K": K, "state_dict": h.state_dict()}, path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
