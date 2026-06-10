"""Train the action-conditioned object-dynamics head h(z_t, a) → Δobject, then
(a) re-run the V3 counterfactual check on it, and (b) if it shows life, run the
full per-(task, regime) Boundary Blindness pass with the h channel as the model's
prediction — the grounded-dynamics fix.

The chain that motivates this (all measured, 2026-06-10):
  V1 ✓ object decodable from latent   V2 ✓ predictor propagates it factually
  V3 ✗ predictor's counterfactual object response is noise (corr ≈ 0.03)
h is supervised *directly* on (z_t, a_t) → obj_{t+1}−obj_t from the cache; the
cross-sample neighbourhood variation (similar state, different action, different
outcome) is exactly the signal the 98k-dim L2 training objective buried.

    python scripts/17_train_object_dynamics.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --epochs 3

Outputs: checkpoints/object_dynamics_<model>.pt and (if the counterfactual gate
passes) results/{dataset}_boundary_dynamics.csv.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from data.latent_cache import REGIME_TO_ID  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.probes import ObjectDynamicsHead, ObjectDynamicsAdapter  # noqa: E402
from scripts._boundary_diagnostic import (  # noqa: E402
    _load_runner_helpers,
    accumulate_cell,
    compute_true_outcome,
    finalize_rows,
)
from stratification import state_neighbours, boundary_score_per_transition  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def split_by_trajectory(records, val_frac, seed):
    tids = sorted({r["tid"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(tids)
    val_tids = set(tids[: max(1, int(len(tids) * val_frac))])
    return ([r for r in records if r["tid"] not in val_tids],
            [r for r in records if r["tid"] in val_tids])


def iter_chunks(records, chunk, rng):
    order = np.arange(len(records))
    rng.shuffle(order)
    for lo in range(0, len(order), chunk):
        sel = [records[int(i)] for i in order[lo: lo + chunk]]
        sel.sort(key=lambda r: r["tid"])
        yield sel


def norm_action(adapter, a_t, device):
    B = a_t.shape[0]
    a = a_t.to(device).float().reshape(B, -1, adapter.action_dim())
    return adapter.normalize_action(a).reshape(B, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", choices=["object", "state"], default="object",
                    help="supervision label: 'object' = Metaworld object slice "
                         "(the boundary proof); 'state' = the cache's full raw "
                         "state diff (DROID transfer: state==proprio, 7-dim "
                         "pose+gripper — proxy label, no object GT)")
    ap.add_argument("--min-corr", type=float, default=0.2,
                    help="counterfactual gate: corr(spread_h, spread_true) must "
                         "exceed this before the full BB pass is run")
    ap.add_argument("--skip-bb", action="store_true")
    ap.add_argument("--out-dir", default="checkpoints")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    dataset_name = cfg["dataset"]["name"]
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, dataset_name)
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    regime_by_traj = read_regimes(cache_path)
    rng = np.random.default_rng(args.seed)

    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=True)
        train_recs, val_recs = split_by_trajectory(records, args.val_frac, args.seed)
        print(f"transitions: train={len(train_recs)} val={len(val_recs)}", flush=True)

        probe_d = helpers.materialize_records(cache, train_recs[:2], step,
                                              want_proprio=False, want_state=True)
        latent_dim = int(probe_d["z_t"].shape[-1])
        flat_action_dim = int(probe_d["a_t"].shape[-1])
        out_dim = 3 if args.label == "object" else int(probe_d["state_t"].shape[-1])
        del probe_d

        head = ObjectDynamicsHead(latent_dim=latent_dim, action_dim=flat_action_dim,
                                  out_dim=out_dim, hidden=args.hidden).to(device).train()
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        print(f"h params: {sum(p.numel() for p in head.parameters())/1e6:.2f}M", flush=True)

        def label_fn(d):
            if args.label == "object":
                return (d["state_t1"][:, OBJECT_SLICE] - d["state_t"][:, OBJECT_SLICE]).float()
            return (d["state_t1"] - d["state_t"]).float()    # full state diff (DROID: ==Δproprio)

        def epoch_pass(recs, train):
            se, n = 0.0, 0
            for sel in iter_chunks(recs, args.chunk, rng if train else np.random.default_rng(1)):
                d = helpers.materialize_records(cache, sel, step,
                                                want_proprio=False, want_state=True)
                dobj = label_fn(d)
                m = d["z_t"].shape[0]
                order = np.arange(m)
                if train:
                    rng.shuffle(order)
                for lo in range(0, m, args.batch_size):
                    idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
                    pred = head(d["z_t"][idx].to(device),
                                norm_action(adapter, d["a_t"][idx], device))
                    loss = ((pred - dobj[idx].to(device)) ** 2).mean()
                    if train:
                        opt.zero_grad(); loss.backward(); opt.step()
                    se += loss.item() * len(idx)
                    n += len(idx)
                del d
                gc.collect()
            return se / max(n, 1)

        for ep in range(args.epochs):
            tr = epoch_pass(train_recs, True)
            head.eval()
            with torch.no_grad():
                va = epoch_pass(val_recs, False)
            head.train()
            print(f"epoch {ep+1}/{args.epochs}: train MSE={tr:.6f} val MSE={va:.6f}", flush=True)
        head.eval()

        # ---- counterfactual gate (V3 rerun, now on h) -------------------------
        pool_idx = np.random.default_rng(0).choice(
            np.arange(len(records)), size=min(cfg["hard_nn"]["pool_size"], len(records)),
            replace=False)
        pool = helpers.materialize_records(cache, [records[int(i)] for i in pool_idx],
                                           step, want_proprio=False, want_state=True)
        # True outcome for the gate: object displacement on Metaworld; the
        # dataset's standard proxy (‖Δz‖) elsewhere — same as the BB runner.
        pool_out = compute_true_outcome(
            dataset_name, pool["z_t"], pool["z_t1"],
            pool.get("state_t"), pool.get("state_t1"))

        sub = [val_recs[int(i)] for i in np.random.default_rng(3).choice(
            len(val_recs), size=min(512, len(val_recs)), replace=False)]
        d = helpers.materialize_records(cache, sub, step, want_proprio=False, want_state=True)
        idx, mask, _ = state_neighbours(d["z_t"], cfg["hard_nn"]["similarity_radius"],
                                        max_neighbours=16, pool_z=pool["z_t"])
        bscore = boundary_score_per_transition(
            d["a_t"], pool["a_t"][idx], torch.as_tensor(pool_out)[idx], mask)
        thr = np.nanquantile(bscore, cfg.get("boundary", {}).get("quantile", 0.75))
        bsel = np.where(np.isfinite(bscore) & (bscore > thr))[0][:128]

        spreads_h, spreads_true = [], []
        with torch.no_grad():
            for i in bsel.tolist():
                acts = pool["a_t"][idx[i]][mask[i]]
                m = acts.shape[0]
                if m < 2:
                    continue
                z_rep = d["z_t"][i: i + 1].expand(m, *d["z_t"].shape[1:]).to(device)
                hv = head(z_rep, norm_action(adapter, acts, device))
                spreads_h.append(float((hv - hv.mean(0, keepdim=True))
                                       .norm(dim=-1).pow(2).mean().sqrt()))
                spreads_true.append(float(torch.as_tensor(pool_out)[idx[i]][mask[i]].std()))
        sh, st = np.asarray(spreads_h), np.asarray(spreads_true)
        corr = float(np.corrcoef(sh, st)[0, 1]) if len(sh) > 2 else float("nan")
        print(f"counterfactual gate: n={len(sh)} median spread_h={np.median(sh):.4f} "
              f"median spread_true={np.median(st):.4f} corr={corr:+.3f} "
              f"(min required {args.min_corr})", flush=True)

        out_dir = ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"object_dynamics_{args.model}.pt"
        torch.save({"model": args.model, "latent_dim": latent_dim,
                    "action_dim": flat_action_dim, "out_dim": out_dim,
                    "hidden": args.hidden, "label": args.label,
                    "state_dict": head.state_dict(), "val_mse": va,
                    "cf_corr": corr, "cf_spread_h": float(np.median(sh)),
                    "cf_spread_true": float(np.median(st))}, ckpt_path)
        print(f"wrote {ckpt_path}", flush=True)

        if args.skip_bb or not (np.isfinite(corr) and corr >= args.min_corr):
            if not args.skip_bb:
                print(f"GATE FAILED (corr {corr:+.3f} < {args.min_corr}) — "
                      f"skipping the BB pass; h did not learn the action dependence.")
            return 0

        # ---- full BB pass with the h channel as the prediction ----------------
        dyn = ObjectDynamicsAdapter(adapter, head, device=str(device))
        pool_outcome = compute_true_outcome(
            dataset_name, pool["z_t"], pool["z_t1"], pool.get("state_t"), pool.get("state_t1"))
        tasks = sorted({r["task"] for r in records})
        min_n = cfg["eval"]["min_transitions_per_cell"]
        name = f"{args.model}+obj_dynamics"
        print(f"\n=== Boundary eval: {name} ===", flush=True)
        acc = {"s_true": [], "s_model": [], "boundary_score": [],
               "traj_tag": [], "task": [], "regime": []}
        for regime_name in cfg["regimes"]:
            rid = REGIME_TO_ID[regime_name]
            for task in tasks:
                cell_records = [r for r in records
                                if r["regime"] == rid and r["task"] == task]
                if len(cell_records) < min_n:
                    continue
                cell_data = helpers.materialize_records(
                    cache, cell_records, step, want_proprio=False, want_state=False)
                out = accumulate_cell(
                    dyn, cell_data, pool["z_t"], pool["a_t"], pool_outcome, device=device,
                    similarity_radius=cfg["hard_nn"]["similarity_radius"],
                    max_neighbours=int(cfg.get("boundary", {}).get("max_neighbours", 16)))
                n = len(out["s_true"])
                for k_src, k_dst in (("s_true", "s_true"), ("s_model", "s_model"),
                                     ("boundary_score", "boundary_score"),
                                     ("traj_tag", "traj_tag")):
                    acc[k_dst].append(out[k_src])
                acc["task"].append(np.array([task] * n))
                acc["regime"].append(np.array([regime_name] * n))
                del cell_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        rows, _ = finalize_rows(
            dataset_name, name,
            s_true=np.concatenate(acc["s_true"]),
            s_model=np.concatenate(acc["s_model"]),
            boundary_score=np.concatenate(acc["boundary_score"]),
            task=np.concatenate(acc["task"]),
            regime=np.concatenate(acc["regime"]),
            traj_tag=np.concatenate(acc["traj_tag"]),
            boundary_quantile=float(cfg.get("boundary", {}).get("quantile", 0.75)),
            n_resamples=cfg["bootstrap"]["n_resamples"], ci=cfg["bootstrap"]["ci"])
        for r in rows:
            print(f"  {r['task']:22s} {r['regime']:20s} n={r['n_transitions']:5d} "
                  f"BB={r['bb']:+.3f} BB_boundary={r['bb_boundary']:+.3f} "
                  f"(n_b={r['n_boundary']})", flush=True)

        import pandas as pd
        out_path = Path(cfg["output"]["csv"]).with_name(f"{dataset_name}_boundary_dynamics.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\nWrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
