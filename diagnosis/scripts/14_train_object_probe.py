"""Train the object-position probe g(z) → obj xyz on cached Metaworld latents,
then run the three validation measurements the metric-level fix depends on:

  V1  held-out probe error          — is the boundary info IN the latent at all?
  V2  probe-through-prediction      — does the *predictor's output* carry it?
      ‖g(F(z_t, a_t)) − obj_{t+1}‖  (vs V1 on real z_{t+1} as the floor)
  V3  counterfactual sensitivity    — does g(F(z_t, a)) MOVE when the action
      changes? (median spread of the object readout across hard_nn neighbour
      actions, on boundary-score-selected anchors — the precheck for BB-under-φ)

If V1 fails (object not decodable) the latent genuinely lacks the boundary →
encoder retraining territory; if V1 passes but V2/V3 fail, the predictor drops
the object channel → the metric fix alone cannot work and we report that.

    python scripts/14_train_object_probe.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --epochs 3
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
from models.adapters import build_adapter  # noqa: E402
from models.probes import ObjectProbe  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
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
    ap.add_argument("--out-dir", default="checkpoints")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model,
                                   cfg["dataset"]["name"])
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
        del probe_d

        probe = ObjectProbe(latent_dim=latent_dim, out_dim=3, hidden=args.hidden
                            ).to(device).train()
        opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
        n_params = sum(p.numel() for p in probe.parameters())
        print(f"probe params: {n_params/1e6:.2f}M", flush=True)

        def epoch_pass(recs, train):
            se, n = 0.0, 0
            for sel in iter_chunks(recs, args.chunk, rng if train else np.random.default_rng(1)):
                d = helpers.materialize_records(cache, sel, step,
                                                want_proprio=False, want_state=True)
                obj = d["state_t"][:, OBJECT_SLICE].float()
                m = d["z_t"].shape[0]
                order = np.arange(m)
                if train:
                    rng.shuffle(order)
                for lo in range(0, m, args.batch_size):
                    idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
                    pred = probe(d["z_t"][idx].to(device))
                    loss = ((pred - obj[idx].to(device)) ** 2).mean()
                    if train:
                        opt.zero_grad(); loss.backward(); opt.step()
                    se += loss.item() * len(idx)
                    n += len(idx)
                del d
                gc.collect()
            return se / max(n, 1)

        for ep in range(args.epochs):
            tr = epoch_pass(train_recs, True)
            probe.eval()
            with torch.no_grad():
                va = epoch_pass(val_recs, False)
            probe.train()
            print(f"epoch {ep+1}/{args.epochs}: train MSE={tr:.6f} val MSE={va:.6f}", flush=True)
        probe.eval()

        # ---- V1 / V2: held-out probe error on z_t1 vs through the prediction ----
        errs_v1, errs_v2, obj_all = [], [], []
        for sel in iter_chunks(val_recs, args.chunk, np.random.default_rng(2)):
            d = helpers.materialize_records(cache, sel, step,
                                            want_proprio=adapter.uses_proprio(),
                                            want_state=True)
            obj1 = d["state_t1"][:, OBJECT_SLICE].float().to(device)
            with torch.no_grad():
                for lo in range(0, d["z_t"].shape[0], args.batch_size):
                    s = slice(lo, lo + args.batch_size)
                    z_t1 = d["z_t1"][s].to(device)
                    errs_v1.append((probe(z_t1) - obj1[s]).norm(dim=-1).cpu().numpy())
                    prop = (d["proprio_t"][s].to(device)
                            if d.get("proprio_t") is not None else None)
                    pred = adapter.predict(d["z_t"][s].to(device),
                                           d["a_t"][s].to(device), proprio_t=prop)
                    errs_v2.append((probe(pred) - obj1[s]).norm(dim=-1).cpu().numpy())
            obj_all.append(d["state_t1"][:, OBJECT_SLICE].float().numpy())
            del d
            gc.collect()
        v1 = np.concatenate(errs_v1); v2 = np.concatenate(errs_v2)
        obj_sd = float(np.concatenate(obj_all).std(axis=0).mean())
        print(f"\nV1 probe error on real z_t1:   median {np.median(v1):.4f} "
              f"(object per-dim sd ≈ {obj_sd:.4f})", flush=True)
        print(f"V2 probe error through F(z,a): median {np.median(v2):.4f}", flush=True)

        # ---- V3: counterfactual sensitivity of g(F(z,a')) on boundary anchors ----
        pool_idx = np.random.default_rng(0).choice(
            np.arange(len(records)), size=min(cfg["hard_nn"]["pool_size"], len(records)),
            replace=False)
        pool = helpers.materialize_records(cache, [records[int(i)] for i in pool_idx],
                                           step, want_proprio=False, want_state=True)
        pool_out = (pool["state_t1"][:, OBJECT_SLICE]
                    - pool["state_t"][:, OBJECT_SLICE]).norm(dim=-1).numpy()

        sub = [val_recs[int(i)] for i in np.random.default_rng(3).choice(
            len(val_recs), size=min(512, len(val_recs)), replace=False)]
        d = helpers.materialize_records(cache, sub, step,
                                        want_proprio=adapter.uses_proprio(), want_state=True)
        idx, mask, _ = state_neighbours(d["z_t"], cfg["hard_nn"]["similarity_radius"],
                                        max_neighbours=16, pool_z=pool["z_t"])
        bscore = boundary_score_per_transition(
            d["a_t"], pool["a_t"][idx], torch.as_tensor(pool_out)[idx], mask)
        thr = np.nanquantile(bscore, cfg.get("boundary", {}).get("quantile", 0.75))
        bsel = np.where(np.isfinite(bscore) & (bscore > thr))[0][:128]

        spreads_g, spreads_true = [], []
        with torch.no_grad():
            for i in bsel.tolist():
                acts = pool["a_t"][idx[i]][mask[i]]
                m = acts.shape[0]
                if m < 2:
                    continue
                z_rep = d["z_t"][i: i + 1].expand(m, *d["z_t"].shape[1:]).to(device)
                prop = (d["proprio_t"][i: i + 1].expand(m, -1).to(device)
                        if d.get("proprio_t") is not None else None)
                g = probe(adapter.predict(z_rep, acts.to(device), proprio_t=prop))
                spreads_g.append(float((g - g.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()))
                t_out = torch.as_tensor(pool_out)[idx[i]][mask[i]]
                spreads_true.append(float(t_out.std()))
        sg, st = np.asarray(spreads_g), np.asarray(spreads_true)
        corr = float(np.corrcoef(sg, st)[0, 1]) if len(sg) > 2 else float("nan")
        print(f"V3 boundary anchors n={len(sg)}: median spread of g(F(z,a')) = "
              f"{np.median(sg):.4f}; median true outcome spread = {np.median(st):.4f}; "
              f"corr(spread_g, spread_true) = {corr:+.3f}", flush=True)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"object_probe_{args.model}.pt"
    torch.save({
        "model": args.model, "latent_dim": latent_dim, "out_dim": 3,
        "hidden": args.hidden, "state_dict": probe.state_dict(),
        "val_mse": va, "v1_median": float(np.median(v1)),
        "v2_median": float(np.median(v2)), "object_sd": obj_sd,
        "v3_spread_g_median": float(np.median(sg)) if len(sg) else float("nan"),
        "v3_spread_true_median": float(np.median(st)) if len(st) else float("nan"),
        "v3_corr": corr,
    }, path)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
