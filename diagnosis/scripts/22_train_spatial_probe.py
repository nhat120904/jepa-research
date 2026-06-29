"""Test 1b: train a SPATIAL object probe (CLS-attention over patch tokens) and
re-measure the representation precision, to separate "encoder limit" from
"pooled-probe limit" (docs/plans/2026-06-18-residual-predictor-design.md §5).

Test 1 found the pooled-token object probe localises the object to a held-out
median 6.4cm (only 27% within the 5cm push radius). But mean+max pooling discards
WHICH patch lit up. This probe lets a CLS query attend over the patch grid, so it
upper-bounds the spatial information the frozen latent carries. If it still cannot
beat the success radius, the ceiling is the ENCODER; if it localises far better,
the ceiling was the readout and the encoder is not the bottleneck.

Reports the same table as scripts/21 (median/mean/p90 cm + fraction within the
5cm / 7cm radii) on the identical held-out split, so the two are directly
comparable. Trains a ~5M-param probe on the cache; encoder + predictor frozen.

    .venv/Scripts/python.exe scripts/22_train_spatial_probe.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld --epochs 8
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
from data.latent_cache import ID_TO_REGIME  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.probes import SpatialObjectProbe  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402

PUSH_RADIUS, PICK_RADIUS = 0.05, 0.07


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
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--out", default="results/representation_precision_spatial.csv")
    # Phase-3 3b: mix off-policy random-action frames into training so the readout is
    # robust on the distribution the planner actually scores (see _offpolicy_frames).
    ap.add_argument("--offpolicy-frac", type=float, default=0.0,
                    help="fraction of each TRAIN batch drawn from off-policy frames "
                         "(0 = original expert-only probe; >0 saves a *_offpolicy.pt)")
    ap.add_argument("--op-episodes", type=int, default=8)
    ap.add_argument("--op-max-steps", type=int, default=60)
    ap.add_argument("--op-collect-every", type=int, default=4)
    ap.add_argument("--op-seed", type=int, default=20000)
    ap.add_argument("--op-tasks", nargs="+", default=["mw-push", "mw-pick-place"])
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

    # Off-policy buffer (Phase-3 3b): random-action sim frames, encoded + obj-labelled.
    op_train_z = op_train_obj = op_val_z = op_val_obj = None
    if args.offpolicy_frac > 0:
        from scripts._offpolicy_frames import collect_offpolicy_frames
        buf = collect_offpolicy_frames(adapter, device, tasks=args.op_tasks,
                                       episodes=args.op_episodes, max_steps=args.op_max_steps,
                                       collect_every=args.op_collect_every, seed=args.op_seed)
        zb, ob = buf["z"], buf["obj"]
        perm = np.random.default_rng(args.op_seed).permutation(zb.shape[0])
        nval = max(1, int(0.1 * zb.shape[0]))
        vi, ti = perm[:nval], perm[nval:]
        op_train_z, op_train_obj = zb[ti], ob[ti]
        op_val_z, op_val_obj = zb[vi], ob[vi]
        print(f"off-policy buffer: train={len(ti)} val={len(vi)} frac={args.offpolicy_frac}",
              flush=True)

    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=True)
        train_recs, val_recs = split_by_trajectory(records, args.val_frac, args.seed)
        print(f"transitions: train={len(train_recs)} val={len(val_recs)}", flush=True)
        d0 = helpers.materialize_records(cache, train_recs[:2], step, want_proprio=False, want_state=True)
        latent_dim = int(d0["z_t"].shape[-1]); del d0
        probe = SpatialObjectProbe(latent_dim=latent_dim, out_dim=3,
                                   n_layers=args.n_layers, hidden=args.hidden).to(device).train()
        opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
        print(f"spatial probe params: {sum(p.numel() for p in probe.parameters())/1e6:.2f}M", flush=True)

        def epoch_pass(recs, train):
            se = n = 0.0
            for sel in iter_chunks(recs, args.chunk, rng if train else np.random.default_rng(1)):
                d = helpers.materialize_records(cache, sel, step, want_proprio=False, want_state=True)
                obj = d["state_t"][:, OBJECT_SLICE].float()
                m = d["z_t"].shape[0]; order = np.arange(m)
                if train: rng.shuffle(order)
                for lo in range(0, m, args.batch_size):
                    idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
                    zb, ob = d["z_t"][idx], obj[idx]
                    if train and op_train_z is not None:
                        k = max(1, int(round(args.offpolicy_frac * len(idx))))
                        oi = rng.integers(0, op_train_z.shape[0], size=k)
                        zb = torch.cat([zb, op_train_z[oi]], 0)
                        ob = torch.cat([ob, op_train_obj[oi]], 0)
                    pred = probe(zb.to(device))
                    loss = ((pred - ob.to(device)) ** 2).mean()
                    if train:
                        opt.zero_grad(); loss.backward(); opt.step()
                    se += loss.item() * len(zb); n += len(zb)
                del d; gc.collect()
            return se / max(n, 1)

        for ep in range(args.epochs):
            tr = epoch_pass(train_recs, True)
            probe.eval()
            with torch.no_grad(): va = epoch_pass(val_recs, False)
            probe.train()
            print(f"epoch {ep+1}/{args.epochs}: train MSE={tr:.6f} val MSE={va:.6f}", flush=True)
        probe.eval()

        # Held-out object-decode error distribution (V1) on real z_t1, same metric
        # as scripts/21 so the spatial vs pooled probe are directly comparable.
        errs, regimes = [], []
        for sel in [val_recs[i:i+args.chunk] for i in range(0, len(val_recs), args.chunk)]:
            sel = sorted(sel, key=lambda r: r["tid"])
            d = helpers.materialize_records(cache, sel, step, want_proprio=False, want_state=True)
            obj_true = d["state_t1"][:, OBJECT_SLICE].float()
            with torch.no_grad():
                for lo in range(0, d["z_t1"].shape[0], args.batch_size):
                    s = slice(lo, lo + args.batch_size)
                    e = (probe(d["z_t1"][s].to(device)) - obj_true[s].to(device)).norm(dim=-1)
                    errs.append(e.cpu().numpy())
            regimes.extend(int(r["regime"]) for r in sel)
            del d; gc.collect()
    err = np.concatenate(errs); reg = np.asarray(regimes)

    def line(name, e):
        return {"group": name, "n": len(e), "median_cm": 100*float(np.median(e)),
                "mean_cm": 100*float(e.mean()), "p90_cm": 100*float(np.percentile(e, 90)),
                "frac_within_push_5cm": float((e < PUSH_RADIUS).mean()),
                "frac_within_pick_7cm": float((e < PICK_RADIUS).mean())}
    rows = [line("ALL", err)] + [line(ID_TO_REGIME.get(r, str(r)), err[reg == r])
                                 for r in sorted(set(reg.tolist()))]
    print(f"\nSPATIAL probe held-out object decode (cf. pooled probe 6.43cm med / 27% <5cm):")
    print(f"{'group':16s} {'med':>6s} {'<5cm':>6s} {'<7cm':>6s}")
    for r in rows:
        print(f"{r['group']:16s} {r['median_cm']:6.2f} {r['frac_within_push_5cm']*100:5.1f}% "
              f"{r['frac_within_pick_7cm']*100:5.1f}%", flush=True)

    # Off-policy held-out precision — the Phase-3 number that must improve vs the
    # expert-only probe (Test-1b expert was 92% <5cm but off-policy collapsed).
    if op_val_z is not None:
        with torch.no_grad():
            oe = [( probe(op_val_z[lo:lo+args.batch_size].to(device))
                    - op_val_obj[lo:lo+args.batch_size].to(device)).norm(dim=-1).cpu().numpy()
                  for lo in range(0, op_val_z.shape[0], args.batch_size)]
        oe = np.concatenate(oe)
        print(f"OFF-POLICY held-out: med={100*np.median(oe):.2f}cm "
              f"<5cm={(oe < PUSH_RADIUS).mean()*100:.1f}% "
              f"<7cm={(oe < PICK_RADIUS).mean()*100:.1f}%", flush=True)

    out_dir = ROOT / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_offpolicy" if args.offpolicy_frac > 0 else ""
    path = out_dir / f"spatial_object_probe_{args.model}{suffix}.pt"
    torch.save({"model": args.model, "probe_type": "spatial", "latent_dim": latent_dim,
                "out_dim": 3, "hidden": args.hidden, "n_layers": args.n_layers, "n_heads": 6,
                "state_dict": probe.state_dict(), "val_mse": va,
                "v1_median": float(np.median(err))}, path)
    import pandas as pd
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {path} and {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
