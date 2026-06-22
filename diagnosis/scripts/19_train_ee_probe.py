"""Train the end-effector-position probe g_ee(z) -> ee xyz on cached Metaworld
latents — the APPROACH signal for the grounded-exploration planner (option B,
docs/plans/2026-06-18-grounded-exploration-design.md).

Identical machinery to scripts/14 (the object probe) but targets EE_SLICE =
state[0:3] instead of OBJECT_SLICE. The end-effector is the most salient moving
thing in the corner2 frame, so its held-out decode error (reported as V1) is
expected to be small — that is the precondition for using g_ee(z) - g_obj(z) as
an imagined ee->object distance the CEM search can minimise to create contact.

Everything frozen except this 0.3 M-param readout; trains on the existing cache.

    .venv/Scripts/python.exe scripts/19_train_ee_probe.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld --epochs 3
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
from stratification.metaworld_regimes import EE_SLICE  # noqa: E402


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
        print(f"ee probe params: {n_params/1e6:.2f}M  (target = state[{EE_SLICE}])", flush=True)

        def epoch_pass(recs, train):
            se, n = 0.0, 0
            for sel in iter_chunks(recs, args.chunk, rng if train else np.random.default_rng(1)):
                d = helpers.materialize_records(cache, sel, step,
                                                want_proprio=False, want_state=True)
                ee = d["state_t"][:, EE_SLICE].float()
                m = d["z_t"].shape[0]
                order = np.arange(m)
                if train:
                    rng.shuffle(order)
                for lo in range(0, m, args.batch_size):
                    idx = torch.as_tensor(order[lo: lo + args.batch_size], dtype=torch.long)
                    pred = probe(d["z_t"][idx].to(device))
                    loss = ((pred - ee[idx].to(device)) ** 2).mean()
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

        # ---- V1: held-out ee decode error (norm, metres) + per-dim sd floor ----
        errs_v1, ee_all = [], []
        for sel in iter_chunks(val_recs, args.chunk, np.random.default_rng(2)):
            d = helpers.materialize_records(cache, sel, step,
                                            want_proprio=False, want_state=True)
            ee1 = d["state_t1"][:, EE_SLICE].float().to(device)
            with torch.no_grad():
                for lo in range(0, d["z_t"].shape[0], args.batch_size):
                    s = slice(lo, lo + args.batch_size)
                    z_t1 = d["z_t1"][s].to(device)
                    errs_v1.append((probe(z_t1) - ee1[s]).norm(dim=-1).cpu().numpy())
            ee_all.append(d["state_t1"][:, EE_SLICE].float().numpy())
            del d
            gc.collect()
        v1 = np.concatenate(errs_v1)
        ee_sd = float(np.concatenate(ee_all).std(axis=0).mean())
        print(f"\nV1 ee probe error on real z_t1: median {np.median(v1):.4f} "
              f"(ee per-dim sd ≈ {ee_sd:.4f})", flush=True)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ee_probe_{args.model}.pt"
    torch.save({
        "model": args.model, "latent_dim": latent_dim, "out_dim": 3,
        "hidden": args.hidden, "state_dict": probe.state_dict(),
        "target_slice": [EE_SLICE.start, EE_SLICE.stop],
        "val_mse": va, "v1_median": float(np.median(v1)), "ee_sd": ee_sd,
    }, path)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
