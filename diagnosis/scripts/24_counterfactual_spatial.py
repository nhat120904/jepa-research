"""V3-spatial: does the FROZEN predictor give action-discriminative object
predictions when read with the SPATIAL probe? (the decisive test after Test 1b/2;
docs/plans/2026-06-18-residual-predictor-design.md, reframing 2026-06-19).

Test 1b/2 showed the encoder localizes the object to ~2cm and the frozen
predictor tracks it FACTUALLY to ~3cm/6 steps — but planning needs COUNTERFACTUAL
discrimination: from one state, do DIFFERENT actions give DIFFERENT predicted
object positions, tracking the true outcome spread? The original V3 (scripts/14)
said no (corr +0.035) — but via the POOLED probe (6.4cm noise). This re-runs the
exact V3 machinery (boundary anchors: similar state / different action / different
true outcome) decoding g(F(z,a')) with BOTH probes side by side.

  corr(spread_g, spread_true) high  -> predictor IS action-discriminative; the
       pooled probe was hiding it -> plan against the spatial-probe object cost.
  corr ~0 with the spatial probe too -> the predictor genuinely smears actions at
       the object level (true Boundary Blindness) -> fix the predictor.

Needs the predictor -> GPU.

    .venv/Scripts/python.exe scripts/24_counterfactual_spatial.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
        --pooled-probe checkpoints/object_probe_dino_wm_metaworld.pt \
        --spatial-probe checkpoints/spatial_object_probe_dino_wm_metaworld.pt
"""

from __future__ import annotations

import argparse
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
from models.probes import load_probe  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
from stratification import state_neighbours, boundary_score_per_transition  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def split_by_trajectory(records, val_frac, seed):
    tids = sorted({r["tid"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(tids)
    val_tids = set(tids[: max(1, int(len(tids) * val_frac))])
    return [r for r in records if r["tid"] in val_tids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pooled-probe", required=True)
    ap.add_argument("--spatial-probe", required=True)
    ap.add_argument("--n-anchors", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/counterfactual_spatial.csv")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    helpers = _load_runner_helpers()
    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, cfg["dataset"]["name"])
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    probes = {"pooled": load_probe(args.pooled_probe, device)[0],
              "spatial": load_probe(args.spatial_probe, device)[0]}
    regime_by_traj = read_regimes(cache_path)

    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=True)
        val_recs = split_by_trajectory(records, args.val_frac, args.seed)

        # Pool of candidate (state, action, true-outcome) for neighbour lookups.
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
        bscore = boundary_score_per_transition(d["a_t"], pool["a_t"][idx],
                                               torch.as_tensor(pool_out)[idx], mask)
        thr = np.nanquantile(bscore, cfg.get("boundary", {}).get("quantile", 0.75))
        bsel = np.where(np.isfinite(bscore) & (bscore > thr))[0][: args.n_anchors]

        spreads = {k: [] for k in probes}; spreads_true = []
        with torch.no_grad():
            for i in bsel.tolist():
                acts = pool["a_t"][idx[i]][mask[i]]
                m = acts.shape[0]
                if m < 2:
                    continue
                z_rep = d["z_t"][i: i + 1].expand(m, *d["z_t"].shape[1:]).to(device)
                prop = (d["proprio_t"][i: i + 1].expand(m, -1).to(device)
                        if d.get("proprio_t") is not None else None)
                pred = adapter.predict(z_rep, acts.to(device), proprio_t=prop)   # (m, latent)
                for name, pr in probes.items():
                    g = pr(pred)
                    spreads[name].append(
                        float((g - g.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()))
                spreads_true.append(float(torch.as_tensor(pool_out)[idx[i]][mask[i]].std()))

    st = np.asarray(spreads_true)
    print(f"\nV3 counterfactual object discrimination — {len(st)} boundary anchors")
    print(f"true outcome spread: median {100*np.median(st):.2f} cm\n")
    print(f"{'probe':8s} {'median spread_g (cm)':>20s} {'corr(spread_g, spread_true)':>28s}")
    rows = []
    for name in probes:
        sg = np.asarray(spreads[name])
        corr = float(np.corrcoef(sg, st)[0, 1]) if len(sg) > 2 else float("nan")
        print(f"{name:8s} {100*np.median(sg):20.2f} {corr:+28.3f}", flush=True)
        rows.append({"probe": name, "n_anchors": len(sg),
                     "median_spread_g_cm": 100*float(np.median(sg)),
                     "median_spread_true_cm": 100*float(np.median(st)),
                     "corr_spread_g_true": corr})
    import pandas as pd
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
