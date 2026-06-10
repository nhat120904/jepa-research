"""BB under the state-grounded (boundary-aware) latent metric — the metric-level
fix's gate. Identical pool/cells/standardisation as ``12``/``13``; the only change
is the space predictions are measured in:

    base       — plain L2 on the raw predicted latent (reference; equals the
                 frozen-baseline row of results/metaworld_boundary.csv)
    obj_only   — predictions mapped to g(ẑ): "spread of the predicted OBJECT
                 position across neighbour actions" (γ=0) — the maximally
                 interpretable variant, directly commensurate with S_true.
    blended    — φ(ẑ) = [ẑ/s_z ‖ β·g(ẑ)/s_g] (γ=1): the planner-ready metric.

Scales: s_z = median ‖z_{t+1}−z_t‖ and s_g = median object-position spread, both
from the shared seed-0 pool, recorded in the CSV. Success = BB drops in
pre_grasp/contact under obj_only/blended while the gate's base numbers stand.

    python scripts/15_eval_metric_boundary.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --probe checkpoints/object_probe_dino_wm_metaworld.pt
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
from models.probes import BoundaryAwareMetricAdapter, load_probe  # noqa: E402
from scripts._boundary_diagnostic import (  # noqa: E402
    _load_runner_helpers,
    accumulate_cell,
    compute_true_outcome,
    finalize_rows,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--skip-base", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    cfg = yaml.safe_load(open(args.config))
    helpers = _load_runner_helpers()

    dataset_name = cfg["dataset"]["name"]
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    min_n = cfg["eval"]["min_transitions_per_cell"]
    similarity_radius = cfg["hard_nn"]["similarity_radius"]
    pool_size = cfg["hard_nn"]["pool_size"]
    bcfg = cfg.get("boundary", {})
    max_neighbours = int(bcfg.get("max_neighbours", cfg["cra"]["K"]))
    boundary_quantile = float(bcfg.get("quantile", 0.75))
    n_resamples = cfg["bootstrap"]["n_resamples"]
    ci = cfg["bootstrap"]["ci"]

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, dataset_name)
    base = build_adapter(args.model, device=str(device)).eval()
    step = base.frames_per_step
    probe, pcfg = load_probe(args.probe, device)
    want_state = dataset_name == "metaworld"

    regime_by_traj = read_regimes(cache_path)
    all_rows = []
    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(
            cache, regime_by_traj, step, per_task=dataset_name == "metaworld")
        rng = np.random.default_rng(0)        # same pool as 12/13
        pool_idx = rng.choice(np.arange(len(records)),
                              size=min(pool_size, len(records)), replace=False)
        pool_records = [records[int(i)] for i in pool_idx]
        pool_data = helpers.materialize_records(cache, pool_records, step,
                                                want_proprio=False, want_state=want_state)
        pool_outcome = compute_true_outcome(
            dataset_name, pool_data["z_t"], pool_data["z_t1"],
            pool_data.get("state_t"), pool_data.get("state_t1"))
        pool_z, pool_a = pool_data["z_t"], pool_data["a_t"]
        tasks = sorted({r["task"] for r in records})

        # Metric scales from the pool (recorded in the CSV rows).
        s_z = float((pool_data["z_t1"] - pool_data["z_t"])
                    .reshape(pool_z.shape[0], -1).norm(dim=-1).median())
        with torch.no_grad():
            g_pool = probe(pool_z.to(device)).cpu()
        s_g = float((g_pool - g_pool.mean(0, keepdim=True)).norm(dim=-1).median())
        print(f"metric scales: s_z={s_z:.2f} s_g={s_g:.4f} beta={args.beta}", flush=True)

        variants = {}
        if not args.skip_base:
            variants[args.model] = base
        variants[f"{args.model}+metric_obj"] = BoundaryAwareMetricAdapter(
            base, probe, s_z=s_z, s_g=s_g, beta=args.beta, gamma=0.0, device=str(device))
        variants[f"{args.model}+metric_blend"] = BoundaryAwareMetricAdapter(
            base, probe, s_z=s_z, s_g=s_g, beta=args.beta, gamma=1.0, device=str(device))

        for name, adapter in variants.items():
            print(f"\n=== Metric-BB eval: {name} ===", flush=True)
            acc = {"s_true": [], "s_model": [], "boundary_score": [],
                   "traj_tag": [], "task": [], "regime": []}
            for regime_name in cfg["regimes"]:
                regime_id = REGIME_TO_ID[regime_name]
                for task in tasks:
                    cell_records = [r for r in records
                                    if r["regime"] == regime_id and r["task"] == task]
                    if len(cell_records) < min_n:
                        continue
                    cell_data = helpers.materialize_records(
                        cache, cell_records, step,
                        want_proprio=base.uses_proprio(), want_state=False)
                    out = accumulate_cell(
                        adapter, cell_data, pool_z, pool_a, pool_outcome, device=device,
                        similarity_radius=similarity_radius, max_neighbours=max_neighbours)
                    n = len(out["s_true"])
                    acc["s_true"].append(out["s_true"])
                    acc["s_model"].append(out["s_model"])
                    acc["boundary_score"].append(out["boundary_score"])
                    acc["traj_tag"].append(out["traj_tag"])
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
                boundary_quantile=boundary_quantile, n_resamples=n_resamples, ci=ci)
            for r in rows:
                r["s_z"], r["s_g"], r["beta"] = s_z, s_g, args.beta
                print(f"  {r['task']:22s} {r['regime']:20s} n={r['n_transitions']:5d} "
                      f"BB={r['bb']:+.3f} BB_boundary={r['bb_boundary']:+.3f} "
                      f"(n_b={r['n_boundary']})", flush=True)
            all_rows.extend(rows)

    import pandas as pd
    out_path = Path(cfg["output"]["csv"]).with_name(f"{dataset_name}_boundary_metric.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
