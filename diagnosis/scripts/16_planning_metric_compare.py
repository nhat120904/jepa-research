"""Planning probe A/B: CEM with the plain L2 cost vs the state-grounded
(boundary-aware) cost, paired per transition — the planning-improvement leg of
the metric-level fix.

For each effectful transition (per regime, horizon = planning.goal_H) we plan
with BOTH costs using the *same* CEM noise seed, so the only difference between
arms is the objective; Action Error (vs the expert actions, the paper's DROID
metric) is then compared pairwise. Arms:

    l2      — upstream objective: MSE(last latent, goal)            [baseline]
    aware   — MSE(z)/s_z² + β²·MSE(g(z_H))/s_g²  (probe readout)    [metric only]
    obj     — β²·MSE(g(z_H))/s_g² only (γ=0 ablation)
    hdyn    — MSE(z)/s_z² + β²·‖(g(z_0)+Σ_t h(ẑ_t,a_t)) − g(goal)‖²/s_g²
              (requires --dyn-head; the grounded-dynamics fix — h is the channel
              whose counterfactual response tracks the true outcome)

Output: results/{dataset}_planning_metric.csv (per regime×arm means + CIs and
the paired per-transition delta), + *_pertrans.npz.

    python scripts/16_planning_metric_compare.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --probe checkpoints/object_probe_dino_wm_metaworld.pt \
        --cem-num-samples 64 --max-per-regime 30
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from data.latent_cache import REGIME_TO_ID  # noqa: E402
from metrics import effect_mask, bootstrap_ci  # noqa: E402
from metrics.action_score import action_error  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.probes import (  # noqa: E402
    load_probe,
    load_dynamics_head,
    boundary_aware_cost,
    grounded_dynamics_cost,
)
from planning.cem_planner import cem_plan  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--dyn-head", default=None,
                    help="object-dynamics checkpoint; adds the 'hdyn' arm")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max-per-regime", type=int, default=30)
    ap.add_argument("--cem-num-samples", type=int, default=64)
    ap.add_argument("--regimes", nargs="*",
                    default=["free_space", "pre_grasp", "contact_manipulation"])
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    cfg = yaml.safe_load(open(args.config))
    helpers = _load("_runner05", "05_run_diagnostic.py")
    probe08 = _load("_probe08", "08_planning_probe.py")

    dataset_name = cfg["dataset"]["name"]
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    pcfg = cfg.get("planning", {})
    horizon = int(pcfg.get("goal_H", 3))
    cem_kw = dict(
        num_samples=args.cem_num_samples,
        iterations=int(pcfg.get("iterations", 15)),
        num_elites=int(pcfg.get("num_elites", 10)),
        var_scale=float(pcfg.get("var_scale", 0.1)),
        max_norms=list(pcfg.get("max_norms", [1.0])),
        max_norm_dims=[list(g) for g in pcfg.get("max_norm_dims", [list(range(20))])],
    )
    n_resamples = cfg["bootstrap"]["n_resamples"]
    ci = cfg["bootstrap"]["ci"]

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, dataset_name)
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = adapter.frames_per_step
    raw_A = adapter.spec.action_dim
    model_A = adapter._model_action_dim
    probe, _ = load_probe(args.probe, device)
    dyn_head = None
    if args.dyn_head:
        dyn_head, _ = load_dynamics_head(args.dyn_head, device)
    regime_by_traj = read_regimes(cache_path)

    rows, pertrans = [], []
    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regime_by_traj, step, per_task=False)
        threshold = helpers.calibrate_effect_threshold_streaming(cache, records, step)

        # Metric scales from the same seed-0 pool construction as 12/13/15.
        pool_idx = np.random.default_rng(0).choice(
            np.arange(len(records)), size=min(cfg["hard_nn"]["pool_size"], len(records)),
            replace=False)
        pool_data = helpers.materialize_records(
            cache, [records[int(i)] for i in pool_idx], step, want_proprio=False)
        s_z = float((pool_data["z_t1"] - pool_data["z_t"])
                    .reshape(len(pool_idx), -1).norm(dim=-1).median())
        with torch.no_grad():
            g_pool = probe(pool_data["z_t"].to(device)).cpu()
        s_g = float((g_pool - g_pool.mean(0, keepdim=True)).norm(dim=-1).median())
        del pool_data
        print(f"H={horizon} samples={cem_kw['num_samples']} s_z={s_z:.2f} "
              f"s_g={s_g:.4f} beta={args.beta}", flush=True)

        def grp(tid):
            return cache.h5["trajectories"][LatentCache._safe_key(tid)]

        for regime_name in args.regimes:
            rid = REGIME_TO_ID[regime_name]
            cell = [r for r in records
                    if r["regime"] == rid
                    and r["idx0"] + horizon * step < int(grp(r["tid"])["z"].shape[0])]
            if not cell:
                continue
            sub = np.random.default_rng(42)
            if len(cell) > args.max_per_regime:
                cell = [cell[int(i)] for i in
                        sub.choice(len(cell), size=args.max_per_regime, replace=False)]
            data = probe08.load_planning_cell(cache, cell, step, horizon,
                                              want_proprio=adapter.uses_proprio())
            n = len(data["z_t"])
            z_t = torch.stack(data["z_t"]).to(device).float()
            z_t1 = torch.stack(data["z_t1"]).to(device).float()
            eff = effect_mask(z_t, z_t1, threshold)
            del z_t, z_t1

            arms = ["l2", "aware", "obj"] + (["hdyn"] if dyn_head is not None else [])
            errs = {arm: np.full(n, np.nan) for arm in arms}
            for i in range(n):
                if not eff[i]:
                    continue
                zi = data["z_t"][i].to(device).float()
                zg = data["z_goal"][i].to(device).float()
                p_t = (data["proprio_t"][i].to(device).float()
                       if adapter.uses_proprio() and data["proprio_t"][i] is not None else None)
                aware_cost = boundary_aware_cost(probe, zg, s_z=s_z, s_g=s_g, beta=args.beta)

                def obj_cost(pred_last, z_goal_, _c=aware_cost):
                    B = pred_last.shape[0]
                    sq_z = ((pred_last.reshape(B, -1) - z_goal_.reshape(1, -1)) ** 2).sum(-1)
                    return _c(pred_last, z_goal_) - sq_z / (s_z ** 2)

                plans = [("l2", None, None), ("aware", aware_cost, None),
                         ("obj", obj_cost, None)]
                if dyn_head is not None:
                    plans.append(("hdyn", None, grounded_dynamics_cost(
                        probe, dyn_head, adapter, zi, zg,
                        s_z=s_z, s_g=s_g, beta=args.beta)))
                for arm, cf, tcf in plans:
                    planned = cem_plan(
                        adapter, zi, zg, horizon=horizon, action_dim=model_A,
                        num_act_stepped=horizon, proprio_t=p_t,
                        generator=torch.Generator(device=device).manual_seed(1000 + i),
                        cost_fn=cf, traj_cost_fn=tcf, **cem_kw)
                    planned_raw = planned.reshape(-1, raw_A).cpu()
                    errs[arm][i] = action_error(planned_raw, data["expert_raw"][i])["total"]

            groups = np.asarray(data["traj_tag"])
            keep = eff & ~np.isnan(errs["l2"])
            for arm in errs:
                if keep.sum() < 1:
                    continue
                eci = bootstrap_ci(errs[arm][keep], n_resamples=n_resamples, ci=ci,
                                   seed=1, groups=groups[keep])
                rows.append({"dataset": dataset_name, "model": args.model,
                             "regime": regime_name, "horizon": horizon, "arm": arm,
                             "n_planned": int(keep.sum()), "beta": args.beta,
                             "s_z": s_z, "s_g": s_g,
                             "action_error": eci.point, "action_error_lo": eci.low,
                             "action_error_hi": eci.high})
            if keep.sum() >= 1:
                for fix_arm in [a for a in ("aware", "hdyn") if a in errs]:
                    delta = errs[fix_arm][keep] - errs["l2"][keep]
                    dci = bootstrap_ci(delta, n_resamples=n_resamples, ci=ci, seed=2,
                                       groups=groups[keep])
                    rows.append({"dataset": dataset_name, "model": args.model,
                                 "regime": regime_name, "horizon": horizon,
                                 "arm": f"{fix_arm}_minus_l2_paired",
                                 "n_planned": int(keep.sum()), "beta": args.beta,
                                 "s_z": s_z, "s_g": s_g,
                                 "action_error": dci.point, "action_error_lo": dci.low,
                                 "action_error_hi": dci.high})
                means = " ".join(f"{a}={np.nanmean(errs[a][keep]):.4f}" for a in errs)
                print(f"  {regime_name:22s} n={int(keep.sum()):3d} {means}", flush=True)
            for i in range(n):
                if keep[i]:
                    pertrans.append({"regime": regime_name, "tid": data["traj_tag"][i],
                                     **{a: errs[a][i] for a in errs}})
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_csv = Path(cfg["output"]["csv"]).with_name(f"{dataset_name}_planning_metric.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    npz = out_csv.with_name(out_csv.stem + "_pertrans.npz")
    if pertrans:
        np.savez(npz, **{k: np.asarray([p[k] for p in pertrans]) for k in pertrans[0]})
    print(f"\nWrote {out_csv} ({len(rows)} rows) + {npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
