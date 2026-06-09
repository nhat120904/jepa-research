"""Planning Action-Score probe — closes the CRA_eff → planning-failure link.

For each (effectful) DROID transition, in one pass, compute BOTH:
  (a) CRA_eff correctness under hard_nn negatives (the 1-step action-ranking signal), and
  (b) the paper's DROID Action Error from a real CEM plan to the goal latent.

Per-transition pairs let ``09_correlate_planning.py`` correlate the two with real
statistical power (thousands of points), not just 4 per-regime means.

The CEM planner, its hyper-parameters, the L2 objective, and the Action-Error
formula are faithful to the upstream DROID dino-wm config (see
``docs/plans/2026-06-05-planning-action-score-design.md``).

Runs on the server off the cached latents (03) + regimes (04). Output:
  results/droid_planning.csv          — per (regime, horizon) means + CIs
  results/droid_planning_pertrans.npz — per-transition pairs for correlation
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
from metrics import (  # noqa: E402
    cra_per_transition,
    effect_mask,
    sample_negatives,
    bootstrap_ci,
)
from metrics.action_score import action_error, rescale_action_score  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from planning.cem_planner import cem_plan  # noqa: E402
from scripts._resource_guard import preflight_model_load  # noqa: E402


def _load_runner_helpers():
    """Import the shared cache helpers from 05_run_diagnostic.py (module name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(
        "_runner05", str(ROOT / "scripts" / "05_run_diagnostic.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _group(cache: LatentCache, tid: str):
    return cache.h5["trajectories"][LatentCache._safe_key(tid)]


def load_planning_cell(cache, cell_records, step, horizon, *, want_proprio):
    """Read the tensors a planning cell needs, one trajectory resident at a time.

    Returns dict of lists indexed per transition:
        z_t, z_t1 (1-step, for CRA), z_goal (horizon steps), a_t_model (model_action_dim, for CRA),
        expert_raw (horizon*fps, raw_action_dim), proprio_t, traj_tag.
    """
    out = {k: [] for k in ["z_t", "z_t1", "z_goal", "a_t_model", "expert_raw", "proprio_t", "traj_tag"]}
    by_tid: dict[str, list[int]] = {}
    for r in cell_records:
        by_tid.setdefault(r["tid"], []).append(int(r["idx0"]))
    for tid, starts in by_tid.items():
        grp = _group(cache, tid)
        z = np.asarray(grp["z"])
        action = np.asarray(grp["action"])          # (La, raw_action_dim)
        proprio = np.asarray(grp["proprio"]) if (want_proprio and "proprio" in grp) else None
        for idx0 in starts:
            out["z_t"].append(torch.from_numpy(z[idx0].astype(np.float32)))
            out["z_t1"].append(torch.from_numpy(z[idx0 + step].astype(np.float32)))
            out["z_goal"].append(torch.from_numpy(z[idx0 + horizon * step].astype(np.float32)))
            a_model = action[idx0: idx0 + step].reshape(-1).astype(np.float32)
            out["a_t_model"].append(torch.from_numpy(a_model))
            expert_raw = action[idx0: idx0 + horizon * step].astype(np.float32)  # (horizon*fps, raw_A)
            out["expert_raw"].append(torch.from_numpy(expert_raw))
            out["proprio_t"].append(
                torch.from_numpy(proprio[idx0].astype(np.float32)) if proprio is not None else None
            )
            out["traj_tag"].append(tid)
    return out


def _pertrans_path(out_csv: Path) -> Path:
    return out_csv.with_name(f"{out_csv.stem}_pertrans.npz")


def main(
    config_path: str,
    *,
    only_model: str | None = None,
    max_planning_transitions: int | None = None,
    cem_num_samples: int | None = None,
    cem_iterations: int | None = None,
    out_csv_override: str | None = None,
) -> int:
    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    cfg = yaml.safe_load(open(config_path))
    helpers = _load_runner_helpers()

    dataset_name = cfg["dataset"]["name"]
    cache_root = cfg["latent_cache"]["root"]
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available()
                          or cfg["eval"]["device"] == "cpu" else "cpu")
    K = cfg["cra"]["K"]
    n_resamples = cfg["bootstrap"]["n_resamples"]
    ci = cfg["bootstrap"]["ci"]

    pcfg = cfg.get("planning", {})
    goal_H = int(pcfg.get("goal_H", 3))
    horizons = sorted({1, goal_H})
    max_plan = int(
        max_planning_transitions
        if max_planning_transitions is not None
        else pcfg.get("max_planning_transitions", 200)
    )
    cem_kw = dict(
        num_samples=int(
            cem_num_samples if cem_num_samples is not None else pcfg.get("num_samples", 300)
        ),
        iterations=int(
            cem_iterations if cem_iterations is not None else pcfg.get("iterations", 15)
        ),
        num_elites=int(pcfg.get("num_elites", 10)),
        var_scale=float(pcfg.get("var_scale", 0.1)),
        max_norms=list(pcfg.get("max_norms", [0.1, 0.75])),
        max_norm_dims=[list(g) for g in pcfg.get("max_norm_dims", [[0, 1, 2, 3, 4, 5], [6]])],
    )
    if max_plan < 1:
        raise ValueError("--max-planning-transitions must be at least 1")
    if cem_kw["iterations"] < 1:
        raise ValueError("--cem-iterations must be at least 1")
    if cem_kw["num_samples"] < cem_kw["num_elites"]:
        raise ValueError("--cem-num-samples must be at least planning.num_elites")

    bounds, l1_radius, gripper_dim, gripper_range, l1_dims = helpers.action_spec(
        dataset_name, cfg["dataset"]["action_dim"], cfg["dataset"]
    )

    rows, pertrans = [], []
    model_names = list(cfg["models"])
    if only_model is not None:
        if only_model not in model_names:
            raise ValueError(f"Unknown model {only_model!r}; configured models: {model_names}")
        model_names = [only_model]
    for model_name in model_names:
        cache_path = latent_cache_path(cache_root, model_name, dataset_name)
        if not cache_path.exists():
            print(f"[skip] {cache_path} missing")
            continue
        print(f"\n=== Planning probe: {model_name} on {dataset_name} ===", flush=True)
        preflight_model_load(model_name, str(device))
        adapter = build_adapter(model_name, device=str(device)).eval()
        step = adapter.frames_per_step
        raw_A = adapter.spec.action_dim
        model_A = adapter._model_action_dim
        fps = step
        regime_by_traj = read_regimes(cache_path)

        with LatentCache(cache_path, mode="r") as cache:
            records = helpers.build_transition_records(cache, regime_by_traj, step=step, per_task=False)
            threshold = helpers.calibrate_effect_threshold_streaming(cache, records, step=step)
            print(f"  step={step} raw_A={raw_A} model_A={model_A} ECS_thr={threshold:.4f} "
                  f"goal_H={goal_H} max_plan={max_plan}", flush=True)

            # Fixed hard-negative pool (same construction as 05).
            pool_size = cfg["hard_nn"]["pool_size"]
            rng = np.random.default_rng(0)
            pool_idx = rng.choice(np.arange(len(records)), size=min(pool_size, len(records)), replace=False)
            pool_data = helpers.materialize_records(cache, [records[int(i)] for i in pool_idx],
                                                    step, want_proprio=False)
            pool_z = pool_data["z_t"].to(device).float()
            pool_a = pool_data["a_t"].to(device).float()

            for horizon in horizons:
                for regime_name in cfg["regimes"]:
                    rid = REGIME_TO_ID[regime_name]
                    # Effectful transitions in this regime, long enough for the horizon.
                    cell = []
                    for r in records:
                        if r["regime"] != rid:
                            continue
                        grp = _group(cache, r["tid"])
                        if r["idx0"] + horizon * step >= int(grp["z"].shape[0]):
                            continue
                        cell.append(r)
                    if not cell:
                        continue
                    # Effect mask requires z_t, z_t1 — read lazily; subsample first for cost.
                    sub_rng = np.random.default_rng(42)
                    if len(cell) > max_plan:
                        keep = sub_rng.choice(len(cell), size=max_plan, replace=False)
                        cell = [cell[int(i)] for i in keep]

                    data = load_planning_cell(cache, cell, step, horizon,
                                              want_proprio=adapter.uses_proprio())
                    n = len(data["z_t"])
                    z_t = torch.stack(data["z_t"]).to(device).float()
                    z_t1 = torch.stack(data["z_t1"]).to(device).float()
                    a_model = torch.stack(data["a_t_model"]).to(device).float()
                    proprio = (torch.stack(data["proprio_t"]).to(device).float()
                               if adapter.uses_proprio() else None)
                    eff = effect_mask(z_t, z_t1, threshold)        # (n,) effectful?

                    # (a) CRA_eff correctness under hard_nn (1-step, horizon-independent).
                    a_neg = sample_negatives("hard_nn", a_t=a_model, K=K, action_bounds=bounds,
                                             z_t=z_t, pool_z=pool_z, pool_a=pool_a,
                                             similarity_radius=cfg["hard_nn"]["similarity_radius"])
                    cra_correct, _, _, _ = cra_per_transition(
                        adapter, z_t, a_model, z_t1, a_neg,
                        distance=adapter.spec.planning_distance, proprio_t=proprio,
                    )

                    # (b) plan each transition → grouped summed-delta Action Error.
                    errs = np.full(n, np.nan, dtype=np.float64)
                    for i in range(n):
                        if not eff[i]:
                            continue
                        p_t = data["proprio_t"][i].to(device).float() if proprio is not None else None
                        planned = cem_plan(
                            adapter, data["z_t"][i].to(device).float(),
                            data["z_goal"][i].to(device).float(),
                            horizon=horizon, action_dim=model_A, num_act_stepped=horizon,
                            proprio_t=p_t,
                            generator=torch.Generator(device=device).manual_seed(1000 + i),
                            **cem_kw,
                        )
                        planned_raw = planned.reshape(-1, raw_A).cpu()       # (horizon*fps, raw_A)
                        errs[i] = action_error(planned_raw, data["expert_raw"][i])["total"]

                    groups = np.asarray(data["traj_tag"])
                    keep = eff & ~np.isnan(errs)
                    for i in range(n):
                        if keep[i]:
                            pertrans.append({
                                "model": model_name, "regime": regime_name, "horizon": horizon,
                                "tid": data["traj_tag"][i], "cra_eff_correct": float(cra_correct[i]),
                                "action_error": float(errs[i]),
                            })
                    if keep.sum() >= 1:
                        err_ci = bootstrap_ci(errs[keep], n_resamples=n_resamples, ci=ci, seed=1,
                                              groups=groups[keep])
                        cra_ci = bootstrap_ci(cra_correct[keep], n_resamples=n_resamples, ci=ci, seed=2,
                                              groups=groups[keep])
                        rows.append({
                            "dataset": dataset_name, "model": model_name, "regime": regime_name,
                            "horizon": horizon, "n_planned": int(keep.sum()),
                            "max_planning_transitions": max_plan,
                            "cem_num_samples": cem_kw["num_samples"],
                            "cem_iterations": cem_kw["iterations"],
                            "cem_num_elites": cem_kw["num_elites"],
                            "action_error": err_ci.point, "action_error_lo": err_ci.low,
                            "action_error_hi": err_ci.high,
                            "cra_eff": cra_ci.point, "cra_eff_lo": cra_ci.low, "cra_eff_hi": cra_ci.high,
                        })
                        print(f"  H={horizon} {regime_name:22s} n={int(keep.sum()):4d} "
                              f"ActErr={err_ci.point:.4f} CRA_eff={cra_ci.point:.3f}", flush=True)
                    del z_t, z_t1, a_model, proprio, a_neg, cra_correct, eff
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    # Rescale Action Error → Action Score with a dataset-wide reference (p95 over all planned).
    if not df.empty:
        pt = pd.DataFrame(pertrans)
        d_ref = float(np.quantile(pt["action_error"], 0.95)) if not pt.empty else 1.0
        df["action_score"] = rescale_action_score(df["action_error"].to_numpy(), d_ref)
        df["d_ref_p95"] = d_ref
    out_csv = (
        Path(out_csv_override)
        if out_csv_override is not None
        else Path(cfg["output"]["csv"]).with_name("droid_planning.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    pertrans_path = _pertrans_path(out_csv)
    np.savez(pertrans_path,
             **{k: np.asarray([p[k] for p in pertrans]) for k in
                (pertrans[0].keys() if pertrans else [])})
    print(f"\nWrote {out_csv} ({len(rows)} rows) + {pertrans_path} ({len(pertrans)} pairs)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only-model")
    ap.add_argument("--max-planning-transitions", type=int)
    ap.add_argument("--cem-num-samples", type=int)
    ap.add_argument("--cem-iterations", type=int)
    ap.add_argument("--out-csv")
    args = ap.parse_args()
    sys.exit(main(
        args.config,
        only_model=args.only_model,
        max_planning_transitions=args.max_planning_transitions,
        cem_num_samples=args.cem_num_samples,
        cem_iterations=args.cem_iterations,
        out_csv_override=args.out_csv,
    ))
