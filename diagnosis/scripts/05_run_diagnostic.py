"""Run CRA / AUG / ECS across all (model × strategy × regime × task) cells.

Inputs: latent caches from ``03_extract_latents.py`` + regimes from
``04_classify_regimes.py``. Output: tidy CSV with one row per cell, with
**trajectory-clustered** bootstrap 95% CIs on every reported number.

Calls the per-transition metric functions in ``metrics/`` directly (the same
ones validated by ``07_validate_synthetic.py``), threads proprioception into the
predictor, and reports both raw CRA and **effect-conditioned** CRA (restricted
to transitions where ||Δz|| exceeds the per-model median).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from data.latent_cache import REGIME_TO_ID  # noqa: E402
from metrics import (  # noqa: E402
    cra_per_transition,
    aug_per_transition,
    effect_mask,
    calibrate_effect_threshold,
    sample_negatives,
    bootstrap_ci,
)
from models.adapters import build_adapter  # noqa: E402


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cache_into_tensors(cache: LatentCache, regime_by_traj: Optional[dict] = None,
                            step: int = 1) -> dict:
    """Materialize the cache into in-memory transition tensors.

    Transitions are built at the model's native granularity: ``z_t`` and
    ``z_t1`` are ``step`` frames apart (``step = adapter.frames_per_step``, the
    frameskip), and ``a_t`` is the stack of the ``step`` raw actions spanning
    them (so ``a_t`` has dim ``step * action_dim`` = the predictor's
    model_action_dim). For step==1 this reduces to consecutive-frame
    transitions. The regime label is taken at the transition's start frame.

    Regimes come from the atomic sidecar (``regime_by_traj``, written by
    04_classify_regimes) keyed by traj_id, falling back to any ``regime`` array
    embedded in the cache (legacy) and finally to -1 (unclassified).
    """
    zs_t, zs_t1, acts, proprios, regimes, traj_tags, grippers = [], [], [], [], [], [], []
    has_proprio = True
    step = max(1, int(step))
    offsets = torch.arange(step)
    for tid in cache.trajectory_ids():
        traj = cache.read_trajectory(tid)
        z = torch.as_tensor(traj["z"])
        a = torch.as_tensor(traj["action"])
        La = a.shape[0]
        n = La // step                         # full frameskip steps
        if n == 0:
            continue
        idx0 = torch.arange(n) * step          # start frame of each model step
        gather = idx0.unsqueeze(1) + offsets    # (n, step) raw-action indices
        zs_t.append(z[idx0])
        zs_t1.append(z[idx0 + step])
        acts.append(a[gather].reshape(n, -1))   # (n, step * action_dim) stacked
        if "proprio" in traj:
            proprios.append(torch.as_tensor(traj["proprio"])[idx0])
        else:
            has_proprio = False
        reg = None
        if regime_by_traj is not None and tid in regime_by_traj:
            reg = regime_by_traj[tid]
        elif "regime" in traj:
            reg = traj["regime"]
        regimes.append(torch.as_tensor(reg, dtype=torch.int8)[idx0] if reg is not None
                       else torch.full((n,), -1, dtype=torch.int8))
        traj_tags.extend([tid] * n)
        if "gripper" in traj:
            grippers.append(torch.as_tensor(traj["gripper"])[idx0])
    return {
        "z_t": torch.cat(zs_t, dim=0),
        "z_t1": torch.cat(zs_t1, dim=0),
        "a_t": torch.cat(acts, dim=0).float(),
        "proprio_t": torch.cat(proprios, dim=0).float() if has_proprio and proprios else None,
        "regime": torch.cat(regimes, dim=0),
        "traj_tag": np.array(traj_tags),
        "gripper_t": torch.cat(grippers, dim=0) if grippers else None,
    }


def action_spec(dataset_name: str, action_dim: int, ds_cfg: dict):
    """Return action sampler settings for the negative samplers."""
    if dataset_name == "metaworld":
        return (-1.0, 1.0), None, 3, (-1.0, 1.0), None
    if dataset_name in ("droid", "robocasa"):
        pose_bound = float(ds_cfg.get("pose_action_bound", 0.1))
        gripper_range = tuple(ds_cfg.get("gripper_action_range", [-0.75, 0.75]))
        bounds = torch.tensor([[-pose_bound, pose_bound]] * (action_dim - 1) + [list(gripper_range)])
        l1_dims = ds_cfg.get("action_l1_dims", list(range(action_dim - 1)))
        return bounds, ds_cfg.get("action_l1_radius", 0.075), action_dim - 1, gripper_range, l1_dims
    return (-1.0, 1.0), None, None, (0.0, 1.0), None


# ---------------------------------------------------------------------------
# Low-memory cache access
# ---------------------------------------------------------------------------

def _group_for_tid(cache: LatentCache, tid: str):
    assert cache.h5 is not None
    return cache.h5["trajectories"][LatentCache._safe_key(tid)]


def build_transition_records(cache: LatentCache, regime_by_traj: Optional[dict],
                             step: int, *, per_task: bool) -> list[dict]:
    """Build a small metadata index without loading the latent tensors.

    The original runner materialized every transition in a model cache at once.
    A 60-traj/task Metaworld cache is ~26 GB on disk and can exceed 20 GB RAM
    during that load on Windows. These records let the metric loop materialize
    only one (task, regime) cell at a time.
    """
    records: list[dict] = []
    step = max(1, int(step))
    for tid in cache.trajectory_ids():
        grp = _group_for_tid(cache, tid)
        La = int(grp["action"].shape[0])
        n = La // step
        if n == 0:
            continue
        reg = None
        if regime_by_traj is not None and tid in regime_by_traj:
            reg = regime_by_traj[tid]
        elif "regime" in grp:
            reg = grp["regime"][:]
        task = tid.split("/")[0] if per_task else "all"
        for k in range(n):
            idx0 = k * step
            regime = int(reg[idx0]) if reg is not None and idx0 < len(reg) else -1
            records.append({"tid": tid, "idx0": idx0, "task": task, "regime": regime})
    return records


def calibrate_effect_threshold_streaming(cache: LatentCache, records: list[dict],
                                         step: int, quantile: float = 0.5) -> float:
    """Median ||z_{t+1}-z_t|| with one trajectory resident at a time."""
    by_tid: dict[str, list[int]] = {}
    for r in records:
        by_tid.setdefault(r["tid"], []).append(int(r["idx0"]))

    norms = []
    for tid, starts in by_tid.items():
        grp = _group_for_tid(cache, tid)
        z = np.asarray(grp["z"])
        idx0 = np.asarray(starts, dtype=np.int64)
        diff = z[idx0 + step] - z[idx0]
        norms.append(np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1))
    if not norms:
        return 0.0
    all_norms = np.concatenate(norms)
    return float(np.quantile(all_norms, quantile))


def materialize_records(cache: LatentCache, records: list[dict], step: int,
                        *, want_proprio: bool) -> dict:
    """Load only the requested transition records into CPU tensors."""
    if not records:
        raise ValueError("materialize_records called with no records")

    first = _group_for_tid(cache, records[0]["tid"])
    z_shape = tuple(first["z"].shape[1:])
    action_dim = int(first["action"].shape[-1]) * step
    proprio_dim = int(first["proprio"].shape[-1]) if want_proprio and "proprio" in first else 0

    N = len(records)
    z_t = torch.empty((N, *z_shape), dtype=torch.float32)
    z_t1 = torch.empty((N, *z_shape), dtype=torch.float32)
    a_t = torch.empty((N, action_dim), dtype=torch.float32)
    proprio_t = torch.empty((N, proprio_dim), dtype=torch.float32) if proprio_dim else None
    traj_tags = [None] * N

    by_tid: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(records):
        by_tid.setdefault(r["tid"], []).append((i, r))

    for tid, items in by_tid.items():
        grp = _group_for_tid(cache, tid)
        starts = np.asarray([int(r["idx0"]) for _, r in items], dtype=np.int64)
        rows = np.asarray([i for i, _ in items], dtype=np.int64)
        rows_t = torch.as_tensor(rows, dtype=torch.long)
        z = np.asarray(grp["z"])
        action = np.asarray(grp["action"])
        z_t[rows_t] = torch.from_numpy(np.asarray(z[starts], dtype=np.float32))
        z_t1[rows_t] = torch.from_numpy(np.asarray(z[starts + step], dtype=np.float32))
        if step == 1:
            a = np.asarray(action[starts], dtype=np.float32)
        else:
            a = np.stack([action[s: s + step].reshape(-1) for s in starts]).astype(np.float32)
        a_t[rows_t] = torch.from_numpy(a)
        if proprio_t is not None:
            proprio = np.asarray(grp["proprio"])
            proprio_t[rows_t] = torch.from_numpy(np.asarray(proprio[starts], dtype=np.float32))
        for i, _ in items:
            traj_tags[i] = tid

    return {
        "z_t": z_t,
        "z_t1": z_t1,
        "a_t": a_t,
        "proprio_t": proprio_t,
        "traj_tag": np.asarray(traj_tags),
    }


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cell(adapter, data, indices, *, strategy, K, bounds, l1_radius, gripper_dim,
                  gripper_range, l1_dims, pool_indices, similarity_radius, distance,
                  batch_size, device, effect_threshold, action_penalty=0.5):
    """Per-transition CRA correctness/MRR, AUG, and effect mask over a slice."""
    z_t_all, z_t1_all, a_t_all = data["z_t"][indices], data["z_t1"][indices], data["a_t"][indices]
    proprio_all = data["proprio_t"][indices] if data["proprio_t"] is not None else None
    pool_z = data["z_t"][pool_indices].to(device).float()
    pool_a = data["a_t"][pool_indices].to(device).float()
    # Candidate next-latents — only hard_effect needs them (true-effect divergence).
    pool_z1 = data["z_t1"][pool_indices].to(device).float()

    cra_c, cra_r, aug_p, eff_m = [], [], [], []
    N = z_t_all.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        z_t = z_t_all[s:e].to(device).float()
        z_t1 = z_t1_all[s:e].to(device).float()
        a_t = a_t_all[s:e].to(device).float()
        proprio_t = proprio_all[s:e].to(device).float() if proprio_all is not None else None

        a_neg = sample_negatives(
            strategy, a_t=a_t, K=K, action_bounds=bounds, l1_radius=l1_radius,
            l1_dims=l1_dims, sigma=0.1, gripper_dim=gripper_dim,
            gripper_range=gripper_range, z_t=z_t, z_t1=z_t1, pool_z=pool_z,
            pool_a=pool_a, pool_z1=pool_z1, similarity_radius=similarity_radius,
            action_penalty=action_penalty,
        )
        correct, recip, _, _ = cra_per_transition(
            adapter, z_t, a_t, z_t1, a_neg, distance=distance, proprio_t=proprio_t
        )
        cra_c.append(correct)
        cra_r.append(recip)
        aug_p.append(aug_per_transition(adapter, z_t, a_t, z_t1, proprio_t=proprio_t))
        eff_m.append(effect_mask(z_t, z_t1, effect_threshold))

    return (np.concatenate(cra_c), np.concatenate(cra_r),
            np.concatenate(aug_p), np.concatenate(eff_m))


@torch.no_grad()
def evaluate_materialized_cell(adapter, data, pool_data, *, strategy, K, bounds, l1_radius,
                               gripper_dim, gripper_range, l1_dims, similarity_radius,
                               distance, batch_size, device, effect_threshold,
                               action_penalty=0.5):
    """Evaluate one materialized cell.

    ``data`` holds only the transitions for a single (task, regime) cell. The
    candidate pool is the fixed per-model hard-negative pool. This keeps RAM
    bounded on machines where the full Metaworld cache cannot fit in memory.
    """
    z_t_all, z_t1_all, a_t_all = data["z_t"], data["z_t1"], data["a_t"]
    proprio_all = data["proprio_t"]

    needs_pool = strategy in ("hard_nn", "hard_effect")
    pool_z = pool_a = pool_z1 = None
    if needs_pool:
        pool_z = pool_data["z_t"].to(device).float()
        pool_a = pool_data["a_t"].to(device).float()
        pool_z1 = pool_data["z_t1"].to(device).float()

    cra_c, cra_r, aug_p, eff_m = [], [], [], []
    N = z_t_all.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        z_t = z_t_all[s:e].to(device).float()
        z_t1 = z_t1_all[s:e].to(device).float()
        a_t = a_t_all[s:e].to(device).float()
        proprio_t = proprio_all[s:e].to(device).float() if proprio_all is not None else None

        a_neg = sample_negatives(
            strategy, a_t=a_t, K=K, action_bounds=bounds, l1_radius=l1_radius,
            l1_dims=l1_dims, sigma=0.1, gripper_dim=gripper_dim,
            gripper_range=gripper_range, z_t=z_t, z_t1=z_t1, pool_z=pool_z,
            pool_a=pool_a, pool_z1=pool_z1, similarity_radius=similarity_radius,
            action_penalty=action_penalty,
        )
        correct, recip, _, _ = cra_per_transition(
            adapter, z_t, a_t, z_t1, a_neg, distance=distance, proprio_t=proprio_t
        )
        cra_c.append(correct)
        cra_r.append(recip)
        aug_p.append(aug_per_transition(adapter, z_t, a_t, z_t1, proprio_t=proprio_t))
        eff_m.append(effect_mask(z_t, z_t1, effect_threshold))

    return (np.concatenate(cra_c), np.concatenate(cra_r),
            np.concatenate(aug_p), np.concatenate(eff_m))


def _ci(arr, groups, n_resamples, ci, seed):
    return bootstrap_ci(arr, n_resamples=n_resamples, ci=ci, seed=seed, groups=groups)


def main(config_path: str, ctd: bool = False) -> int:
    torch_threads = int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2"))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, torch_threads)))
    except RuntimeError:
        pass

    cfg = yaml.safe_load(open(config_path))
    only_model = os.environ.get("CAI_JEPA_ONLY_MODEL")
    if only_model:
        cfg["models"] = [m for m in cfg["models"] if m == only_model]
        if not cfg["models"]:
            raise ValueError(f"CAI_JEPA_ONLY_MODEL={only_model!r} is not in config models")
    dataset_name = cfg["dataset"]["name"]
    action_dim = cfg["dataset"]["action_dim"]
    cache_root = cfg["latent_cache"]["root"]
    batch_size = cfg["eval"]["batch_size"]
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available()
                          or cfg["eval"]["device"] == "cpu" else "cpu")
    K = cfg["cra"]["K"]
    n_resamples = cfg["bootstrap"]["n_resamples"]
    ci = cfg["bootstrap"]["ci"]
    min_n = cfg["eval"]["min_transitions_per_cell"]

    bounds, l1_radius, gripper_dim, gripper_range, l1_dims = action_spec(
        dataset_name, action_dim, cfg["dataset"]
    )

    rows: List[dict] = []
    for model_name in cfg["models"]:
        cache_path = latent_cache_path(cache_root, model_name, dataset_name)
        if not cache_path.exists():
            print(f"[skip] {cache_path} missing")
            continue

        print(f"\n=== Diagnostic: {model_name} on {dataset_name} ===")
        adapter = build_adapter(model_name, device=str(device)).eval()
        # Use the model's own planner distance (all baselines: L2) unless overridden.
        distance = cfg.get("distance", adapter.spec.planning_distance)

        regime_by_traj = read_regimes(cache_path)
        if regime_by_traj is None:
            print(f"  [warn] no regime sidecar for {cache_path.name} — run "
                  "04_classify_regimes.py first (all cells will be insufficient_data).")
        with LatentCache(cache_path, mode="r") as cache:
            per_task = (dataset_name == "metaworld")
            records = build_transition_records(cache, regime_by_traj, step=adapter.frames_per_step,
                                               per_task=per_task)
            if not records:
                print(f"  [warn] no transitions found for {cache_path.name}", flush=True)
                continue

            print(f"  frames_per_step (frameskip)={adapter.frames_per_step}; "
                  f"{len(records)} transitions; low_memory_cell_streaming=True; "
                  f"torch_threads={torch_threads}", flush=True)

            threshold = calibrate_effect_threshold_streaming(cache, records, step=adapter.frames_per_step)
            print(f"  ECS threshold (median ||Δz||): {threshold:.4f}; "
                  f"proprio={adapter.uses_proprio()}", flush=True)

            pool_size = cfg["hard_nn"]["pool_size"]
            rng = np.random.default_rng(0)
            pool_idx = rng.choice(np.arange(len(records)), size=min(pool_size, len(records)), replace=False)
            pool_records = [records[int(i)] for i in pool_idx]
            pool_data = materialize_records(cache, pool_records, adapter.frames_per_step,
                                            want_proprio=False)
            print(f"  hard-negative pool: {len(pool_records)} transitions", flush=True)

            tasks = sorted({r["task"] for r in records})
            for regime_name in cfg["regimes"]:
                regime_id = REGIME_TO_ID[regime_name]
                for task in tasks:
                    cell_records = [r for r in records if r["regime"] == regime_id and r["task"] == task]
                    if len(cell_records) < min_n:
                        for strategy in cfg["negative_strategies"]:
                            rows.append({"dataset": dataset_name, "task": task, "model": model_name,
                                         "strategy": strategy, "regime": regime_name,
                                         "n_transitions": int(len(cell_records)),
                                         "status": "insufficient_data"})
                        continue

                    data = materialize_records(cache, cell_records, adapter.frames_per_step,
                                               want_proprio=adapter.uses_proprio())
                    groups = data["traj_tag"]
                    for strategy in cfg["negative_strategies"]:
                        cra_c, cra_r, aug_p, eff_m = evaluate_materialized_cell(
                            adapter, data, pool_data, strategy=strategy, K=K, bounds=bounds,
                            l1_radius=l1_radius, gripper_dim=gripper_dim,
                            gripper_range=gripper_range, l1_dims=l1_dims,
                            similarity_radius=cfg["hard_nn"]["similarity_radius"], distance=distance,
                            batch_size=batch_size, device=device, effect_threshold=threshold,
                            action_penalty=cfg.get("hard_effect", {}).get("action_penalty", 0.5),
                        )
                        eff_groups = groups[eff_m]

                        cra_top1 = _ci(cra_c, groups, n_resamples, ci, 1)
                        cra_mrr = _ci(cra_r, groups, n_resamples, ci, 2)
                        aug = _ci(aug_p, groups, n_resamples, ci, 3)
                        cra_eff = _ci(cra_c[eff_m], eff_groups, n_resamples, ci, 4) if eff_m.sum() else None
                        ecs = _ci(aug_p[eff_m], eff_groups, n_resamples, ci, 5) if eff_m.sum() else None

                        rows.append({
                            "dataset": dataset_name, "task": task, "model": model_name,
                            "strategy": strategy, "regime": regime_name,
                            "n_transitions": int(len(cell_records)), "n_effect": int(eff_m.sum()),
                            "effect_threshold": threshold,
                            "cra_top1": cra_top1.point, "cra_top1_lo": cra_top1.low, "cra_top1_hi": cra_top1.high,
                            "cra_top1_eff": cra_eff.point if cra_eff else float("nan"),
                            "cra_top1_eff_lo": cra_eff.low if cra_eff else float("nan"),
                            "cra_top1_eff_hi": cra_eff.high if cra_eff else float("nan"),
                            "cra_mrr": cra_mrr.point, "cra_mrr_lo": cra_mrr.low, "cra_mrr_hi": cra_mrr.high,
                            "aug": aug.point, "aug_lo": aug.low, "aug_hi": aug.high,
                            "ecs": ecs.point if ecs else float("nan"),
                            "ecs_lo": ecs.low if ecs else float("nan"),
                            "ecs_hi": ecs.high if ecs else float("nan"),
                            "status": "ok",
                        })
                        print(f"  {task:22s} {strategy:8s} {regime_name:20s} n={len(cell_records):5d} "
                              f"CRA={cra_top1.point:.3f} CRA_eff={rows[-1]['cra_top1_eff']:.3f} "
                              f"AUG={aug.point:+.4f}", flush=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    del data
                    gc.collect()

            del pool_data

        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = Path(os.environ.get("CAI_JEPA_OUTPUT_CSV", cfg["output"]["csv"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ctd", action="store_true", help="Also compute CTD (multi-step). Slower.")
    args = parser.parse_args()
    sys.exit(main(args.config, ctd=args.ctd))
