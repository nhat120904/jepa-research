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

from data import LatentCache, latent_cache_path  # noqa: E402
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

def load_cache_into_tensors(cache: LatentCache) -> dict:
    """Materialize the cache into in-memory transition tensors."""
    zs_t, zs_t1, acts, proprios, regimes, traj_tags, grippers = [], [], [], [], [], [], []
    has_proprio = True
    for tid in cache.trajectory_ids():
        traj = cache.read_trajectory(tid)
        z = torch.as_tensor(traj["z"])
        a = torch.as_tensor(traj["action"])
        T = a.shape[0]
        zs_t.append(z[:T])
        zs_t1.append(z[1 : T + 1])
        acts.append(a)
        if "proprio" in traj:
            proprios.append(torch.as_tensor(traj["proprio"])[:T])
        else:
            has_proprio = False
        regimes.append(torch.as_tensor(traj["regime"], dtype=torch.int8) if "regime" in traj
                       else torch.full((T,), -1, dtype=torch.int8))
        traj_tags.extend([tid] * T)
        if "gripper" in traj:
            grippers.append(torch.as_tensor(traj["gripper"])[:T])
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
    """Return (bounds, l1_radius, gripper_dim) for the negative samplers."""
    if dataset_name == "metaworld":
        return (-1.0, 1.0), None, 3
    if dataset_name in ("droid", "robocasa"):
        bounds = torch.tensor([[-0.1, 0.1]] * (action_dim - 1) + [[0.0, 1.0]])
        return bounds, ds_cfg.get("action_l1_radius", 0.075), action_dim - 1
    return (-1.0, 1.0), None, None


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cell(adapter, data, indices, *, strategy, K, bounds, l1_radius, gripper_dim,
                  pool_indices, similarity_radius, distance, batch_size, device, effect_threshold):
    """Per-transition CRA correctness/MRR, AUG, and effect mask over a slice."""
    z_t_all, z_t1_all, a_t_all = data["z_t"][indices], data["z_t1"][indices], data["a_t"][indices]
    proprio_all = data["proprio_t"][indices] if data["proprio_t"] is not None else None
    pool_z = data["z_t"][pool_indices].to(device).float()
    pool_a = data["a_t"][pool_indices].to(device).float()

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
            sigma=0.1, gripper_dim=gripper_dim, z_t=z_t, pool_z=pool_z, pool_a=pool_a,
            similarity_radius=similarity_radius,
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
    cfg = yaml.safe_load(open(config_path))
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

    bounds, l1_radius, gripper_dim = action_spec(dataset_name, action_dim, cfg["dataset"])

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

        with LatentCache(cache_path, mode="r") as cache:
            data = load_cache_into_tensors(cache)

        threshold = calibrate_effect_threshold(data["z_t"], data["z_t1"])
        print(f"  ECS threshold (median ||Δz||): {threshold:.4f}; proprio="
              f"{data['proprio_t'] is not None and adapter.uses_proprio()}")

        pool_size = cfg["hard_nn"]["pool_size"]
        all_idx = np.arange(data["z_t"].shape[0])
        rng = np.random.default_rng(0)
        pool_indices = rng.choice(all_idx, size=min(pool_size, len(all_idx)), replace=False)

        per_task = (dataset_name == "metaworld")
        traj_tags = data["traj_tag"]
        tasks = (np.array([t.split("/")[0] for t in traj_tags]) if per_task
                 else np.array(["all"] * len(all_idx)))

        for strategy in cfg["negative_strategies"]:
            for regime_name in cfg["regimes"]:
                regime_id = REGIME_TO_ID[regime_name]
                base_mask = (data["regime"].numpy() == regime_id)
                for task in sorted(set(tasks)):
                    mask = base_mask & (tasks == task)
                    indices = np.nonzero(mask)[0]
                    if len(indices) < min_n:
                        rows.append({"dataset": dataset_name, "task": task, "model": model_name,
                                     "strategy": strategy, "regime": regime_name,
                                     "n_transitions": int(len(indices)), "status": "insufficient_data"})
                        continue

                    cra_c, cra_r, aug_p, eff_m = evaluate_cell(
                        adapter, data, indices, strategy=strategy, K=K, bounds=bounds,
                        l1_radius=l1_radius, gripper_dim=gripper_dim, pool_indices=pool_indices,
                        similarity_radius=cfg["hard_nn"]["similarity_radius"], distance=distance,
                        batch_size=batch_size, device=device, effect_threshold=threshold,
                    )
                    groups = traj_tags[indices]
                    eff_groups = groups[eff_m]

                    cra_top1 = _ci(cra_c, groups, n_resamples, ci, 1)
                    cra_mrr = _ci(cra_r, groups, n_resamples, ci, 2)
                    aug = _ci(aug_p, groups, n_resamples, ci, 3)
                    cra_eff = _ci(cra_c[eff_m], eff_groups, n_resamples, ci, 4) if eff_m.sum() else None
                    ecs = _ci(aug_p[eff_m], eff_groups, n_resamples, ci, 5) if eff_m.sum() else None

                    rows.append({
                        "dataset": dataset_name, "task": task, "model": model_name,
                        "strategy": strategy, "regime": regime_name,
                        "n_transitions": int(len(indices)), "n_effect": int(eff_m.sum()),
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
                    print(f"  {task:22s} {strategy:8s} {regime_name:20s} n={len(indices):5d} "
                          f"CRA={cra_top1.point:.3f} CRA_eff={rows[-1]['cra_top1_eff']:.3f} "
                          f"AUG={aug.point:+.4f}")

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = Path(cfg["output"]["csv"])
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
