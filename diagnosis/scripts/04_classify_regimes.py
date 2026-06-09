"""Annotate every cached transition with a regime label.

Reads ``data/precomputed_latents/{dataset}__{model}.h5`` **read-only**, applies
the dataset's stratifier, and writes per-trajectory ``regime`` arrays to an
atomic JSON sidecar (``{dataset}__{model}.h5.regimes.json``). The large latent
cache is never re-opened for writing, so a kill mid-stratification can never
truncate/corrupt it; the sidecar is written via os.replace (atomic).

* Metaworld: proxy regimes from the 39-dim ``state`` vector (ee/object/gripper).
  Not MuJoCo contact GT — see stratification/metaworld_regimes.py.
* DROID / RoboCasa: proprioception + latent-change heuristics (no object GT;
  RoboCasa state is the 7-dim droid format).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, write_regimes  # noqa: E402
from data.latent_cache import REGIME_TO_ID  # noqa: E402
from stratification import classify_metaworld_regime, classify_droid_regime  # noqa: E402


def classify_metaworld(cache: LatentCache) -> dict:
    out: dict = {}
    for tid in tqdm(cache.trajectory_ids(), desc="metaworld regimes"):
        traj = cache.read_trajectory(tid)
        T = len(traj["action"])
        if "state" not in traj:
            raise KeyError(
                f"{tid} has no cached 'state'; re-run 03_extract_latents.py "
                "(Metaworld stratification needs the 39-dim state vector)."
            )
        state = np.asarray(traj["state"])
        ids = np.zeros(T, dtype=np.int8)
        for t in range(T):
            ids[t] = REGIME_TO_ID[classify_metaworld_regime(state[t], state[t + 1])]
        out[tid] = ids
    return out


def _classify_with_latent_heuristic(cache: LatentCache, desc: str) -> dict:
    """Shared DROID/RoboCasa heuristic: gripper delta + latent-change proxy."""
    # First pass: global median latent change.
    diffs = []
    for tid in cache.trajectory_ids():
        z = torch.as_tensor(cache.read_trajectory(tid)["z"])
        d = (z[1:] - z[:-1]).reshape(z.shape[0] - 1, -1).norm(dim=-1)
        diffs.append(d)
    baseline = float(torch.cat(diffs).median().item())
    print(f"  {desc} baseline median latent change: {baseline:.4f}")

    out: dict = {}
    for tid in tqdm(cache.trajectory_ids(), desc=f"{desc} regimes"):
        traj = cache.read_trajectory(tid)
        T = len(traj["action"])
        has_grip = "gripper" in traj
        ids = np.zeros(T, dtype=np.int8)
        for t in range(T):
            info = {
                "gripper_state_t": float(traj["gripper"][t]) if has_grip else 0.5,
                "gripper_state_t1": float(traj["gripper"][t + 1]) if has_grip else 0.5,
                "global_median_latent_change": baseline,
            }
            ids[t] = REGIME_TO_ID[classify_droid_regime(info, traj["z"][t], traj["z"][t + 1])]
        out[tid] = ids
    return out


def classify_droid(cache: LatentCache) -> dict:
    return _classify_with_latent_heuristic(cache, "droid")


def classify_robocasa(cache: LatentCache) -> dict:
    # RoboCasa state is the 7-dim droid format (no object pos) → same heuristic.
    return _classify_with_latent_heuristic(cache, "robocasa")


def classify_franka_custom(cache: LatentCache) -> dict:
    # Franka custom uses the same 7-dim pose+gripper format as DROID.
    return _classify_with_latent_heuristic(cache, "franka_custom")


def classify_free_space_only(cache: LatentCache) -> dict:
    """Sanity/navigation datasets do not use the manipulation regime ontology."""
    out: dict = {}
    for tid in tqdm(cache.trajectory_ids(), desc="free_space regimes"):
        traj = cache.read_trajectory(tid)
        out[tid] = np.full(len(traj["action"]), REGIME_TO_ID["free_space"], dtype=np.int8)
    return out


CLASSIFIERS = {
    "metaworld": classify_metaworld,
    "droid": classify_droid,
    "robocasa": classify_robocasa,
    "franka_custom": classify_franka_custom,
    "pusht": classify_free_space_only,
    "point_maze": classify_free_space_only,
    "wall": classify_free_space_only,
}


def main(config_path: str) -> int:
    cfg = yaml.safe_load(open(config_path))
    dataset_name = cfg["dataset"]["name"]
    cache_root = cfg["latent_cache"]["root"]
    only_model = os.environ.get("CAI_JEPA_ONLY_MODEL")
    models = cfg["models"]
    if only_model:
        models = [m for m in models if m == only_model]
        if not models:
            raise ValueError(f"CAI_JEPA_ONLY_MODEL={only_model!r} is not in config models")

    classifier = CLASSIFIERS[dataset_name]
    for model_name in models:
        path = latent_cache_path(cache_root, model_name, dataset_name)
        if not path.exists():
            print(f"[skip] no cache at {path} — run 03_extract_latents.py first")
            continue
        print(f"\n=== Stratifying {model_name} on {dataset_name} ===")
        # Open the (expensive) latent cache read-only; never re-write it.
        with LatentCache(path, mode="r") as cache:
            regime_by_traj = classifier(cache)
        sidecar = write_regimes(path, regime_by_traj)
        print(f"  Wrote {sidecar} ({len(regime_by_traj)} trajectories)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    sys.exit(main(args.config))
