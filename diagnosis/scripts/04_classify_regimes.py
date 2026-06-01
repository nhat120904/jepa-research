"""Annotate every cached transition with a regime label.

Reads ``data/precomputed_latents/{dataset}__{model}.h5``, applies the dataset's
stratifier, and writes a per-trajectory ``regime`` array back into the file.

* Metaworld: proxy regimes from the 39-dim ``state`` vector (ee/object/gripper).
  Not MuJoCo contact GT — see stratification/metaworld_regimes.py.
* DROID / RoboCasa: proprioception + latent-change heuristics (no object GT;
  RoboCasa state is the 7-dim droid format).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path  # noqa: E402
from data.latent_cache import REGIME_TO_ID  # noqa: E402
from stratification import classify_metaworld_regime, classify_droid_regime  # noqa: E402


def classify_metaworld(cache: LatentCache) -> None:
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
        cache.write_regime(tid, ids)


def _classify_with_latent_heuristic(cache: LatentCache, desc: str) -> None:
    """Shared DROID/RoboCasa heuristic: gripper delta + latent-change proxy."""
    # First pass: global median latent change.
    diffs = []
    for tid in cache.trajectory_ids():
        z = torch.as_tensor(cache.read_trajectory(tid)["z"])
        d = (z[1:] - z[:-1]).reshape(z.shape[0] - 1, -1).norm(dim=-1)
        diffs.append(d)
    baseline = float(torch.cat(diffs).median().item())
    print(f"  {desc} baseline median latent change: {baseline:.4f}")

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
        cache.write_regime(tid, ids)


def classify_droid(cache: LatentCache) -> None:
    _classify_with_latent_heuristic(cache, "droid")


def classify_robocasa(cache: LatentCache) -> None:
    # RoboCasa state is the 7-dim droid format (no object pos) → same heuristic.
    _classify_with_latent_heuristic(cache, "robocasa")


CLASSIFIERS = {
    "metaworld": classify_metaworld,
    "droid": classify_droid,
    "robocasa": classify_robocasa,
}


def main(config_path: str) -> int:
    cfg = yaml.safe_load(open(config_path))
    dataset_name = cfg["dataset"]["name"]
    cache_root = cfg["latent_cache"]["root"]

    classifier = CLASSIFIERS[dataset_name]
    for model_name in cfg["models"]:
        path = latent_cache_path(cache_root, model_name, dataset_name)
        if not path.exists():
            print(f"[skip] no cache at {path} — run 03_extract_latents.py first")
            continue
        print(f"\n=== Stratifying {model_name} on {dataset_name} ===")
        with LatentCache(path, mode="a") as cache:
            classifier(cache)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    sys.exit(main(args.config))
