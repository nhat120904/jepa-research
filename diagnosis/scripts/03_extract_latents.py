"""Encode every observation once and cache to HDF5.

Per (model, dataset): walk the trajectory iterator, encode each frame through
``adapter.encode`` (which applies the model's own /255 + transform + frozen
encoder), and write (z, action, proprio, state, gripper) to
``data/precomputed_latents/{dataset}__{model}.h5``.

This is the most expensive step; cache aggressively. All metrics run on the
cache, never re-encoding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    LatentCache,
    iterate_metaworld_trajectories,
    iterate_droid_trajectories,
    iterate_robocasa_trajectories,
    latent_cache_path,
)
from models.adapters import build_adapter  # noqa: E402


def get_iterator(dataset_name: str, ds_cfg: dict):
    external_root = ds_cfg.get("external_root", "external/jepa-wms")
    if dataset_name == "metaworld":
        tasks = ds_cfg["tasks"]
        all_tasks = tasks["easy"] + tasks["medium"] + tasks["hard"]
        return iterate_metaworld_trajectories(
            root=ds_cfg["root"],
            tasks=all_tasks,
            max_trajectories_per_task=ds_cfg.get("max_trajectories_per_task", 1000),
            external_root=external_root,
        )
    if dataset_name == "droid":
        return iterate_droid_trajectories(
            root=ds_cfg["root"],
            max_transitions=ds_cfg.get("max_transitions", 50000),
            external_root=external_root,
            dataset_kwargs=ds_cfg.get("dataset_kwargs"),
        )
    if dataset_name == "robocasa":
        return iterate_robocasa_trajectories(
            root=ds_cfg["root"],
            max_transitions=ds_cfg.get("max_transitions", 20000),
            external_root=external_root,
            dataset_kwargs=ds_cfg.get("dataset_kwargs"),
        )
    raise ValueError(f"Unknown dataset {dataset_name}")


@torch.no_grad()
def encode_trajectory(adapter, visual: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    """Encode each frame independently (T=1) in mini-batches.

    visual: (T, C, H, W) in [0, 255]. Returns visual latent (T, V, H, W, D).
    """
    T = visual.shape[0]
    chunks = []
    for s in range(0, T, batch_size):
        frames = visual[s : s + batch_size]          # (b, C, H, W)
        z = adapter.encode(frames.unsqueeze(1))       # (b, 1, V, H, W, D)
        chunks.append(z[:, 0].cpu())                  # (b, V, H, W, D)
    return torch.cat(chunks, dim=0)


def main(config_path: str) -> int:
    cfg = yaml.safe_load(open(config_path))
    dataset_name = cfg["dataset"]["name"]
    device = cfg["eval"].get("device", "cuda")
    batch_size = cfg["eval"].get("batch_size", 64)
    cache_root = cfg["latent_cache"]["root"]

    for model_name in cfg["models"]:
        print(f"\n=== Encoding {model_name} on {dataset_name} ===", flush=True)
        cache_path = latent_cache_path(cache_root, model_name, dataset_name)
        if cache_path.exists():
            print(f"  Cache exists at {cache_path} (delete to re-encode).")
            continue

        adapter = build_adapter(model_name, device=device).eval()
        with LatentCache(cache_path, mode="w",
                          compression=cfg["latent_cache"].get("compression", "gzip")) as cache:
            for traj in tqdm(get_iterator(dataset_name, cfg["dataset"]), desc=model_name):
                z = encode_trajectory(adapter, traj.obs_visual, batch_size=batch_size)
                cache.write_trajectory(
                    traj_id=traj.traj_id,
                    z=z,
                    action=traj.action,
                    gripper=traj.gripper,
                    proprio=traj.proprio,
                    state=traj.state,
                )
        print(f"  Wrote {cache_path}")
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="e.g. configs/diagnostic_metaworld.yaml")
    args = parser.parse_args()
    sys.exit(main(args.config))
