"""DROID pipeline sanity gate.

This used to be a 2-way CRA test with a hard-coded >0.90 threshold from a
qualitative Terver-style open/close visualization. That was too brittle for
this diagnostic pipeline: it compared hard-coded actions against arbitrary
dataset next-latents, so it could fail even when the loader, cached latents, and
model API were wired correctly.

The current gate checks the plumbing we actually rely on:

1. DROID action/proprio temporal alignment: action[-1] must equal the sampled
   gripper-position delta between consecutive frames.
2. Optional cache determinism: with a fixed dataset seed, cached action/proprio
   arrays must match the loader for the first few clips.
3. Model action path sensitivity: predictions with factual, zero, open, and
   close gripper actions must not collapse to exactly the same latent.

It does not require the model to be highly action-grounded. Low CRA remains a
valid diagnostic outcome; this script only catches pipeline bugs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, iterate_droid_trajectories, latent_cache_path  # noqa: E402
from metrics.distances import get_distance  # noqa: E402
from models.adapters import build_adapter  # noqa: E402

GRIPPER_IDX = -1
Z_IDX = 2


def _iter_droid(ds_cfg: dict, max_transitions: int):
    return iterate_droid_trajectories(
        ds_cfg["root"],
        max_transitions=max_transitions,
        external_root=ds_cfg.get("external_root", "external/jepa-wms"),
        dataset_kwargs=ds_cfg.get("dataset_kwargs"),
    )


def _check_loader_alignment(ds_cfg: dict, max_transitions: int) -> tuple[int, float, tuple[float, float]]:
    total = 0
    max_gripper_err = 0.0
    action_min = float("inf")
    action_max = -float("inf")
    for traj in _iter_droid(ds_cfg, max_transitions):
        action = traj.action
        gripper = traj.proprio[:, GRIPPER_IDX]
        expected = gripper[1:1 + action.shape[0]] - gripper[:action.shape[0]]
        err = (action[:, GRIPPER_IDX] - expected).abs().max().item()
        max_gripper_err = max(max_gripper_err, float(err))
        action_min = min(action_min, float(action[:, GRIPPER_IDX].min().item()))
        action_max = max(action_max, float(action[:, GRIPPER_IDX].max().item()))
        total += int(action.shape[0])
        if total >= max_transitions:
            break
    return total, max_gripper_err, (action_min, action_max)


def _check_cache(ds_cfg: dict, model_name: str, max_trajectories: int = 5) -> tuple[bool, float]:
    cache_path = latent_cache_path(ds_cfg.get("latent_cache_root", "data/precomputed_latents"),
                                   model_name, "droid")
    if not cache_path.exists():
        return False, float("nan")

    max_err = 0.0
    with LatentCache(cache_path, mode="r") as cache:
        for i, traj in enumerate(_iter_droid(ds_cfg, max_transitions=10_000)):
            if i >= max_trajectories:
                break
            cached = cache.read_trajectory(traj.traj_id)
            for key, live in (("action", traj.action), ("proprio", traj.proprio), ("state", traj.state)):
                live_np = live.detach().cpu().numpy()
                cached_np = np.asarray(cached[key])
                n = min(len(live_np), len(cached_np))
                max_err = max(max_err, float(np.max(np.abs(live_np[:n] - cached_np[:n]))))
    return True, max_err


@torch.no_grad()
def _check_model_sensitivity(ds_cfg: dict, model_name: str, max_transitions: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = build_adapter(model_name, device=device).eval()
    dist_fn = get_distance(adapter.spec.planning_distance)

    rows = []
    total = 0
    for traj in _iter_droid(ds_cfg, max_transitions):
        T = traj.action.shape[0]
        z = torch.cat(
            [adapter.encode(traj.obs_visual[i:i + 1].unsqueeze(1))[:, 0].cpu()
             for i in range(T + 1)],
            0,
        ).to(device).float()
        z_t, z_t1 = z[:T], z[1:T + 1]
        a = traj.action[:T].to(device).float()

        zero = torch.zeros_like(a)
        open_a = zero.clone()
        close_a = zero.clone()
        open_a[:, Z_IDX] = 0.05
        close_a[:, Z_IDX] = 0.05
        open_a[:, GRIPPER_IDX] = -0.75
        close_a[:, GRIPPER_IDX] = 0.75

        pred_fact = adapter.predict(z_t, a)
        pred_zero = adapter.predict(z_t, zero)
        pred_open = adapter.predict(z_t, open_a)
        pred_close = adapter.predict(z_t, close_a)

        rows.append({
            "fact_vs_zero": float(dist_fn(pred_fact, pred_zero).mean().item()),
            "open_vs_close": float(dist_fn(pred_open, pred_close).mean().item()),
            "fact_to_next": float(dist_fn(pred_fact, z_t1).mean().item()),
            "zero_to_next": float(dist_fn(pred_zero, z_t1).mean().item()),
        })
        total += T
        if total >= max_transitions:
            break

    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = {"n_transitions": total}
    for key in rows[0]:
        out[key] = float(np.mean([r[key] for r in rows]))
    return out


def main(config_path: str, model_name: str, max_transitions: int = 512,
         cache_trajectories: int = 5) -> int:
    cfg = yaml.safe_load(open(config_path))
    ds_cfg = dict(cfg["dataset"])
    ds_cfg["latent_cache_root"] = cfg["latent_cache"]["root"]

    total, grip_err, grip_range = _check_loader_alignment(ds_cfg, max_transitions)
    cache_present, cache_err = _check_cache(ds_cfg, model_name, cache_trajectories)
    sens = _check_model_sensitivity(ds_cfg, model_name, max_transitions)

    print(f"[droid-pipeline] transitions checked: {total}")
    print(f"[droid-pipeline] gripper delta max error: {grip_err:.3e}")
    print(f"[droid-pipeline] sampled gripper action range: [{grip_range[0]:.3f}, {grip_range[1]:.3f}]")
    if cache_present:
        print(f"[droid-pipeline] cache/loader max action-proprio-state error: {cache_err:.3e}")
    else:
        print("[droid-pipeline] cache/loader check skipped: cache missing")
    print("[droid-pipeline] model sensitivity:")
    for key, value in sens.items():
        if key == "n_transitions":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6f}")

    ok = True
    ok = ok and total > 0
    ok = ok and grip_err < 1e-5
    if cache_present:
        ok = ok and cache_err < 1e-5
    ok = ok and sens["fact_vs_zero"] > 1e-6
    ok = ok and sens["open_vs_close"] > 1e-6

    print("  PASS: DROID pipeline plumbing looks consistent" if ok else
          "  FAIL: DROID pipeline plumbing is suspect")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/diagnostic_droid.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--max-transitions", type=int, default=512)
    p.add_argument("--cache-trajectories", type=int, default=5)
    args = p.parse_args()
    sys.exit(main(args.config, args.model, args.max_transitions, args.cache_trajectories))
