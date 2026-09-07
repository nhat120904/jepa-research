#!/usr/bin/env python3
"""Replay one Stage-A anchor and collect true swapped-order branch outcomes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def require_compute_node() -> None:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("PushT rendering must be run through scripts/slurm_collect.sh")


def load_tensor(path: Path) -> np.ndarray:
    return torch.load(path, map_location="cpu", weights_only=False).detach().cpu().numpy()


def add_original_source(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"official DINO-WM checkout not found: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def physics_state(env: Any) -> np.ndarray:
    return np.asarray(
        [
            *env.agent.position,
            *env.agent.velocity,
            *env.block.position,
            *env.block.velocity,
            env.block.angle,
            env.block.angular_velocity,
        ],
        dtype=np.float64,
    )


def make_env(dino_wm_root: Path, *, shape: str):
    """Build the environment from gaoyuezhou/dino_wm, never jepa-wms."""

    add_original_source(dino_wm_root)
    # Importing ``env.pusht`` executes the authors' env/__init__.py, which
    # eagerly imports the unrelated PointMaze MuJoCo stack.  Loading this exact
    # source file avoids that optional dependency without modifying the
    # original checkout or changing PushT dynamics.
    module_name = "order_jepa_original_pusht_env"
    module = sys.modules.get(module_name)
    if module is None:
        source = dino_wm_root / "env" / "pusht" / "pusht_env.py"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load original PushT environment: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    PushTEnv = module.PushTEnv

    return PushTEnv(
        with_velocity=True,
        with_target=True,
        relative=True,
        action_scale=100.0,
        render_size=224,
        shape=shape,
    )


def reset_and_replay(
    dino_wm_root: Path,
    shape: str,
    initial_state: np.ndarray,
    warmup_actions: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    env = make_env(dino_wm_root, shape=shape)
    env.seed(seed)
    env.reset_to_state = initial_state
    observation, _ = env.reset()
    observations = [observation]
    infos: list[dict[str, Any]] = []
    for action in warmup_actions:
        observation, _, _, info = env.step(action)
        observations.append(observation)
        infos.append(info)
    return env, observations, infos


def action_block(actions: np.ndarray, ref: list[int], block_steps: int) -> np.ndarray:
    episode, start = int(ref[0]), int(ref[1])
    block = actions[episode, start : start + block_steps]
    if block.shape != (block_steps, 2):
        raise ValueError(f"invalid action block {ref}: {block.shape}")
    return block


def rollout_candidate(
    dino_wm_root: Path,
    shape: str,
    initial_state: np.ndarray,
    warmup_actions: np.ndarray,
    candidate_actions: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    env, observations, _ = reset_and_replay(
        dino_wm_root, shape, initial_state, warmup_actions, seed=seed
    )
    anchor_physics = physics_state(env)
    anchor_visual = np.asarray(observations[-1]["visual"], dtype=np.uint8)
    contacts = []
    endpoint_obs = observations[-1]
    endpoint_info = {"state": env._get_obs()}
    for action in candidate_actions.reshape(-1, 2):
        endpoint_obs, _, _, endpoint_info = env.step(action)
        contacts.append(int(endpoint_info.get("n_contacts", 0)))
    result = {
        "anchor_physics": anchor_physics,
        "anchor_visual": anchor_visual,
        "endpoint_visual": np.asarray(endpoint_obs["visual"], dtype=np.uint8),
        "endpoint_proprio": np.asarray(endpoint_obs["proprio"], dtype=np.float32),
        "endpoint_state": np.asarray(endpoint_info["state"], dtype=np.float32),
        "endpoint_physics": physics_state(env),
        "contacts": np.asarray(contacts, dtype=np.int16),
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-index", type=int, required=True)
    parser.add_argument(
        "--dino-wm-root",
        type=Path,
        required=True,
        help="checkout of the authors' gaoyuezhou/dino_wm repository",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    require_compute_node()
    manifest = json.loads(args.manifest.read_text())
    source_commit = git_value(args.dino_wm_root, "rev-parse", "HEAD")
    source_origin = git_value(args.dino_wm_root, "remote", "get-url", "origin")
    source_dirty = git_value(
        args.dino_wm_root, "status", "--porcelain", "--untracked-files=no"
    )
    if source_commit != manifest["source_commit"]:
        raise ValueError(
            f"DINO-WM source {source_commit} != manifest pin {manifest['source_commit']}"
        )
    if "gaoyuezhou/dino_wm" not in source_origin or source_dirty:
        raise ValueError("collector requires a clean checkout of gaoyuezhou/dino_wm")
    record = manifest["anchors"][args.anchor_index]
    dataset = Path(manifest["dataset"])
    block_steps = int(manifest["block_steps"])
    num_hist = int(manifest["num_hist"])

    states = load_tensor(dataset / "states.pth")
    velocities = load_tensor(dataset / "velocities.pth")
    actions = load_tensor(dataset / "rel_actions.pth").astype(np.float32) / 100.0
    shapes_path = dataset / "shapes.pkl"
    if shapes_path.exists():
        with shapes_path.open("rb") as handle:
            shapes = pickle.load(handle)
    else:
        # This is also the fallback in the original DINO-WM PushTDataset.
        shapes = ["T"] * len(states)
    episode, step = int(record["episode"]), int(record["step"])
    shape = str(shapes[episode])
    initial_state = np.concatenate([states[episode, 0, :5], velocities[episode, 0, :2]])
    warmup_actions = actions[episode, :step]

    base_env, observations, warmup_infos = reset_and_replay(
        args.dino_wm_root,
        shape,
        initial_state,
        warmup_actions,
        seed=int(record["sim_seed"]),
    )
    hist_indices = [step - k * block_steps for k in reversed(range(num_hist))]
    history_visual = np.stack(
        [np.asarray(observations[index]["visual"], dtype=np.uint8) for index in hist_indices]
    )
    history_proprio = np.stack(
        [np.asarray(observations[index]["proprio"], dtype=np.float32) for index in hist_indices]
    )
    anchor_state = np.asarray(base_env._get_obs(), dtype=np.float32)
    anchor_physics_reference = physics_state(base_env)
    warmup_contacts = np.asarray(
        [int(info.get("n_contacts", 0)) for info in warmup_infos[-block_steps:]], dtype=np.int16
    )
    base_env.close()

    source_actions = actions[episode, step : step + 2 * block_steps].reshape(2, block_steps, 2)
    goal = rollout_candidate(
        args.dino_wm_root,
        shape,
        initial_state,
        warmup_actions,
        source_actions,
        seed=int(record["sim_seed"]),
    )

    candidate_actions = []
    candidate_results = []
    repeat_endpoint_states = []
    for candidate_index, candidate in enumerate(record["candidates"]):
        chunks = np.stack(
            [action_block(actions, ref, block_steps) for ref in candidate["blocks"]]
        ).astype(np.float32)
        candidate_actions.append(chunks)
        repeats = [
            rollout_candidate(
                args.dino_wm_root,
                shape,
                initial_state,
                warmup_actions,
                chunks,
                seed=int(record["sim_seed"]),
            )
            for _ in range(args.repeats)
        ]
        candidate_results.append(repeats[0])
        repeat_endpoint_states.append(np.stack([item["endpoint_physics"] for item in repeats]))

    anchor_physics = np.stack([item["anchor_physics"] for item in candidate_results])
    anchor_visual = np.stack([item["anchor_visual"] for item in candidate_results])
    reset_physics_max_abs = float(np.max(np.abs(anchor_physics - anchor_physics_reference)))
    reset_pixel_max_abs = int(
        np.max(np.abs(anchor_visual.astype(np.int16) - history_visual[-1].astype(np.int16)))
    )

    arrays = {
        "history_visual": history_visual,
        "history_proprio": history_proprio,
        "anchor_state": anchor_state,
        "anchor_physics": anchor_physics_reference,
        "warmup_contacts": warmup_contacts,
        "goal_visual": goal["endpoint_visual"],
        "goal_proprio": goal["endpoint_proprio"],
        "goal_state": goal["endpoint_state"],
        "source_actions": source_actions,
        "candidate_actions": np.stack(candidate_actions),
        "candidate_visual": np.stack([item["endpoint_visual"] for item in candidate_results]),
        "candidate_proprio": np.stack([item["endpoint_proprio"] for item in candidate_results]),
        "candidate_state": np.stack([item["endpoint_state"] for item in candidate_results]),
        "candidate_physics": np.stack([item["endpoint_physics"] for item in candidate_results]),
        "candidate_contacts": np.stack([item["contacts"] for item in candidate_results]),
        "repeat_endpoint_physics": np.stack(repeat_endpoint_states),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / f"anchor_{int(record['anchor_id']):04d}.npz"
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        "schema": "order-jepa-pusht-branches-v2-original-dino-wm",
        "implementation": "gaoyuezhou/dino_wm",
        "source_commit": manifest["source_commit"],
        "source_origin": source_origin,
        "anchor_id": int(record["anchor_id"]),
        "episode": episode,
        "step": step,
        "shape": shape,
        "sim_seed": int(record["sim_seed"]),
        "block_steps": block_steps,
        "num_hist": num_hist,
        "candidate_kinds": [candidate["kind"] for candidate in record["candidates"]],
        "order_pairs": record["order_pairs"],
        "reset_physics_max_abs": reset_physics_max_abs,
        "reset_pixel_max_abs": reset_pixel_max_abs,
        "repeat_endpoint_max_abs": float(
            np.max(np.ptp(arrays["repeat_endpoint_physics"], axis=1))
        ),
        "npz": npz_path.name,
    }
    meta_path = npz_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
