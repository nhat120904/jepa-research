#!/usr/bin/env python3
"""Collect and score the preregistered ORDER Stage-A audit on LeWM Reacher."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from order_jepa.core import (  # noqa: E402
    angular_distance,
    order_vector_metrics,
    reacher_physical_cost,
    reacher_success,
    selection_regret,
)
from order_jepa.lewm_reacher import (  # noqa: E402
    EXPECTED_ACTION_BLOCK,
    EXPECTED_HISTORY,
    EXPECTED_IMAGE_SIZE,
    EXPECTED_RAW_ACTION_DIM,
    OFFICIAL_REPO,
    build_provenance,
)


ACTION_STD = np.sqrt(1.0 / 3.0)  # exact population std for Uniform[-1, 1]
JOINT_TOLERANCE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-worldmodel-source", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=200)
    parser.add_argument("--anchors-per-episode", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--ordinary-candidates", type=int, default=32)
    parser.add_argument("--order-pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_transform():
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=EXPECTED_IMAGE_SIZE),
        ]
    )


def transform_images(images: np.ndarray, transform, device: str) -> torch.Tensor:
    from torchvision import tv_tensors

    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    chw = np.transpose(images, (0, 3, 1, 2))
    return torch.stack(
        [transform(tv_tensors.Image(image)) for image in chw]
    ).to(device)


def episode_actions(seed: int, episode: int, steps: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed + 1_000_003 * episode)
    return rng.uniform(-1.0, 1.0, size=(steps, EXPECTED_RAW_ACTION_DIM)).astype(
        np.float32
    )


def physics_state(raw_env) -> np.ndarray:
    return np.asarray(raw_env.env.physics.get_state(), dtype=np.float64).copy()


def restore_physics(raw_env, state: np.ndarray, reset_seed: int) -> None:
    raw_env.reset(seed=reset_seed)
    raw_env.env.physics.set_state(np.asarray(state, dtype=np.float64))
    raw_env.env.physics.forward()


def render(raw_env) -> np.ndarray:
    return np.asarray(
        raw_env.render(width=EXPECTED_IMAGE_SIZE, height=EXPECTED_IMAGE_SIZE)
    ).copy()


def step_sequence(raw_env, actions: np.ndarray) -> None:
    for action in np.asarray(actions).reshape(-1, EXPECTED_RAW_ACTION_DIM):
        raw_env.step(action.astype(np.float32))


def build_anchor(raw_env, args: argparse.Namespace, anchor_id: int) -> dict:
    episode = anchor_id // args.anchors_per_episode
    slot = anchor_id % args.anchors_per_episode
    reset_seed = args.seed + 50_000 + episode
    schedule = episode_actions(args.seed, episode)
    context_start = 10 + 20 * slot

    raw_env.reset(seed=reset_seed)
    step_sequence(raw_env, schedule[:context_start])
    frames = [render(raw_env)]
    history_blocks = []
    for block_id in range(EXPECTED_HISTORY - 1):
        lo = context_start + block_id * EXPECTED_ACTION_BLOCK
        hi = lo + EXPECTED_ACTION_BLOCK
        block = schedule[lo:hi]
        history_blocks.append(block.copy())
        step_sequence(raw_env, block)
        frames.append(render(raw_env))

    state = physics_state(raw_env)
    qpos = np.asarray(raw_env.env.physics.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(raw_env.env.physics.data.qvel, dtype=np.float64).copy()
    pixel = frames[-1]

    restore_physics(raw_env, state, reset_seed)
    reset_state = physics_state(raw_env)
    reset_pixel = render(raw_env)
    return {
        "anchor_id": anchor_id,
        "episode": episode,
        "step": context_start + (EXPECTED_HISTORY - 1) * EXPECTED_ACTION_BLOCK,
        "reset_seed": reset_seed,
        "frames": np.stack(frames),
        "history_blocks": np.stack(history_blocks),
        "state": state,
        "qpos": qpos,
        "qvel": qvel,
        "reset_physics_max_abs": float(np.max(np.abs(reset_state - state))),
        "reset_pixel_max_abs": int(
            np.max(np.abs(reset_pixel.astype(np.int16) - pixel.astype(np.int16)))
        ),
    }


def build_candidates(args: argparse.Namespace, anchor_id: int) -> tuple[np.ndarray, list[str], list[dict]]:
    rng = np.random.default_rng(args.seed + 9_000_001 + anchor_id)
    shape = (5, EXPECTED_ACTION_BLOCK, EXPECTED_RAW_ACTION_DIM)
    witness = rng.uniform(-0.8, 0.8, size=shape)
    actions: list[np.ndarray] = []
    kinds: list[str] = []
    sigmas = (0.02, 0.05, 0.10, 0.25)
    per_sigma = args.ordinary_candidates // len(sigmas)
    for sigma in sigmas:
        count = per_sigma
        if sigma == sigmas[-1]:
            count += args.ordinary_candidates - per_sigma * len(sigmas)
        for _ in range(count):
            candidate = np.clip(witness + rng.normal(0.0, sigma, size=shape), -1.0, 1.0)
            actions.append(candidate.astype(np.float32))
            kinds.append(f"ordinary_sigma_{sigma:.2f}")

    witness_index = len(actions)
    actions.append(witness.astype(np.float32))
    kinds.append("witness")
    pairs = []
    for pair_id in range(args.order_pairs):
        base = rng.uniform(-0.9, 0.9, size=shape).astype(np.float32)
        ij = base.copy()
        ji = base.copy()
        ji[-2], ji[-1] = base[-1].copy(), base[-2].copy()
        ij_index = len(actions)
        actions.extend([ij, ji])
        kinds.extend(["order_pair", "order_pair"])
        pairs.append({"pair_id": pair_id, "ij": ij_index, "ji": ij_index + 1})
    assert witness_index == args.ordinary_candidates
    return np.stack(actions), kinds, pairs


def execute_candidates(raw_env, anchor: dict, actions: np.ndarray) -> dict:
    images, qpos, qvel, states = [], [], [], []
    for candidate in actions:
        restore_physics(raw_env, anchor["state"], anchor["reset_seed"])
        step_sequence(raw_env, candidate)
        images.append(render(raw_env))
        qpos.append(np.asarray(raw_env.env.physics.data.qpos, dtype=np.float64).copy())
        qvel.append(np.asarray(raw_env.env.physics.data.qvel, dtype=np.float64).copy())
        states.append(physics_state(raw_env))
    return {
        "images": np.stack(images),
        "qpos": np.stack(qpos),
        "qvel": np.stack(qvel),
        "states": np.stack(states),
    }


@torch.inference_mode()
def encode_and_rollout(model, transform, anchor: dict, actions: np.ndarray, goal_image: np.ndarray, device: str):
    context_pixels = transform_images(anchor["frames"], transform, device)
    endpoint_pixels = transform_images(actions["images"], transform, device)
    goal_pixels = transform_images(goal_image, transform, device)

    history = anchor["history_blocks"].reshape(EXPECTED_HISTORY - 1, -1) / ACTION_STD
    future = actions["raw"].reshape(len(actions["raw"]), 5, -1) / ACTION_STD
    action_history = torch.as_tensor(history, dtype=torch.float32, device=device)
    action_history = action_history[None, None].expand(1, len(future), -1, -1)
    action_sequence = torch.as_tensor(future, dtype=torch.float32, device=device)[None]
    info = {
        "pixels": context_pixels[None, None],
        "action_history": action_history,
    }
    predicted = model.rollout(info, action_sequence, history_size=EXPECTED_HISTORY)[
        "predicted_emb"
    ][0, :, -1]
    true = model.encode({"pixels": endpoint_pixels[:, None]})["emb"][:, -1]
    goal = model.encode({"pixels": goal_pixels[:, None]})["emb"][:, -1]
    return (
        predicted.float().cpu().numpy(),
        true.float().cpu().numpy(),
        goal[0].float().cpu().numpy(),
    )


def score_anchor(raw_env, model, transform, provenance: dict, args: argparse.Namespace, anchor_id: int) -> dict:
    anchor = build_anchor(raw_env, args, anchor_id)
    raw_actions, kinds, pairs = build_candidates(args, anchor_id)
    endpoints = execute_candidates(raw_env, anchor, raw_actions)
    goal_index = args.ordinary_candidates
    goal_image = endpoints["images"][goal_index]
    goal_qpos = endpoints["qpos"][goal_index]

    predicted_z, true_z, goal_z = encode_and_rollout(
        model,
        transform,
        anchor,
        {"raw": raw_actions, "images": endpoints["images"]},
        goal_image,
        args.device,
    )
    predicted_cost = np.sum(np.square(predicted_z - goal_z), axis=-1)
    true_cost = np.sum(np.square(true_z - goal_z), axis=-1)
    physical_cost = reacher_physical_cost(endpoints["qpos"], goal_qpos)
    success = reacher_success(endpoints["qpos"], goal_qpos)

    true_selected, true_regret = selection_regret(true_cost, physical_cost)
    pred_selected, pred_regret = selection_regret(predicted_cost, physical_cost)
    candidate_rows = []
    for index, kind in enumerate(kinds):
        candidate_rows.append(
            {
                "index": index,
                "kind": kind,
                "predicted_latent_cost": float(predicted_cost[index]),
                "true_latent_cost": float(true_cost[index]),
                "physical_cost": float(physical_cost[index]),
                "success": bool(success[index]),
                "max_joint_error_rad": float(
                    np.max(angular_distance(endpoints["qpos"][index], goal_qpos))
                ),
            }
        )

    pair_rows = []
    repeat_endpoint_max_abs = 0.0
    for pair in pairs:
        ij, ji = pair["ij"], pair["ji"]
        metrics = order_vector_metrics(true_z[ij], true_z[ji], predicted_z[ij], predicted_z[ji])
        repeat_noise = []
        for index in (ij, ji):
            restore_physics(raw_env, anchor["state"], anchor["reset_seed"])
            step_sequence(raw_env, raw_actions[index])
            repeated = np.asarray(raw_env.env.physics.data.qpos, dtype=np.float64).copy()
            raw_diff = float(np.max(np.abs(repeated - endpoints["qpos"][index])))
            repeat_endpoint_max_abs = max(repeat_endpoint_max_abs, raw_diff)
            repeat_noise.append(
                float(np.max(angular_distance(repeated, endpoints["qpos"][index])) / JOINT_TOLERANCE)
            )
        qpos_effect = float(
            np.max(angular_distance(endpoints["qpos"][ij], endpoints["qpos"][ji]))
            / JOINT_TOLERANCE
        )
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "ij": ij,
                "ji": ji,
                **asdict(metrics),
                "physical_delta": float(physical_cost[ij] - physical_cost[ji]),
                "true_latent_delta": float(true_cost[ij] - true_cost[ji]),
                "predicted_latent_delta": float(predicted_cost[ij] - predicted_cost[ji]),
                "object_effect": qpos_effect,
                "qpos_order_effect": qpos_effect,
                "repeat_object_noise": float(max(repeat_noise)),
            }
        )

    return {
        "schema": "order-jepa-reacher-stage-a-scores-v1-official-lewm",
        "anchor_id": anchor_id,
        "episode": anchor["episode"],
        "step": anchor["step"],
        "planning_context_frames": EXPECTED_HISTORY,
        "action_block": EXPECTED_ACTION_BLOCK,
        "action_scaler": {
            "source": "analytic Uniform[-1,1] population statistics",
            "mean": 0.0,
            "scale": float(ACTION_STD),
        },
        "provenance": provenance,
        "reset": {
            "physics_max_abs": anchor["reset_physics_max_abs"],
            "pixel_max_abs": anchor["reset_pixel_max_abs"],
            "repeat_endpoint_max_abs": repeat_endpoint_max_abs,
        },
        "selection": {
            "physical_oracle_index": int(np.argmin(physical_cost)),
            "true_latent_index": true_selected,
            "predicted_latent_index": pred_selected,
            "true_latent_regret": true_regret,
            "predicted_latent_regret": pred_regret,
            "dynamics_excess_regret": pred_regret - true_regret,
            "oracle_has_success": bool(np.any(success)),
            "true_latent_success": bool(success[true_selected]),
            "predicted_latent_success": bool(success[pred_selected]),
        },
        "candidates": candidate_rows,
        "order_pairs": pair_rows,
    }


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Reacher simulation/model scoring must run on Slurm")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but no GPU is visible")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    if args.anchor_count % args.anchors_per_episode:
        raise ValueError("anchor-count must be divisible by anchors-per-episode")

    import stable_worldmodel as swm

    recorded = json.loads(args.provenance.read_text())
    checkpoint_dir = args.stablewm_home / "checkpoints/models--quentinll--lewm-reacher"
    current = build_provenance(
        args.stable_worldmodel_source,
        checkpoint_dir,
        checkpoint_revision=recorded.get("checkpoint_revision"),
    ).to_dict()
    if current != recorded:
        raise RuntimeError("checkpoint/source provenance changed after preparation")

    model = swm.wm.utils.load_pretrained(OFFICIAL_REPO).to(args.device).eval()
    model.requires_grad_(False)
    transform = make_transform()
    world = swm.World(
        "swm/ReacherDMControl-v0",
        num_envs=1,
        image_shape=(EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE),
        max_episode_steps=1000,
        task="qpos_match",
    )
    raw_env = world.envs.envs[0].unwrapped
    try:
        if tuple(raw_env.action_space.shape) != (EXPECTED_RAW_ACTION_DIM,):
            raise RuntimeError(f"unexpected Reacher action space: {raw_env.action_space}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for anchor_id in range(args.shard_index, args.anchor_count, args.num_shards):
            out = args.out_dir / f"anchor_{anchor_id:04d}.json"
            payload = score_anchor(raw_env, model, transform, current, args, anchor_id)
            out.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "anchor": anchor_id,
                        "pairs": len(payload["order_pairs"]),
                        "reset": payload["reset"],
                        "out": str(out),
                    }
                )
            )
    finally:
        world.close()


if __name__ == "__main__":
    main()
