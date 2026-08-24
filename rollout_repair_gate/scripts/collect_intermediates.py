#!/usr/bin/env python3
"""Replay persisted CEM populations and save every block-boundary frame."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "diagnosis"
sys.path.insert(0, str(REPO))

from rollout_repair_gate.core import normalized_to_raw, split_for_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "physical_search_distillation/outputs/h0/manifest.json",
    )
    parser.add_argument(
        "--population-dir", type=Path,
        default=REPO / "physical_search_distillation/outputs/h0/populations",
    )
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--latent-atol", type=float, default=2e-4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


@torch.inference_mode()
def encode_images(
    model: Any, images: np.ndarray, transform: Any, audit: Any, batch_size: int
) -> np.ndarray:
    device = str(next(model.parameters()).device)
    flat = np.asarray(images)
    output: list[np.ndarray] = []
    for start in range(0, len(flat), batch_size):
        pixels = audit.transform_images(flat[start : start + batch_size], transform, device)
        embedding = model.encode({"pixels": pixels})["emb"][:, -1].float()
        output.append(embedding.cpu().numpy())
    return np.concatenate(output)


def replay_boundaries(
    raw_env: Any,
    init_row: dict[str, Any],
    goal_row: dict[str, Any],
    actions_raw: np.ndarray,
    audit: Any,
) -> dict[str, np.ndarray]:
    """Execute candidates once and render after each planner action block."""

    target = audit.goal_field(goal_row, "block_0_pos")
    n_candidates, horizon, _, _ = actions_raw.shape
    frames = np.empty((n_candidates, horizon, 224, 224, 3), dtype=np.uint8)
    physical = np.empty((n_candidates, horizon), dtype=np.float32)
    valid = np.zeros((n_candidates, horizon), dtype=bool)
    executed = np.zeros(n_candidates, dtype=np.int16)
    qpos = np.empty((n_candidates, horizon, raw_env._data.qpos.size), dtype=np.float32)
    qvel = np.empty((n_candidates, horizon, raw_env._data.qvel.size), dtype=np.float32)

    for candidate, sequence in enumerate(actions_raw):
        corrected.restore_complete(
            raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit
        )
        terminated = False
        last_frame: np.ndarray | None = None
        for step, block in enumerate(sequence):
            if not terminated:
                valid[candidate, step] = True
                for action in block:
                    _, _, done, truncated, _ = raw_env.step(action)
                    executed[candidate] += 1
                    if done or truncated:
                        terminated = True
                        break
                last_frame = audit.resize_render(raw_env.render())
            if last_frame is None:
                raise RuntimeError("no frame rendered for first horizon")
            frames[candidate, step] = last_frame
            physical[candidate, step] = audit.cube_distance(raw_env, target)
            qpos[candidate, step] = raw_env._data.qpos
            qvel[candidate, step] = raw_env._data.qvel
    return {
        "frames": frames,
        "physical_distance_m": physical,
        "valid_horizon": valid,
        "executed_steps": executed,
        "qpos": qpos,
        "qvel": qvel,
    }


# Filled after loading the corrected evaluator. Keeping replay_boundaries easy to
# unit-test avoids importing MuJoCo/stable-worldmodel on the login node.
corrected: Any = None


def main() -> None:
    global corrected

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("intermediate collection requires a GPU Slurm allocation")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "rrg_audit")
    corrected = load_module(
        DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "rrg_corrected"
    )
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    row = manifest[args.snapshot_index]
    snapshot = audit.Snapshot(**row)
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("manifest order/index mismatch")

    source_path = args.population_dir / f"snapshot_{snapshot.order:03d}/populations.npz"
    with np.load(source_path) as source:
        source_arrays = {key: np.asarray(source[key]) for key in source.files}
    actions_normalized = source_arrays["actions_normalized"].astype(np.float32)
    if actions_normalized.shape[:2] != (2, 96):
        raise RuntimeError(f"unexpected locked population shape {actions_normalized.shape}")

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = StandardScaler().fit(action_data)

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)
    world, raw_env, visual_hash, visual_shapes = corrected.make_world(swm, snapshot)
    try:
        raw_dim = int(np.prod(world.envs.single_action_space.shape))
        actions_raw = normalized_to_raw(
            actions_normalized, scaler, args.horizon, args.action_block, raw_dim
        )
        true = [
            replay_boundaries(raw_env, init_row, goal_row, population, audit)
            for population in actions_raw
        ]
        rendered_goal = corrected.render_state(
            raw_env,
            corrected.resolve_goal(goal_row, "qpos"),
            corrected.resolve_goal(goal_row, "qvel"),
            goal_row,
            audit,
        )

        base = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
        config = swm.PlanConfig(
            horizon=args.horizon,
            receding_horizon=args.horizon,
            action_block=args.action_block,
            history_len=1,
            warm_start=True,
        )
        preparation_solver = swm.planning.CEMSolver(
            cost=base,
            batch_size=1,
            num_samples=actions_normalized.shape[1],
            n_steps=1,
            topk=1,
            device="cuda",
            seed=0,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=preparation_solver,
            config=config,
            process={"action": scaler},
            transform={"pixels": transform, "goal": transform},
        )
        policy.set_env(world.envs)
        raw_info = {
            "pixels": np.asarray(init_row["pixels"])[None, None],
            "goal": np.asarray(goal_row["goal"])[None, None],
            "action": np.full((1, 1, raw_dim), np.nan, dtype=np.float32),
        }
        predicted, current, dataset_goal = [], [], []
        for population in range(2):
            prepared = policy._prepare_info(raw_info)
            # CEMSolver.solve inserts the candidate/sample axis before calling
            # the cost evaluator.  We bypass solve here because the candidate
            # populations are locked, so reproduce that expansion exactly.
            # Without it, default_goal_encode interprets the RGB channel axis
            # as LeWM's temporal axis and sends a 3-D tensor into ViT.
            expanded: dict[str, Any] = {}
            for key, value in prepared.items():
                if torch.is_tensor(value):
                    value = value.cuda()
                    expanded[key] = value.unsqueeze(1).expand(
                        value.shape[0], actions_normalized.shape[1], *value.shape[1:]
                    )
                elif isinstance(value, np.ndarray):
                    expanded[key] = np.repeat(
                        value[:, None, ...], actions_normalized.shape[1], axis=1
                    )
                else:
                    expanded[key] = value
            action_tensor = torch.from_numpy(actions_normalized[population]).cuda()[None]
            rolled = base._rollout(expanded, action_tensor)
            pred = rolled["predicted_emb"][0, :, -args.horizon :].float().cpu().numpy()
            predicted.append(pred)
            current.append(rolled["emb"][0, 0, -1].float().cpu().numpy())
            dataset_goal.append(rolled["goal_emb"][0, -1].float().cpu().numpy())
    finally:
        world.close()

    frames = np.stack([item["frames"] for item in true])
    flat_frames = frames.reshape(-1, *frames.shape[-3:])
    future_embeddings = encode_images(model, flat_frames, transform, audit, args.encode_batch)
    future_embeddings = future_embeddings.reshape(*frames.shape[:3], -1)
    rendered_goal_embedding = encode_images(
        model, rendered_goal[None], transform, audit, args.encode_batch
    )[0]
    predicted_embeddings = np.stack(predicted)
    current_embedding = np.stack(current)
    dataset_goal_embedding = np.stack(dataset_goal)

    source_endpoint = source_arrays["predicted_endpoint"]
    replay_error = float(np.max(np.abs(predicted_embeddings[:, :, -1] - source_endpoint)))
    if replay_error > args.latent_atol:
        raise RuntimeError(
            f"frozen rollout replay mismatch {replay_error:.3g} > {args.latent_atol}"
        )
    if not np.array_equal(
        np.stack([item["executed_steps"] for item in true]),
        source_arrays["executed_steps"],
    ):
        raise RuntimeError("physical executed-step replay mismatch")
    physical_error = float(
        np.max(
            np.abs(
                np.stack([item["physical_distance_m"][:, -1] for item in true])
                - source_arrays["physical_distance_m"]
            )
        )
    )
    if physical_error > 1e-5:
        raise RuntimeError(f"physical replay mismatch {physical_error:.3g}")

    # Store z0 once per population; training expands it over candidates.
    out = args.out_dir / f"snapshot_{snapshot.order:03d}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "intermediates.npz",
        actions_normalized=actions_normalized,
        actions_raw=actions_raw.astype(np.float32),
        true_frames=frames,
        true_future_embeddings=future_embeddings.astype(np.float32),
        predicted_future_embeddings=predicted_embeddings.astype(np.float32),
        current_embedding=current_embedding.astype(np.float32),
        dataset_goal_embedding=dataset_goal_embedding.astype(np.float32),
        rendered_goal_embedding=rendered_goal_embedding.astype(np.float32),
        physical_distance_m=np.stack([item["physical_distance_m"] for item in true]),
        valid_horizon=np.stack([item["valid_horizon"] for item in true]),
        executed_steps=np.stack([item["executed_steps"] for item in true]),
        qpos=np.stack([item["qpos"] for item in true]),
        qvel=np.stack([item["qvel"] for item in true]),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
    )
    summary = {
        "snapshot": snapshot.order,
        "episode": snapshot.episode,
        "split": split_for_order(snapshot.order),
        "source": str(source_path),
        "source_sha256": digest(actions_normalized),
        "frame_boundaries": list(range(1, args.horizon + 1)),
        "primitive_frames_persisted": False,
        "num_populations": int(frames.shape[0]),
        "num_candidates": int(frames.shape[1]),
        "valid_target_fraction": float(
            np.stack([item["valid_horizon"] for item in true]).mean()
        ),
        "rollout_replay_max_abs": replay_error,
        "physical_replay_max_abs": physical_error,
        "visual_signature": visual_hash,
        "visual_signature_shapes": visual_shapes,
        "arrays_sha256": {
            "true_frames": digest(frames),
            "future_embeddings": digest(future_embeddings),
            "physical": digest(np.stack([item["physical_distance_m"] for item in true])),
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
