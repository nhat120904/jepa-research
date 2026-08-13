#!/usr/bin/env python3
"""Corrected, renderer-controlled OGBench true-endpoint audit.

The script reuses the locked Stage-0 final CEM candidates, but evaluates them
from a fully reset MuJoCo state in two independently compiled worlds.  Both
dataset-goal and same-renderer-goal true-endpoint costs are reported.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/ogb_stage0/audit_locked/manifest.json"),
    )
    parser.add_argument("--candidate-artifacts", type=Path, required=True)
    parser.add_argument("--reference-physical-shards", type=Path, required=True)
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--physical-atol", type=float, default=1e-5)
    parser.add_argument("--latent-atol", type=float, default=1e-5)
    parser.add_argument("--domain-ratio-max", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_stage0_module() -> Any:
    source = Path(__file__).with_name("72_ogb_stage0_candidate_audit.py")
    spec = importlib.util.spec_from_file_location(
        "ogb_stage0_audit_for_corrected_endpoint", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_goal(goal_row: dict[str, Any], field: str) -> np.ndarray:
    key = f"goal_{field}"
    if key not in goal_row:
        raise KeyError(f"missing {key}; keys={sorted(goal_row)}")
    return np.asarray(goal_row[key])


def restore_complete(
    raw_env: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_row: dict[str, Any],
    audit: Any,
) -> None:
    """Reset all MjData fields, then install one dataset state."""

    import mujoco

    mujoco.mj_resetData(raw_env._model, raw_env._data)
    raw_env._reset_next_step = False
    raw_env.set_state(qpos=np.asarray(qpos), qvel=np.asarray(qvel))
    raw_env.set_target_pos(
        cube_id=0,
        target_pos=audit.goal_field(goal_row, "block_0_pos"),
        target_quat=audit.goal_field(goal_row, "block_0_quat"),
    )
    mujoco.mj_forward(raw_env._model, raw_env._data)
    if hasattr(raw_env, "_prev_qpos"):
        raw_env._prev_qpos = raw_env._data.qpos.copy()
    if hasattr(raw_env, "_prev_qvel"):
        raw_env._prev_qvel = raw_env._data.qvel.copy()
    raw_env.post_step()


def render_state(
    raw_env: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_row: dict[str, Any],
    audit: Any,
) -> np.ndarray:
    restore_complete(raw_env, qpos, qvel, goal_row, audit)
    return audit.resize_render(raw_env.render())


def model_visual_signature(model: Any) -> tuple[str, dict[str, list[int]]]:
    """Hash static arrays that determine camera/lighting/material appearance."""

    names = (
        "cam_pos",
        "cam_quat",
        "cam_mat0",
        "cam_fovy",
        "light_pos",
        "light_dir",
        "light_diffuse",
        "light_ambient",
        "light_specular",
        "geom_rgba",
        "mat_rgba",
        "tex_rgb",
        "tex_adr",
        "tex_height",
        "tex_width",
    )
    digest = hashlib.sha256()
    shapes: dict[str, list[int]] = {}
    for name in names:
        if not hasattr(model, name):
            continue
        value = np.ascontiguousarray(np.asarray(getattr(model, name)))
        shapes[name] = list(value.shape)
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest(), shapes


def make_world(swm: Any, snapshot: Any) -> tuple[Any, Any, str, dict[str, list[int]]]:
    import mujoco

    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=1,
        max_episode_steps=50,
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
        image_shape=(224, 224),
    )
    raw_env = world.envs.envs[0].unwrapped
    # Apply the baseline visual configuration once.  Its contaminated dynamic
    # state is discarded by restore_complete before every measurement.
    raw_env.reset(seed=snapshot.reset_seed, options={"variation": []})
    raw_env._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    signature, shapes = model_visual_signature(raw_env._model)
    return world, raw_env, signature, shapes


def rollout_population(
    raw_env: Any,
    init_row: dict[str, Any],
    goal_row: dict[str, Any],
    actions_raw: np.ndarray,
    audit: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = audit.goal_field(goal_row, "block_0_pos")
    endpoints: list[np.ndarray] = []
    physical = np.empty(len(actions_raw), dtype=np.float64)
    success = np.zeros(len(actions_raw), dtype=bool)
    executed = np.empty(len(actions_raw), dtype=np.int64)
    for index, sequence in enumerate(actions_raw):
        restore_complete(
            raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit
        )
        terminated = False
        n_executed = 0
        for action in sequence.reshape(-1, sequence.shape[-1]):
            _, _, terminated, truncated, _ = raw_env.step(action)
            n_executed += 1
            if terminated or truncated:
                break
        endpoints.append(audit.resize_render(raw_env.render()))
        physical[index] = audit.cube_distance(raw_env, target)
        success[index] = bool(terminated) or physical[index] <= 0.04
        executed[index] = n_executed
    return np.stack(endpoints), physical, success, executed


@torch.inference_mode()
def encode_images(
    model: Any,
    images: np.ndarray,
    transform: Any,
    batch_size: int,
) -> np.ndarray:
    device = str(next(model.parameters()).device)
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    output: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        pixels = load_stage0_transform_images(
            images[start : start + batch_size], transform, device
        )
        emb = model.encode({"pixels": pixels})["emb"][:, -1].float()
        output.append(emb.reshape(len(emb), -1).cpu().numpy())
    return np.concatenate(output)


# Set in main after loading the Stage-0 module. Keeping this adapter at module
# scope lets the encoding helper remain easy to unit-test without importing the
# heavyweight stable-worldmodel stack on the login node.
load_stage0_transform_images: Any = None


def squared_l2(embeddings: np.ndarray, goal: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64).reshape(1, -1)
    return np.square(embeddings - goal).sum(axis=1)


def domain_diagnostics(
    dataset_init_emb: np.ndarray,
    rendered_init_emb: np.ndarray,
    dataset_goal_emb: np.ndarray,
    rendered_goal_emb: np.ndarray,
    ratio_max: float,
) -> dict[str, Any]:
    di = np.asarray(dataset_init_emb).reshape(-1)
    ri = np.asarray(rendered_init_emb).reshape(-1)
    dg = np.asarray(dataset_goal_emb).reshape(-1)
    rg = np.asarray(rendered_goal_emb).reshape(-1)

    def distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.square(a.astype(np.float64) - b.astype(np.float64)).sum())

    delta_init = distance(ri, di)
    delta_goal = distance(rg, dg)
    task_data = distance(di, dg)
    task_sim = distance(ri, rg)
    denominator = min(task_data, task_sim)
    ratio = float("inf") if denominator <= 1e-12 else max(delta_init, delta_goal) / denominator
    init_identity = delta_init < distance(ri, dg)
    goal_identity = delta_goal < distance(rg, di)
    return {
        "same_state_init_cost": delta_init,
        "same_state_goal_cost": delta_goal,
        "dataset_task_cost": task_data,
        "rendered_task_cost": task_sim,
        "domain_ratio": ratio,
        "domain_ratio_max": float(ratio_max),
        "rendered_init_identity": bool(init_identity),
        "rendered_goal_identity": bool(goal_identity),
        "pass": bool(ratio <= ratio_max and init_identity and goal_identity),
    }


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.max(np.abs(np.asarray(a).astype(np.float64) - np.asarray(b).astype(np.float64)))
    )


def pixel_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise RuntimeError(f"pixel shape mismatch {a.shape} versus {b.shape}")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def population_pixel_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if np.asarray(a).shape != np.asarray(b).shape:
        raise RuntimeError(f"endpoint shape mismatch {np.shape(a)} versus {np.shape(b)}")
    maximum = 0.0
    total = 0.0
    count = 0
    for left, right in zip(a, b):
        diff = np.abs(left.astype(np.int16) - right.astype(np.int16))
        maximum = max(maximum, float(diff.max()))
        total += float(diff.sum())
        count += int(diff.size)
    return {"max_abs": maximum, "mean_abs": total / count}


def selector_metrics(
    costs: np.ndarray,
    physical: np.ndarray,
    success: np.ndarray,
    audit: Any,
) -> dict[str, float | int]:
    costs = np.asarray(costs, dtype=np.float64)
    selected = int(np.argmin(costs))
    physical_order = np.argsort(physical)
    cost_order = np.argsort(costs)
    n_top = max(1, int(np.ceil(0.10 * len(costs))))
    best = int(physical_order[0])
    return {
        "selected_candidate": selected,
        "selected_physical_distance_m": float(physical[selected]),
        "selected_success": float(success[selected]),
        "selection_regret_m": float(physical[selected] - physical[best]),
        "success_gap": float(int(success.any()) - int(success[selected])),
        "spearman_physical": audit.spearman(costs, physical),
        "top10pct_recall_physical": len(
            set(cost_order[:n_top].tolist()) & set(physical_order[:n_top].tolist())
        )
        / n_top,
        "false_elite_rate_physical": float(
            np.mean(physical[cost_order[:n_top]] > np.median(physical))
        ),
    }


def load_reference_physical(
    root: Path, snapshot_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = root / str(snapshot_index) / "candidate_costs.csv.gz"
    rows: list[dict[str, str]] = []
    with gzip.open(path, "rt", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["candidate"]))
    expected = list(range(len(rows)))
    if [int(row["candidate"]) for row in rows] != expected:
        raise RuntimeError(f"reference candidates are incomplete in {path}")
    return (
        np.asarray([float(row["physical_distance_m"]) for row in rows]),
        np.asarray([bool(int(row["success"])) for row in rows]),
        np.asarray([int(row["executed_steps"]) for row in rows]),
    )


def write_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def main() -> None:
    global load_stage0_transform_images

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("corrected true-endpoint audit requires a GPU Slurm allocation")
    if args.domain_ratio_max <= 0:
        raise ValueError("domain-ratio-max must be positive")

    import mujoco
    import stable_worldmodel as swm
    from stable_worldmodel.world.world import _extract_init_goal

    audit = load_stage0_module()
    load_stage0_transform_images = audit.transform_images
    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest):
        raise ValueError("snapshot index outside persisted manifest")
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("persisted manifest order/index mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = (
        args.candidate_artifacts / f"snapshot_{snapshot.order:03d}_final.npz"
    )
    with np.load(artifact_path, allow_pickle=False) as artifact:
        actions_raw = np.asarray(artifact["actions_raw"])
        learned_cost = np.asarray(artifact["learned_cost"])
    expected_actions = (
        len(learned_cost),
        args.horizon,
        args.action_block,
        actions_raw.shape[-1],
    )
    if actions_raw.shape != expected_actions:
        raise RuntimeError(f"unexpected action shape {actions_raw.shape}")

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]
    goal_qpos = resolve_goal(goal_row, "qpos")
    goal_qvel = resolve_goal(goal_row, "qvel")

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)

    runs: list[dict[str, Any]] = []
    for replicate in range(2):
        world, raw_env, visual_hash, visual_shapes = make_world(swm, snapshot)
        try:
            if int(np.prod(world.envs.single_action_space.shape)) != actions_raw.shape[-1]:
                raise RuntimeError("environment/artifact action dimension mismatch")
            rendered_init = render_state(
                raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit
            )
            rendered_goal = render_state(
                raw_env, goal_qpos, goal_qvel, goal_row, audit
            )
            endpoints, physical, success, executed = rollout_population(
                raw_env, init_row, goal_row, actions_raw, audit
            )
        finally:
            world.close()
        endpoint_emb = encode_images(
            model, endpoints, transform, args.encode_batch
        )
        rendered_goal_emb = encode_images(
            model, rendered_goal, transform, args.encode_batch
        )[0]
        dataset_goal_emb = encode_images(
            model, np.asarray(goal_row["goal"]), transform, args.encode_batch
        )[0]
        runs.append(
            {
                "visual_hash": visual_hash,
                "visual_shapes": visual_shapes,
                "rendered_init": rendered_init,
                "rendered_goal": rendered_goal,
                "endpoints": endpoints,
                "physical": physical,
                "success": success,
                "executed": executed,
                "endpoint_emb": endpoint_emb,
                "true_dataset_goal": squared_l2(endpoint_emb, dataset_goal_emb),
                "true_rendered_goal": squared_l2(endpoint_emb, rendered_goal_emb),
            }
        )

    first, second = runs
    repeat = {
        "visual_signature_match": first["visual_hash"] == second["visual_hash"],
        "initial_pixels": pixel_error(first["rendered_init"], second["rendered_init"]),
        "goal_pixels": pixel_error(first["rendered_goal"], second["rendered_goal"]),
        "endpoint_pixels": population_pixel_error(first["endpoints"], second["endpoints"]),
        "physical_distance_m_max_abs": max_abs(first["physical"], second["physical"]),
        "success_disagreements": int(np.sum(first["success"] != second["success"])),
        "executed_steps_max_abs": max_abs(first["executed"], second["executed"]),
        "true_dataset_goal_cost_max_abs": max_abs(
            first["true_dataset_goal"], second["true_dataset_goal"]
        ),
        "true_rendered_goal_cost_max_abs": max_abs(
            first["true_rendered_goal"], second["true_rendered_goal"]
        ),
        "true_dataset_goal_selected_match": int(np.argmin(first["true_dataset_goal"]))
        == int(np.argmin(second["true_dataset_goal"])),
        "true_rendered_goal_selected_match": int(np.argmin(first["true_rendered_goal"]))
        == int(np.argmin(second["true_rendered_goal"])),
    }

    dataset_init = audit.resize_render(np.asarray(init_row["pixels"]))
    dataset_goal = audit.resize_render(np.asarray(goal_row["goal"]))
    four_emb = encode_images(
        model,
        np.stack(
            [dataset_init, first["rendered_init"], dataset_goal, first["rendered_goal"]]
        ),
        transform,
        args.encode_batch,
    )
    domain = domain_diagnostics(*four_emb, ratio_max=args.domain_ratio_max)
    domain["dataset_vs_rendered_init_pixels"] = pixel_error(
        dataset_init, first["rendered_init"]
    )
    domain["dataset_vs_rendered_goal_pixels"] = pixel_error(
        dataset_goal, first["rendered_goal"]
    )

    reference_physical, reference_success, reference_executed = load_reference_physical(
        args.reference_physical_shards, snapshot.order
    )
    reference = {
        "physical_distance_m_max_abs": max_abs(first["physical"], reference_physical),
        "success_disagreements": int(np.sum(first["success"] != reference_success)),
        "executed_steps_max_abs": max_abs(first["executed"], reference_executed),
    }

    checks = {
        "visual_signature": bool(repeat["visual_signature_match"]),
        "initial_pixels_exact": repeat["initial_pixels"]["max_abs"] == 0,
        "goal_pixels_exact": repeat["goal_pixels"]["max_abs"] == 0,
        "endpoint_pixels_exact": repeat["endpoint_pixels"]["max_abs"] == 0,
        "physical_repeat": repeat["physical_distance_m_max_abs"] <= args.physical_atol,
        "success_repeat": repeat["success_disagreements"] == 0,
        "executed_repeat": repeat["executed_steps_max_abs"] == 0,
        "dataset_goal_cost_repeat": repeat["true_dataset_goal_cost_max_abs"]
        <= args.latent_atol,
        "rendered_goal_cost_repeat": repeat["true_rendered_goal_cost_max_abs"]
        <= args.latent_atol,
        "dataset_goal_selection_repeat": bool(
            repeat["true_dataset_goal_selected_match"]
        ),
        "rendered_goal_selection_repeat": bool(
            repeat["true_rendered_goal_selected_match"]
        ),
        "reference_physical": reference["physical_distance_m_max_abs"]
        <= args.physical_atol,
        "reference_success": reference["success_disagreements"] == 0,
        "reference_executed": reference["executed_steps_max_abs"] == 0,
        "domain_match": bool(domain["pass"]),
    }
    gate = {"pass": bool(all(checks.values())), "checks": checks}

    costs = {
        "learned_predicted_goal": learned_cost,
        "true_dataset_goal": first["true_dataset_goal"],
        "true_rendered_goal": first["true_rendered_goal"],
    }
    metric_rows = [
        {
            "snapshot": snapshot.order,
            "selector": selector,
            **selector_metrics(cost, first["physical"], first["success"], audit),
        }
        for selector, cost in costs.items()
    ]
    with (args.out_dir / "snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    candidate_rows = []
    for candidate in range(len(learned_cost)):
        candidate_rows.append(
            {
                "snapshot": snapshot.order,
                "candidate": candidate,
                **{name: float(value[candidate]) for name, value in costs.items()},
                "physical_distance_m": float(first["physical"][candidate]),
                "success": int(first["success"][candidate]),
                "executed_steps": int(first["executed"][candidate]),
            }
        )
    with gzip.open(args.out_dir / "candidate_costs.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    for name, image in {
        "dataset_init.png": dataset_init,
        "rendered_init.png": first["rendered_init"],
        "dataset_goal.png": dataset_goal,
        "rendered_goal.png": first["rendered_goal"],
    }.items():
        write_image(args.out_dir / name, image)
    np.savez_compressed(
        args.out_dir / "repeat_debug.npz",
        first_physical=first["physical"],
        second_physical=second["physical"],
        reference_physical=reference_physical,
        first_success=first["success"],
        second_success=second["success"],
        first_executed=first["executed"],
        second_executed=second["executed"],
        first_true_dataset_goal=first["true_dataset_goal"],
        second_true_dataset_goal=second["true_dataset_goal"],
        first_true_rendered_goal=first["true_rendered_goal"],
        second_true_rendered_goal=second["true_rendered_goal"],
    )

    summary = {
        "config": {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "snapshot_index": snapshot.order,
            "goal_offset": args.goal_offset,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "num_candidates": len(learned_cost),
            "independent_worlds": 2,
            "physical_atol": args.physical_atol,
            "latent_atol": args.latent_atol,
            "domain_ratio_max": args.domain_ratio_max,
            "warmstart_disabled": True,
            "visual_variation": [],
        },
        "snapshot": asdict(snapshot),
        "visual_signature": {
            "first": first["visual_hash"],
            "second": second["visual_hash"],
            "array_shapes": first["visual_shapes"],
        },
        "repeat": repeat,
        "reference_corrected_physical": reference,
        "domain": domain,
        "gate": gate,
        "oracle_candidate_success": int(first["success"].any()),
        "oracle_physical_distance_m": float(first["physical"].min()),
        "scope": (
            "offline diagnostic on locked candidates; simulator images/state are "
            "never exposed to a deployable selector"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (args.out_dir / "manifest_row.json").write_text(
        json.dumps(asdict(snapshot), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))
    print("OGB_TRUE_ENDPOINT_CORRECTED_PASS" if gate["pass"] else "OGB_TRUE_ENDPOINT_CORRECTED_BLOCKED")
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
