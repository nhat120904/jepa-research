#!/usr/bin/env python3
"""Same-candidate PFCG reranking pilot on the locked OGBench Stage-0 data.

Deployable selector indices are fixed from the frozen model before MuJoCo is
queried. Simulator replay supplies physical distance and success only after
selection; no simulator rendering enters a selector or diagnostic cost.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from planning.pfcg import (  # noqa: E402
    diagonal_response_cost,
    fit_symmetric_probe_geometry,
    latent_l2,
    matched_random_geometry,
    pfcg_cost,
    projected_cost,
)


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
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--probe-pairs", type=int, default=32)
    parser.add_argument("--relative-eigen-floor", type=float, default=1e-6)
    parser.add_argument("--ridge-fraction", type=float, default=0.1)
    parser.add_argument("--replay-physical-atol", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_stage0_module():
    source = Path(__file__).with_name("72_ogb_stage0_candidate_audit.py")
    spec = importlib.util.spec_from_file_location("ogb_stage0_audit_for_pfcg", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expand_info(
    prepared: dict[str, Any], samples: int, device: str, dtype: torch.dtype
) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in prepared.items():
        if torch.is_tensor(value):
            target_dtype = dtype if value.is_floating_point() else None
            value = value.to(device=device, dtype=target_dtype)
            expanded[key] = value.unsqueeze(1).expand(
                value.shape[0], samples, *value.shape[1:]
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:, None, ...], samples, axis=1)
        else:
            expanded[key] = value
    return expanded


@torch.inference_mode()
def predicted_terminal_embeddings(
    model: Any,
    prepared: dict[str, Any],
    actions_normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import stable_worldmodel as swm

    device = str(next(model.parameters()).device)
    dtype = next(model.parameters()).dtype
    actions = torch.as_tensor(
        actions_normalized, device=device, dtype=dtype
    ).unsqueeze(0)
    info = expand_info(prepared, len(actions_normalized), device, dtype)
    evaluator = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
    rolled = evaluator._rollout(info, actions)
    endpoints = rolled["predicted_emb"][0, :, -1].float().cpu().numpy()
    goal = rolled["goal_emb"][0, -1].float().cpu().numpy()
    return endpoints, goal


def restore_physical(
    raw_env: Any,
    init_row: dict[str, Any],
    goal_row: dict[str, Any],
    seed: int,
    audit: Any,
) -> None:
    """Restore the complete MuJoCo dynamics state without episode randomization."""

    import mujoco

    # CubeEnv.initialize_episode executes two random actions to render a goal,
    # then restores only qpos/qvel. Calling env.reset for every candidate would
    # therefore leak random ctrl/qacc_warmstart into the rollout. Reset mjData
    # itself so all solver, actuator, force, time, and warm-start state is clean.
    del seed
    mujoco.mj_resetData(raw_env._model, raw_env._data)
    raw_env._reset_next_step = False
    raw_env.set_state(qpos=init_row["qpos"], qvel=init_row["qvel"])
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


def rollout_physical_population(
    raw_env: Any,
    init_row: dict[str, Any],
    goal_row: dict[str, Any],
    reset_seed: int,
    actions_raw: np.ndarray,
    audit: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay candidates and return only stable physical measurements."""

    target = audit.goal_field(goal_row, "block_0_pos")
    distances = np.empty(len(actions_raw), dtype=np.float64)
    successes = np.zeros(len(actions_raw), dtype=bool)
    executed = np.empty(len(actions_raw), dtype=np.int64)
    for index, sequence in enumerate(actions_raw):
        restore_physical(raw_env, init_row, goal_row, reset_seed, audit)
        n_executed = 0
        terminated = False
        for action in sequence.reshape(-1, sequence.shape[-1]):
            _, _, terminated, truncated, _ = raw_env.step(action)
            n_executed += 1
            if terminated or truncated:
                break
        distances[index] = audit.cube_distance(raw_env, target)
        successes[index] = bool(terminated) or distances[index] <= 0.04
        executed[index] = n_executed
    return distances, successes, executed


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
    physical_best = int(physical_order[0])
    n_top = max(1, int(np.ceil(0.10 * len(costs))))
    return {
        "selected_candidate": selected,
        "selected_physical_distance_m": float(physical[selected]),
        "selected_success": float(success[selected]),
        "selection_regret_m": float(physical[selected] - physical[physical_best]),
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


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray, atol: float) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    if not np.allclose(actual, expected, rtol=1e-5, atol=atol):
        raise RuntimeError(f"{name} mismatch: max_abs={error}")
    return error


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PFCG pilot must run in a GPU Slurm allocation")
    if args.probe_pairs < 2:
        raise ValueError("probe_pairs must be at least two")

    import stable_worldmodel as swm
    import mujoco
    from stable_worldmodel.world.world import _extract_init_goal

    audit = load_stage0_module()
    manifest_payload = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest_payload):
        raise ValueError("snapshot index outside persisted manifest")
    snapshot = audit.Snapshot(**manifest_payload[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("persisted manifest order/index mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = (
        args.candidate_artifacts / f"snapshot_{snapshot.order:03d}_final.npz"
    )
    with np.load(artifact_path, allow_pickle=False) as artifact:
        actions_normalized = np.asarray(artifact["actions_normalized"])
        actions_raw = np.asarray(artifact["actions_raw"])
        stored_learned = np.asarray(artifact["learned_cost"])

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)

    raw_action_dim = int(actions_raw.shape[-1])
    model_action_dim = raw_action_dim * args.action_block
    expected_shape = (len(actions_normalized), args.horizon, model_action_dim)
    if actions_normalized.shape != expected_shape:
        raise RuntimeError(
            f"candidate shape {actions_normalized.shape} != expected {expected_shape}"
        )
    if actions_raw.shape != (
        len(actions_normalized),
        args.horizon,
        args.action_block,
        raw_action_dim,
    ):
        raise RuntimeError(f"unexpected raw action shape {actions_raw.shape}")

    # This is the exact image preprocessing performed by
    # WorldModelPolicy._prepare_info in Stage 0, without constructing an
    # environment merely to configure an otherwise-unused solver.
    prepared = {
        "pixels": audit.transform_images(
            np.asarray(init_row["pixels"]), transform, "cuda"
        ),
        "goal": audit.transform_images(
            np.asarray(goal_row["goal"]), transform, "cuda"
        ),
        "action": torch.full(
            (1, 1, raw_action_dim), float("nan"), device="cuda"
        ),
    }

    # Candidate prediction and probe geometry are completed before any MuJoCo
    # outcome is requested.  These selectors are the deployable arms.
    predicted, predicted_goal = predicted_terminal_embeddings(
        model, prepared, actions_normalized
    )
    recomputed_l2 = latent_l2(predicted, predicted_goal)
    learned_cost_error = assert_close(
        "learned latent cost", recomputed_l2, stored_learned, atol=2e-4
    )

    rng = np.random.default_rng(args.seed + 50_000 + snapshot.order)
    positive_actions = rng.standard_normal(
        (args.probe_pairs, args.horizon, model_action_dim), dtype=np.float32
    )
    probe_actions = np.concatenate([positive_actions, -positive_actions], axis=0)
    probe_endpoints, probe_goal = predicted_terminal_embeddings(
        model, prepared, probe_actions
    )
    assert_close("probe/candidate goal embedding", probe_goal, predicted_goal, atol=1e-6)
    plus = probe_endpoints[: args.probe_pairs]
    minus = probe_endpoints[args.probe_pairs :]
    geometry = fit_symmetric_probe_geometry(
        plus,
        minus,
        relative_eigen_floor=args.relative_eigen_floor,
        ridge_fraction=args.ridge_fraction,
    )
    random_geometry = matched_random_geometry(
        geometry, predicted.shape[1], args.seed + 70_000 + snapshot.order
    )
    deployable_costs = {
        "pred_l2": recomputed_l2,
        "pred_pfcg": pfcg_cost(predicted, predicted_goal, geometry),
        "pred_projected": projected_cost(predicted, predicted_goal, geometry),
        "pred_diag": diagonal_response_cost(
            predicted,
            predicted_goal,
            plus,
            minus,
            relative_variance_floor=args.relative_eigen_floor,
            ridge_fraction=args.ridge_fraction,
        ),
        "pred_random": pfcg_cost(predicted, predicted_goal, random_geometry),
    }
    deployable_selected = {
        name: int(np.argmin(cost)) for name, cost in deployable_costs.items()
    }

    # Hidden evaluator begins here. Compile one physical model, then replay the
    # population twice with a reset before every candidate. Independent model
    # compilations are handled as a separate robustness replication.
    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=1,
        max_episode_steps=2 * args.horizon * args.action_block,
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
    # Same-state candidate evaluation must not inherit the contact solver's
    # acceleration warm start from a reset/previous rollout.
    raw_env._model.opt.disableflags |= int(
        mujoco.mjtDisableBit.mjDSBL_WARMSTART
    )
    env_action_dim = int(np.prod(world.envs.single_action_space.shape))
    if env_action_dim != raw_action_dim:
        raise RuntimeError(
            f"artifact action dim {raw_action_dim} != env action dim {env_action_dim}"
        )
    replays = []
    for _ in range(2):
        replays.append(
            rollout_physical_population(
                raw_env,
                init_row,
                goal_row,
                snapshot.reset_seed,
                actions_raw,
                audit,
            )
        )
    world.close()
    physical, success, executed = replays[0]
    repeat_physical, repeat_success, repeat_executed = replays[1]
    repeat_physical_error = assert_close(
        "repeat replay physical distance",
        physical,
        repeat_physical,
        atol=args.replay_physical_atol,
    )
    repeat_success_error = assert_close(
        "repeat replay success", success, repeat_success, atol=0.0
    )
    repeat_executed_error = assert_close(
        "repeat replay executed steps", executed, repeat_executed, atol=0.0
    )
    with np.load(artifact_path, allow_pickle=False) as artifact:
        stored_physical = np.asarray(artifact["physical_distance_m"])
        stored_success = np.asarray(artifact["success"])
        stored_executed = np.asarray(artifact["executed_steps"])
    np.savez_compressed(
        args.out_dir / "reproduction_debug.npz",
        recomputed_physical=physical,
        stored_physical=stored_physical,
        recomputed_success=success,
        stored_success=stored_success,
        recomputed_executed=executed,
        stored_executed=stored_executed,
        repeat_physical=repeat_physical,
        repeat_success=repeat_success,
        repeat_executed=repeat_executed,
    )
    artifact_physical_error = float(np.max(np.abs(physical - stored_physical)))
    artifact_success_disagreements = int(np.sum(success != stored_success))
    artifact_executed_error = float(np.max(np.abs(executed - stored_executed)))
    all_costs = deployable_costs
    metric_rows = []
    for selector, costs in all_costs.items():
        metric_rows.append(
            {
                "snapshot": snapshot.order,
                "selector": selector,
                **selector_metrics(costs, physical, success, audit),
            }
        )

    candidate_rows = []
    for candidate in range(len(actions_normalized)):
        candidate_rows.append(
            {
                "snapshot": snapshot.order,
                "candidate": candidate,
                **{name: float(cost[candidate]) for name, cost in all_costs.items()},
                "physical_distance_m": float(physical[candidate]),
                "success": int(success[candidate]),
                "executed_steps": int(executed[candidate]),
            }
        )

    with (args.out_dir / "snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with gzip.open(
        args.out_dir / "candidate_costs.csv.gz", "wt", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    summary = {
        "config": {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "snapshot_index": snapshot.order,
            "goal_offset": args.goal_offset,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "num_candidates": len(actions_normalized),
            "probe_pairs": args.probe_pairs,
            "relative_eigen_floor": args.relative_eigen_floor,
            "ridge_fraction": args.ridge_fraction,
            "replay_physical_atol": args.replay_physical_atol,
            "seed": args.seed,
        },
        "snapshot": asdict(snapshot),
        "geometry": {
            "rank": geometry.rank,
            "latent_dim": int(predicted.shape[1]),
            "ridge": geometry.ridge,
            "largest_eigenvalue": float(geometry.eigenvalues[0]),
            "smallest_retained_eigenvalue": float(geometry.eigenvalues[-1]),
            "response_energy_retained": geometry.response_energy_retained,
        },
        "artifact_reproduction": {
            "learned_cost": learned_cost_error,
            "physical_distance_m": artifact_physical_error,
            "success_disagreements": artifact_success_disagreements,
            "executed_steps": artifact_executed_error,
        },
        "repeat_replay": {
            "physical_distance_m": repeat_physical_error,
            "success": repeat_success_error,
            "executed_steps": repeat_executed_error,
        },
        "deployable_selected_before_simulator": deployable_selected,
        "scope": (
            "PFCG choices fixed before MuJoCo evaluation; one compiled physical "
            "model is replayed twice and supplies distance/success only"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (args.out_dir / "manifest_row.json").write_text(
        json.dumps(asdict(snapshot), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("OGB_PFCG_PILOT_DONE")


if __name__ == "__main__":
    main()
