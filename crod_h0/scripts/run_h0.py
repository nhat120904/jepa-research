#!/usr/bin/env python3
"""Locked matched-budget CROD test on fresh OGBench-Cube snapshots."""

from __future__ import annotations

import argparse
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
PROJECT = REPO / "crod_h0"
DIAG = REPO / "diagnosis"
PHASE0D = REPO / "counterfactual_flow/scripts/run_ogb_phase0d_deployed_action.py"
sys.path.insert(0, str(PROJECT))
from core import (  # noqa: E402
    directional_ordinal_score,
    mean_action_distance,
    select_arms,
)


ARM_NAMES = (
    "crod_directional",
    "action_diverse",
    "rejected_action_diverse",
    "dino_best_rejected",
    "native_uncertainty_diverse",
    "random_rejected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--native-checkpoint", default="quentinll/lewm-cube")
    parser.add_argument(
        "--auxiliary-checkpoint",
        default="crod_dinowm_cube_seed42/weights_epoch_10.pt",
    )
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--alternatives", type=int, default=8)
    parser.add_argument("--instability-repeats", type=int, default=4)
    parser.add_argument("--jitter-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--physical-atol", type=float, default=1e-5)
    parser.add_argument("--min-physical-gap-m", type=float, default=0.02)
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
def score_actions(
    cost: Any,
    prepared: dict[str, Any],
    actions: np.ndarray,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    action_tensor = torch.as_tensor(actions, device=device, dtype=dtype).unsqueeze(0)
    values = cost.get_cost(
        expand_info(prepared, len(actions), device, dtype), action_tensor
    )
    return values[0].detach().float().cpu().numpy()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CROD H0 requires a GPU Slurm allocation")
    if not 0 < args.alternatives < args.num_samples:
        raise ValueError("invalid acquisition budget")
    if args.instability_repeats < 2 or args.jitter_scale <= 0:
        raise ValueError("invalid native-uncertainty settings")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "crod_audit")
    corrected = load_module(
        DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "crod_corrected"
    )
    phase0d = load_module(PHASE0D, "crod_phase0d")
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest):
        raise ValueError("snapshot-index outside CROD manifest")
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("manifest order/index mismatch")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]
    scaler = StandardScaler()
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler.fit(action_data)

    native_model = swm.wm.utils.load_pretrained(args.native_checkpoint).cuda().eval()
    native_model.requires_grad_(False)
    native_model.interpolate_pos_encoding = True
    auxiliary_model = swm.wm.utils.load_pretrained(args.auxiliary_checkpoint).cuda().eval()
    auxiliary_model.requires_grad_(False)
    auxiliary_model.interpolate_pos_encoding = True
    if auxiliary_model.__class__.__name__ != "PreJEPA":
        raise RuntimeError("auxiliary checkpoint is not the action-only DINO-WM/PreJEPA")
    if set(auxiliary_model.extra_encoders) != {"action"}:
        raise RuntimeError(
            f"auxiliary must be action-only, got {sorted(auxiliary_model.extra_encoders)}"
        )

    transform = audit.make_transform(224)
    world, raw_env, visual_hash, visual_shapes = corrected.make_world(swm, snapshot)
    try:
        raw_dim = int(np.prod(world.envs.single_action_space.shape))
        recorder = phase0d.FinalPopulationRecorder(args.cem_steps - 1)
        native_cost_model = swm.planning.ShootingCostEvaluator(
            native_model, swm.planning.GoalMSE()
        )
        solver = swm.planning.CEMSolver(
            cost=native_cost_model,
            batch_size=1,
            num_samples=args.num_samples,
            var_scale=args.var_scale,
            n_steps=args.cem_steps,
            topk=args.topk,
            device="cuda",
            seed=args.seed + snapshot.order,
            callbacks=[recorder],
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(
                horizon=args.horizon,
                receding_horizon=args.horizon,
                action_block=args.action_block,
                history_len=1,
                warm_start=True,
            ),
            process={"action": scaler},
            transform={"pixels": transform, "goal": transform},
        )
        policy.set_env(world.envs)
        raw_info = {
            "pixels": np.asarray(init_row["pixels"])[None, None],
            "goal": np.asarray(goal_row["goal"])[None, None],
            "action": np.full((1, 1, raw_dim), np.nan, dtype=np.float32),
            "id": np.asarray([[snapshot.order]], dtype=np.int64),
            "step_idx": np.asarray([[snapshot.start_step]], dtype=np.int64),
        }
        prepared = policy._prepare_info(raw_info)
        with torch.inference_mode():
            result = solver.solve(prepared)
        if recorder.record is None:
            raise RuntimeError("final CEM population was not recorded")
        anchor_norm = np.asarray(result["actions"][0], dtype=np.float32)
        final_norm = np.asarray(recorder.record["actions_normalized"], dtype=np.float32)
        union_norm = np.concatenate([anchor_norm[None], final_norm], axis=0)

        native_union_cost = score_actions(
            native_cost_model, prepared, union_norm, "cuda", solver.dtype
        )
        auxiliary_union_cost = score_actions(
            auxiliary_model, prepared, union_norm, "cuda", torch.float32
        )
        if not np.isfinite(native_union_cost).all() or not np.isfinite(auxiliary_union_cost).all():
            raise RuntimeError("non-finite native or auxiliary candidate cost")
        if float(np.std(auxiliary_union_cost)) <= 1e-10:
            raise RuntimeError("auxiliary cost is constant across the candidate set")
        anchor_native, final_native = float(native_union_cost[0]), native_union_cost[1:]
        anchor_auxiliary, final_auxiliary = (
            float(auxiliary_union_cost[0]),
            auxiliary_union_cost[1:],
        )
        crod_score, native_rank, auxiliary_rank = directional_ordinal_score(
            final_native, final_auxiliary, anchor_native, anchor_auxiliary
        )

        proposal_scale = np.asarray(result["var"][0][0], dtype=np.float32)
        rng = np.random.default_rng(args.seed + 20_000 + snapshot.order)
        jitter_costs = [final_native.astype(np.float64)]
        for _ in range(args.instability_repeats):
            noise = rng.standard_normal(final_norm.shape).astype(np.float32)
            jittered = final_norm + args.jitter_scale * proposal_scale[None] * noise
            jitter_costs.append(
                score_actions(
                    native_cost_model, prepared, jittered, "cuda", solver.dtype
                ).astype(np.float64)
            )
        jitter_stack = np.stack(jitter_costs)
        native_uncertainty = jitter_stack.std(axis=0) / np.maximum(
            np.abs(jitter_stack).mean(axis=0), 1e-6
        )
        selections = select_arms(
            final_norm,
            anchor_norm,
            final_native,
            final_auxiliary,
            anchor_native,
            crod_score,
            native_uncertainty,
            args.alternatives,
            args.seed + 30_000 + snapshot.order,
        )

        # No physical outcome has been inspected above this line.
        anchor_raw = phase0d.normalized_to_raw(
            anchor_norm, scaler, args.horizon, args.action_block, raw_dim
        )
        final_raw = phase0d.normalized_to_raw(
            final_norm, scaler, args.horizon, args.action_block, raw_dim
        )
        selected_union = np.asarray(
            sorted({int(i) for indices in selections.values() for i in indices}),
            dtype=np.int64,
        )
        _, selected_distance, selected_success, selected_steps = corrected.rollout_population(
            raw_env, init_row, goal_row, final_raw[selected_union], audit
        )
        physical_by_index = {
            int(index): (float(selected_distance[j]), bool(selected_success[j]), int(selected_steps[j]))
            for j, index in enumerate(selected_union)
        }
        _, anchor_distance_arr, anchor_success_arr, anchor_steps_arr = corrected.rollout_population(
            raw_env, init_row, goal_row, anchor_raw[None], audit
        )
        _, replay_distance, replay_success, replay_steps = corrected.rollout_population(
            raw_env, init_row, goal_row, anchor_raw[None], audit
        )
    finally:
        world.close()

    anchor_distance = float(anchor_distance_arr[0])
    anchor_success = bool(anchor_success_arr[0])
    repeat_gate = {
        "physical_max_abs": corrected.max_abs(anchor_distance_arr, replay_distance),
        "success_disagreements": int(np.sum(anchor_success_arr != replay_success)),
        "executed_max_abs": corrected.max_abs(anchor_steps_arr, replay_steps),
    }
    repeat_gate["pass"] = bool(
        repeat_gate["physical_max_abs"] <= args.physical_atol
        and repeat_gate["success_disagreements"] == 0
        and repeat_gate["executed_max_abs"] == 0
    )
    if not repeat_gate["pass"]:
        raise RuntimeError(f"anchor replay failed: {repeat_gate}")

    artifact: dict[str, np.ndarray] = {
        "anchor_actions_normalized": anchor_norm,
        "anchor_actions_raw": anchor_raw,
        "anchor_native_cost": np.asarray(anchor_native),
        "anchor_auxiliary_cost": np.asarray(anchor_auxiliary),
        "anchor_physical_distance_m": np.asarray(anchor_distance),
        "anchor_success": np.asarray(anchor_success),
        "final_actions_normalized": final_norm,
        "final_native_cost": final_native,
        "final_auxiliary_cost": final_auxiliary,
        "native_rank_fraction": native_rank,
        "auxiliary_rank_fraction": auxiliary_rank,
        "crod_score": crod_score,
        "native_uncertainty": native_uncertainty,
        "physically_executed_unique_indices": selected_union,
        "physically_executed_distance_m": selected_distance,
        "physically_executed_success": selected_success,
    }
    arm_summary: dict[str, Any] = {}
    for name in ARM_NAMES:
        indices = selections[name]
        physical = np.asarray([physical_by_index[int(i)][0] for i in indices])
        success = np.asarray([physical_by_index[int(i)][1] for i in indices])
        executed = np.asarray([physical_by_index[int(i)][2] for i in indices])
        rejected = final_native[indices] > anchor_native
        aux_prefers = final_auxiliary[indices] < anchor_auxiliary
        advantage = anchor_distance - physical
        corrective = rejected & (advantage >= args.min_physical_gap_m)
        arm_summary[name] = {
            "n_alternatives": int(len(indices)),
            "accounted_physical_queries": int(len(indices) + 1),
            "native_rejected_fraction": float(rejected.mean()),
            "auxiliary_prefers_fraction": float(aux_prefers.mean()),
            "directional_positive_fraction": float((crod_score[indices] > 0).mean()),
            "mean_crod_score": float(crod_score[indices].mean()),
            "mean_native_rank_fraction": float(native_rank[indices].mean()),
            "mean_auxiliary_rank_fraction": float(auxiliary_rank[indices].mean()),
            "mean_action_distance_from_anchor": mean_action_distance(
                final_norm[indices], anchor_norm
            ),
            "inversion_hit": int(corrective.any()),
            "best_corrective_advantage_m": float(np.max(advantage[corrective], initial=0.0)),
            "best_any_advantage_m": float(np.max(advantage, initial=0.0)),
            "best_improvement_per_query_m": float(
                np.max(advantage, initial=0.0) / args.alternatives
            ),
            "any_success_gain": int((not anchor_success) and success.any()),
            "best_selected_distance_m": float(physical.min()),
            "selected_success_any": int(success.any()),
            "selected_indices": [int(i) for i in indices],
        }
        artifact[f"{name}_indices"] = indices
        artifact[f"{name}_physical_distance_m"] = physical
        artifact[f"{name}_success"] = success
        artifact[f"{name}_executed_steps"] = executed
    np.savez_compressed(args.out_dir / "crod_h0.npz", **artifact)

    summary = {
        "snapshot": snapshot.order,
        "episode": snapshot.episode,
        "start_step": snapshot.start_step,
        "scope": (
            "All choices are locked from LeWM/DINO-WM costs before physics. "
            "Each arm is charged one exact CEM-returned anchor plus eight alternatives."
        ),
        "config": {
            "dataset": args.dataset,
            "native_checkpoint": args.native_checkpoint,
            "auxiliary_checkpoint": args.auxiliary_checkpoint,
            "auxiliary_representation": "frozen DINOv2-Small, action-only predictor",
            "goal_offset": args.goal_offset,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "num_samples": args.num_samples,
            "cem_steps": args.cem_steps,
            "topk": args.topk,
            "alternatives": args.alternatives,
            "physical_budget_per_arm": args.alternatives + 1,
            "min_physical_gap_m": args.min_physical_gap_m,
            "seed": args.seed,
        },
        "visual_signature": visual_hash,
        "visual_signature_shapes": visual_shapes,
        "repeat_gate": repeat_gate,
        "anchor": {
            "native_cost": anchor_native,
            "auxiliary_cost": anchor_auxiliary,
            "physical_distance_m": anchor_distance,
            "success": int(anchor_success),
            "executed_steps": int(anchor_steps_arr[0]),
            "actions_raw_sha256": phase0d.array_hash(anchor_raw),
        },
        "diagnostics": {
            "native_rejected_candidates": int(np.sum(final_native > anchor_native)),
            "directional_positive_candidates": int(np.sum(crod_score > 0)),
            "auxiliary_cost_std": float(np.std(auxiliary_union_cost)),
            "native_callback_rescore_max_abs": float(
                np.max(np.abs(final_native - recorder.record["learned_cost"]))
            ),
            "unique_physical_alternatives_executed": int(len(selected_union)),
        },
        "arms": arm_summary,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
