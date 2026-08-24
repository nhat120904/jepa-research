#!/usr/bin/env python3
"""Locked, physical-budget-matched acquisition test for ordinal inversions.

Each arm selects eight final-CEM candidates using only the frozen model's CEM
artifacts.  After all choices are locked, it replays exactly the returned CEM
mean plus the eight alternatives from the same restored MuJoCo state.  It never
scans the full population in physics, so a hit is an attainable acquisition
event rather than an oracle-picked counterfactual.
"""

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
PHASE0D = Path(__file__).with_name("run_ogb_phase0d_deployed_action.py")
ARM_NAMES = (
    "proxy_rejected_stratified_diverse",
    "random_final_population",
    "proxy_hard",
    "action_diverse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--alternatives", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
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


def rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks / max(len(values) - 1, 1)


def farthest_point_indices(actions: np.ndarray, anchor: np.ndarray, count: int) -> np.ndarray:
    """Deterministic action-space coverage, independent of physical outcomes."""
    flat = np.asarray(actions, dtype=np.float64).reshape(len(actions), -1)
    anchor_flat = np.asarray(anchor, dtype=np.float64).reshape(1, -1)
    dist_anchor = np.square(flat - anchor_flat).sum(axis=1)
    chosen: list[int] = [int(np.argmax(dist_anchor))]
    min_dist = np.square(flat - flat[chosen[0]]).sum(axis=1)
    while len(chosen) < count:
        min_dist[chosen] = -np.inf
        chosen.append(int(np.argmax(min_dist)))
        min_dist = np.minimum(min_dist, np.square(flat - flat[chosen[-1]]).sum(axis=1))
    return np.asarray(chosen, dtype=np.int64)


def select_stratified_diverse(
    actions: np.ndarray, proxy: np.ndarray, anchor: np.ndarray, anchor_proxy: float, count: int
) -> np.ndarray:
    """One action-novel candidate from each proxy-rejected rank stratum."""
    eligible = np.flatnonzero(proxy > anchor_proxy)
    if len(eligible) < count:
        raise RuntimeError("too few proxy-rejected candidates for locked acquisition budget")
    ordered = eligible[np.argsort(proxy[eligible], kind="stable")]
    novelty = np.square(
        actions.reshape(len(actions), -1) - np.asarray(anchor).reshape(1, -1)
    ).sum(axis=1)
    edges = np.linspace(0, len(ordered), count + 1, dtype=int)
    result: list[int] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        bucket = ordered[lower:upper]
        result.append(int(bucket[np.argmax(novelty[bucket])]))
    if len(set(result)) != count:
        raise RuntimeError("stratified selector produced duplicate candidates")
    return np.asarray(result, dtype=np.int64)


def select_indices(
    actions: np.ndarray,
    proxy: np.ndarray,
    anchor: np.ndarray,
    anchor_proxy: float,
    count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if len(actions) < count:
        raise ValueError("candidate population is smaller than physical-query budget")
    rng = np.random.default_rng(seed)
    return {
        "proxy_rejected_stratified_diverse": select_stratified_diverse(
            actions, proxy, anchor, anchor_proxy, count
        ),
        "random_final_population": np.sort(rng.choice(len(actions), size=count, replace=False)),
        "proxy_hard": np.argsort(proxy, kind="stable")[-count:][::-1],
        "action_diverse": farthest_point_indices(actions, anchor, count),
    }


def mean_distance(actions: np.ndarray, anchor: np.ndarray) -> float:
    flat = np.asarray(actions, dtype=np.float64).reshape(len(actions), -1)
    return float(np.sqrt(np.square(flat - np.asarray(anchor).reshape(1, -1)).sum(axis=1)).mean())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1a requires a GPU Slurm allocation")
    if args.alternatives <= 0 or args.alternatives >= args.num_samples:
        raise ValueError("alternatives must lie in [1, num_samples)")
    if args.topk > args.num_samples or args.cem_steps < 2:
        raise ValueError("invalid CEM configuration")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "phase1a_audit")
    corrected = load_module(DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "phase1a_corrected")
    phase0d = load_module(PHASE0D, "phase1a_phase0d")
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest):
        raise ValueError("snapshot-index outside Phase-1a manifest")
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("Phase-1a manifest order/index mismatch")
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

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)
    world, raw_env, visual_hash, visual_shapes = corrected.make_world(swm, snapshot)
    try:
        raw_dim = int(np.prod(world.envs.single_action_space.shape))
        recorder = phase0d.FinalPopulationRecorder(args.cem_steps - 1)
        cost = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
        solver = swm.planning.CEMSolver(
            cost=cost, batch_size=1, num_samples=args.num_samples,
            var_scale=args.var_scale, n_steps=args.cem_steps, topk=args.topk,
            device="cuda", seed=args.seed + snapshot.order, callbacks=[recorder],
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(
                horizon=args.horizon, receding_horizon=args.horizon,
                action_block=args.action_block, history_len=1, warm_start=True,
            ),
            process={"action": scaler}, transform={"pixels": transform, "goal": transform},
        )
        policy.set_env(world.envs)
        raw_info = {
            "pixels": np.asarray(init_row["pixels"])[None, None],
            "goal": np.asarray(goal_row["goal"])[None, None],
            "action": np.full((1, 1, raw_dim), np.nan, dtype=np.float32),
        }
        prepared = policy._prepare_info(raw_info)
        with torch.inference_mode():
            result = solver.solve(prepared)
            anchor_norm = np.asarray(result["actions"][0], dtype=np.float32)
            anchor_tensor = torch.as_tensor(result["actions"], device="cuda")
            anchor_proxy = float(cost.get_cost(
                phase0d.expand_info_for_cost(prepared, "cuda", solver.dtype),
                anchor_tensor.unsqueeze(1),
            )[0, 0].detach().cpu())
        if recorder.record is None:
            raise RuntimeError("final CEM population was not recorded")
        final_norm = recorder.record["actions_normalized"]
        final_proxy = recorder.record["learned_cost"]
        selections = select_indices(
            final_norm, final_proxy, anchor_norm, anchor_proxy, args.alternatives,
            args.seed + 10_000 + snapshot.order,
        )
        # All selection decisions above this line use only CEM/model quantities.
        anchor_raw = phase0d.normalized_to_raw(
            anchor_norm, scaler, args.horizon, args.action_block, raw_dim
        )
        final_raw = phase0d.normalized_to_raw(
            final_norm, scaler, args.horizon, args.action_block, raw_dim
        )
        _, anchor_distance_arr, anchor_success_arr, anchor_steps_arr = corrected.rollout_population(
            raw_env, init_row, goal_row, anchor_raw[None], audit
        )
        _, replay_distance, replay_success, replay_steps = corrected.rollout_population(
            raw_env, init_row, goal_row, anchor_raw[None], audit
        )
        arms_physical: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, indices in selections.items():
            _, distance, success, steps = corrected.rollout_population(
                raw_env, init_row, goal_row, final_raw[indices], audit
            )
            arms_physical[name] = (distance, success, steps)
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

    arm_summary: dict[str, Any] = {}
    choice_arrays: dict[str, np.ndarray] = {}
    for name in ARM_NAMES:
        indices = selections[name]
        physical, success, executed = arms_physical[name]
        proxy_rejected = final_proxy[indices] > anchor_proxy
        advantage = anchor_distance - physical
        corrective = proxy_rejected & (advantage >= args.min_physical_gap_m)
        arm_summary[name] = {
            "n_alternatives": int(len(indices)),
            "accounted_physical_queries": int(len(indices) + 1),
            "all_proxy_rejected": int(proxy_rejected.all()),
            "proxy_rejected_fraction": float(proxy_rejected.mean()),
            "mean_proxy_rank_fraction": float(rank_fraction(final_proxy)[indices].mean()),
            "mean_action_distance_from_anchor": mean_distance(final_norm[indices], anchor_norm),
            "inversion_hit": int(corrective.any()),
            "best_corrective_advantage_m": float(max(0.0, advantage[corrective].max(initial=0.0))),
            "best_any_advantage_m": float(max(0.0, advantage.max(initial=0.0))),
            "any_success_gain": int((not anchor_success) and success.any()),
            "best_selected_distance_m": float(physical.min()),
            "selected_success_any": int(success.any()),
            "selected_indices": [int(index) for index in indices],
        }
        choice_arrays[f"{name}_indices"] = indices
        choice_arrays[f"{name}_physical_distance_m"] = physical
        choice_arrays[f"{name}_success"] = success
        choice_arrays[f"{name}_proxy_cost"] = final_proxy[indices]
        choice_arrays[f"{name}_executed_steps"] = executed
    np.savez_compressed(
        args.out_dir / "acquisition_audit.npz",
        anchor_actions_normalized=anchor_norm,
        anchor_actions_raw=anchor_raw,
        anchor_proxy_cost=np.asarray(anchor_proxy),
        anchor_physical_distance_m=np.asarray(anchor_distance),
        anchor_success=np.asarray(anchor_success),
        final_proxy_cost=final_proxy,
        **choice_arrays,
    )
    summary = {
        "snapshot": snapshot.order,
        "episode": snapshot.episode,
        "start_step": snapshot.start_step,
        "scope": (
            "Selections are locked from frozen CEM/model quantities before any physical replay. "
            "Every arm is charged the same returned-plan-plus-K physical-query budget."
        ),
        "config": {
            "dataset": args.dataset, "checkpoint": args.checkpoint,
            "goal_offset": args.goal_offset, "horizon": args.horizon,
            "action_block": args.action_block, "num_samples": args.num_samples,
            "cem_steps": args.cem_steps, "topk": args.topk, "var_scale": args.var_scale,
            "alternatives": args.alternatives, "physical_budget_per_arm": args.alternatives + 1,
            "min_physical_gap_m": args.min_physical_gap_m, "seed": args.seed,
        },
        "visual_signature": visual_hash,
        "visual_signature_shapes": visual_shapes,
        "repeat_gate": repeat_gate,
        "anchor": {
            "proxy_cost": anchor_proxy, "physical_distance_m": anchor_distance,
            "success": int(anchor_success), "executed_steps": int(anchor_steps_arr[0]),
            "actions_raw_sha256": phase0d.array_hash(anchor_raw),
        },
        "arms": arm_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
