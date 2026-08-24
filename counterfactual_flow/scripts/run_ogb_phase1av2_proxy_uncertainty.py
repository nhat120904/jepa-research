#!/usr/bin/env python3
"""Fresh matched-budget test of model-only inversion-acquisition signals.

The primary selector estimates local instability of the frozen proxy by
rescoring each final-CEM candidate under small proposal-scaled action
perturbations, then applies action-space coverage inside the most unstable
quartile. A second model-only selector combines CEM proposal likelihood with
proxy rejection. Choices are locked before any physical branch is executed.
"""

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
DIAG = REPO / "diagnosis"
PHASE0D = Path(__file__).with_name("run_ogb_phase0d_deployed_action.py")
PHASE1A = Path(__file__).with_name("run_ogb_phase1a_acquisition.py")
ARM_NAMES = (
    "proxy_instability_diverse",
    "cem_disagreement_diverse",
    "action_diverse",
    "random_final_population",
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
    parser.add_argument("--instability-repeats", type=int, default=4)
    parser.add_argument("--jitter-scale", type=float, default=0.25)
    parser.add_argument("--selector-pool-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260819)
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
    tensor = torch.as_tensor(actions, device=device, dtype=dtype).unsqueeze(0)
    values = cost.get_cost(expand_info(prepared, len(actions), device, dtype), tensor)
    return values[0].detach().float().cpu().numpy()


def top_score_diverse(
    actions: np.ndarray,
    anchor: np.ndarray,
    proxy: np.ndarray,
    anchor_proxy: float,
    score: np.ndarray,
    count: int,
    pool_fraction: float,
    phase1a: Any,
) -> np.ndarray:
    eligible = np.flatnonzero(proxy > anchor_proxy)
    pool_size = max(count, int(np.ceil(pool_fraction * len(eligible))))
    ranked = eligible[np.argsort(score[eligible], kind="stable")[::-1]]
    pool = ranked[:pool_size]
    local = phase1a.farthest_point_indices(actions[pool], anchor, count)
    return pool[local]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1a-v2 requires a GPU Slurm allocation")
    if not 0 < args.selector_pool_frac <= 1:
        raise ValueError("selector-pool-frac must lie in (0, 1]")
    if args.instability_repeats < 2 or args.jitter_scale <= 0:
        raise ValueError("invalid proxy-instability configuration")
    if args.alternatives <= 0 or args.alternatives >= args.num_samples:
        raise ValueError("invalid alternatives count")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "phase1av2_audit")
    corrected = load_module(DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "phase1av2_corrected")
    phase0d = load_module(PHASE0D, "phase1av2_phase0d")
    phase1a = load_module(PHASE1A, "phase1av2_phase1a")
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest):
        raise ValueError("snapshot-index outside Phase-1a-v2 manifest")
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("Phase-1a-v2 manifest order/index mismatch")
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
            proposal_scale = np.asarray(result["var"][0][0], dtype=np.float32)
            anchor_proxy = float(score_actions(
                cost, prepared, anchor_norm[None], "cuda", solver.dtype
            )[0])
        if recorder.record is None:
            raise RuntimeError("final CEM population was not recorded")
        final_norm = recorder.record["actions_normalized"]
        final_proxy = recorder.record["learned_cost"]
        rng = np.random.default_rng(args.seed + 20_000 + snapshot.order)
        jitter_costs = [final_proxy.astype(np.float64)]
        for _ in range(args.instability_repeats):
            noise = rng.standard_normal(final_norm.shape).astype(np.float32)
            jittered = final_norm + args.jitter_scale * proposal_scale[None] * noise
            jitter_costs.append(score_actions(
                cost, prepared, jittered, "cuda", solver.dtype
            ).astype(np.float64))
        jitter_costs_array = np.stack(jitter_costs)
        cost_scale = np.maximum(np.abs(jitter_costs_array).mean(axis=0), 1e-6)
        proxy_instability = jitter_costs_array.std(axis=0) / cost_scale

        positive_scale = proposal_scale[proposal_scale > 0]
        fallback_scale = float(np.median(positive_scale)) if len(positive_scale) else 1.0
        safe_scale = np.maximum(proposal_scale, max(1e-4, 0.1 * fallback_scale))
        proposal_distance = np.square(
            (final_norm - anchor_norm[None]) / safe_scale[None]
        ).mean(axis=(1, 2))
        cem_disagreement = (
            phase1a.rank_fraction(final_proxy) - phase1a.rank_fraction(proposal_distance)
        )
        selections = {
            "proxy_instability_diverse": top_score_diverse(
                final_norm, anchor_norm, final_proxy, anchor_proxy, proxy_instability,
                args.alternatives, args.selector_pool_frac, phase1a,
            ),
            "cem_disagreement_diverse": top_score_diverse(
                final_norm, anchor_norm, final_proxy, anchor_proxy, cem_disagreement,
                args.alternatives, args.selector_pool_frac, phase1a,
            ),
            "action_diverse": phase1a.farthest_point_indices(
                final_norm, anchor_norm, args.alternatives
            ),
            "random_final_population": np.sort(rng.choice(
                len(final_norm), size=args.alternatives, replace=False
            )),
        }
        # All model-based scores and selections are fixed before this line.
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
        arm_physical: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, indices in selections.items():
            _, distance, success, steps = corrected.rollout_population(
                raw_env, init_row, goal_row, final_raw[indices], audit
            )
            arm_physical[name] = (distance, success, steps)
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
    artifact: dict[str, np.ndarray] = {
        "anchor_actions_normalized": anchor_norm,
        "anchor_actions_raw": anchor_raw,
        "anchor_proxy_cost": np.asarray(anchor_proxy),
        "anchor_physical_distance_m": np.asarray(anchor_distance),
        "anchor_success": np.asarray(anchor_success),
        "final_proxy_cost": final_proxy,
        "proxy_instability": proxy_instability,
        "cem_disagreement": cem_disagreement,
        "proposal_distance": proposal_distance,
    }
    proxy_rank = phase1a.rank_fraction(final_proxy)
    for name in ARM_NAMES:
        indices = selections[name]
        physical, success, executed = arm_physical[name]
        rejected = final_proxy[indices] > anchor_proxy
        advantage = anchor_distance - physical
        corrective = rejected & (advantage >= args.min_physical_gap_m)
        arm_summary[name] = {
            "n_alternatives": int(len(indices)),
            "accounted_physical_queries": int(len(indices) + 1),
            "proxy_rejected_fraction": float(rejected.mean()),
            "mean_proxy_rank_fraction": float(proxy_rank[indices].mean()),
            "mean_action_distance_from_anchor": phase1a.mean_distance(
                final_norm[indices], anchor_norm
            ),
            "mean_proxy_instability": float(proxy_instability[indices].mean()),
            "mean_cem_disagreement": float(cem_disagreement[indices].mean()),
            "inversion_hit": int(corrective.any()),
            "best_corrective_advantage_m": float(np.max(advantage[corrective], initial=0.0)),
            "best_any_advantage_m": float(np.max(advantage, initial=0.0)),
            "any_success_gain": int((not anchor_success) and success.any()),
            "best_selected_distance_m": float(physical.min()),
            "selected_success_any": int(success.any()),
            "selected_indices": [int(index) for index in indices],
        }
        artifact[f"{name}_indices"] = indices
        artifact[f"{name}_physical_distance_m"] = physical
        artifact[f"{name}_success"] = success
        artifact[f"{name}_proxy_cost"] = final_proxy[indices]
        artifact[f"{name}_executed_steps"] = executed
    np.savez_compressed(args.out_dir / "acquisition_audit.npz", **artifact)
    summary = {
        "snapshot": snapshot.order,
        "episode": snapshot.episode,
        "start_step": snapshot.start_step,
        "scope": (
            "All selectors use frozen model/CEM quantities only and are locked before physics. "
            "Every arm is charged the same returned-anchor-plus-K branch budget."
        ),
        "config": {
            "dataset": args.dataset, "checkpoint": args.checkpoint,
            "goal_offset": args.goal_offset, "horizon": args.horizon,
            "action_block": args.action_block, "num_samples": args.num_samples,
            "cem_steps": args.cem_steps, "topk": args.topk, "var_scale": args.var_scale,
            "alternatives": args.alternatives,
            "physical_budget_per_arm": args.alternatives + 1,
            "instability_repeats": args.instability_repeats,
            "jitter_scale": args.jitter_scale,
            "selector_pool_frac": args.selector_pool_frac,
            "min_physical_gap_m": args.min_physical_gap_m,
            "seed": args.seed,
        },
        "visual_signature": visual_hash,
        "visual_signature_shapes": visual_shapes,
        "repeat_gate": repeat_gate,
        "anchor": {
            "proxy_cost": anchor_proxy, "physical_distance_m": anchor_distance,
            "success": int(anchor_success), "executed_steps": int(anchor_steps_arr[0]),
            "actions_raw_sha256": phase0d.array_hash(anchor_raw),
        },
        "diagnostics": {
            "mean_proxy_instability": float(proxy_instability.mean()),
            "mean_abs_cem_disagreement": float(np.abs(cem_disagreement).mean()),
            "median_proposal_scale": float(np.median(proposal_scale)),
        },
        "arms": arm_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
