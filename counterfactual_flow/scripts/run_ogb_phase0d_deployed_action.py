#!/usr/bin/env python3
"""Instrument the actual CEM plan returned by LeWM on OGBench-Cube.

Phase 0c could only score the proxy argmin inside a persisted population.  In
the upstream solver that is not the deployed plan: CEM returns the mean after
the last elite refit.  This script re-runs the frozen solver, persists that
returned mean, and evaluates it and every final-step candidate from the exact
same restored MuJoCo state.  The simulator remains strictly post-hoc.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=DIAG / "results/ogb_stage0/audit_locked/manifest.json",
    )
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260810)
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


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(value.tobytes()).hexdigest()


class FinalPopulationRecorder:
    """Persist the final sampled population before its elite update."""

    name = "phase0d_final_population"

    def __init__(self, final_step: int) -> None:
        self.final_step = final_step
        self.record: dict[str, np.ndarray] | None = None
        self.history: list[Any] = []

    @property
    def output_key(self) -> str:
        return self.name

    def reset(self) -> None:
        self.record = None
        self.history = []

    def start_batch(self) -> None:
        pass

    def end_solve(self) -> None:
        pass

    def __call__(self, **state: Any) -> None:
        if int(state["step"]) != self.final_step:
            return
        self.record = {
            "actions_normalized": state["candidates"][0].detach().float().cpu().numpy(),
            "learned_cost": state["costs"][0].detach().float().cpu().numpy(),
        }


def expand_info_for_cost(info: dict[str, Any], device: str, dtype: torch.dtype) -> dict[str, Any]:
    """Mirror CEMSolver's one-sample expansion for exact mean-plan scoring."""
    expanded: dict[str, Any] = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            target_dtype = dtype if value.is_floating_point() else None
            expanded[key] = value.to(device=device, dtype=target_dtype).unsqueeze(1)
        elif isinstance(value, np.ndarray):
            expanded[key] = value[:, None, ...]
        else:
            expanded[key] = value
    return expanded


def normalized_to_raw(
    normalized: np.ndarray, scaler: Any, horizon: int, action_block: int, raw_dim: int
) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float32)
    if normalized.shape[-2:] != (horizon, raw_dim * action_block):
        raise RuntimeError(f"unexpected normalized plan shape {normalized.shape}")
    flat = normalized.reshape(-1, raw_dim)
    return scaler.inverse_transform(flat).reshape(
        *normalized.shape[:-2], horizon, action_block, raw_dim
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 0d requires a GPU Slurm allocation")
    if args.cem_steps < 2 or args.topk > args.num_samples:
        raise ValueError("invalid CEM configuration")
    if args.min_physical_gap_m <= 0 or args.physical_atol <= 0:
        raise ValueError("physical thresholds must be positive")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "phase0d_audit")
    corrected = load_module(
        DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "phase0d_corrected"
    )
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest):
        raise ValueError("snapshot-index outside locked manifest")
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("locked-manifest index mismatch")

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
        recorder = FinalPopulationRecorder(final_step=args.cem_steps - 1)
        cost = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
        solver = swm.planning.CEMSolver(
            cost=cost,
            batch_size=1,
            num_samples=args.num_samples,
            var_scale=args.var_scale,
            n_steps=args.cem_steps,
            topk=args.topk,
            device="cuda",
            seed=args.seed + snapshot.order,
            callbacks=[recorder],
        )
        config = swm.PlanConfig(
            horizon=args.horizon,
            receding_horizon=args.horizon,
            action_block=args.action_block,
            history_len=1,
            warm_start=True,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
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
        prepared = policy._prepare_info(raw_info)
        with torch.inference_mode():
            result = solver.solve(prepared)
            returned_norm = np.asarray(result["actions"][0], dtype=np.float32)
            returned_tensor = torch.as_tensor(result["actions"], device="cuda")
            returned_proxy = float(
                cost.get_cost(
                    expand_info_for_cost(prepared, "cuda", solver.dtype),
                    returned_tensor.unsqueeze(1),
                )[0, 0].detach().cpu()
            )
        if recorder.record is None:
            raise RuntimeError("final CEM population was not recorded")
        final_norm = recorder.record["actions_normalized"]
        final_proxy = recorder.record["learned_cost"]
        returned_raw = normalized_to_raw(
            returned_norm, scaler, args.horizon, args.action_block, raw_dim
        )
        final_raw = normalized_to_raw(
            final_norm, scaler, args.horizon, args.action_block, raw_dim
        )

        _, final_physical, final_success, final_executed = corrected.rollout_population(
            raw_env, init_row, goal_row, final_raw, audit
        )
        _, returned_physical_arr, returned_success_arr, returned_executed_arr = corrected.rollout_population(
            raw_env, init_row, goal_row, returned_raw[None], audit
        )
        _, replay_physical, replay_success, replay_executed = corrected.rollout_population(
            raw_env, init_row, goal_row, returned_raw[None], audit
        )
    finally:
        world.close()

    returned_physical = float(returned_physical_arr[0])
    returned_success = bool(returned_success_arr[0])
    returned_executed = int(returned_executed_arr[0])
    repeat_gate = {
        "physical_max_abs": corrected.max_abs(returned_physical_arr, replay_physical),
        "success_disagreements": int(np.sum(returned_success_arr != replay_success)),
        "executed_max_abs": corrected.max_abs(returned_executed_arr, replay_executed),
    }
    repeat_gate["pass"] = bool(
        repeat_gate["physical_max_abs"] <= args.physical_atol
        and repeat_gate["success_disagreements"] == 0
        and repeat_gate["executed_max_abs"] == 0
    )
    if not repeat_gate["pass"]:
        raise RuntimeError(f"returned-plan replay failed: {repeat_gate}")

    physical_best = int(np.argmin(final_physical))
    corrective_mask = (final_proxy > returned_proxy) & (
        final_physical <= returned_physical - args.min_physical_gap_m
    )
    corrective = (
        int(np.flatnonzero(corrective_mask)[np.argmin(final_physical[corrective_mask])])
        if corrective_mask.any()
        else None
    )
    np.savez_compressed(
        args.out_dir / "deployed_action_audit.npz",
        returned_actions_normalized=returned_norm,
        returned_actions_raw=returned_raw,
        returned_proxy_cost=np.asarray(returned_proxy),
        returned_physical_distance_m=np.asarray(returned_physical),
        final_actions_normalized=final_norm,
        final_actions_raw=final_raw,
        final_proxy_cost=final_proxy,
        final_physical_distance_m=final_physical,
        final_success=final_success,
        final_executed_steps=final_executed,
    )
    summary: dict[str, Any] = {
        "snapshot": snapshot.order,
        "episode": snapshot.episode,
        "start_step": snapshot.start_step,
        "scope": (
            "CEM-returned final proposal mean replayed from a complete MuJoCo state; "
            "simulator outcomes are post-hoc and were not exposed to CEM."
        ),
        "config": {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "goal_offset": args.goal_offset,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "num_samples": args.num_samples,
            "cem_steps": args.cem_steps,
            "topk": args.topk,
            "var_scale": args.var_scale,
            "seed": args.seed,
        },
        "visual_signature": visual_hash,
        "visual_signature_shapes": visual_shapes,
        "repeat_gate": repeat_gate,
        "returned": {
            "proxy_cost": returned_proxy,
            "physical_distance_m": returned_physical,
            "success": int(returned_success),
            "executed_steps": returned_executed,
            "actions_raw_sha256": array_hash(returned_raw),
        },
        "final_population": {
            "n_candidates": int(len(final_proxy)),
            "physical_oracle_candidate": physical_best,
            "physical_oracle_distance_m": float(final_physical[physical_best]),
            "physical_oracle_success": int(final_success.any()),
            "returned_selection_regret_m": float(returned_physical - final_physical[physical_best]),
            "returned_success_gap": int(final_success.any()) - int(returned_success),
            "proxy_better_than_returned_fraction": float(np.mean(final_proxy < returned_proxy)),
        },
        "verified_proxy_rejected_corrective": {
            "exists": int(corrective is not None),
            "criterion": (
                "final candidate has strictly higher frozen proxy cost than the returned "
                f"CEM mean and at least {args.min_physical_gap_m * 100:.1f} cm lower physical distance"
            ),
        },
    }
    if corrective is not None:
        summary["verified_proxy_rejected_corrective"].update({
            "candidate": corrective,
            "proxy_cost": float(final_proxy[corrective]),
            "physical_distance_m": float(final_physical[corrective]),
            "physical_advantage_m": float(returned_physical - final_physical[corrective]),
            "success": int(final_success[corrective]),
        })
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
