#!/usr/bin/env python3
"""Collect exact same-state physical labels for two planner-induced populations."""

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
from physical_search_distillation.core import hard_elite_refit, split_for_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=96)
    parser.add_argument("--cem-steps", type=int, default=12)
    parser.add_argument("--topk", type=int, default=10)
    # No argparse default: action="append" appends to it rather than replacing
    # it, so a caller-supplied pair silently became [0, 11, ...].  The
    # fallback below reproduces the old default exactly at --cem-steps 12.
    parser.add_argument("--record-step", type=int, action="append", default=None)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--physical-atol", type=float, default=1e-5)
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


def normalized_to_raw(
    normalized: np.ndarray, scaler: Any, horizon: int, action_block: int, raw_dim: int
) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float32)
    if normalized.shape[-2:] != (horizon, raw_dim * action_block):
        raise RuntimeError(f"unexpected normalized shape {normalized.shape}")
    return scaler.inverse_transform(normalized.reshape(-1, raw_dim)).reshape(
        *normalized.shape[:-2], horizon, action_block, raw_dim
    )


class RecordingEvaluator(torch.nn.Module):
    """GoalMSE evaluator exposing the deployable rollout features to a callback."""

    def __init__(self, base: Any) -> None:
        super().__init__()
        self.base = base
        self.last: dict[str, torch.Tensor] | None = None

    def get_cost(self, info: dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        rolled = self.base._rollout(info, actions)
        cost = self.base.objective(rolled)
        goal = rolled["goal_emb"][:, -1]
        endpoint = rolled["predicted_emb"][:, :, -1]
        current = rolled["emb"][:, :, -1]
        self.last = {
            "endpoint": endpoint.detach(), "goal": goal.detach(), "current": current.detach()
        }
        return cost


class PopulationRecorder:
    name = "perd_populations"

    def __init__(self, evaluator: RecordingEvaluator, steps: set[int]) -> None:
        self.evaluator = evaluator
        self.steps = steps
        self.records: list[dict[str, np.ndarray]] = []
        self.history: list[Any] = []

    @property
    def output_key(self) -> str:
        return self.name

    def reset(self) -> None:
        self.records = []
        self.history = []

    def start_batch(self) -> None:
        pass

    def end_solve(self) -> None:
        pass

    def __call__(self, **state: Any) -> None:
        step = int(state["step"])
        if step not in self.steps:
            return
        if self.evaluator.last is None:
            raise RuntimeError("callback did not receive rollout features")
        self.records.append({
            "step": np.asarray(step, dtype=np.int64),
            "actions_normalized": state["candidates"][0].detach().float().cpu().numpy(),
            "native_cost": state["costs"][0].detach().float().cpu().numpy(),
            "predicted_endpoint": self.evaluator.last["endpoint"][0].float().cpu().numpy(),
            "current_embedding": self.evaluator.last["current"][0, 0].float().cpu().numpy(),
            "goal_embedding": self.evaluator.last["goal"][0].float().cpu().numpy(),
            "proposal_mean": state["prev_mean"][0].detach().float().cpu().numpy(),
            "proposal_std": state["prev_var"][0].detach().float().cpu().numpy(),
            "native_elite": state["topk_inds"][0].detach().cpu().numpy(),
        })


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("collection requires a GPU Slurm allocation")
    record_steps = sorted(set(
        args.record_step if args.record_step else [0, args.cem_steps - 1]))
    if record_steps != [0, args.cem_steps - 1]:
        raise ValueError("H0 protocol requires first and final CEM populations")
    if not 1 < args.topk <= args.num_samples:
        raise ValueError("invalid topk")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "perd_audit")
    corrected = load_module(DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "perd_corrected")
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    row = manifest[args.snapshot_index]
    snapshot = audit.Snapshot(**row)
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("manifest order/index mismatch")

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
    base = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
    evaluator = RecordingEvaluator(base)
    recorder = PopulationRecorder(evaluator, set(record_steps))
    transform = audit.make_transform(224)
    world, raw_env, visual_hash, visual_shapes = corrected.make_world(swm, snapshot)
    try:
        raw_dim = int(np.prod(world.envs.single_action_space.shape))
        solver = swm.planning.CEMSolver(
            cost=evaluator, batch_size=1, num_samples=args.num_samples,
            var_scale=args.var_scale, n_steps=args.cem_steps, topk=args.topk,
            device="cuda", seed=args.seed + snapshot.order, callbacks=[recorder],
        )
        config = swm.PlanConfig(
            horizon=args.horizon, receding_horizon=args.horizon,
            action_block=args.action_block, history_len=1, warm_start=True,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process={"action": scaler},
            transform={"pixels": transform, "goal": transform},
        )
        policy.set_env(world.envs)
        raw_info = {
            "pixels": np.asarray(init_row["pixels"])[None, None],
            "goal": np.asarray(goal_row["goal"])[None, None],
            "action": np.full((1, 1, raw_dim), np.nan, dtype=np.float32),
        }
        result = solver.solve(policy._prepare_info(raw_info))
        if len(recorder.records) != 2:
            raise RuntimeError(f"expected two recorded populations, got {len(recorder.records)}")

        physical, success, executed, teacher_elite, teacher_mean, teacher_std = [], [], [], [], [], []
        for record in recorder.records:
            actions_raw = normalized_to_raw(
                record["actions_normalized"], scaler, args.horizon, args.action_block, raw_dim
            )
            _, dist, succ, n_exec = corrected.rollout_population(
                raw_env, init_row, goal_row, actions_raw, audit
            )
            elite, mean, std = hard_elite_refit(
                record["actions_normalized"], dist, args.topk
            )
            physical.append(dist); success.append(succ); executed.append(n_exec)
            teacher_elite.append(elite); teacher_mean.append(mean); teacher_std.append(std)

        returned_norm = np.asarray(result["actions"][0], dtype=np.float32)
        returned_raw = normalized_to_raw(
            returned_norm, scaler, args.horizon, args.action_block, raw_dim
        )
        _, ret_dist, ret_success, ret_exec = corrected.rollout_population(
            raw_env, init_row, goal_row, returned_raw[None], audit
        )
        _, replay_dist, replay_success, replay_exec = corrected.rollout_population(
            raw_env, init_row, goal_row, returned_raw[None], audit
        )
    finally:
        world.close()

    if (
        abs(float(ret_dist[0]) - float(replay_dist[0])) > args.physical_atol
        or bool(ret_success[0]) != bool(replay_success[0])
        or int(ret_exec[0]) != int(replay_exec[0])
    ):
        raise RuntimeError("same-state replay gate failed")

    arrays = {key: np.stack([record[key] for record in recorder.records]) for key in (
        "step", "actions_normalized", "native_cost", "predicted_endpoint",
        "current_embedding", "goal_embedding", "proposal_mean", "proposal_std", "native_elite",
    )}
    out = args.out_dir / f"snapshot_{snapshot.order:03d}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "populations.npz", **arrays,
        physical_distance_m=np.stack(physical), success=np.stack(success),
        executed_steps=np.stack(executed), teacher_elite=np.stack(teacher_elite),
        teacher_mean=np.stack(teacher_mean), teacher_std=np.stack(teacher_std),
        returned_actions_normalized=returned_norm,
        returned_physical_distance_m=ret_dist[0], returned_success=ret_success[0],
    )
    summary = {
        "snapshot": snapshot.order, "episode": snapshot.episode,
        "split": split_for_order(snapshot.order), "record_steps": record_steps,
        "visual_signature": visual_hash, "visual_signature_shapes": visual_shapes,
        "candidate_rollouts": int(len(record_steps) * args.num_samples),
        "teacher_best_distance_m": [float(np.min(x)) for x in physical],
        "native_returned_distance_m": float(ret_dist[0]),
        "native_returned_success": int(ret_success[0]),
        "returned_action_sha256": sha256(returned_raw), "repeat_gate": True,
        "config": vars(args) | {"manifest": str(args.manifest), "out_dir": str(args.out_dir)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
