#!/usr/bin/env python3
"""Evaluate native and learned objectives with zero physical queries inside CEM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from torch import nn


REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "diagnosis"
sys.path.insert(0, str(REPO))
from physical_search_distillation.core import ARMS, load_checkpoint, split_for_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=96)
    parser.add_argument("--cem-steps", type=int, default=12)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--physical-atol", type=float, default=1e-5)
    return parser.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def normalized_to_raw(normalized: np.ndarray, scaler: Any, horizon: int, block: int, raw_dim: int) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float32)
    return scaler.inverse_transform(normalized.reshape(-1, raw_dim)).reshape(
        *normalized.shape[:-2], horizon, block, raw_dim
    )


def robust_features_torch(cost: torch.Tensor) -> torch.Tensor:
    median = cost.median(dim=1, keepdim=True).values
    mad = (cost - median).abs().median(dim=1, keepdim=True).values.clamp_min(1e-6)
    robust = (cost - median) / mad
    order = torch.argsort(cost, dim=1, stable=True)
    rank = torch.empty_like(cost)
    values = torch.arange(cost.shape[1], device=cost.device, dtype=cost.dtype)
    values = values[None].expand_as(cost) / max(cost.shape[1] - 1, 1)
    rank.scatter_(1, order, values)
    return torch.stack([robust, rank, torch.log1p(cost.clamp_min(0))], dim=-1)


class LearnedCostEvaluator(nn.Module):
    def __init__(self, base: nn.Module, models: list[nn.Module], payloads: list[dict]) -> None:
        super().__init__()
        self.base = base
        self.models = nn.ModuleList(models)
        self.arm = payloads[0]["arm"]
        if any(payload["arm"] != self.arm for payload in payloads):
            raise ValueError("mixed-arm ensemble")
        device = next(models[0].parameters()).device
        self.register_buffer(
            "feature_mean", torch.from_numpy(payloads[0]["feature_mean"]).float().to(device)
        )
        self.register_buffer(
            "feature_std", torch.from_numpy(payloads[0]["feature_std"]).float().to(device)
        )

    def get_cost(self, info: dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        rolled = self.base._rollout(info, actions)
        native = self.base.objective(rolled)
        native_features = robust_features_torch(native)
        endpoint = rolled["predicted_emb"][:, :, -1].float()
        current = rolled["emb"][:, :, -1].float()
        goal = rolled["goal_emb"][:, -1].float()[:, None].expand_as(endpoint)
        if self.arm == "operator_metric":
            predictions = [model(actions.float(), native_features, endpoint, goal) for model in self.models]
        else:
            residual = endpoint - goal
            context = current - goal
            norms = torch.stack(
                [residual.norm(dim=-1), actions.float().flatten(start_dim=2).norm(dim=-1)], dim=-1
            )
            features = torch.cat([
                actions.float().flatten(start_dim=2), native_features, endpoint, residual,
                residual.square(), context, norms,
            ], dim=-1)
            features = (features - self.feature_mean) / self.feature_std
            predictions = [model(features) for model in self.models]
        return torch.stack(predictions).mean(dim=0)


def load_ensemble(root: Path, arm: str) -> tuple[list[nn.Module], list[dict]]:
    paths = sorted((root / arm).glob("seed_*.pt"))
    if len(paths) != 3:
        raise RuntimeError(f"expected three checkpoints for {arm}, got {paths}")
    loaded = [load_checkpoint(path, "cuda") for path in paths]
    return [item[0] for item in loaded], [item[1] for item in loaded]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires a GPU Slurm allocation")
    manifest = json.loads(args.manifest.read_text())
    row = manifest[args.snapshot_index]
    if split_for_order(row["order"]) != "test":
        raise ValueError("evaluation accepts held-out test states only")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "perd_eval_audit")
    corrected = load_module(DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "perd_eval_corrected")
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    snapshot = audit.Snapshot(**row)
    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = StandardScaler().fit(action_data)
    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False); model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)
    world, raw_env, visual_hash, _ = corrected.make_world(swm, snapshot)
    results: dict[str, dict] = {}
    plans: dict[str, np.ndarray] = {}
    try:
        raw_dim = int(np.prod(world.envs.single_action_space.shape))
        raw_info = {
            "pixels": np.asarray(init_row["pixels"])[None, None],
            "goal": np.asarray(goal_row["goal"])[None, None],
            "action": np.full((1, 1, raw_dim), np.nan, dtype=np.float32),
        }
        config = swm.PlanConfig(
            horizon=args.horizon, receding_horizon=args.horizon,
            action_block=args.action_block, history_len=1, warm_start=True,
        )
        base = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
        objectives: dict[str, nn.Module] = {"native": base}
        for arm in args.arms:
            models, payloads = load_ensemble(args.checkpoints, arm)
            objectives[arm] = LearnedCostEvaluator(base, models, payloads)

        for name, objective in objectives.items():
            solver = swm.planning.CEMSolver(
                cost=objective, batch_size=1, num_samples=args.num_samples,
                var_scale=args.var_scale, n_steps=args.cem_steps, topk=args.topk,
                device="cuda", seed=args.seed + snapshot.order,
            )
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process={"action": scaler},
                transform={"pixels": transform, "goal": transform},
            )
            policy.set_env(world.envs)
            start = time.perf_counter()
            solved = solver.solve(policy._prepare_info(raw_info))
            solve_seconds = time.perf_counter() - start
            plan_norm = np.asarray(solved["actions"][0], dtype=np.float32)
            plan_raw = normalized_to_raw(plan_norm, scaler, args.horizon, args.action_block, raw_dim)
            _, distance, success, executed = corrected.rollout_population(
                raw_env, init_row, goal_row, plan_raw[None], audit
            )
            plans[name] = plan_norm
            results[name] = {
                "physical_distance_m": float(distance[0]), "success": int(success[0]),
                "executed_steps": int(executed[0]), "solve_seconds": solve_seconds,
                "physical_queries_during_planning": 0,
            }

        native_raw = normalized_to_raw(plans["native"], scaler, args.horizon, args.action_block, raw_dim)
        _, replay_distance, replay_success, replay_executed = corrected.rollout_population(
            raw_env, init_row, goal_row, native_raw[None], audit
        )
    finally:
        world.close()

    native = results["native"]
    repeat_gate = bool(
        abs(native["physical_distance_m"] - float(replay_distance[0])) <= args.physical_atol
        and native["success"] == int(replay_success[0])
        and native["executed_steps"] == int(replay_executed[0])
    )
    if not repeat_gate:
        raise RuntimeError("native same-state replay failed")
    out = args.out_dir / f"snapshot_{snapshot.order:03d}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "plans.npz", **{f"{name}_normalized": value for name, value in plans.items()})
    summary = {
        "snapshot": snapshot.order, "episode": snapshot.episode, "split": "test",
        "visual_signature": visual_hash, "repeat_gate": repeat_gate, "results": results,
        "protocol": "zero-query CEM; one post-hoc physical execution per returned plan",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
