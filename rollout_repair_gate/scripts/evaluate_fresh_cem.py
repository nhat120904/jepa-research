#!/usr/bin/env python3
"""Re-optimize CEM under every checkpoint and execute the returned plan."""

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
sys.path.insert(0, str(REPO))

from rollout_repair_gate.core import (
    dynamics_state_dict,
    load_dynamics_state,
    normalized_to_raw,
    split_for_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=96)
    parser.add_argument("--cem-steps", type=int, default=12)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--repeat-atol", type=float, default=1e-5)
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


def checkpoint_specs(root: Path) -> list[tuple[str, dict]]:
    specs = []
    signatures = set()
    for path in sorted(root.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        specs.append((f"{metadata['arm']}_seed{metadata['seed']}", payload["dynamics"]))
        signatures.add(json.dumps(metadata["same_compute_signature"], sort_keys=True))
    if len(specs) != 9 or len(signatures) != 1:
        raise RuntimeError(
            f"checkpoint completeness/same-compute failure: n={len(specs)}, signatures={len(signatures)}"
        )
    return specs


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("fresh CEM evaluation requires a GPU Slurm allocation")
    if split_for_order(args.snapshot_index) != "test":
        raise ValueError("fresh CEM evaluation is restricted to immutable test snapshots")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "rrg_cem_audit")
    corrected = load_module(
        DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "rrg_cem_corrected"
    )
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = StandardScaler().fit(action_data)
    transform = audit.make_transform(224)
    world, raw_env, _, _ = corrected.make_world(swm, snapshot)
    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    specs = [("native", dynamics_state_dict(model))] + checkpoint_specs(args.checkpoint_dir)
    raw_dim = int(np.prod(world.envs.single_action_space.shape))
    raw_info = {
        "pixels": np.asarray(init_row["pixels"])[None, None],
        "goal": np.asarray(goal_row["goal"])[None, None],
        "action": np.full((1, 1, raw_dim), np.nan, dtype=np.float32),
    }

    rows = []
    try:
        for label, state in specs:
            load_dynamics_state(model, state)
            model.eval()
            cost = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
            solver = swm.planning.CEMSolver(
                cost=cost,
                batch_size=1,
                num_samples=args.num_samples,
                var_scale=1.0,
                n_steps=args.cem_steps,
                topk=args.topk,
                device="cuda",
                seed=args.seed + snapshot.order,
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
            with torch.inference_mode():
                result = solver.solve(policy._prepare_info(raw_info))
            normalized = np.asarray(result["actions"][0], dtype=np.float32)
            raw = normalized_to_raw(
                normalized[None], scaler, args.horizon, args.action_block, raw_dim
            )
            _, distance, success, executed = corrected.rollout_population(
                raw_env, init_row, goal_row, raw, audit
            )
            _, distance_repeat, success_repeat, executed_repeat = corrected.rollout_population(
                raw_env, init_row, goal_row, raw, audit
            )
            if (
                abs(float(distance[0]) - float(distance_repeat[0])) > args.repeat_atol
                or bool(success[0]) != bool(success_repeat[0])
                or int(executed[0]) != int(executed_repeat[0])
            ):
                raise RuntimeError(f"physical replay failed for {label}")
            rows.append(
                {
                    "snapshot": snapshot.order,
                    "episode": snapshot.episode,
                    "label": label,
                    "physical_distance_m": float(distance[0]),
                    "success": int(success[0]),
                    "executed_steps": int(executed[0]),
                    "returned_normalized_actions": normalized.tolist(),
                }
            )
    finally:
        world.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"snapshot_{snapshot.order:03d}.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"snapshot": snapshot.order, "models": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

