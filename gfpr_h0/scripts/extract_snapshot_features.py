#!/usr/bin/env python3
"""Extract deployable GFPR features for one fully-labelled Phase-0d snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / "gfpr_h0"
DIAG = REPO / "diagnosis"
sys.path.insert(0, str(REPO))
from gfpr_h0.core import build_feature_views  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DIAG / "results/ogb_stage0/audit_locked/manifest.json",
    )
    parser.add_argument(
        "--phase0d-root",
        type=Path,
        default=REPO
        / "counterfactual_flow/outputs/ogbench_cube_phase0/phase0d_shards",
    )
    parser.add_argument(
        "--complementarity-root",
        type=Path,
        default=REPO / "crod_h0/outputs/complementarity/shards",
    )
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--cost-atol", type=float, default=2e-2)
    parser.add_argument("--cost-rtol", type=float, default=2e-4)
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
    result: dict[str, Any] = {}
    for key, value in prepared.items():
        if torch.is_tensor(value):
            target_dtype = dtype if value.is_floating_point() else None
            value = value.to(device=device, dtype=target_dtype)
            result[key] = value.unsqueeze(1).expand(
                value.shape[0], samples, *value.shape[1:]
            )
        elif isinstance(value, np.ndarray):
            result[key] = np.repeat(value[:, None, ...], samples, axis=1)
        else:
            result[key] = value
    return result


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def rank_diagnostics(extracted: np.ndarray, persisted: np.ndarray) -> dict[str, Any]:
    """Check that small device-level numeric drift preserves candidate ordering."""
    extracted = np.asarray(extracted, dtype=np.float64)
    persisted = np.asarray(persisted, dtype=np.float64)
    extracted_order = np.argsort(extracted, kind="stable")
    persisted_order = np.argsort(persisted, kind="stable")
    extracted_rank = np.empty(len(extracted), dtype=np.int64)
    persisted_rank = np.empty(len(persisted), dtype=np.int64)
    extracted_rank[extracted_order] = np.arange(len(extracted))
    persisted_rank[persisted_order] = np.arange(len(persisted))
    correlation = float(np.corrcoef(extracted_rank, persisted_rank)[0, 1])
    return {
        "spearman": correlation,
        "argmin_match": bool(extracted_order[0] == persisted_order[0]),
        "top10_set_match": bool(
            set(extracted_order[:10].tolist()) == set(persisted_order[:10].tolist())
        ),
        "max_rank_shift": int(np.max(np.abs(extracted_rank - persisted_rank))),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("feature extraction requires a GPU Slurm allocation")
    if args.cost_atol <= 0 or args.cost_rtol <= 0:
        raise ValueError("cost tolerances must be positive")

    audit = load_module(
        DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "gfpr_audit"
    )
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest):
        raise ValueError("snapshot-index outside manifest")
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("manifest order mismatch")

    phase_path = (
        args.phase0d_root / str(args.snapshot_index) / "deployed_action_audit.npz"
    )
    comp_path = (
        args.complementarity_root
        / str(args.snapshot_index)
        / "complementarity_audit.npz"
    )
    with np.load(phase_path, allow_pickle=False) as artifact:
        anchor_action = np.asarray(
            artifact["returned_actions_normalized"], dtype=np.float32
        )
        candidate_actions = np.asarray(
            artifact["final_actions_normalized"], dtype=np.float32
        )
        anchor_native = float(artifact["returned_proxy_cost"])
        candidate_native = np.asarray(
            artifact["final_proxy_cost"], dtype=np.float64
        )
        anchor_physical = float(artifact["returned_physical_distance_m"])
        candidate_physical = np.asarray(
            artifact["final_physical_distance_m"], dtype=np.float64
        )
        candidate_success = np.asarray(artifact["final_success"], dtype=bool)
    with np.load(comp_path, allow_pickle=False) as artifact:
        candidate_auxiliary = np.asarray(
            artifact["final_auxiliary_cost"], dtype=np.float64
        )

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler = StandardScaler().fit(action_data)

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)
    evaluator = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
    dummy_solver = swm.planning.CEMSolver(
        cost=evaluator,
        batch_size=1,
        num_samples=2,
        n_steps=2,
        topk=1,
        device="cuda",
        seed=0,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=dummy_solver,
        config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        process={"action": scaler},
        transform={"pixels": transform, "goal": transform},
    )
    raw_info = {
        "pixels": np.asarray(init_rows[0]["pixels"])[None, None],
        "goal": np.asarray(goal_rows[0]["goal"])[None, None],
        "action": np.full((1, 1, action_data.shape[1]), np.nan, dtype=np.float32),
        "id": np.asarray([[snapshot.order]], dtype=np.int64),
        "step_idx": np.asarray([[snapshot.start_step]], dtype=np.int64),
    }
    prepared = policy._prepare_info(raw_info)
    all_actions = np.concatenate([anchor_action[None], candidate_actions], axis=0)
    action_tensor = torch.as_tensor(
        all_actions, device="cuda", dtype=next(model.parameters()).dtype
    ).unsqueeze(0)
    with torch.inference_mode():
        rolled = evaluator._rollout(
            expand_info(prepared, len(all_actions), "cuda", action_tensor.dtype),
            action_tensor,
        )
        extracted_cost = evaluator.criterion(rolled, action_tensor)[0]
        predicted = rolled["predicted_emb"][0, :, -1]
        current = rolled["emb"][0, 0, -1]
        goal = rolled["goal_emb"][0, -1]
        extracted_cost = extracted_cost.detach().float().cpu().numpy()
        predicted = predicted.detach().float().cpu().numpy()
        current = current.detach().float().cpu().numpy()
        goal = goal.detach().float().cpu().numpy()

    persisted_cost = np.r_[anchor_native, candidate_native]
    cost_error = max_abs(extracted_cost, persisted_cost)
    cost_relative_error = cost_error / max(float(np.max(np.abs(persisted_cost))), 1.0)
    ordering = rank_diagnostics(extracted_cost[1:], persisted_cost[1:])
    reproduction_pass = bool(
        cost_error <= args.cost_atol
        and cost_relative_error <= args.cost_rtol
        and ordering["spearman"] >= 0.9999
        and ordering["argmin_match"]
        and ordering["top10_set_match"]
    )
    if not reproduction_pass:
        raise RuntimeError(
            "extracted LeWM cost does not reproduce Phase-0d: "
            f"max_abs={cost_error}, max_relative={cost_relative_error}, "
            f"ordering={ordering}"
        )

    views = build_feature_views(
        candidate_actions,
        anchor_action,
        candidate_native,
        candidate_auxiliary,
        predicted[1:],
        predicted[0],
        current,
        goal,
    )
    physical_gain_cm = (anchor_physical - candidate_physical) * 100.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "features.npz",
        snapshot_index=np.asarray(snapshot.order),
        episode=np.asarray(snapshot.episode),
        anchor_action=anchor_action,
        candidate_actions=candidate_actions,
        anchor_native_cost=np.asarray(anchor_native),
        candidate_native_cost=candidate_native,
        candidate_auxiliary_cost=candidate_auxiliary,
        anchor_physical_distance_m=np.asarray(anchor_physical),
        candidate_physical_distance_m=candidate_physical,
        candidate_success=candidate_success,
        physical_gain_cm=physical_gain_cm,
        action_only=views["action_only"],
        proxy_action=views["proxy_action"],
        latent_context=views["latent_context"],
    )
    summary = {
        "snapshot_index": snapshot.order,
        "episode": snapshot.episode,
        "n_candidates": int(len(candidate_actions)),
        "feature_dimensions": {key: int(value.shape[1]) for key, value in views.items()},
        "cost_reproduction_max_abs": cost_error,
        "cost_reproduction_atol": args.cost_atol,
        "cost_reproduction_rtol": args.cost_rtol,
        "cost_ordering_gate": ordering,
        "cost_reproduction_max_relative": cost_relative_error,
        "cost_reproduction_pass": reproduction_pass,
        "corrective_candidates_2cm": int(np.sum(physical_gain_cm >= 2.0)),
        "physical_oracle_gain_cm": float(max(0.0, np.max(physical_gain_cm))),
        "contains_physical_input_features": False,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
