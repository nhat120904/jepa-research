#!/usr/bin/env python3
"""Add DINO-WM ranks to the already fully-labelled Phase-0d populations."""

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
DIAG = REPO / "diagnosis"
PROJECT = REPO / "crod_h0"
sys.path.insert(0, str(PROJECT))
from core import directional_ordinal_score  # noqa: E402


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
        default=REPO / "counterfactual_flow/outputs/ogbench_cube_phase0/phase0d_shards",
    )
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument(
        "--auxiliary-checkpoint",
        default="crod_dinowm_cube_seed42/weights_epoch_10.pt",
    )
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--physical-margin-m", type=float, default=0.02)
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


@torch.inference_mode()
def score(model: Any, prepared: dict[str, Any], actions: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(actions, device="cuda", dtype=torch.float32).unsqueeze(0)
    values = model.get_cost(
        expand_info(prepared, len(actions), "cuda", torch.float32), tensor
    )
    return values[0].detach().float().cpu().numpy()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("complementarity audit requires a GPU Slurm allocation")
    if args.physical_margin_m <= 0:
        raise ValueError("physical margin must be positive")
    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "crod_audit_info")
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    artifact_path = args.phase0d_root / str(args.snapshot_index) / "deployed_action_audit.npz"
    with np.load(artifact_path, allow_pickle=False) as artifact:
        anchor_actions = np.asarray(artifact["returned_actions_normalized"], dtype=np.float32)
        final_actions = np.asarray(artifact["final_actions_normalized"], dtype=np.float32)
        anchor_native = float(artifact["returned_proxy_cost"])
        final_native = np.asarray(artifact["final_proxy_cost"], dtype=np.float64)
        anchor_physical = float(artifact["returned_physical_distance_m"])
        final_physical = np.asarray(artifact["final_physical_distance_m"], dtype=np.float64)

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    scaler = StandardScaler()
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    scaler.fit(action_data)
    transform = audit.make_transform(224)
    auxiliary = swm.wm.utils.load_pretrained(args.auxiliary_checkpoint).cuda().eval()
    auxiliary.requires_grad_(False)
    auxiliary.interpolate_pos_encoding = True
    if auxiliary.__class__.__name__ != "PreJEPA" or set(auxiliary.extra_encoders) != {"action"}:
        raise RuntimeError("audit auxiliary is not action-only DINO-WM")

    # A policy object provides exactly the preprocessing used by planning; its
    # solver is never called in this rescore-only audit.
    dummy_cost = swm.planning.ShootingCostEvaluator(auxiliary, swm.planning.GoalMSE())
    dummy_solver = swm.planning.CEMSolver(
        cost=dummy_cost,
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
    auxiliary_cost = score(
        auxiliary, prepared, np.concatenate([anchor_actions[None], final_actions], axis=0)
    )
    anchor_auxiliary = float(auxiliary_cost[0])
    final_auxiliary = auxiliary_cost[1:]
    crod_score, native_rank, auxiliary_rank = directional_ordinal_score(
        final_native, final_auxiliary, anchor_native, anchor_auxiliary
    )

    candidate_better = final_physical <= anchor_physical - args.physical_margin_m
    anchor_better = anchor_physical <= final_physical - args.physical_margin_m
    informative = candidate_better | anchor_better
    native_prefers_candidate = final_native < anchor_native
    auxiliary_prefers_candidate = final_auxiliary < anchor_auxiliary
    native_wrong = informative & (native_prefers_candidate != candidate_better)
    auxiliary_wrong = informative & (auxiliary_prefers_candidate != candidate_better)
    disagreement = informative & (native_prefers_candidate != auxiliary_prefers_candidate)
    native_rejected = final_native > anchor_native
    directional_support = native_rejected & auxiliary_prefers_candidate
    corrective = native_rejected & candidate_better

    counts = {
        "informative": int(informative.sum()),
        "native_wrong": int(native_wrong.sum()),
        "auxiliary_wrong": int(auxiliary_wrong.sum()),
        "both_wrong": int((native_wrong & auxiliary_wrong).sum()),
        "disagreement": int(disagreement.sum()),
        "native_wrong_and_disagreement": int((native_wrong & disagreement).sum()),
        "native_rejected": int(native_rejected.sum()),
        "corrective": int(corrective.sum()),
        "directional_support": int(directional_support.sum()),
        "corrective_in_directional_support": int(
            (corrective & directional_support).sum()
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "complementarity_audit.npz",
        final_native_cost=final_native,
        final_auxiliary_cost=final_auxiliary,
        final_physical_distance_m=final_physical,
        native_rank_fraction=native_rank,
        auxiliary_rank_fraction=auxiliary_rank,
        crod_score=crod_score,
        informative=informative,
        native_wrong=native_wrong,
        auxiliary_wrong=auxiliary_wrong,
        disagreement=disagreement,
        corrective=corrective,
        directional_support=directional_support,
    )
    summary = {
        "snapshot": snapshot.order,
        "episode": snapshot.episode,
        "scope": "Post-hoc DINO-WM rescore of a fully physics-labelled Phase-0d population.",
        "physical_margin_m": args.physical_margin_m,
        "auxiliary_checkpoint": args.auxiliary_checkpoint,
        "counts": counts,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
