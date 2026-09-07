#!/usr/bin/env python3
"""Score fixed PushT candidate shards with one original DINO-WM model load."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from order_jepa.core import (  # noqa: E402
    angular_distance,
    order_vector_metrics,
    pusht_physical_cost,
    pusht_success,
    selection_regret,
)
from order_jepa.original_dino_wm import OriginalDinoWMPushT  # noqa: E402


def require_compute_node() -> None:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("checkpoint loading/encoding must run through scripts/slurm_score.sh")


def terminal_cost(
    encoded: dict[str, torch.Tensor],
    goal: dict[str, torch.Tensor],
    *,
    alpha: float,
) -> torch.Tensor:
    """Exact objective_fn_last from the original repository."""

    visual = (encoded["visual"][:, -1:] - goal["visual"]).square().mean(
        dim=tuple(range(1, encoded["visual"].ndim))
    )
    proprio = (encoded["proprio"][:, -1:] - goal["proprio"]).square().mean(
        dim=tuple(range(1, encoded["proprio"].ndim))
    )
    return visual + alpha * proprio


def np_visual(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def score_anchor(
    args: argparse.Namespace, anchor_index: int, adapter: OriginalDinoWMPushT
) -> Path:
    npz_path = args.branches_dir / f"anchor_{anchor_index:04d}.npz"
    meta_path = npz_path.with_suffix(".json")
    branch_meta = json.loads(meta_path.read_text())
    if branch_meta.get("implementation") != "gaoyuezhou/dino_wm":
        raise ValueError(f"refusing non-original branch shard: {meta_path}")
    arrays = np.load(npz_path, allow_pickle=False)

    if int(branch_meta["block_steps"]) != adapter.frameskip:
        raise ValueError(
            f"branch block={branch_meta['block_steps']} != checkpoint frameskip={adapter.frameskip}"
        )
    if branch_meta["source_commit"] != adapter.provenance.source_commit:
        raise ValueError("branch source commit and scoring source commit differ")

    n_candidates = int(arrays["candidate_actions"].shape[0])
    anchor_visual = torch.from_numpy(arrays["history_visual"][-1:][None]).repeat(
        n_candidates, 1, 1, 1, 1
    )
    anchor_proprio = torch.from_numpy(arrays["history_proprio"][-1:][None]).repeat(
        n_candidates, 1, 1
    )
    actions = torch.from_numpy(arrays["candidate_actions"])
    candidate_visual = torch.from_numpy(arrays["candidate_visual"][:, None])
    candidate_proprio = torch.from_numpy(arrays["candidate_proprio"][:, None])
    goal_visual = torch.from_numpy(arrays["goal_visual"][None, None])
    goal_proprio = torch.from_numpy(arrays["goal_proprio"][None, None])

    with torch.no_grad():
        predicted = adapter.rollout(anchor_visual, anchor_proprio, actions)
        encoded_true = adapter.encode_observations(candidate_visual, candidate_proprio)
        encoded_goal = adapter.encode_observations(goal_visual, goal_proprio)
        predicted_cost = terminal_cost(predicted, encoded_goal, alpha=args.alpha)
        true_latent_cost = terminal_cost(encoded_true, encoded_goal, alpha=args.alpha)

    predicted_cost_np = predicted_cost.detach().cpu().double().numpy()
    true_latent_cost_np = true_latent_cost.detach().cpu().double().numpy()
    physical_cost = pusht_physical_cost(arrays["candidate_state"], arrays["goal_state"])
    success = pusht_success(arrays["candidate_state"], arrays["goal_state"])
    true_selected, true_regret = selection_regret(true_latent_cost_np, physical_cost)
    pred_selected, pred_regret = selection_regret(predicted_cost_np, physical_cost)
    oracle_selected = int(np.argmin(physical_cost))

    candidate_rows = []
    for index in range(n_candidates):
        state = arrays["candidate_state"][index]
        goal = arrays["goal_state"]
        candidate_rows.append(
            {
                "index": index,
                "kind": branch_meta["candidate_kinds"][index],
                "predicted_latent_cost": float(predicted_cost_np[index]),
                "true_latent_cost": float(true_latent_cost_np[index]),
                "physical_cost": float(physical_cost[index]),
                "success": bool(success[index]),
                "contacts": int(np.sum(arrays["candidate_contacts"][index] > 0)),
                "agent_position_error": float(np.linalg.norm(state[:2] - goal[:2])),
                "object_position_error": float(np.linalg.norm(state[2:4] - goal[2:4])),
                "object_angle_error": float(angular_distance(state[4], goal[4])),
            }
        )

    true_visual = np_visual(encoded_true["visual"][:, -1])
    predicted_visual = np_visual(predicted["visual"][:, -1])
    repeat = arrays["repeat_endpoint_physics"]
    repeat_object_noise = np.linalg.norm(np.ptp(repeat[:, :, 4:6], axis=1), axis=-1) / 20.0
    repeat_object_noise += np.ptp(repeat[:, :, 8], axis=1) / (np.pi / 9.0)
    pair_rows = []
    for pair in branch_meta["order_pairs"]:
        ij, ji = int(pair["ij"]), int(pair["ji"])
        metrics = order_vector_metrics(
            true_visual[ij], true_visual[ji], predicted_visual[ij], predicted_visual[ji]
        )
        state_ij, state_ji = arrays["candidate_state"][ij], arrays["candidate_state"][ji]
        object_effect = float(
            np.linalg.norm(state_ij[2:4] - state_ji[2:4]) / 20.0
            + angular_distance(state_ij[4], state_ji[4]) / (np.pi / 9.0)
        )
        pair_rows.append(
            {
                "pair_id": int(pair["pair_id"]),
                "ij": ij,
                "ji": ji,
                **asdict(metrics),
                "physical_delta": float(physical_cost[ij] - physical_cost[ji]),
                "true_latent_delta": float(true_latent_cost_np[ij] - true_latent_cost_np[ji]),
                "predicted_latent_delta": float(predicted_cost_np[ij] - predicted_cost_np[ji]),
                "object_effect": object_effect,
                "repeat_object_noise": float(
                    max(repeat_object_noise[ij], repeat_object_noise[ji])
                ),
                "contact_steps_ij": int(np.sum(arrays["candidate_contacts"][ij] > 0)),
                "contact_steps_ji": int(np.sum(arrays["candidate_contacts"][ji] > 0)),
            }
        )

    payload = {
        "schema": "order-jepa-stage-a-scores-v1-original-dino-wm",
        "anchor_id": int(branch_meta["anchor_id"]),
        "episode": int(branch_meta["episode"]),
        "step": int(branch_meta["step"]),
        "alpha": args.alpha,
        "planning_context_frames": 1,
        "retained_history_frames": int(branch_meta["num_hist"]),
        "provenance": asdict(adapter.provenance),
        "reset": {
            "physics_max_abs": float(branch_meta["reset_physics_max_abs"]),
            "pixel_max_abs": int(branch_meta["reset_pixel_max_abs"]),
            "repeat_endpoint_max_abs": float(branch_meta["repeat_endpoint_max_abs"]),
        },
        "selection": {
            "physical_oracle_index": oracle_selected,
            "true_latent_index": true_selected,
            "predicted_latent_index": pred_selected,
            "true_latent_regret": true_regret,
            "predicted_latent_regret": pred_regret,
            "dynamics_excess_regret": pred_regret - true_regret,
            "random_expected_regret": float(np.mean(physical_cost) - np.min(physical_cost)),
            "oracle_has_success": bool(np.any(success)),
            "true_latent_success": bool(success[true_selected]),
            "predicted_latent_success": bool(success[pred_selected]),
        },
        "candidates": candidate_rows,
        "order_pairs": pair_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"anchor_{anchor_index:04d}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(out), "selection": payload["selection"]}, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches-dir", type=Path, required=True)
    parser.add_argument("--anchor-index", type=int, help="score exactly one anchor")
    parser.add_argument("--anchor-count", type=int, default=200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dino-wm-root", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="official OSF outputs/pusht directory containing hydra.yaml and checkpoints/",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    require_compute_node()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    indices = (
        [args.anchor_index]
        if args.anchor_index is not None
        else list(range(args.shard_index, args.anchor_count, args.num_shards))
    )
    adapter = OriginalDinoWMPushT(
        args.dino_wm_root,
        args.model_dir,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    for anchor_index in indices:
        score_anchor(args, anchor_index, adapter)


if __name__ == "__main__":
    main()
