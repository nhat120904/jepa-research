#!/usr/bin/env python3
"""Evaluate all dynamics checkpoints on untouched held-out candidate pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rollout_repair_gate.core import (
    autoregressive_rollout,
    dynamics_state_dict,
    load_dynamics_state,
    split_for_order,
    squared_goal_cost,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint_specs(root: Path) -> list[tuple[str, dict]]:
    specs = []
    signatures = set()
    for path in sorted(root.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        label = f"{metadata['arm']}_seed{metadata['seed']}"
        signatures.add(json.dumps(metadata["same_compute_signature"], sort_keys=True))
        specs.append((label, payload["dynamics"]))
    if len(specs) != 9:
        raise RuntimeError(f"expected 9 arm/seed checkpoints, got {len(specs)}")
    if len(signatures) != 1:
        raise RuntimeError("same-compute checkpoint metadata mismatch")
    return specs


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("fixed-pool evaluation requires a GPU Slurm allocation")
    if split_for_order(args.snapshot_index) != "test":
        raise ValueError("fixed-pool evaluation is restricted to immutable test snapshots")

    path = args.data_dir / f"snapshot_{args.snapshot_index:03d}/intermediates.npz"
    with np.load(path) as data:
        actions = data["actions_normalized"].astype(np.float32)
        current = data["current_embedding"].astype(np.float32)
        true_future = data["true_future_embeddings"].astype(np.float32)
        dataset_goal = data["dataset_goal_embedding"].astype(np.float32)
        rendered_goal = data["rendered_goal_embedding"].astype(np.float32)
        physical = data["physical_distance_m"].astype(np.float32)
        valid = data["valid_horizon"].astype(bool)

    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    base_state = dynamics_state_dict(model)
    specs = [("native", base_state)] + load_checkpoint_specs(args.checkpoint_dir)
    predicted_costs = []
    predicted_latents = []
    prediction_mse = []
    with torch.inference_mode():
        for label, state in specs:
            load_dynamics_state(model, state)
            per_population = []
            for population in range(actions.shape[0]):
                initial = torch.from_numpy(
                    np.repeat(current[population][None], actions.shape[1], axis=0)
                ).cuda()
                action_tensor = torch.from_numpy(actions[population]).cuda()
                pred = autoregressive_rollout(
                    model, initial, action_tensor, args.history_size
                ).float().cpu().numpy()
                per_population.append(pred)
            pred = np.stack(per_population)
            predicted_latents.append(pred.astype(np.float32))
            prediction_mse.append(
                np.square(pred.astype(np.float64) - true_future.astype(np.float64)).mean(
                    axis=-1
                ).astype(np.float32)
            )
            predicted_costs.append(
                np.stack(
                    [
                        squared_goal_cost(pred[p], dataset_goal[p])
                        for p in range(pred.shape[0])
                    ]
                ).astype(np.float32)
            )

    true_dataset_cost = np.stack(
        [squared_goal_cost(true_future[p], dataset_goal[p]) for p in range(len(true_future))]
    ).astype(np.float32)
    true_rendered_cost = np.stack(
        [squared_goal_cost(true_future[p], rendered_goal) for p in range(len(true_future))]
    ).astype(np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"snapshot_{args.snapshot_index:03d}.npz"
    np.savez_compressed(
        output,
        labels=np.asarray([label for label, _ in specs]),
        predicted_cost=np.stack(predicted_costs),
        predicted_latent=np.stack(predicted_latents).astype(np.float16),
        prediction_mse=np.stack(prediction_mse),
        true_dataset_cost=true_dataset_cost,
        true_rendered_cost=true_rendered_cost,
        physical_distance_m=physical,
        valid_horizon=valid,
    )
    summary = {
        "snapshot": args.snapshot_index,
        "split": "test",
        "models": [label for label, _ in specs],
        "output": str(output),
    }
    (args.out_dir / f"snapshot_{args.snapshot_index:03d}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
