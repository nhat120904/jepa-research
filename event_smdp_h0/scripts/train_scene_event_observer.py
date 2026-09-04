#!/usr/bin/env python3
"""Train a current-q observer on deduplicated Scene canonical roots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.scene_event_perception import EventStateObserver  # noqa: E402
from event_smdp_h0.scripts.train_scene_h1 import (  # noqa: E402
    binary_pos_weight,
    fit_model,
    load_split,
    seed_everything,
    select,
    standardizer,
    task_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-view", choices=("latent", "privileged"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def unique_roots(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = np.stack(
        [data["task_id"], data["reset_seed"], data["path_id"], data["root_index"]],
        axis=1,
    )
    _, indices = np.unique(keys, axis=0, return_index=True)
    indices = np.sort(indices)
    return {key: value[indices] for key, value in data.items()}


def tensors(
    data: dict[str, np.ndarray],
    view: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    goal_mean: np.ndarray,
    goal_std: np.ndarray,
) -> dict[str, torch.Tensor]:
    feature = (data[f"before_{view}"].astype(np.float32) - feature_mean) / feature_std
    goal = (data["goal"].astype(np.float32) - goal_mean) / goal_std
    return {
        "feature": torch.from_numpy(feature).cuda(),
        "goal": torch.from_numpy(goal).cuda(),
        "task": torch.from_numpy(task_matrix(data["task_id"])).cuda(),
        "skill": torch.zeros(len(feature), dtype=torch.int64, device="cuda"),
        "cube": torch.from_numpy(data["before_cube_stage"].astype(np.int64)).cuda(),
        "window": torch.from_numpy(data["before_window_stage"].astype(np.int64)).cuda(),
        "stable": torch.from_numpy(
            (data["before_stable_count"] >= 3).astype(np.float32)
        ).cuda(),
    }


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("observer training must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("observer training requires a GPU allocation")
    seed_everything(args.seed)
    train_np, train_paths = load_split(args.data_root, "train")
    val_np, val_paths = load_split(args.data_root, "val")
    train_np = unique_roots(train_np)
    val_np = unique_roots(val_np)
    overlap = set(train_np["reset_seed"].tolist()) & set(val_np["reset_seed"].tolist())
    if overlap:
        raise RuntimeError(f"train/val reset leakage: {sorted(overlap)}")
    feature_key = f"before_{args.feature_view}"
    feature_mean, feature_std = standardizer(train_np[feature_key])
    goal_mean, goal_std = standardizer(train_np["goal"])
    train = tensors(
        train_np, args.feature_view, feature_mean, feature_std, goal_mean, goal_std
    )
    val = tensors(val_np, args.feature_view, feature_mean, feature_std, goal_mean, goal_std)
    feature_dim = int(train["feature"].shape[1])
    goal_dim = int(train["goal"].shape[1])
    model = EventStateObserver(feature_dim, goal_dim, width=args.width).cuda()
    stable_weight = binary_pos_weight(train["stable"])

    def loss_fn(data: dict[str, torch.Tensor], index: torch.Tensor | None):
        out = model(
            select(data, index, "feature"),
            select(data, index, "goal"),
            select(data, index, "task"),
        )
        return (
            F.cross_entropy(out["cube_logits"], select(data, index, "cube"))
            + F.cross_entropy(out["window_logits"], select(data, index, "window"))
            + 0.5
            * F.binary_cross_entropy_with_logits(
                out["stable_logit"],
                select(data, index, "stable"),
                pos_weight=stable_weight,
            )
        )

    state, fit = fit_model(
        model,
        train,
        val,
        loss_fn,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 2309,
    )
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        out = model(val["feature"], val["goal"], val["task"])
        cube_correct = out["cube_logits"].argmax(-1) == val["cube"]
        window_correct = out["window_logits"].argmax(-1) == val["window"]
        stable_correct = (torch.sigmoid(out["stable_logit"]) >= 0.5) == val["stable"].bool()
        metrics = {
            "cube_accuracy": float(cube_correct.float().mean().item()),
            "window_accuracy": float(window_correct.float().mean().item()),
            "stable_accuracy": float(stable_correct.float().mean().item()),
            "exact_q_accuracy": float(
                (cube_correct & window_correct & stable_correct).float().mean().item()
            ),
            "stable_brier": float(
                F.mse_loss(torch.sigmoid(out["stable_logit"]), val["stable"]).item()
            ),
        }
    metadata = {
        "protocol": "scene_event_perception_v1",
        "feature_view": args.feature_view,
        "seed": args.seed,
        "fit": fit,
        "metrics": metrics,
        "num_unique_train_roots": len(train_np["task_id"]),
        "num_unique_val_roots": len(val_np["task_id"]),
        "train_reset_seeds": sorted(set(train_np["reset_seed"].tolist())),
        "val_reset_seeds": sorted(set(val_np["reset_seed"].tolist())),
        "source_train_shards": train_paths,
        "source_val_shards": val_paths,
    }
    payload = {
        "protocol": "scene_event_perception_v1",
        "feature_view": args.feature_view,
        "feature_dim": feature_dim,
        "goal_dim": goal_dim,
        "width": args.width,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "goal_mean": goal_mean.astype(np.float32),
        "goal_std": goal_std.astype(np.float32),
        "state_dict": state,
        "metadata": metadata,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_dir / "observer.pt")
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()

