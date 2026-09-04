#!/usr/bin/env python3
"""Train the event-state-closed Scene SMDP for the H1b mechanism audit."""

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

from event_smdp_h0.scene_abstract_smdp import AbstractSMDPHead  # noqa: E402
from event_smdp_h0.scripts.train_scene_h1 import (  # noqa: E402
    binary_pos_weight,
    fit_model,
    load_split,
    milestone_matrix,
    seed_everything,
    select,
    task_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def make_tensors(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    return {
        "task": torch.from_numpy(task_matrix(data["task_id"])).to(device),
        "milestone": torch.from_numpy(milestone_matrix(data, "before")).to(device),
        "skill": torch.from_numpy(data["skill"].astype(np.int64)).to(device),
        "cube": torch.from_numpy(data["after_cube_stage"].astype(np.int64)).to(device),
        "window": torch.from_numpy(data["after_window_stage"].astype(np.int64)).to(device),
        "terminal": torch.from_numpy(
            data["native_success_after"].astype(np.float32)
        ).to(device),
        "duration": torch.from_numpy(data["duration"].astype(np.float32)).to(device),
        "no_effect": torch.from_numpy(data["no_effect"].astype(np.float32)).to(device),
    }


def abstraction_ambiguity(data: dict[str, np.ndarray]) -> dict[str, float | int]:
    groups: dict[tuple[int, int, int, int, int], set[tuple[int, int, int]]] = {}
    for i in range(len(data["skill"])):
        key = (
            int(data["task_id"][i]),
            int(data["before_cube_stage"][i]),
            int(data["before_window_stage"][i]),
            int(data["before_stable_count"][i]),
            int(data["skill"][i]),
        )
        target = (
            int(data["after_cube_stage"][i]),
            int(data["after_window_stage"][i]),
            int(data["native_success_after"][i]),
        )
        groups.setdefault(key, set()).add(target)
    ambiguous = sum(len(targets) > 1 for targets in groups.values())
    return {
        "num_abstract_state_actions": len(groups),
        "num_ambiguous_state_actions": ambiguous,
        "ambiguous_fraction": ambiguous / max(len(groups), 1),
    }


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("H1b training must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("H1b training requires a GPU allocation")
    seed_everything(args.seed)

    train_np, train_paths = load_split(args.data_root, "train")
    val_np, val_paths = load_split(args.data_root, "val")
    overlap = set(train_np["reset_seed"].tolist()) & set(val_np["reset_seed"].tolist())
    if overlap:
        raise RuntimeError(f"train/val reset leakage: {sorted(overlap)}")
    train = make_tensors(train_np)
    val = make_tensors(val_np)
    model = AbstractSMDPHead(width=args.width).cuda()
    stable_weight = binary_pos_weight(train["terminal"])
    no_effect_weight = binary_pos_weight(train["no_effect"])

    def loss_fn(data: dict[str, torch.Tensor], index: torch.Tensor | None):
        out = model(
            select(data, index, "task"),
            select(data, index, "milestone"),
            select(data, index, "skill"),
        )
        duration = select(data, index, "duration").clamp_min(1).log()
        return (
            F.cross_entropy(out["cube_logits"], select(data, index, "cube"))
            + F.cross_entropy(out["window_logits"], select(data, index, "window"))
            + 0.5
            * F.binary_cross_entropy_with_logits(
                out["stable_logit"],
                select(data, index, "terminal"),
                pos_weight=stable_weight,
            )
            + 0.25
            * F.binary_cross_entropy_with_logits(
                out["no_effect_logit"],
                select(data, index, "no_effect"),
                pos_weight=no_effect_weight,
            )
            + 0.1 * F.smooth_l1_loss(out["log_duration"], duration)
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
        seed=args.seed + 1701,
    )
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        out = model(val["task"], val["milestone"], val["skill"])
        cube_correct = out["cube_logits"].argmax(-1) == val["cube"]
        window_correct = out["window_logits"].argmax(-1) == val["window"]
        stable_probability = torch.sigmoid(out["stable_logit"])
        metrics = {
            "cube_accuracy": float(cube_correct.float().mean().item()),
            "window_accuracy": float(window_correct.float().mean().item()),
            "joint_stage_accuracy": float((cube_correct & window_correct).float().mean().item()),
            "stable_brier": float(F.mse_loss(stable_probability, val["terminal"]).item()),
            "duration_mae": float(
                (out["log_duration"].exp() - val["duration"]).abs().mean().item()
            ),
        }
    metadata = {
        "protocol": "scene_h1b_abstract_closure_v1",
        "seed": args.seed,
        "width": args.width,
        "fit": fit,
        "metrics": metrics,
        "train_ambiguity": abstraction_ambiguity(train_np),
        "val_ambiguity": abstraction_ambiguity(val_np),
        "num_train": int(len(train["skill"])),
        "num_val": int(len(val["skill"])),
        "train_shards": train_paths,
        "val_shards": val_paths,
        "train_reset_seeds": sorted(set(train_np["reset_seed"].tolist())),
        "val_reset_seeds": sorted(set(val_np["reset_seed"].tolist())),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "abstract_smdp.pt"
    torch.save(
        {
            "protocol": "scene_h1b_abstract_closure_v1",
            "width": args.width,
            "state_dict": state,
            "metadata": metadata,
        },
        checkpoint,
    )
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"checkpoint": str(checkpoint), **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()

