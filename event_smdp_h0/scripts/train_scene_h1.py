#!/usr/bin/env python3
"""Train the shared skill dynamics and three H1 Scene readouts."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.scene_learning import (  # noqa: E402
    HEADS,
    SkillDynamics,
    checkpoint_payload,
    make_head,
    task_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-view", choices=("latent", "privileged"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(root: Path, split: str) -> tuple[dict[str, np.ndarray], list[str]]:
    paths = sorted((root / split).glob("*/transitions.npz"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards under {root}")
    pieces: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path) as payload:
            for key in payload.files:
                pieces.setdefault(key, []).append(np.asarray(payload[key]))
    return {key: np.concatenate(value) for key, value in pieces.items()}, [str(p) for p in paths]


def standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values, dtype=np.float32).mean(axis=0)
    std = np.asarray(values, dtype=np.float32).std(axis=0)
    return mean, np.maximum(std, 1e-4)


def task_matrix(task_ids: np.ndarray) -> np.ndarray:
    return np.stack([task_vector(int(task)) for task in task_ids]).astype(np.float32)


def milestone_matrix(data: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for cube, window, stable in zip(
        data[f"{prefix}_cube_stage"],
        data[f"{prefix}_window_stage"],
        data[f"{prefix}_stable_count"],
    ):
        cube_hot = np.eye(6, dtype=np.float32)[int(cube)]
        window_hot = np.eye(4, dtype=np.float32)[int(window)]
        rows.append(
            np.concatenate(
                [cube_hot, window_hot, np.asarray([min(int(stable), 3) / 3.0], dtype=np.float32)]
            )
        )
    return np.stack(rows)


def tensors(
    data: dict[str, np.ndarray],
    view: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    goal_mean: np.ndarray,
    goal_std: np.ndarray,
) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    before = (data[f"before_{view}"].astype(np.float32) - feature_mean) / feature_std
    after = (data[f"after_{view}"].astype(np.float32) - feature_mean) / feature_std
    goal = (data["goal"].astype(np.float32) - goal_mean) / goal_std
    return {
        "before": torch.from_numpy(before).to(device),
        "after": torch.from_numpy(after).to(device),
        "goal": torch.from_numpy(goal).to(device),
        "task": torch.from_numpy(task_matrix(data["task_id"])).to(device),
        "skill": torch.from_numpy(data["skill"].astype(np.int64)).to(device),
        "milestone": torch.from_numpy(milestone_matrix(data, "before")).to(device),
        "cube": torch.from_numpy(data["after_cube_stage"].astype(np.int64)).to(device),
        "window": torch.from_numpy(data["after_window_stage"].astype(np.int64)).to(device),
        "terminal": torch.from_numpy(data["native_success_after"].astype(np.float32)).to(device),
        "predicates": torch.from_numpy(data["after_predicates"].astype(np.float32)).to(device),
        "duration": torch.from_numpy(data["duration"].astype(np.float32)).to(device),
        "no_effect": torch.from_numpy(data["no_effect"].astype(np.float32)).to(device),
    }


def batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator, device="cpu")
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size].cuda(non_blocking=True)


def fit_model(
    model: torch.nn.Module,
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    loss_fn: Callable[[dict[str, torch.Tensor], torch.Tensor | None], torch.Tensor],
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    for epoch in range(epochs):
        model.train()
        for index in batches(len(train["skill"]), batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(train, index)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(val, None).item())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("optimizer never produced a finite validation state")
    model.load_state_dict(best_state)
    return best_state, {
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_ran": epoch + 1,
    }


def select(data: dict[str, torch.Tensor], index: torch.Tensor | None, key: str) -> torch.Tensor:
    return data[key] if index is None else data[key][index]


def binary_pos_weight(target: torch.Tensor, cap: float = 30.0) -> torch.Tensor:
    positive = target.sum(dim=0)
    negative = target.shape[0] - positive
    return (negative / positive.clamp_min(1.0)).clamp(1.0, cap)


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("training must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a GPU allocation")
    seed_everything(args.seed)

    train_np, train_paths = load_split(args.data_root, "train")
    val_np, val_paths = load_split(args.data_root, "val")
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
    feature_dim = int(train["before"].shape[1])
    goal_dim = int(train["goal"].shape[1])

    dynamics = SkillDynamics(feature_dim, width=args.width).cuda()

    def dynamics_loss(data: dict[str, torch.Tensor], index: torch.Tensor | None) -> torch.Tensor:
        prediction = dynamics(
            select(data, index, "before"), select(data, index, "skill")
        )
        return F.smooth_l1_loss(prediction, select(data, index, "after"))

    dynamics_state, dynamics_fit = fit_model(
        dynamics,
        train,
        val,
        dynamics_loss,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 101,
    )
    dynamics.load_state_dict(dynamics_state)
    dynamics.eval()
    with torch.inference_mode():
        val_dynamics_mse = float(
            F.mse_loss(dynamics(val["before"], val["skill"]), val["after"]).item()
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    head_metrics: dict[str, Any] = {}
    for head_index, head_name in enumerate(HEADS):
        readout = make_head(head_name, feature_dim, goal_dim, width=args.width).cuda()
        if head_name == "terminal":
            pos_weight = binary_pos_weight(train["terminal"])

            def head_loss(data, index, model=readout, weight=pos_weight):
                logits = model(
                    select(data, index, "after"),
                    select(data, index, "goal"),
                    select(data, index, "task"),
                ).squeeze(-1)
                return F.binary_cross_entropy_with_logits(
                    logits, select(data, index, "terminal"), pos_weight=weight
                )

        elif head_name == "event_bce":
            pos_weight = binary_pos_weight(train["predicates"])

            def head_loss(data, index, model=readout, weight=pos_weight):
                logits = model(
                    select(data, index, "after"),
                    select(data, index, "goal"),
                    select(data, index, "task"),
                )
                return F.binary_cross_entropy_with_logits(
                    logits, select(data, index, "predicates"), pos_weight=weight
                )

        else:
            stable_weight = binary_pos_weight(train["terminal"])
            no_effect_weight = binary_pos_weight(train["no_effect"])

            def head_loss(
                data,
                index,
                model=readout,
                stable_pos=stable_weight,
                no_effect_pos=no_effect_weight,
            ):
                out = model(
                    select(data, index, "before"),
                    select(data, index, "goal"),
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
                        pos_weight=stable_pos,
                    )
                    + 0.25
                    * F.binary_cross_entropy_with_logits(
                        out["no_effect_logit"],
                        select(data, index, "no_effect"),
                        pos_weight=no_effect_pos,
                    )
                    + 0.1 * F.smooth_l1_loss(out["log_duration"], duration)
                )

        _, fit = fit_model(
            readout,
            train,
            val,
            head_loss,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + 1009 + head_index,
        )
        readout.eval()
        with torch.inference_mode():
            if head_name == "terminal":
                probability = torch.sigmoid(
                    readout(val["after"], val["goal"], val["task"]).squeeze(-1)
                )
                metrics = {
                    "brier": float(F.mse_loss(probability, val["terminal"]).item()),
                    "accuracy": float(((probability >= 0.5) == val["terminal"].bool()).float().mean().item()),
                    "prevalence": float(val["terminal"].mean().item()),
                }
            elif head_name == "event_bce":
                probability = torch.sigmoid(readout(val["after"], val["goal"], val["task"]))
                metrics = {
                    "mean_brier": float(F.mse_loss(probability, val["predicates"]).item()),
                    "bit_accuracy": float(((probability >= 0.5) == val["predicates"].bool()).float().mean().item()),
                    "exact_match": float(((probability >= 0.5) == val["predicates"].bool()).all(dim=1).float().mean().item()),
                }
            else:
                out = readout(
                    val["before"], val["goal"], val["task"], val["milestone"], val["skill"]
                )
                metrics = {
                    "cube_accuracy": float((out["cube_logits"].argmax(-1) == val["cube"]).float().mean().item()),
                    "window_accuracy": float((out["window_logits"].argmax(-1) == val["window"]).float().mean().item()),
                    "stable_brier": float(F.mse_loss(torch.sigmoid(out["stable_logit"]), val["terminal"]).item()),
                    "no_effect_brier": float(F.mse_loss(torch.sigmoid(out["no_effect_logit"]), val["no_effect"]).item()),
                    "duration_mae": float((out["log_duration"].exp() - val["duration"]).abs().mean().item()),
                }
        metadata = {
            "seed": args.seed,
            "dynamics_fit": dynamics_fit,
            "dynamics_val_mse": val_dynamics_mse,
            "head_fit": fit,
            "head_metrics": metrics,
            "num_train": int(len(train["skill"])),
            "num_val": int(len(val["skill"])),
            "train_shards": train_paths,
            "val_shards": val_paths,
            "train_reset_seeds": sorted(set(train_np["reset_seed"].tolist())),
            "val_reset_seeds": sorted(set(val_np["reset_seed"].tolist())),
        }
        torch.save(
            checkpoint_payload(
                feature_view=args.feature_view,
                head=head_name,
                feature_dim=feature_dim,
                goal_dim=goal_dim,
                width=args.width,
                feature_mean=feature_mean,
                feature_std=feature_std,
                goal_mean=goal_mean,
                goal_std=goal_std,
                dynamics=dynamics,
                readout=readout,
                metadata=metadata,
            ),
            args.out_dir / f"{head_name}.pt",
        )
        head_metrics[head_name] = metadata

    summary = {
        "protocol": "scene_h1_learnability_v1",
        "feature_view": args.feature_view,
        "seed": args.seed,
        "feature_dim": feature_dim,
        "goal_dim": goal_dim,
        "dynamics_val_mse": val_dynamics_mse,
        "heads": head_metrics,
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
