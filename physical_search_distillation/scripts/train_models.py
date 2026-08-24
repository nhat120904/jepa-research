#!/usr/bin/env python3
"""Train matched physical-supervision arms on planner-induced populations."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from physical_search_distillation.core import (
    ARMS, CheckpointSpec, MetricOperatorScorer, PopulationScorer, build_features_np,
    fit_standardizer, normalized_operator_loss, robust_population_features,
    save_checkpoint, split_for_order, straight_through_refit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--student-temperature", type=float, default=0.5)
    parser.add_argument("--teacher-temperature-m", type=float, default=0.02)
    parser.add_argument("--elite-weight", type=float, default=0.25)
    return parser.parse_args()


def load_groups(data_dir: Path) -> list[dict]:
    groups: list[dict] = []
    paths = sorted(data_dir.glob("snapshot_*/populations.npz"))
    file_counts = {name: 0 for name in ("train", "val", "test")}
    for path in paths:
        order = int(path.parent.name.split("_")[-1])
        split = split_for_order(order)
        file_counts[split] += 1
        # A mechanical leakage guard: the training process never opens held-out arrays.
        if split == "test":
            continue
        with np.load(path) as data:
            for population in range(len(data["step"])):
                actions = data["actions_normalized"][population].astype(np.float32)
                native = data["native_cost"][population].astype(np.float32)
                endpoint = data["predicted_endpoint"][population].astype(np.float32)
                current = data["current_embedding"][population].astype(np.float32)
                goal = data["goal_embedding"][population].astype(np.float32)
                groups.append({
                    "order": order, "population": population, "split": split,
                    "actions": actions, "native_features": robust_population_features(native),
                    "endpoint": endpoint, "current": current, "goal": goal,
                    "features": build_features_np(actions, native, endpoint, current, goal),
                    "physical": data["physical_distance_m"][population].astype(np.float32),
                    "teacher_elite": data["teacher_elite"][population].astype(np.int64),
                    "teacher_mean": data["teacher_mean"][population].astype(np.float32),
                    "teacher_std": data["teacher_std"][population].astype(np.float32),
                    "proposal_std": data["proposal_std"][population].astype(np.float32),
                })
    if not groups:
        raise RuntimeError(f"no populations found under {data_dir}")
    if file_counts != {"train": 80, "val": 16, "test": 32}:
        raise RuntimeError(f"locked split incomplete: {file_counts}")
    return groups


def tensor_group(group: dict, mean: np.ndarray, std: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    result = {
        "features": torch.from_numpy((group["features"] - mean) / std).to(device),
        "actions": torch.from_numpy(group["actions"]).to(device),
        "native_features": torch.from_numpy(group["native_features"]).to(device),
        "endpoint": torch.from_numpy(group["endpoint"]).to(device),
        "goal": torch.from_numpy(np.repeat(group["goal"][None], len(group["actions"]), axis=0)).to(device),
        "physical": torch.from_numpy(group["physical"]).to(device),
        "teacher_mean": torch.from_numpy(group["teacher_mean"]).to(device),
        "teacher_std": torch.from_numpy(group["teacher_std"]).to(device),
        "proposal_std": torch.from_numpy(group["proposal_std"]).to(device),
    }
    target = torch.zeros(len(group["actions"]), device=device)
    target[torch.from_numpy(group["teacher_elite"]).to(device)] = 1.0
    result["elite_target"] = target
    return result


def score_group(model: torch.nn.Module, arm: str, group: dict[str, torch.Tensor]) -> torch.Tensor:
    if arm == "operator_metric":
        return model(group["actions"], group["native_features"], group["endpoint"], group["goal"])
    return model(group["features"])


def group_loss(
    model: torch.nn.Module,
    arm: str,
    group: dict[str, torch.Tensor],
    args: argparse.Namespace,
    target_mean: float,
    target_std: float,
) -> torch.Tensor:
    scores = score_group(model, arm, group)
    if arm == "pointwise":
        target = (group["physical"] - target_mean) / target_std
        return F.smooth_l1_loss(scores, target)
    if arm == "listwise":
        target_prob = torch.softmax(-group["physical"] / args.teacher_temperature_m, dim=0)
        return -(target_prob * torch.log_softmax(-scores / args.student_temperature, dim=0)).sum()
    if arm == "elite":
        return F.binary_cross_entropy_with_logits(-scores, group["elite_target"])
    student_mean, student_std, _ = straight_through_refit(
        group["actions"], scores, args.topk, args.student_temperature
    )
    operator = normalized_operator_loss(
        student_mean, student_std, group["teacher_mean"], group["teacher_std"],
        group["proposal_std"],
    )
    auxiliary = F.binary_cross_entropy_with_logits(-scores, group["elite_target"])
    return operator + args.elite_weight * auxiliary


def mean_loss(
    model: torch.nn.Module, arm: str, groups: list[dict[str, torch.Tensor]],
    args: argparse.Namespace, target_mean: float, target_std: float,
) -> float:
    model.eval()
    with torch.no_grad():
        values = [group_loss(model, arm, group, args, target_mean, target_std).item() for group in groups]
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a GPU Slurm allocation")
    groups = load_groups(args.data_dir)
    train_np = [g for g in groups if g["split"] == "train"]
    val_np = [g for g in groups if g["split"] == "val"]
    feature_mean, feature_std = fit_standardizer(g["features"] for g in train_np)
    target_values = np.concatenate([g["physical"] for g in train_np])
    target_mean = float(target_values.mean())
    target_std = max(float(target_values.std()), 1e-6)
    device = "cuda"
    train = [tensor_group(g, feature_mean, feature_std, device) for g in train_np]
    val = [tensor_group(g, feature_mean, feature_std, device) for g in val_np]
    sample = train_np[0]
    spec_base = dict(
        feature_dim=sample["features"].shape[1], embedding_dim=sample["endpoint"].shape[1],
        action_shape=tuple(sample["actions"].shape[1:]), feature_mean=feature_mean,
        feature_std=feature_std,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for arm in args.arms:
        report[arm] = {}
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if arm == "operator_metric":
                model = MetricOperatorScorer(
                    spec_base["embedding_dim"], int(np.prod(spec_base["action_shape"]))
                ).to(device)
            else:
                model = PopulationScorer(spec_base["feature_dim"]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_loss, best_state, best_epoch, stale = float("inf"), None, -1, 0
            generator = np.random.default_rng(seed)
            history = []
            for epoch in range(args.epochs):
                model.train()
                order = generator.permutation(len(train))
                train_values = []
                for index in order:
                    optimizer.zero_grad(set_to_none=True)
                    loss = group_loss(model, arm, train[index], args, target_mean, target_std)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    train_values.append(float(loss.detach()))
                validation = mean_loss(model, arm, val, args, target_mean, target_std)
                history.append({"epoch": epoch, "train": float(np.mean(train_values)), "val": validation})
                if validation < best_loss - 1e-5:
                    best_loss, best_epoch, stale = validation, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                if stale >= args.patience:
                    break
            if best_state is None:
                raise RuntimeError("training produced no checkpoint")
            model.load_state_dict(best_state)
            spec = CheckpointSpec(arm=arm, **spec_base)
            path = args.out_dir / arm / f"seed_{seed}.pt"
            extra = {
                "best_epoch": best_epoch, "best_val_loss": best_loss,
                "target_mean": target_mean, "target_std": target_std,
                "topk": args.topk, "student_temperature": args.student_temperature,
                "teacher_temperature_m": args.teacher_temperature_m,
            }
            save_checkpoint(path, model, spec, seed, extra)
            (path.with_suffix(".history.json")).write_text(json.dumps(history, indent=2) + "\n")
            report[arm][str(seed)] = extra | {"checkpoint": str(path)}
            print(json.dumps({"arm": arm, "seed": seed, **extra}, sort_keys=True), flush=True)
    (args.out_dir / "training_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
