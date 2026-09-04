#!/usr/bin/env python3
"""Train one arm of the Scene event-observer coverage/history factorial.

The four arms share this script, this architecture and this optimiser; they
differ only in ``--history`` (current frame versus full prefix) and
``--coverage`` (canonical milestone roots versus roots plus counterfactual
endpoints).  See docs/SCENE_EVENT_HISTORY_PROTOCOL.md.
"""

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

from event_smdp_h0.scene_event_history import (  # noqa: E402
    ABLATIONS,
    NO_SKILL,
    PROTOCOL,
    HistoryEventObserver,
    canonical_skill_paths,
)
from event_smdp_h0.scene_history_dataset import build_sequences, restrict  # noqa: E402
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
    parser.add_argument("--feature-view", choices=("latent", "privileged"), default="latent")
    parser.add_argument("--history", choices=("frame", "full"), required=True)
    parser.add_argument("--coverage", choices=("canonical", "full"), required=True)
    # `none` leaves the original 2x2 factorial untouched; the other settings
    # decompose the history input into its action and observation halves.
    parser.add_argument("--ablation", choices=ABLATIONS, default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def assert_canonical_paths_match_collector() -> None:
    """The pure copy used for dataset construction must equal the collector."""

    from event_smdp_h0.scripts.collect_scene_h1 import canonical_paths

    for task_id in (4, 5):
        if tuple(canonical_paths(task_id)) != canonical_skill_paths(task_id):
            raise RuntimeError(f"canonical path mismatch for task {task_id}")


def to_tensors(
    dataset: dict[str, np.ndarray],
    *,
    history: str,
    ablation: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    goal_mean: np.ndarray,
    goal_std: np.ndarray,
) -> dict[str, torch.Tensor]:
    feature = (dataset["feature"].astype(np.float32) - feature_mean) / feature_std
    prev_skill = dataset["prev_skill"].astype(np.int64)
    length = dataset["length"].astype(np.int64)
    if history == "frame":
        rows = np.arange(len(length))
        last = length - 1
        feature = feature[rows, last][:, None]
        prev_skill = np.full((len(length), 1), NO_SKILL, dtype=np.int64)
        length = np.ones_like(length)
    else:
        # Padding beyond `length` is masked out by the gather in the model, but
        # zero it explicitly so a shape bug cannot silently leak information.
        mask = np.arange(feature.shape[1])[None] < length[:, None]
        feature = feature * mask[..., None]
        prev_skill = np.where(mask, prev_skill, NO_SKILL)
    if ablation == "action_only":
        # Zero after standardisation so the deployed evaluator can reproduce
        # this exactly without knowing the training statistics.
        feature = np.zeros_like(feature)
    elif ablation == "obs_history":
        prev_skill = np.full_like(prev_skill, NO_SKILL)
    goal = (dataset["goal"].astype(np.float32) - goal_mean) / goal_std
    device = torch.device("cuda")
    return {
        "feature": torch.from_numpy(np.ascontiguousarray(feature)).to(device),
        "prev_skill": torch.from_numpy(np.ascontiguousarray(prev_skill)).to(device),
        "length": torch.from_numpy(length).to(device),
        "goal": torch.from_numpy(goal).to(device),
        "task": torch.from_numpy(task_matrix(dataset["task_id"])).to(device),
        "cube": torch.from_numpy(dataset["cube"].astype(np.int64)).to(device),
        "window": torch.from_numpy(dataset["window"].astype(np.int64)).to(device),
        "stable": torch.from_numpy(dataset["stable"].astype(np.float32)).to(device),
        "is_endpoint": torch.from_numpy(dataset["is_endpoint"].astype(np.int64)).to(device),
        # `fit_model` sizes its batches from this key.
        "skill": torch.from_numpy(dataset["length"].astype(np.int64)).to(device),
    }


def accuracy_block(model: HistoryEventObserver, data: dict[str, torch.Tensor]) -> dict:
    with torch.inference_mode():
        out = model(
            data["feature"], data["prev_skill"], data["goal"], data["task"], data["length"]
        )
        cube_correct = out["cube_logits"].argmax(-1) == data["cube"]
        window_correct = out["window_logits"].argmax(-1) == data["window"]
        stable_correct = (torch.sigmoid(out["stable_logit"]) >= 0.5) == data["stable"].bool()
        exact = cube_correct & window_correct & stable_correct

        def rate(mask: torch.Tensor) -> dict:
            if int(mask.sum().item()) == 0:
                return {"n": 0, "exact_q_accuracy": None}
            return {
                "n": int(mask.sum().item()),
                "exact_q_accuracy": float(exact[mask].float().mean().item()),
            }

        return {
            "cube_accuracy": float(cube_correct.float().mean().item()),
            "window_accuracy": float(window_correct.float().mean().item()),
            "stable_accuracy": float(stable_correct.float().mean().item()),
            "exact_q_accuracy": float(exact.float().mean().item()),
            "canonical_roots": rate(data["is_endpoint"] == 0),
            "off_canonical_endpoints": rate(data["is_endpoint"] == 1),
        }


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("observer training must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("observer training requires a GPU allocation")
    assert_canonical_paths_match_collector()
    seed_everything(args.seed)

    train_np, train_paths = load_split(args.data_root, "train")
    val_np, val_paths = load_split(args.data_root, "val")
    overlap = set(train_np["reset_seed"].tolist()) & set(val_np["reset_seed"].tolist())
    if overlap:
        raise RuntimeError(f"train/val reset leakage: {sorted(overlap)}")

    train_full = build_sequences(train_np, args.feature_view)
    val_full = build_sequences(val_np, args.feature_view)
    train_set = restrict(train_full, args.coverage)
    val_set = restrict(val_full, args.coverage)

    flat_train = train_set["feature"].reshape(-1, train_set["feature"].shape[-1])
    steps = np.arange(train_set["feature"].shape[1])[None] < train_set["length"][:, None]
    feature_mean, feature_std = standardizer(flat_train[steps.reshape(-1)])
    goal_mean, goal_std = standardizer(train_set["goal"])

    if args.ablation != "none" and args.history != "full":
        raise ValueError("input ablations only apply to the full-history arm")
    kwargs = {
        "history": args.history,
        "ablation": args.ablation,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "goal_mean": goal_mean,
        "goal_std": goal_std,
    }
    train = to_tensors(train_set, **kwargs)
    val = to_tensors(val_set, **kwargs)
    # Held-out diagnostic on the complete validation state distribution, which
    # the canonical arms never train on.  Never used for model selection.
    val_all = to_tensors(val_full, **kwargs)

    feature_dim = int(train["feature"].shape[-1])
    goal_dim = int(train["goal"].shape[1])
    model = HistoryEventObserver(feature_dim, goal_dim, width=args.width).cuda()
    stable_weight = binary_pos_weight(train["stable"])

    def loss_fn(data: dict[str, torch.Tensor], index: torch.Tensor | None):
        out = model(
            select(data, index, "feature"),
            select(data, index, "prev_skill"),
            select(data, index, "goal"),
            select(data, index, "task"),
            select(data, index, "length"),
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
        seed=args.seed + 4409,
    )
    model.load_state_dict(state)
    model.eval()

    if args.ablation == "none":
        arm = f"{args.history if args.history == 'frame' else 'history'}_{args.coverage}"
    else:
        arm = f"{args.ablation}_{args.coverage}"
    metadata = {
        "protocol": PROTOCOL,
        "arm": arm,
        "ablation": args.ablation,
        "history": args.history,
        "coverage": args.coverage,
        "feature_view": args.feature_view,
        "seed": args.seed,
        "fit": fit,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "metrics_matched_val": accuracy_block(model, val),
        "metrics_full_val": accuracy_block(model, val_all),
        "num_train_samples": int(len(train_set["cube"])),
        "num_val_samples": int(len(val_set["cube"])),
        "num_full_val_samples": int(len(val_full["cube"])),
        "max_history_steps": int(train_full["feature"].shape[1]),
        "train_reset_seeds": sorted(set(train_np["reset_seed"].tolist())),
        "val_reset_seeds": sorted(set(val_np["reset_seed"].tolist())),
        "source_train_shards": train_paths,
        "source_val_shards": val_paths,
    }
    payload = {
        "protocol": PROTOCOL,
        "arm": arm,
        "ablation": args.ablation,
        "history_length": 1 if args.history == "frame" else int(train_full["feature"].shape[1]),
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
    print(json.dumps({k: v for k, v in metadata.items() if not k.startswith("source_")}, sort_keys=True))


if __name__ == "__main__":
    main()
