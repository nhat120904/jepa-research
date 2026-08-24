#!/usr/bin/env python3
"""Train one matched dynamics arm from the released LeWM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rollout_repair_gate.core import (
    ARMS,
    autoregressive_rollout,
    concatenate_sequences,
    dynamics_state_dict,
    masked_prediction_mse,
    set_dynamics_trainable,
    split_for_order,
    teacher_forced_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--offpolicy-dir", type=Path, required=True)
    parser.add_argument("--expert-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def state_hash(state: dict[str, dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for module in sorted(state):
        for name in sorted(state[module]):
            value = state[module][name].detach().cpu().contiguous()
            digest.update(module.encode())
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_expert(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for path in sorted(root.glob("expert_shard_*.npz")):
        with np.load(path) as data:
            arrays.append(
                (
                    data["true_embeddings"].astype(np.float32),
                    data["actions_normalized"].astype(np.float32),
                    data["valid_horizon"].astype(bool),
                )
            )
    if len(arrays) != 16:
        raise RuntimeError(f"expected 16 expert shards, got {len(arrays)}")
    return concatenate_sequences(arrays)


def load_offpolicy(
    root: Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    counts = {"train": 0, "val": 0, "test": 0}
    for path in sorted(root.glob("snapshot_*/intermediates.npz")):
        order = int(path.parent.name.split("_")[-1])
        row_split = split_for_order(order)
        counts[row_split] += 1
        if row_split != split:
            continue
        with np.load(path) as data:
            future = data["true_future_embeddings"].astype(np.float32)
            current = data["current_embedding"].astype(np.float32)
            actions = data["actions_normalized"].astype(np.float32)
            valid = data["valid_horizon"].astype(bool)
        expanded = np.repeat(current[:, None, None], future.shape[1], axis=1)
        sequence = np.concatenate([expanded, future], axis=2)
        arrays.append(
            (
                sequence.reshape(-1, *sequence.shape[2:]),
                actions.reshape(-1, *actions.shape[2:]),
                valid.reshape(-1, valid.shape[-1]),
            )
        )
    if counts != {"train": 80, "val": 16, "test": 32}:
        raise RuntimeError(f"off-policy split incomplete: {counts}")
    return concatenate_sequences(arrays)


def batch_loss(
    model: torch.nn.Module,
    arm: str,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    valid: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    if arm == "one_step_expert":
        predicted = teacher_forced_rollout(model, embeddings, actions, history_size)
    else:
        predicted = autoregressive_rollout(model, embeddings[:, 0], actions, history_size)
    return masked_prediction_mse(predicted, embeddings[:, 1:], valid)


@torch.inference_mode()
def validation_loss(
    model: torch.nn.Module,
    arm: str,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    history_size: int,
) -> float:
    embeddings, actions, valid = arrays
    values, weights = [], []
    for start in range(0, len(embeddings), batch_size):
        stop = min(start + batch_size, len(embeddings))
        z = torch.from_numpy(embeddings[start:stop]).cuda()
        a = torch.from_numpy(actions[start:stop]).cuda()
        mask = torch.from_numpy(valid[start:stop]).cuda()
        loss = batch_loss(model, arm, z, a, mask, history_size)
        values.append(float(loss.item()) * int(mask.sum()))
        weights.append(int(mask.sum()))
    return float(sum(values) / max(sum(weights), 1))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("predictor training requires a GPU Slurm allocation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the off-policy train split for every arm. Besides being arm 3's
    # training data, its termination mask is the common supervision schedule:
    # index-matched minibatches therefore contain exactly the same number of
    # loss-bearing targets at every horizon in all three arms.
    offpolicy_train = load_offpolicy(args.offpolicy_dir, "train")
    if args.arm.endswith("expert"):
        expert_embeddings, expert_actions, _ = load_expert(args.expert_dir)
        if expert_embeddings.shape[:2] != offpolicy_train[0].shape[:2]:
            raise RuntimeError(
                "expert/off-policy sequence shape mismatch for mask matching: "
                f"{expert_embeddings.shape[:2]} vs {offpolicy_train[0].shape[:2]}"
            )
        train = (
            expert_embeddings,
            expert_actions,
            offpolicy_train[2].copy(),
        )
        training_mask_source = "index-matched off-policy train mask"
    else:
        train = offpolicy_train
        training_mask_source = "native off-policy train mask"
    validation = load_offpolicy(args.offpolicy_dir, "val")
    expected_sequences = 80 * 2 * 96
    if len(train[0]) != expected_sequences:
        raise RuntimeError(
            f"same-data-count gate failed: {len(train[0])} != {expected_sequences}"
        )

    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.interpolate_pos_encoding = True
    initial_state = dynamics_state_dict(model)
    trainable_parameters = set_dynamics_trainable(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(args.seed)
    history = []

    embeddings, actions, valid = train
    targets_per_horizon = valid.sum(axis=0).astype(np.int64)
    if np.any(targets_per_horizon == 0):
        raise RuntimeError(f"empty supervised horizon: {targets_per_horizon.tolist()}")
    valid_mask_sha256 = array_hash(valid.astype(np.uint8))
    for step in range(1, args.steps + 1):
        indices = generator.integers(0, len(embeddings), size=args.batch_size)
        z = torch.from_numpy(embeddings[indices]).cuda()
        a = torch.from_numpy(actions[indices]).cuda()
        mask = torch.from_numpy(valid[indices]).cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, args.arm, z, a, mask, args.history_size)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip))
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "grad_norm": grad_norm,
            }
            # A common off-policy validation set is diagnostic only; final-step
            # checkpoints are used for every arm, so this does not change compute.
            if step % (5 * args.log_every) == 0 or step == args.steps:
                record["offpolicy_val_loss"] = validation_loss(
                    model,
                    "one_step_expert" if args.arm == "one_step_expert" else "multistep_offpolicy",
                    validation,
                    args.batch_size,
                    args.history_size,
                )
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    final_state = dynamics_state_dict(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"{args.arm}_seed{args.seed}.pt"
    metadata = {
        "arm": args.arm,
        "seed": args.seed,
        "base_checkpoint": args.checkpoint,
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "sequences_available": len(embeddings),
        "prediction_calls_per_sequence": int(actions.shape[1]),
        "supervised_targets_total": int(valid.sum()),
        "supervised_targets_per_horizon": targets_per_horizon.tolist(),
        "training_mask_source": training_mask_source,
        "valid_mask_sha256": valid_mask_sha256,
        "trainable_parameters": trainable_parameters,
        "trainable_modules": ["action_encoder", "predictor", "pred_proj"],
        "encoder_frozen": True,
        "same_compute_signature": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "sequence_count": len(embeddings),
            "horizon": int(actions.shape[1]),
            "supervised_targets_total": int(valid.sum()),
            "supervised_targets_per_horizon": targets_per_horizon.tolist(),
            "valid_mask_sha256": valid_mask_sha256,
            "trainable_parameters": trainable_parameters,
        },
        "initial_dynamics_sha256": state_hash(initial_state),
        "final_dynamics_sha256": state_hash(final_state),
        "history": history,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save({"dynamics": final_state, "metadata": metadata}, output)
    (args.out_dir / f"{args.arm}_seed{args.seed}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
