#!/usr/bin/env python3
"""Train the frozen low-level XY-conditioned Ant skill used by all Direction-A arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from skill_handoff_wm.data import GoalConditionedDataset  # noqa: E402
from skill_handoff_wm.policy import GoalPolicy  # noqa: E402
from skill_handoff_wm.sim import require_slurm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--min-future", type=int, default=1)
    parser.add_argument("--max-future", type=int, default=100)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--goal-scale", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=2_000)
    parser.add_argument("--val-batches", type=int, default=50)
    return parser.parse_args()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensors(batch, mean, std, goal_scale, device):
    states, goals, actions, _ = batch
    states = torch.as_tensor(states, device=device)
    goals = torch.as_tensor(goals, device=device)
    actions = torch.as_tensor(actions, device=device)
    mean_t = torch.as_tensor(mean, device=device)
    std_t = torch.as_tensor(std, device=device)
    normalized_states = (states - mean_t) / std_t
    normalized_delta = (goals - states[:, :2]) / goal_scale
    return normalized_states, normalized_delta, actions


@torch.inference_mode()
def validate(model, dataset, rng, args, mean, std, device) -> float:
    losses = []
    model.eval()
    for _ in range(args.val_batches):
        batch = dataset.sample(rng, args.batch_size, args.min_future, args.max_future)
        states, goals, actions = tensors(batch, mean, std, args.goal_scale, device)
        losses.append(nn.functional.mse_loss(model(states, goals), actions).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    require_slurm()
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GCBC training requires a CUDA allocation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    train_data = GoalConditionedDataset(args.train_data)
    val_data = GoalConditionedDataset(args.val_data)
    state_mean, state_std = train_data.normalization()
    state_dim = train_data.observations.shape[1]
    action_dim = train_data.actions.shape[1]
    device = torch.device("cuda")
    model = GoalPolicy(state_dim, action_dim, args.width, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1_000_003)
    config = vars(args) | {"state_dim": state_dim, "action_dim": action_dim}
    ledger = {
        "protocol": "direction_a_gcbc_v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hostname": os.uname().nodename,
        "train_sha256": file_sha256(args.train_data),
        "val_sha256": file_sha256(args.val_data),
        "train_rows": len(train_data.observations),
        "val_rows": len(val_data.observations),
        "config": config,
    }
    (out_dir / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    best_val = float("inf")
    start_time = time.monotonic()
    with (out_dir / "metrics.jsonl").open("w") as metrics:
        for step in range(1, args.steps + 1):
            batch = train_data.sample(train_rng, args.batch_size, args.min_future, args.max_future)
            states, goals, actions = tensors(batch, state_mean, state_std, args.goal_scale, device)
            prediction = model(states, goals)
            loss = nn.functional.mse_loss(prediction, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                val_loss = validate(model, val_data, val_rng, args, state_mean, state_std, device)
                record = {
                    "step": step,
                    "train_mse": float(loss.item()),
                    "val_mse": val_loss,
                    "elapsed_seconds": time.monotonic() - start_time,
                }
                print(json.dumps(record), flush=True)
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                payload = {
                    "model": model.state_dict(),
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "config": config,
                    "step": step,
                    "val_mse": val_loss,
                }
                torch.save(payload, out_dir / "latest.pt")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(payload, out_dir / "best.pt")


if __name__ == "__main__":
    main()
