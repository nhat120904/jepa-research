#!/usr/bin/env python3
"""Train the A2 state-dependent termination baseline from oracle branch states."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from skill_handoff_wm.sim import require_slurm  # noqa: E402
from skill_handoff_wm.termination import TerminationNet, make_features  # noqa: E402


def load_examples(path: Path):
    examples = []
    handoff_id = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        if episode["arm"] != "oracle":
            continue
        episode_key = (int(episode["task_id"]), int(episode["seed"]))
        for handoff in episode["handoffs"]:
            branches = handoff.get("oracle_branches")
            if not branches:
                continue
            chosen = int(handoff["chosen_duration"])
            for branch in branches:
                if "exit_observation" not in branch:
                    raise ValueError("collection lacks branch exit observations; rerun with A2 code")
                examples.append(
                    {
                        "episode_key": episode_key,
                        "handoff_id": handoff_id,
                        "state": branch["exit_observation"],
                        "current_target": handoff["target_xy"],
                        "next_target": handoff["next_target_xy"],
                        "elapsed": int(branch["duration"]),
                        "is_final": bool(handoff["is_final_waypoint"]),
                        "label": int(branch["duration"] == chosen),
                        "chosen": chosen,
                    }
                )
            handoff_id += 1
    if not examples:
        raise ValueError("no oracle branch examples found")
    return examples


@torch.inference_mode()
def predict(model, features, device, batch_size=4096):
    model.eval()
    values = []
    for start in range(0, len(features), batch_size):
        logits = model(torch.as_tensor(features[start : start + batch_size], device=device))
        values.append(torch.sigmoid(logits).cpu().numpy())
    model.train()
    return np.concatenate(values)


def tune_threshold(examples, probabilities):
    grouped = {}
    for example, probability in zip(examples, probabilities):
        grouped.setdefault(example["handoff_id"], []).append((example, float(probability)))
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        correct = 0
        absolute_error = 0.0
        for rows in grouped.values():
            rows.sort(key=lambda item: item[0]["elapsed"])
            predicted = rows[-1][0]["elapsed"]
            for example, probability in rows:
                if probability >= threshold:
                    predicted = example["elapsed"]
                    break
            chosen = rows[0][0]["chosen"]
            correct += int(predicted == chosen)
            absolute_error += abs(predicted - chosen)
        candidate = (correct / len(grouped), -absolute_error / len(grouped), float(threshold))
        if best is None or candidate > best:
            best = candidate
    return {"exact_accuracy": best[0], "mean_absolute_error": -best[1], "threshold": best[2]}


def main() -> None:
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--goal-scale", type=float, default=4.0)
    parser.add_argument("--max-duration", type=float, default=80.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("termination training requires a CUDA allocation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = load_examples(Path(args.episodes))
    episode_keys = sorted(set(example["episode_key"] for example in examples))
    split_rng = np.random.default_rng(args.seed)
    split_rng.shuffle(episode_keys)
    cut = max(1, int(0.8 * len(episode_keys)))
    train_keys = set(episode_keys[:cut])
    train_examples = [example for example in examples if example["episode_key"] in train_keys]
    val_examples = [example for example in examples if example["episode_key"] not in train_keys]
    if not val_examples:
        raise ValueError("need at least two collection episodes for a grouped validation split")

    train_states = np.asarray([example["state"] for example in train_examples], dtype=np.float32)
    state_mean = train_states.mean(axis=0)
    state_std = np.maximum(train_states.std(axis=0), 1e-3)

    def features(rows):
        return make_features(
            np.asarray([row["state"] for row in rows]),
            np.asarray([row["current_target"] for row in rows]),
            np.asarray([row["next_target"] for row in rows]),
            np.asarray([row["elapsed"] for row in rows]),
            args.max_duration,
            np.asarray([row["is_final"] for row in rows]),
            state_mean,
            state_std,
            args.goal_scale,
        )

    train_x = features(train_examples)
    train_y = np.asarray([row["label"] for row in train_examples], dtype=np.float32)
    val_x = features(val_examples)
    device = torch.device("cuda")
    model = TerminationNet(train_x.shape[1] - 6, args.width, args.depth).to(device)
    positives = max(float(train_y.sum()), 1.0)
    pos_weight = torch.tensor((len(train_y) - positives) / positives, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed + 17)
    for step in range(1, args.steps + 1):
        indices = rng.integers(0, len(train_x), size=args.batch_size)
        batch_x = torch.as_tensor(train_x[indices], device=device)
        batch_y = torch.as_tensor(train_y[indices], device=device)
        logits = model(batch_x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 1000 == 0 or step == args.steps:
            print(json.dumps({"step": step, "train_bce": float(loss.item())}), flush=True)

    probabilities = predict(model, val_x, device)
    tuning = tune_threshold(val_examples, probabilities)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"state_dim": int(train_x.shape[1] - 6)}
    payload = {
        "model": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "threshold": tuning["threshold"],
        "config": config,
    }
    torch.save(payload, out_dir / "termination.pt")
    summary = {
        "protocol": "direction_a_a2_termination_v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "num_collection_episodes": len(episode_keys),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "positive_rate_train": float(train_y.mean()),
        "validation_duration": tuning,
        "config": config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
