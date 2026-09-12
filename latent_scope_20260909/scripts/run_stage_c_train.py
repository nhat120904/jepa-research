#!/usr/bin/env python3
"""Train one matched Stage-C arm on a locked latent-feature manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage_c.data import SegmentWindowDataset, require_binary_training_labels
from stage_c.models import ModelDimensions, build_arm


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_stage_b_gate(config: dict) -> dict:
    variable = config["gate"]["required_environment_variable"]
    result_path = os.environ.get(variable)
    if not result_path:
        raise RuntimeError(f"{variable} must point to the completed Stage-B1 result")
    result = json.loads(Path(result_path).read_text())
    required = config["gate"]["required_stage_b_verdict"]
    if result.get("verdict") != required:
        raise RuntimeError(f"Stage-B1 verdict is {result.get('verdict')}, required {required}")
    return {"path": result_path, "verdict": result["verdict"]}


def dimensions(config: dict) -> ModelDimensions:
    model = config["model"]
    data = config["data"]
    return ModelDimensions(
        cameras=model["camera_count"],
        camera_feature_dim=model["visual_feature_dim"],
        proprio_dim=model["proprio_dim"],
        action_dim=model["action_dim"],
        latent_dim=model["fused_dim"],
        context_dim=model["memory_dim"],
        summary_width=model["summary_dim"],
        summary_tokens=model["summary_tokens"],
        predictor_width=model["fused_dim"],
        task_embed_dim=32,
        num_tasks=model["task_count"],
        progress_dim=model["progress_dim"],
        max_segment_steps=data["segment_steps"],
    )


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def binary_auc(labels: list[float], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label >= 0.5]
    negatives = [score for label, score in zip(labels, scores) if label < 0.5]
    if not positives or not negatives:
        return None
    wins = sum(pos > neg for pos in positives for neg in negatives)
    ties = sum(pos == neg for pos in positives for neg in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


@torch.no_grad()
def evaluate(model, loader, device, weights) -> tuple[dict, list[dict]]:
    model.eval()
    total = 0.0
    count = 0
    scores: list[float] = []
    labels: list[float] = []
    candidates: list[dict] = []
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _, logits = model.loss(batch, weights)
        size = len(logits)
        total += float(loss) * size
        count += size
        batch_scores = logits.float().cpu().tolist()
        batch_labels = batch["eventual_success"].float().cpu().tolist()
        scores.extend(batch_scores)
        labels.extend(batch_labels)
        for group, candidate, label, score in zip(
            batch["candidate_group_id"].cpu().tolist(),
            batch["candidate_index"].cpu().tolist(),
            batch_labels,
            batch_scores,
        ):
            if group >= 0 and candidate >= 0:
                candidates.append(
                    {"group": group, "candidate": candidate, "label": label, "score": score}
                )
    predictions = [score >= 0 for score in scores]
    accuracy = float(np.mean([prediction == (label >= 0.5) for prediction, label in zip(predictions, labels)]))
    return {
        "loss": total / max(1, count),
        "accuracy": accuracy,
        "auc": binary_auc(labels, scores),
        "examples": count,
        "positive_rate": float(np.mean(labels)),
    }, candidates


def candidate_metrics(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    baseline = []
    selected = []
    oracle = []
    complete = 0
    for rows_in_group in groups.values():
        if not any(row["candidate"] == 0 for row in rows_in_group):
            continue
        complete += 1
        baseline.append(next(row["label"] for row in rows_in_group if row["candidate"] == 0))
        selected.append(max(rows_in_group, key=lambda row: row["score"])["label"])
        oracle.append(max(row["label"] for row in rows_in_group))
    if not complete:
        return None
    baseline_rate = float(np.mean(baseline))
    selected_rate = float(np.mean(selected))
    oracle_rate = float(np.mean(oracle))
    return {
        "groups": complete,
        "baseline_success_rate": baseline_rate,
        "selected_success_rate": selected_rate,
        "oracle_success_rate": oracle_rate,
        "selected_gain_percentage_points": 100.0 * (selected_rate - baseline_rate),
        "oracle_regret_percentage_points": 100.0 * (oracle_rate - selected_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Stage-C training must run inside an sbatch job")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-C training requires a CUDA batch node")

    config = json.loads(args.config.read_text())
    if args.arm not in config["training"]["arms"]:
        raise ValueError(f"Arm {args.arm} is not locked in the config")
    result_path = args.output_dir / "stage_c_arm_result.json"
    result = {
        "verdict": "STAGE_C_ARM_RUNNING",
        "arm": args.arm,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
    write_json(result_path, result)
    try:
        result["stage_b_gate"] = assert_stage_b_gate(config)
        training = config["training"]
        data = config["data"]
        seed = int(training["seed"])
        seed_all(seed)
        train_data = SegmentWindowDataset(
            data["manifest"],
            "train",
            data["history_steps"],
            data["segment_steps"],
            data["window_stride"],
            progress_dim=config["model"]["progress_dim"],
            training=True,
            forbidden_training_source=data["forbidden_training_source"],
        )
        val_data = SegmentWindowDataset(
            data["manifest"],
            "val",
            data["history_steps"],
            data["segment_steps"],
            data["window_stride"],
            progress_dim=config["model"]["progress_dim"],
        )
        candidate_data = SegmentWindowDataset(
            data["manifest"],
            "candidate_eval",
            data["history_steps"],
            data["segment_steps"],
            stride=data["segment_steps"],
            progress_dim=config["model"]["progress_dim"],
        )
        result["training_class_balance"] = require_binary_training_labels(train_data)
        generator = torch.Generator().manual_seed(seed)
        loader_args = {
            "batch_size": training["batch_size"],
            "num_workers": training["num_workers"],
            "pin_memory": True,
            "persistent_workers": training["num_workers"] > 0,
        }
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
        val_loader = DataLoader(val_data, shuffle=False, **loader_args)
        candidate_loader = DataLoader(candidate_data, shuffle=False, **loader_args)
        device = torch.device("cuda")
        model = build_arm(args.arm, dimensions(config)).to(device)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        weights = training["loss_weights"]
        result["trainable_parameters"] = sum(parameter.numel() for parameter in parameters)
        history = []
        best_loss = float("inf")
        started = time.monotonic()
        for epoch in range(training["epochs"]):
            model.train()
            running = 0.0
            seen = 0
            for batch in train_loader:
                batch = move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, _, _ = model.loss(batch, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, training["gradient_clip_norm"])
                optimizer.step()
                model.update_ema(training["ema_decay"])
                running += float(loss.detach()) * len(batch["task_id"])
                seen += len(batch["task_id"])
            validation, _ = evaluate(model, val_loader, device, weights)
            epoch_row = {
                "epoch": epoch + 1,
                "train_loss": running / max(1, seen),
                "validation": validation,
            }
            history.append(epoch_row)
            write_json(args.output_dir / "history.json", {"epochs": history})
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                torch.save(
                    {
                        "arm": args.arm,
                        "model": model.state_dict(),
                        "dimensions": dimensions(config).__dict__,
                        "config": config,
                    },
                    args.output_dir / "best.pt",
                )
            print(
                f"{args.arm} epoch {epoch + 1}/{training['epochs']} "
                f"train={epoch_row['train_loss']:.6f} val={validation['loss']:.6f}",
                flush=True,
            )

        checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        validation, _ = evaluate(model, val_loader, device, weights)
        _, candidate_rows = evaluate(model, candidate_loader, device, weights)
        result.update(
            {
                "verdict": "STAGE_C_ARM_COMPLETE",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "best_validation": validation,
                "candidate_ranking": candidate_metrics(candidate_rows),
                "checkpoint": str(args.output_dir / "best.pt"),
            }
        )
        write_json(result_path, result)
    except Exception as error:
        result["verdict"] = "STAGE_C_ARM_ERROR"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
