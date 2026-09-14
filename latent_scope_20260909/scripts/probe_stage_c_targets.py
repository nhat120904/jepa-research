#!/usr/bin/env python3
"""Small frozen-feature readout diagnostic, not a training/headroom authorization gate.

Uses disjoint manifest source splits; never trains on Stage-B validation branches.
Native progress is explicitly a privileged reference, not a deployable visual baseline.
An optional *training-only* checkpoint supplies a compressed target for comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from stage_c.data import SegmentWindowDataset, StageCDataError
from stage_c.models import ModelDimensions, build_arm
from stage_c.metrics import candidate_metrics


def ridge_predict(train, labels, test):
    mean = train.mean(0)
    scale = train.std(0, unbiased=False).clamp_min(1e-3)
    x = (train - mean) / scale
    z = (test - mean) / scale
    center = labels.mean(0)
    # Normalize kernel by feature count; fixed regularization, no candidate-label tuning.
    kernel = x @ x.T / x.shape[1]
    dual = torch.linalg.solve(kernel + 0.1 * torch.eye(len(x)), labels - center)
    return z @ x.T / x.shape[1] @ dual + center


@torch.no_grad()
def examples(dataset, cap, target_model=None):
    if any(int(e.get("candidate_group_id", -1)) >= 0 for e in dataset.entries):
        if len(dataset) > cap:
            raise StageCDataError("Candidate panel exceeds cap: increase cap, never truncate groups")
        indices = list(range(len(dataset)))
    else:
        indices = torch.linspace(0, len(dataset) - 1, min(len(dataset), cap)).long().unique().tolist()
    views = {"history_endpoint": [], "ordered_true_sequence": []}
    if target_model is not None:
        views["compressed_true_sequence"] = []
    progress_available = True
    progress = []
    labels, metadata = [], []
    for index in indices:
        row = dataset[index]
        history = torch.cat([row["visual_history"].flatten(), row["proprio_history"].flatten()])
        future = torch.cat([row["visual_future"].flatten(1), row["proprio_future"]], dim=-1)
        task = torch.nn.functional.one_hot(row["task_id"], num_classes=2).float()
        shared = torch.cat([history, task])
        views["history_endpoint"].append(torch.cat([shared, future[-1]]))
        views["ordered_true_sequence"].append(torch.cat([shared, future.flatten()]))
        if target_model is not None:
            batch = {key: value.unsqueeze(0) for key, value in row.items()}
            summary, _ = target_model.target_encoder(target_model.future(batch))
            views["compressed_true_sequence"].append(torch.cat([shared, summary.flatten()]))
        progress_available = progress_available and bool(row["has_progress"])
        progress.append(row["progress_future"][-1])
        labels.append(row["eventual_success"].reshape(1))
        metadata.append({"task": int(row["task_id"]), "group": int(row["candidate_group_id"]),
                         "candidate": int(row["candidate_index"]), "candidate_count": int(row["candidate_count"])})
    if progress_available:
        views["privileged_native_monitor"] = progress
    return {key: torch.stack(value) for key, value in views.items()}, torch.stack(labels), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-windows", type=int, default=256)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Readout analysis must run in a CPU sbatch job")
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text())
    data = cfg["data"]
    result = {"verdict": "TARGET_PROBE_NOT_READY", "job_id": os.environ["SLURM_JOB_ID"],
              "stage_c_authorized": False, "diagnostic_only": True,
              "feature_mode": cfg["encoder"].get("feature_mode", "cls")}
    try:
        manifest = Path(data["manifest"])
        if not manifest.is_file():
            raise StageCDataError(f"Missing feature manifest: {manifest}")
        result["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        target = None
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if checkpoint.get("manifest_sha256") != result["manifest_sha256"]:
                raise StageCDataError("Target checkpoint must declare this locked manifest")
            if checkpoint.get("implementation_protocol") != "qualification_v1_ordered_matched":
                raise StageCDataError("Target checkpoint predates the corrected implementation")
            target = build_arm(checkpoint["arm"], ModelDimensions(**checkpoint["dimensions"])).eval()
            if not hasattr(target, "target_encoder"):
                raise StageCDataError("Checkpoint has no compressed trajectory target")
            target.load_state_dict(checkpoint["model"])
        datasets = {split: SegmentWindowDataset(manifest, split, data["history_steps"], data["segment_steps"],
                                                stride=data["window_stride"], training=split == "train",
                                                progress_dim=cfg["model"]["progress_dim"])
                    for split in ("train", "val", "candidate_eval")}
        train, labels, _ = examples(datasets["train"], args.max_windows, target)
        if labels.min() == labels.max():
            raise StageCDataError("Readout training sample needs both success and failure labels")
        result["train_examples"] = len(labels)
        result["readouts"] = {}
        for split in ("val", "candidate_eval"):
            features, truth, metadata = examples(datasets[split], args.max_windows, target)
            result["readouts"][split] = {}
            for name in train.keys() & features.keys():
                scores = ridge_predict(train[name], labels, features[name]).squeeze(1)
                row = {"examples": len(truth), "mse": float((scores - truth.squeeze(1)).square().mean()),
                       "accuracy": float(((scores >= .5) == (truth.squeeze(1) >= .5)).float().mean())}
                if split == "candidate_eval":
                    ranked = [dict(meta, label=float(label), score=float(score))
                              for meta, label, score in zip(metadata, truth.squeeze(1), scores)]
                    row["candidate_ranking"] = candidate_metrics(ranked)
                result["readouts"][split][name] = row
        result["compressed_target_tested"] = target is not None
        result["verdict"] = "TARGET_PROBE_DIAGNOSTIC_COMPLETE"
        result["limitations"] = ["Fixed ridge readout, unequal feature dimensions; not a capacity-matched method comparison",
                                  "True-future inputs estimate information availability, not predictive planning",
                                  "Native monitor is privileged; ordered learned simple-progress control is still required",
                                  "Small capped windows; no population claim or automatic training gate"]
    except (StageCDataError, FileNotFoundError) as error:
        result["blocker"] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["verdict"], flush=True)


if __name__ == "__main__":
    main()
