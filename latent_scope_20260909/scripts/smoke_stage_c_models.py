#!/usr/bin/env python3
"""Small compute-node forward/backward smoke test for every Stage-C arm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from stage_c.models import ModelDimensions, build_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Stage-C model smoke must run inside an sbatch job")
    torch.manual_seed(7)
    dims = ModelDimensions(
        cameras=3,
        camera_feature_dim=32,
        proprio_dim=16,
        action_dim=12,
        latent_dim=64,
        context_dim=32,
        summary_width=32,
        summary_tokens=2,
        predictor_width=64,
        task_embed_dim=8,
        num_tasks=2,
        progress_dim=7,
        max_segment_steps=8,
    )
    batch_size, history, future = 2, 3, 8
    batch = {
        "visual_history": torch.randn(batch_size, history, 3, 32),
        "proprio_history": torch.randn(batch_size, history, 16),
        "actions": torch.randn(batch_size, future, 12),
        "visual_future": torch.randn(batch_size, future, 3, 32),
        "proprio_future": torch.randn(batch_size, future, 16),
        "progress_future": torch.randn(batch_size, future, 7),
        "has_progress": torch.ones(batch_size, dtype=torch.bool),
        "eventual_success": torch.tensor([0.0, 1.0]),
        "task_id": torch.tensor([0, 1]),
    }
    weights = {
        "endpoint": 1.0,
        "frame": 1.0,
        "segment": 1.0,
        "reconstruction": 0.25,
        "composition": 0.5,
        "split_consistency": 0.5,
        "progress": 1.0,
        "value": 1.0,
    }
    rows = {}
    for name in (
        "endpoint_only",
        "frame_rollout",
        "simple_progress",
        "unstructured_segment",
        "compositional_segment",
        "direct_value",
    ):
        model = build_arm(name, dims)
        total, losses, logits = model.loss(batch, weights)
        total.backward()
        model.update_ema(0.99)
        if not torch.isfinite(total) or not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite output in {name}")
        rows[name] = {
            "loss": float(total.detach()),
            "loss_terms": sorted(losses),
            "logit_shape": list(logits.shape),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    payload = {"verdict": "STAGE_C_MODEL_SMOKE_PASS", "arms": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["verdict"])


if __name__ == "__main__":
    main()
