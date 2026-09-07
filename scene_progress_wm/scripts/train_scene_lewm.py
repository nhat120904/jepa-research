"""Train LeWM from scratch on the OGBench-Scene play cache.

The objective, optimiser family and regulariser weight are the published LeWM ones.
What differs is the arena (Scene at 64x64 instead of Cube at 224x224) and the budget,
which is expressed in optimiser steps rather than epochs so that every arm of this
program is trained for exactly the same amount of compute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.scene_data import EPISODE_LEN  # noqa: E402
from scene_progress_wm.scene_lewm import (  # noqa: E402
    PROTOCOL,
    LeWMConfig,
    SceneWindowDataset,
    build_lewm,
    image_stats,
    lewm_losses,
    normalise_pixels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--val-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=10000)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def infinite(loader: DataLoader):
    while True:
        yield from loader


@torch.no_grad()
def validate(model, sigreg, loader, config, device, mean, std, batches: int) -> dict:
    model.eval()
    totals = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
    seen = 0
    for batch in loader:
        if seen >= batches:
            break
        pixels = normalise_pixels(batch["pixels"].to(device, non_blocking=True), mean, std)
        action = batch["action"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = lewm_losses(model, sigreg, pixels, action, config)
        for key in totals:
            totals[key] += float(out[key])
        seen += 1
    model.train()
    return {key: value / max(seen, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model, config, args, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": PROTOCOL,
            "config": config.to_json(),
            "seed": args.seed,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "action_normalised": False,
            "image_mean": list(map(float, image_stats(torch.device("cpu"))[0].flatten())),
            "image_std": list(map(float, image_stats(torch.device("cpu"))[1].flatten())),
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("training must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a GPU allocation")

    seed_everything(args.seed)
    device = torch.device("cuda")
    config = LeWMConfig()

    train_meta = json.loads((args.cache_dir / "meta.json").read_text())
    val_meta = json.loads((args.val_cache_dir / "meta.json").read_text())
    for name, meta in (("train", train_meta), ("val", val_meta)):
        if meta["render_info"] != train_meta["render_info"]:
            raise RuntimeError("train and val caches were drawn by different renderers")
        if meta["pixel_stride"] != train_meta["pixel_stride"]:
            raise RuntimeError("train and val caches disagree on the render grid")
        if meta["render_size"] != config.img_size:
            raise RuntimeError(f"{name} cache render size != model image size")
        report = args.cache_dir if name == "train" else args.val_cache_dir
        complete = json.loads((report / "cache_complete.json").read_text())
        if not complete["complete"]:
            raise RuntimeError(f"{name} cache has {complete['num_blank_rows']} unrendered rows")

    stride = int(train_meta["pixel_stride"])
    train_set = SceneWindowDataset(args.cache_dir, config, EPISODE_LEN, pixel_stride=stride)
    val_set = SceneWindowDataset(
        args.val_cache_dir, config, EPISODE_LEN, pixel_stride=stride
    )
    print(f"train windows {len(train_set)}  val windows {len(val_set)}", flush=True)

    # The cache renders one frame every `stride` rows and leaves the rest zero, so a
    # window that drifted off the grid would train on black images without erroring.
    probe_rng = np.random.default_rng(args.seed + 17)
    for index in probe_rng.integers(0, len(train_set), size=16):
        frames = train_set[int(index)]["pixels"].numpy()
        if frames.reshape(frames.shape[0], -1).max(axis=1).min() == 0:
            raise RuntimeError(f"training window {int(index)} contains an unrendered frame")

    generator = torch.Generator().manual_seed(args.seed + 4409)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=3 if args.num_workers > 0 else None,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=min(args.num_workers, 4),
        pin_memory=True,
    )

    from stable_worldmodel.wm.loss import SIGReg

    model = build_lewm(config).to(device)
    sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_proj).to(device)
    params = int(sum(p.numel() for p in model.parameters()))
    print(f"parameters {params}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    mean, std = image_stats(device)

    history = []
    best = {"step": -1, "val_pred_loss": float("inf"), "val_loss": float("inf")}
    stream = infinite(train_loader)
    started = time.time()
    model.train()

    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        batch = next(stream)
        pixels = normalise_pixels(batch["pixels"].to(device, non_blocking=True), mean, std)
        action = batch["action"].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = lewm_losses(model, sigreg, pixels, action, config)
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_every == 0:
            rate = (step + 1) / max(time.time() - started, 1e-6)
            print(
                f"step {step + 1}/{args.steps} loss {float(loss):.5f} "
                f"pred {float(out['pred_loss']):.5f} sigreg {float(out['sigreg_loss']):.5f} "
                f"lr {lr_at(step, args):.2e} {rate:.2f} steps/s",
                flush=True,
            )

        if (step + 1) % args.val_every == 0 or (step + 1) == args.steps:
            metrics = validate(
                model, sigreg, val_loader, config, device, mean, std, args.val_batches
            )
            metrics["step"] = step + 1
            history.append(metrics)
            print(json.dumps({"val": metrics}, sort_keys=True), flush=True)
            # Select on prediction loss alone. SIGReg is a batch-distribution
            # regulariser: under model.eval() the projector's BatchNorm switches to
            # running statistics, which shifts the embedding distribution and inflates
            # the term by more than an order of magnitude (train 1.6 vs val 35.8 at
            # step 4000). Selecting on the summed loss would pick checkpoints by that
            # artifact rather than by how well the model predicts.
            if metrics["pred_loss"] < best["val_pred_loss"]:
                best = {
                    "step": step + 1,
                    "val_pred_loss": metrics["pred_loss"],
                    "val_loss": metrics["loss"],
                }
                save_checkpoint(
                    args.out_dir / "lewm_best.pt",
                    model,
                    config,
                    args,
                    {"selected_at_step": step + 1, "val": metrics},
                )

        if (step + 1) % args.save_every == 0:
            save_checkpoint(
                args.out_dir / "lewm_last.pt", model, config, args, {"step": step + 1}
            )

    save_checkpoint(
        args.out_dir / "lewm_last.pt", model, config, args, {"step": args.steps}
    )

    summary = {
        "protocol": PROTOCOL,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "seed": args.seed,
        "config": config.to_json(),
        "parameters": params,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "action_normalised": False,
        "num_train_windows": len(train_set),
        "num_val_windows": len(val_set),
        "train_cache": str(args.cache_dir),
        "val_cache": str(args.val_cache_dir),
        "train_cache_meta": train_meta,
        "val_cache_meta": val_meta,
        "best": best,
        "val_history": history,
        "wall_seconds": time.time() - started,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"best": best, "parameters": params}, sort_keys=True))


if __name__ == "__main__":
    main()
