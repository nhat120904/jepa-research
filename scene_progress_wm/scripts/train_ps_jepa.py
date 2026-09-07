#!/usr/bin/env python3
"""Train a predictive-state JEPA over the frozen LeWM latents.

One arm per run.  Every arm shares this architecture and parameter count and
differs only in which inputs it may read, so a difference between arms is a
difference in information, not in capacity.

The world model is frozen and its latents are read from the cache, which keeps
this comparable to the recurrent-LeWM baseline (same encoder, same data, same
horizon) and makes the belief the only thing being learned.  The declared cost is
that the belief cannot recover what the frozen encoder discarded; if history
aliasing lives in the encoder rather than in the frame, no filter on top can fix
it, and the arms will say so together rather than separately.
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

from scene_progress_wm.ps_jepa import (  # noqa: E402
    ARMS,
    PROTOCOL,
    PSJEPA,
    PSJEPAConfig,
    SceneSegmentDataset,
    collapse_metrics,
    ps_jepa_losses,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--val-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--latents-name", default="latents.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--segment-blocks", type=int, default=48)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--predict-blocks", type=int, default=5)
    parser.add_argument("--belief-dim", type=int, default=256)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def endless(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def validate(model, sigreg, loader, config, device, batches: int) -> dict:
    model.eval()
    totals = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
    horizons = None
    collapse = None
    seen = 0
    for batch in loader:
        latents = batch["latents"].to(device)
        blocks = batch["blocks"].to(device)
        out = ps_jepa_losses(model, sigreg, latents, blocks, config)
        for key in totals:
            totals[key] += float(out[key])
        step = out["per_horizon"].cpu().numpy()
        horizons = step if horizons is None else horizons + step
        if collapse is None:
            collapse = collapse_metrics(out["beliefs"], out["pred_loss"])
        seen += 1
        if seen >= batches:
            break
    model.train()
    result = {key: value / max(seen, 1) for key, value in totals.items()}
    result["per_horizon"] = (horizons / max(seen, 1)).tolist()
    result.update(collapse or {})
    return result


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("predictive-state JEPA training must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("predictive-state JEPA training requires a GPU allocation")

    seed_everything(args.seed)
    device = torch.device("cuda")

    latent_meta = json.loads((args.cache_dir / "latents_meta.json").read_text())
    config = PSJEPAConfig(
        latent_dim=int(latent_meta["latent_dim"]),
        belief_dim=args.belief_dim,
        segment_blocks=args.segment_blocks,
        burn_in=args.burn_in,
        predict_blocks=args.predict_blocks,
        ema_decay=args.ema_decay,
        **ARMS[args.arm],
    )

    train = SceneSegmentDataset(args.cache_dir, config, args.latents_name)
    val = SceneSegmentDataset(args.val_cache_dir, config, args.latents_name)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = PSJEPA(config).to(device)
    from stable_worldmodel.wm.loss import SIGReg

    sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_proj).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    params = sum(p.numel() for p in trainable)
    print(
        f"arm {args.arm}  trainable {params}  segments train={len(train)} val={len(val)}",
        flush=True,
    )

    stream = endless(train_loader)
    history: list[dict] = []
    best: dict | None = None
    started = time.time()

    for step in range(args.steps):
        for group in optimiser.param_groups:
            group["lr"] = lr_at(step, args)
        batch = next(stream)
        latents = batch["latents"].to(device, non_blocking=True)
        blocks = batch["blocks"].to(device, non_blocking=True)

        out = ps_jepa_losses(model, sigreg, latents, blocks, config)
        optimiser.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimiser.step()
        # the target is an EMA of the online filter, so it moves after the step
        model.update_target()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            metrics = validate(model, sigreg, val_loader, config, device, args.val_batches)
            metrics["step"] = step + 1
            history.append(metrics)
            print(
                f"step {step + 1:6d}  train {float(out['loss']):.5f}  "
                f"val {metrics['loss']:.5f}  pred {metrics['pred_loss']:.5f}  "
                f"sigreg {metrics['sigreg_loss']:.4f}  "
                f"rank {metrics['effective_rank']:.1f}/{metrics['dims']}  "
                f"norm_err {metrics['normalised_error']:.4f}  "
                f"{time.time() - started:.0f}s",
                flush=True,
            )
            # selection is on prediction loss, not the regularised total: SIGReg
            # dominating the sum is exactly the failure the LeWM run hit
            if best is None or metrics["pred_loss"] < best["val_pred_loss"]:
                best = {
                    "step": step + 1,
                    "val_loss": metrics["loss"],
                    "val_pred_loss": metrics["pred_loss"],
                    "effective_rank": metrics["effective_rank"],
                    "normalised_error": metrics["normalised_error"],
                }

    out_dir = args.out_dir / args.arm / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": PROTOCOL,
            "arm": args.arm,
            "config": config.to_json(),
            "encoder": model.encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "target": model.target.state_dict(),
        },
        out_dir / "ps_jepa.pt",
    )
    summary = {
        "protocol": PROTOCOL,
        "arm": args.arm,
        "seed": args.seed,
        "config": config.to_json(),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "parameters": params,
        "num_train_segments": len(train),
        "num_val_segments": len(val),
        "train_cache": str(args.cache_dir),
        "val_cache": str(args.val_cache_dir),
        "latents_meta": latent_meta,
        "best": best,
        "val_history": history,
        "wall_seconds": time.time() - started,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_dir / 'ps_jepa.pt'}", flush=True)


if __name__ == "__main__":
    main()
