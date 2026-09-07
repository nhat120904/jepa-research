"""Train the progress head on frozen world-model latents.

Primary signal is self-supervised: how many action blocks remain until a goal drawn from
the same trajectory's future, regressed with an expectile because play data is not
expert data. The supervised arm adds an auxiliary term on the five components OGBench's
own success predicate scores, which is privileged at training time and never available
at deployment -- that arm exists to measure how much of the gap a perfect progress
signal would close, not to be deployed.

Every arm shares this script, this architecture and this optimiser. They differ only in
``--arm``, which sets which inputs the head may read and whether progress is forced to
latch.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.progress_head import (  # noqa: E402
    ARMS,
    PROTOCOL,
    ProgressConfig,
    ProgressHead,
    expectile_loss,
)
from scene_progress_wm.scene_data import EPISODE_LEN, goal_match  # noqa: E402
from scene_progress_wm.scene_eval import state_row  # noqa: E402
from scene_progress_wm.scene_lewm import LeWMConfig  # noqa: E402

COMPONENTS = ("cube", "button_0", "button_1", "drawer", "window")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--val-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--latents-name", type=str, default="latents.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-blocks", type=int, default=8)
    parser.add_argument("--goal-blocks", type=int, default=40, help="max blocks to a goal")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--expectile", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LatentSequences:
    """Windows of consecutive grid frames, with a goal drawn from their own future."""

    def __init__(self, cache_dir: Path, latents_name: str, window: int, goal_span: int):
        cache_dir = Path(cache_dir)
        self.meta = json.loads((cache_dir / "meta.json").read_text())
        self.stride = int(self.meta["pixel_stride"])
        self.latents = np.load(cache_dir / latents_name, mmap_mode="r")
        self.actions = np.load(cache_dir / "actions.npy", mmap_mode="r")
        self.track = np.load(cache_dir / "state.npy", mmap_mode="r")
        self.buttons = np.load(cache_dir / "button_states.npy", mmap_mode="r")
        self.window = window
        self.goal_span = goal_span

        total = int(self.meta["num_rows"])
        grid = np.arange(0, total, self.stride, dtype=np.int64)
        if grid.size != self.latents.shape[0]:
            raise RuntimeError("latent cache does not match the render grid")
        self.grid = grid
        episode = grid // EPISODE_LEN
        # a run must stay inside one episode and leave room for a goal beyond it
        span = window + goal_span
        ok = np.zeros(grid.size, dtype=bool)
        if grid.size > span:
            ok[: grid.size - span] = (
                episode[: grid.size - span] == episode[span : grid.size]
            )
        # the first frame of a window needs the block that led into it
        ok[0] = False
        self.starts = np.flatnonzero(ok)
        if self.starts.size == 0:
            raise RuntimeError("no usable windows in this cache")

    def action_block(self, grid_index: int) -> np.ndarray:
        """The primitive actions leaving the previous grid frame, flattened."""
        row = int(self.grid[grid_index])
        return np.asarray(
            self.actions[row - self.stride : row], dtype=np.float32
        ).reshape(-1)

    def component_match(self, grid_index: int, goal_grid_index: int) -> np.ndarray:
        row = int(self.grid[grid_index])
        goal_row = int(self.grid[goal_grid_index])
        match = goal_match(
            state_row(self.track, row),
            self.buttons[row],
            state_row(self.track, goal_row),
            self.buttons[goal_row],
        )
        return match.vector

    def sample(self, rng: np.random.Generator, batch: int) -> dict[str, np.ndarray]:
        starts = rng.choice(self.starts, size=batch, replace=True)
        goal_gap = rng.integers(1, self.goal_span + 1, size=batch)

        latents = np.empty((batch, self.window, self.latents.shape[1]), dtype=np.float32)
        blocks = np.empty(
            (batch, self.window, self.stride * self.actions.shape[1]), dtype=np.float32
        )
        remaining = np.empty((batch, self.window), dtype=np.float32)
        aux = np.empty((batch, self.window, len(COMPONENTS)), dtype=np.float32)
        goal_latents = np.empty((batch, self.latents.shape[1]), dtype=np.float32)

        for b in range(batch):
            start = int(starts[b])
            goal_index = start + self.window - 1 + int(goal_gap[b])
            goal_latents[b] = self.latents[goal_index]
            for t in range(self.window):
                index = start + t
                latents[b, t] = self.latents[index]
                blocks[b, t] = self.action_block(index)
                remaining[b, t] = min(
                    (goal_index - index) / float(self.goal_span), 1.0
                )
                aux[b, t] = self.component_match(index, goal_index)
        return {
            "latents": latents,
            "blocks": blocks,
            "remaining": remaining,
            "aux": aux,
            "goal_latents": goal_latents,
        }


def to_device(batch: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def compute_losses(head: ProgressHead, batch, args) -> dict[str, torch.Tensor]:
    hiddens, progresses, _final = head.scan(batch["latents"], batch["blocks"])
    steps = hiddens.shape[1]
    goal = batch["goal_latents"].unsqueeze(1).expand(-1, steps, -1)
    values = head.readout(
        torch.cat([hiddens, progresses, goal], dim=-1)
    ).squeeze(-1)

    losses = {"value": expectile_loss(values, batch["remaining"], args.expectile)}
    losses["mae"] = (values - batch["remaining"]).abs().mean().detach()
    if args.arm == "prog_sup":
        logits = head.aux(progresses)
        losses["aux"] = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["aux"]
        )
        losses["loss"] = losses["value"] + args.aux_weight * losses["aux"]
    else:
        losses["loss"] = losses["value"]
    return losses


def lr_at(step: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("progress-head training must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("progress-head training requires a GPU allocation")

    seed_everything(args.seed)
    device = torch.device("cuda")

    train = LatentSequences(
        args.cache_dir, args.latents_name, args.window_blocks, args.goal_blocks
    )
    val = LatentSequences(
        args.val_cache_dir, args.latents_name, args.window_blocks, args.goal_blocks
    )
    lewm = LeWMConfig()
    config = ProgressConfig(
        latent_dim=int(train.latents.shape[1]),
        action_input_dim=lewm.action_input_dim,
        **ARMS[args.arm],
    )
    head = ProgressHead(config).to(device)
    params = int(sum(p.numel() for p in head.parameters()))
    print(f"arm {args.arm}  parameters {params}  windows {train.starts.size}", flush=True)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed + 991)
    val_rng_seed = args.seed + 7717

    history = []
    best = {"step": -1, "val_loss": float("inf")}
    started = time.time()

    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        batch = to_device(train.sample(rng, args.batch_size), device)
        losses = compute_losses(head, batch, args)
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_every == 0:
            rate = (step + 1) / max(time.time() - started, 1e-6)
            print(
                f"step {step + 1}/{args.steps} loss {float(losses['loss']):.5f} "
                f"value {float(losses['value']):.5f} mae {float(losses['mae']):.4f} "
                f"{rate:.2f} steps/s",
                flush=True,
            )

        if (step + 1) % args.val_every == 0 or (step + 1) == args.steps:
            head.eval()
            val_rng = np.random.default_rng(val_rng_seed)
            totals = {"loss": 0.0, "value": 0.0, "mae": 0.0}
            with torch.no_grad():
                for _ in range(args.val_batches):
                    vb = to_device(val.sample(val_rng, args.batch_size), device)
                    vl = compute_losses(head, vb, args)
                    for key in totals:
                        totals[key] += float(vl[key])
            head.train()
            metrics = {k: v / args.val_batches for k, v in totals.items()}
            metrics["step"] = step + 1
            history.append(metrics)
            print(json.dumps({"val": metrics}, sort_keys=True), flush=True)
            if metrics["loss"] < best["val_loss"]:
                best = {"step": step + 1, "val_loss": metrics["loss"], "val_mae": metrics["mae"]}
                args.out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "protocol": PROTOCOL,
                        "arm": args.arm,
                        "config": config.to_json(),
                        "seed": args.seed,
                        "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                        "metadata": {"selected_at_step": step + 1, "val": metrics},
                    },
                    args.out_dir / "progress_head.pt",
                )

    summary = {
        "protocol": PROTOCOL,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "arm": args.arm,
        "config": config.to_json(),
        "seed": args.seed,
        "parameters": params,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "window_blocks": args.window_blocks,
        "goal_blocks": args.goal_blocks,
        "expectile": args.expectile,
        "aux_weight": args.aux_weight if args.arm == "prog_sup" else None,
        "num_train_windows": int(train.starts.size),
        "num_val_windows": int(val.starts.size),
        "train_cache": str(args.cache_dir),
        "val_cache": str(args.val_cache_dir),
        "best": best,
        "val_history": history,
        "wall_seconds": time.time() - started,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"arm": args.arm, "best": best}, sort_keys=True))


if __name__ == "__main__":
    main()
