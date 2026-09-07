"""Encode every cached frame once with the frozen world model.

The progress head trains on the world model's latents, not on pixels, and every arm of
Stage 3 trains on the same ones. Encoding the grid once (200200 frames -> 154 MB at
192 floats each) turns each arm's training into a cheap read, and guarantees that no arm
sees a different view of the data than any other.

The encoder is the trained LeWM: ``encoder -> CLS -> projector``, exactly the path the
planner's rollout uses for its context frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.scene_lewm import (  # noqa: E402
    LeWMConfig,
    build_lewm,
    image_stats,
)

PROTOCOL = "scene_progress_wm_latents_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-name", type=str, default="latents.npy")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("latent encoding must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("latent encoding requires a GPU allocation")

    meta = json.loads((args.cache_dir / "meta.json").read_text())
    complete = json.loads((args.cache_dir / "cache_complete.json").read_text())
    if not complete["complete"]:
        raise RuntimeError("cache is incomplete")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = LeWMConfig(**payload["config"])
    if config.img_size != int(meta["render_size"]):
        raise RuntimeError("checkpoint image size disagrees with the cache")

    device = torch.device("cuda")
    model = build_lewm(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    pixels = np.load(args.cache_dir / "pixels.npy", mmap_mode="r")
    stride = int(meta["pixel_stride"])
    total = int(meta["num_rows"])
    grid = np.arange(0, total, stride, dtype=np.int64)

    out = np.lib.format.open_memmap(
        args.cache_dir / args.out_name,
        mode="w+",
        dtype=np.float32,
        shape=(grid.size, config.embed_dim),
    )
    mean, std = image_stats(device)
    started = time.time()

    with torch.inference_mode():
        for start in range(0, grid.size, args.batch_size):
            rows = grid[start : start + args.batch_size]
            frames = torch.from_numpy(np.asarray(pixels[rows])).to(device)
            # (B, H, W, 3) -> (B, 1, 3, H, W), the shape LeWM.encode expects
            batch = frames.permute(0, 3, 1, 2).unsqueeze(1).float().div_(255.0)
            batch = (batch - mean) / std
            emb = model.encode({"pixels": batch})["emb"][:, 0]
            out[start : start + rows.size] = emb.float().cpu().numpy()
            if (start + rows.size) % (args.batch_size * 40) == 0 or start + rows.size >= grid.size:
                done = start + rows.size
                rate = done / max(time.time() - started, 1e-6)
                print(f"encoded {done}/{grid.size} ({rate:.0f} rows/s)", flush=True)
    out.flush()

    summary = {
        "protocol": PROTOCOL,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "pixel_stride": stride,
        "num_latents": int(grid.size),
        "latent_dim": int(config.embed_dim),
        "out_name": args.out_name,
        "wall_seconds": time.time() - started,
    }
    (args.cache_dir / f"{Path(args.out_name).stem}_meta.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
