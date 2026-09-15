#!/usr/bin/env python3
"""Encode all ScrubCuttingBoard demonstrations for the offline composition pilot.

Frozen DINOv2-S (pinned revision). Each 256x256 frame is resized to 224x224 without
cropping (the processor's centre crop would drop 16 px per border), normalized with
ImageNet statistics, and its 16x16 patch tokens are average-pooled to 4x4. All frames are
kept (20 Hz); downsampling can happen later without re-encoding.

Per episode output (fp16 features): visual [T, 3, 16, 384], proprio [T, 16],
actions [T, 12], frame_index [T]. Splits are assigned by episode before any window exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as parquet
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def stable_split(seed: int, episode: int, ratios: list[float]) -> str:
    digest = hashlib.sha256(f"{seed}:ScrubCuttingBoard:{episode}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


class EpisodeVideos(torch.utils.data.Dataset):
    def __init__(self, root: Path, episodes: list[int], cameras: list[str]) -> None:
        self.root, self.episodes, self.cameras = root, episodes, cameras

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict:
        episode = self.episodes[index]
        table = parquet.read_table(self.root / f"data/chunk-000/episode_{episode:06d}.parquet")
        actions = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
        proprio = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        frames = []
        for camera in self.cameras:
            path = self.root / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
            with av.open(str(path)) as container:
                frames.append(np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)]))
        lengths = {len(actions), len(proprio), *(len(f) for f in frames)}
        if len(lengths) != 1:
            raise RuntimeError(f"episode {episode}: length mismatch {lengths}")
        return {
            "episode": episode,
            "video": torch.from_numpy(np.stack(frames)),  # [C, T, H, W, 3] uint8
            "actions": torch.from_numpy(actions),
            "proprio": torch.from_numpy(proprio),
        }


@torch.inference_mode()
def encode_camera(frames: torch.Tensor, model, batch: int, grid: int, device: torch.device) -> torch.Tensor:
    out = []
    for start in range(0, len(frames), batch):
        x = frames[start : start + batch].to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        x = (x - MEAN.to(device)) / STD.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = model(pixel_values=x).last_hidden_state[:, 1:]
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise RuntimeError("expected a square patch grid without register tokens")
        spatial = tokens.float().transpose(1, 2).reshape(len(tokens), -1, side, side)
        pooled = F.adaptive_avg_pool2d(spatial, grid).flatten(2).transpose(1, 2)  # [B, grid*grid, D]
        out.append(pooled.half().cpu())
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("encoding must run inside an sbatch job")
    if not torch.cuda.is_available():
        raise RuntimeError("encoding requires a CUDA node")
    from transformers import AutoModel

    torch.multiprocessing.set_sharing_strategy("file_system")  # large uint8 videos; avoid /dev/shm limits
    config = json.loads(args.config.read_text())
    data_cfg, enc_cfg = config["data"], config["encoder"]
    root = Path(data_cfg["dataset_root"])
    out_dir = Path(data_cfg["feature_root"])
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "encode_result.json"
    result = {
        "verdict": "ENCODING_RUNNING",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "node": platform.node(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": enc_cfg,
    }
    write_json(result_path, result)
    try:
        episodes = sorted(
            json.loads(line)["episode_index"]
            for line in (root / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        )
        todo = [e for e in episodes if not (out_dir / "episodes" / f"episode_{e:06d}.pt").is_file()]
        device = torch.device("cuda")
        model = AutoModel.from_pretrained(enc_cfg["model_id"], revision=enc_cfg["revision"]).eval().to(device)
        loader = torch.utils.data.DataLoader(
            EpisodeVideos(root, todo, data_cfg["cameras"]),
            batch_size=None,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 2,
            prefetch_factor=2,
        )
        start = time.time()
        frames_done = 0
        for index, item in enumerate(loader):
            episode = item["episode"]
            visual = torch.stack(
                [encode_camera(item["video"][c], model, enc_cfg["batch_frames"], enc_cfg["patch_grid"], device) for c in range(len(data_cfg["cameras"]))],
                dim=1,
            )  # [T, 3, grid*grid, D]
            path = out_dir / "episodes" / f"episode_{episode:06d}.pt"
            tmp = path.with_suffix(".tmp")
            torch.save(
                {
                    "visual": visual,
                    "proprio": item["proprio"],
                    "actions": item["actions"],
                    "frame_index": torch.arange(len(visual)),
                },
                tmp,
            )
            tmp.replace(path)
            frames_done += len(visual)
            if index % 25 == 0:
                print(f"encoded {index + 1}/{len(todo)} episodes, {frames_done / (time.time() - start):.0f} frames/s", flush=True)

        manifest = []
        for episode in episodes:
            path = out_dir / "episodes" / f"episode_{episode:06d}.pt"
            meta = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            manifest.append(
                {
                    "episode": episode,
                    "split": stable_split(data_cfg["split_seed"], episode, data_cfg["split_ratios"]),
                    "frames": int(len(meta["visual"])),
                    "path": str(path.relative_to(out_dir)),
                }
            )
        write_json(out_dir / "manifest.json", {"schema_version": 1, "encoder": enc_cfg, "data": data_cfg, "episodes": manifest})
        counts = {s: sum(m["split"] == s for m in manifest) for s in ("train", "val", "test")}
        result.update(
            {
                "verdict": "ENCODING_COMPLETE",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "episodes": len(manifest),
                "frames": sum(m["frames"] for m in manifest),
                "split_counts": counts,
                "encode_seconds": time.time() - start,
            }
        )
        write_json(result_path, result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        result.update({"verdict": "ENCODING_ERROR", "error": repr(error), "traceback": traceback.format_exc()})
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
