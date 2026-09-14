#!/usr/bin/env python3
"""Encode complete RoboCasa LeRobot videos with a pinned frozen DINOv2 backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as parquet
import torch
from transformers import AutoImageProcessor, AutoModel


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def stable_split(seed: int, task: str, episode: int, ratios: list[float]) -> str:
    digest = hashlib.sha256(f"{seed}:{task}:{episode}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def video_file(root: Path, info: dict, episode: int, key: str) -> Path:
    relative = info["video_path"].format(
        episode_chunk=episode // info["chunks_size"],
        video_key=key,
        episode_index=episode,
    )
    return root / relative


def data_file(root: Path, info: dict, episode: int) -> Path:
    relative = info["data_path"].format(
        episode_chunk=episode // info["chunks_size"], episode_index=episode
    )
    return root / relative


def decode_rgb(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


@torch.inference_mode()
def encode_frames(frames, processor, model, batch_size: int, device: torch.device,
                  feature_mode: str = "cls", patch_grid_size: int = 2) -> torch.Tensor:
    chunks = []
    for start in range(0, len(frames), batch_size):
        inputs = processor(images=frames[start : start + batch_size], return_tensors="pt")
        pixels = inputs["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = model(pixel_values=pixels).last_hidden_state
            if feature_mode == "cls":
                output = tokens[:, 0]
            elif feature_mode == "patch_grid":
                patches = tokens[:, 1:]
                side = int(patches.shape[1] ** 0.5)
                if side * side != patches.shape[1]:
                    raise ValueError("Expected a square DINOv2 patch grid without register tokens")
                spatial = patches.transpose(1, 2).reshape(len(patches), -1, side, side)
                output = torch.nn.functional.adaptive_avg_pool2d(spatial, patch_grid_size).flatten(1)
            else:
                raise ValueError(f"Unknown feature_mode: {feature_mode}")
        chunks.append(output.float().cpu())
    return torch.cat(chunks).half()


def tensor_column(table, name: str, dtype: np.dtype) -> torch.Tensor:
    return torch.from_numpy(np.asarray(table[name].combine_chunks().to_pylist(), dtype=dtype))


def assert_complete(task_cfg: dict, data_cfg: dict) -> tuple[dict, list[dict]]:
    root = Path(task_cfg["root"])
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    expected = task_cfg["expected_episodes"]
    if info["total_episodes"] != expected or len(episodes) != expected:
        raise RuntimeError(
            f"{task_cfg['name']} metadata has {len(episodes)}/{info['total_episodes']}, expected {expected}"
        )
    missing = []
    for episode in range(expected):
        paths = [data_file(root, info, episode)]
        paths.extend(video_file(root, info, episode, key) for key in data_cfg["camera_keys"])
        missing.extend(str(path) for path in paths if not path.is_file())
    if missing:
        preview = missing[:8]
        raise RuntimeError(
            f"{task_cfg['name']} is incomplete: {len(missing)} parquet/video files missing; first={preview}"
        )
    return info, episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("DINO encoding must run inside an sbatch job")
    if not torch.cuda.is_available():
        raise RuntimeError("DINO encoding requires a CUDA batch node")

    config = json.loads(args.config.read_text())
    data_cfg = config["data"]
    encoder_cfg = config["encoder"]
    task_cfg = data_cfg["tasks"][args.task_index]
    task = task_cfg["name"]
    result_path = args.output_dir / "encode_result.json"
    result = {
        "verdict": "ENCODING_RUNNING",
        "task": task,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "node": platform.node(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_path, result)
    try:
        info, episodes = assert_complete(task_cfg, data_cfg)
        device = torch.device("cuda")
        processor = AutoImageProcessor.from_pretrained(
            encoder_cfg["model_id"], revision=encoder_cfg["revision"]
        )
        model = AutoModel.from_pretrained(
            encoder_cfg["model_id"], revision=encoder_cfg["revision"]
        ).eval().to(device)
        task_root = args.output_dir / task
        task_root.mkdir(parents=True, exist_ok=True)
        entries = []
        source_root = Path(task_cfg["root"])
        for episode_row in episodes:
            episode = int(episode_row["episode_index"])
            output_path = task_root / f"episode_{episode:06d}.pt"
            table = parquet.read_table(data_file(source_root, info, episode))
            actions = tensor_column(table, "action", np.float32)
            proprio = tensor_column(table, "observation.state", np.float32)
            rewards = tensor_column(table, "next.reward", np.float32).reshape(-1)
            frames = len(actions)
            if len(proprio) != frames or len(rewards) != frames:
                raise RuntimeError(f"Column length mismatch in {task} episode {episode}")
            visual = torch.empty(
                frames,
                len(data_cfg["camera_keys"]),
                encoder_cfg["feature_dim"],
                dtype=torch.float16,
            )
            for camera_index, key in enumerate(data_cfg["camera_keys"]):
                decoded = decode_rgb(video_file(source_root, info, episode, key))
                if len(decoded) != frames:
                    raise RuntimeError(
                        f"{task} episode {episode} {key}: {len(decoded)} frames, expected {frames}"
                    )
                features = encode_frames(
                    decoded, processor, model, encoder_cfg["batch_frames"], device,
                    encoder_cfg.get("feature_mode", "cls"), encoder_cfg.get("patch_grid_size", 2)
                )
                if features.shape != (frames, encoder_cfg["feature_dim"]):
                    raise RuntimeError(f"Unexpected DINO output {tuple(features.shape)}")
                visual[:, camera_index] = features
            eventual = torch.from_numpy(
                np.maximum.accumulate(rewards.numpy()[::-1])[::-1].copy() > 0
            )
            torch.save(
                {
                    "visual_features": visual,
                    "proprio": proprio,
                    "actions": actions,
                    "eventual_success": eventual,
                },
                output_path,
            )
            split = stable_split(
                data_cfg["split_seed"], task, episode, data_cfg["split_ratios"]
            )
            entries.append(
                {
                    "episode_uid": f"offline:{task}:{episode}",
                    "episode_id": args.task_index * 1_000_000 + episode,
                    "task_id": task_cfg["task_id"],
                    "task": task,
                    "split": split,
                    "num_frames": frames,
                    "path": str(output_path.relative_to(args.output_dir)),
                    "source": "offline_demonstration",
                    "allow_training": True,
                    "has_progress": False,
                    "positive_frames": int(eventual.sum()),
                    "negative_frames": int((~eventual).sum()),
                }
            )
            write_json(
                args.output_dir / f"manifest_{task}.json",
                {
                    "schema_version": 1,
                    "encoder": encoder_cfg,
                    "episodes": entries,
                },
            )
            print(f"{task} encoded {episode + 1}/{len(episodes)}", flush=True)
        result.update(
            {
                "verdict": "ENCODING_COMPLETE",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "episodes": len(entries),
                "manifest": str(args.output_dir / f"manifest_{task}.json"),
            }
        )
        write_json(result_path, result)
    except Exception as error:
        result["verdict"] = "ENCODING_ERROR"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
