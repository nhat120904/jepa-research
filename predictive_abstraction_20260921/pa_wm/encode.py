"""Frozen spatial feature encoding. Compute-node/GPU only."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from .runtime import require_slurm


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.dataset_root is not None:
        config["dataset_root"] = str(args.dataset_root)
    if "test" in config["splits"]:
        raise ValueError("Development encoder must not read the sealed test split")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "features").mkdir()

    import numpy as np
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen ViT-L encoding requires the requested GPU")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = torch.hub.load(config["model_source"], config["model_name"], source="local",
                           weights=str(checkpoint)).eval().cuda()
    model.requires_grad_(False)

    dataset_root = Path(config["dataset_root"])
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    selected = [record for record in manifest["episodes"] if record["split"] in config["splits"]]
    expected = {split: sum(r["split"] == split for r in selected) for split in config["splits"]}
    if not selected or any(count == 0 for count in expected.values()):
        raise RuntimeError(f"Missing requested split: {expected}")
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    output_records = []
    auxiliary_cache = {}
    feature_shape = None
    for episode_index, record in enumerate(selected):
        source = dataset_root / record["path"]
        if sha256(source) != record["sha256"]:
            raise RuntimeError(f"Dataset hash mismatch: {source}")
        with np.load(source) as data:
            rgb = data["rgb"]
            actions = data["actions"]
            anchors = data["anchors"] if "anchors" in data else rgb[:0]
            history = data["history_rgb"] if "history_rgb" in data else rgb[:0]
        if len(rgb) != len(actions) + 1 or rgb.dtype != np.uint8:
            raise RuntimeError(f"Bad temporal/RGB contract: {source}")
        chunks = []
        all_rgb = np.concatenate([rgb, anchors, history])
        for begin in range(0, len(all_rgb), config["batch_size"]):
            batch = torch.from_numpy(all_rgb[begin:begin + config["batch_size"]]).cuda()
            batch = batch.permute(0, 3, 1, 2).float().div_(255)
            batch = F.interpolate(batch, (config["image_size"], config["image_size"]),
                                  mode="bicubic", align_corners=False, antialias=True)
            batch = (batch - mean) / std
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                patches = model.forward_features(batch)["x_norm_patchtokens"]
                side = int(round(patches.shape[1] ** 0.5))
                if side * side != patches.shape[1]:
                    raise RuntimeError(f"Non-square patch grid: {patches.shape}")
                grid = patches.transpose(1, 2).reshape(len(batch), patches.shape[2], side, side)
                pooled = F.adaptive_avg_pool2d(grid, config["pool_grid"])
                pooled = pooled.flatten(2).transpose(1, 2).half().cpu().numpy()
            chunks.append(pooled)
        all_features = np.concatenate(chunks)
        features = all_features[:len(rgb)]
        if not np.isfinite(all_features).all():
            raise RuntimeError(f"Nonfinite features: {source}")
        feature_shape = list(features.shape[1:])
        relative = Path("features") / f"{record['id']}.npz"
        destination = args.run_dir / relative
        extras = {}
        if len(anchors):
            extras["anchor_features"] = all_features[len(rgb):len(rgb) + len(anchors)]
            extras["history_features"] = all_features[len(rgb) + len(anchors):]
            # Canonical shared features avoid mixed-precision batch-size variation.
            key = record["prefix_id"]
            digest = hashlib.sha256(anchors.tobytes() + history.tobytes()).hexdigest()
            if key in auxiliary_cache:
                old_digest, extras = auxiliary_cache[key]
                if old_digest != digest:
                    raise AssertionError("A prefix changed its anchors/history")
            else:
                auxiliary_cache[key] = (digest, extras)
        np.savez_compressed(destination, features=features.astype(np.float16), actions=actions,
                            **extras)
        output_records.append({**record,
                               "source_sha256": record["sha256"],
                               "source": record["path"], "path": str(relative),
                               "frames": len(features), "feature_shape": feature_shape,
                               "sha256": sha256(destination)})
        print(f"encoded {episode_index + 1}/{len(selected)} {record['id']} {features.shape}",
              flush=True)

    # Self-check a deterministic repeated batch after all encoding.
    with np.load(dataset_root / selected[0]["path"]) as data:
        check_rgb = data["rgb"][:2]
    def encode_check(images):
        batch = torch.from_numpy(images).cuda().permute(0, 3, 1, 2).float().div_(255)
        batch = F.interpolate(batch, (config["image_size"], config["image_size"]),
                              mode="bicubic", align_corners=False, antialias=True)
        with torch.inference_mode():
            return model.forward_features((batch - mean) / std)["x_norm_patchtokens"].cpu()
    first, second = encode_check(check_rgb), encode_check(check_rgb)
    if not torch.equal(first, second):
        raise AssertionError("Frozen encoder is not bitwise deterministic on repeated input")
    atomic_json(args.run_dir / "manifest.json", {
        "schema_version": 1, "status": "COMPLETED", "job_id": os.environ["SLURM_JOB_ID"],
        "dataset_root": str(dataset_root), "dataset_kind": manifest.get("kind", "episodes"),
        "model_name": config["model_name"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "model_source": config["model_source"],
        "image_size": config["image_size"], "pool_grid": config["pool_grid"],
        "feature_shape": feature_shape, "dtype": config["dtype"],
        "splits": config["splits"], "split_counts": expected,
        "test_split_read": False, "records": output_records,
        "temporal_contract": "features[t] -- actions[t] --> features[t+1]",
    })
    print("COMPLETED frozen encoding; test split not read.", flush=True)


if __name__ == "__main__":
    main()
