"""Bounded CPU-only construction/audit of label-free targets from cached features.

No training, no value/probe labels, no simulator, no encoder loading, no test episodes.
The fixed random projection is a plumbing choice, NOT a claim of representation adequacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
import traceback
from pathlib import Path

import torch

from . import TARGET_SCHEMA, TEMPORAL_CONVENTION
from .signatures import chen, flatten, integrated_lift, interval, signature


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def select_episodes(manifest, config):
    rng = random.Random(config["seed"])
    chosen = []
    for split, key in (("train", "max_train_episodes"), ("val", "max_val_episodes")):
        pool = sorted((m for m in manifest["episodes"] if m["split"] == split), key=lambda m: m["episode"])
        if config[key] < 1 or len(pool) < config[key]:
            raise ValueError(f"need {config[key]} {split} episodes")
        chosen.extend(sorted(rng.sample(pool, config[key]), key=lambda m: m["episode"]))
    return chosen


def run(config, out, result):
    began = time.monotonic()
    def deadline():
        if time.monotonic() - began > config["max_seconds"]:
            raise TimeoutError("bounded step-2 CPU budget exceeded")

    root = Path(config["feature_root"])
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if not math.isclose(1. / config["dt_seconds"], manifest["encoder"]["frame_rate_hz"]):
        raise ValueError("feature cadence does not match lift dt")
    chosen = select_episodes(manifest, config)
    result.update({"selected_episodes": [{"episode": m["episode"], "split": m["split"]} for m in chosen],
                   "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                   "encoder": manifest["encoder"], "test_episodes_loaded": 0,
                   "labels_loaded": False, "model_checkpoints_loaded": False})
    write_json(out / "result.json", result)
    gen = torch.Generator().manual_seed(config["seed"])
    projection = None
    projected = []
    for i, entry in enumerate(chosen):
        deadline()
        p = torch.load(root / entry["path"], map_location="cpu", weights_only=True, mmap=True)
        visual, proprio = p["visual"], p["proprio"]
        if len(visual) != entry["frames"] or len(proprio) != len(visual):
            raise ValueError("unaligned feature file")
        if not torch.equal(p["frame_index"], torch.arange(len(visual))):
            raise ValueError("targets require consecutive original-cadence frames")
        flat = visual.flatten(1).float()
        if projection is None:
            projection = {
                "visual": torch.randn(flat.shape[-1], config["visual_channels"], generator=gen) / math.sqrt(flat.shape[-1]),
                "proprio": torch.randn(proprio.shape[-1], config["proprio_channels"], generator=gen) / math.sqrt(proprio.shape[-1]),
            }
        x = torch.cat([flat @ projection["visual"], proprio.float() @ projection["proprio"]], -1).double()
        if not bool(torch.isfinite(x).all()):
            raise ValueError("non-finite projected features")
        projected.append((entry, x))
        if (i + 1) % 8 == 0:
            print(f"projected {i+1}/{len(chosen)} cached episodes (CPU)", flush=True)
        del p, flat, visual, proprio

    # Fixed, train-only affine normalization. No per-segment centering or normalization.
    train_x = torch.cat([x for m, x in projected if m["split"] == "train"])
    mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)
    torch.save({"weights": projection, "mean": mean, "std": std, "schema": TARGET_SCHEMA,
                "config": config}, out / "projection.pt")
    degree = config["degree"]
    summaries, audits, train_targets = [], [], []
    tolerance = {"atol": config["algebra_atol"], "rtol": config["algebra_rtol"]}

    for i, (entry, raw) in enumerate(projected):
        deadline()
        x = (raw - mean) / std
        path = integrated_lift(x, config["dt_seconds"])
        prefix = signature(path, degree, stream=True)
        starts, lengths, targets = [], [], []
        max_error = 0.
        checks = 0
        for length in config["lengths"]:
            for start in range(0, len(x) - length, config["window_stride"]):
                end = start + length
                whole = interval(prefix, start, end)
                starts.append(start)
                lengths.append(length)
                targets.append(flatten(whole))
                # Split audits on actual windows. A deliberately non-aligned split
                # prevents the tests from only exercising half/stride boundaries.
                middle = start + max(1, length // 3)
                merged = chen(interval(prefix, start, middle), interval(prefix, middle, end))
                for a, b in zip(whole, merged):
                    torch.testing.assert_close(a, b, **tolerance)
                    max_error = max(max_error, float((a - b).abs().max()))
                checks += 1
                if start == 0 or start == config["window_stride"]:
                    # Independent direct construction: checks prefix-inversion too.
                    direct = signature(integrated_lift(x[start:end+1], config["dt_seconds"]), degree)
                    for a, b in zip(whole, direct):
                        torch.testing.assert_close(a, b, **tolerance)
                        max_error = max(max_error, float((a - b).abs().max()))
                    checks += 1
        if not targets:
            raise ValueError("selected episode has no eligible windows")
        targets = torch.stack(targets)
        if not bool(torch.isfinite(targets).all()):
            raise ValueError("non-finite targets")
        # Store raw algebra coordinates, NOT loss-normalized tensors.
        path_out = out / "targets" / f"episode_{entry['episode']:06d}.pt"
        torch.save({"schema": TARGET_SCHEMA, "temporal_convention": TEMPORAL_CONVENTION,
                    "episode": entry["episode"], "split": entry["split"], "degree": degree,
                    "projected_observations": x, "start": torch.tensor(starts),
                    "length": torch.tensor(lengths), "signature": targets}, path_out)
        if entry["split"] == "train":
            train_targets.append(targets)
        summaries.append({"episode": entry["episode"], "split": entry["split"],
                          "frames": len(x), "windows": len(starts), "path": str(path_out.relative_to(out))})
        audits.append({"episode": entry["episode"], "checks": checks, "max_absolute_error": max_error})
        if (i + 1) % 8 == 0:
            print(f"built/audited {i+1}/{len(projected)} episodes", flush=True)

    train_targets = torch.cat(train_targets)
    # Optional loss preprocessing only: invert this transform before Chen products.
    tm = train_targets.mean(0)
    ts = train_targets.std(0, unbiased=False).clamp_min(1e-6)
    torch.save({"mean": tm, "std": ts, "fit_split": "train", "raw_product_required": True}, out / "target_loss_stats.pt")
    write_json(out / "manifest.json", {"schema": TARGET_SCHEMA, "temporal_convention": TEMPORAL_CONVENTION,
               "config": config, "episodes": summaries, "development_only": True,
               "normalization": "train-only; raw signatures stored; level zero omitted"})
    result.update({"verdict": "STEP2_PLUMBING_PASS_NOT_METHOD_EVIDENCE", "episodes": len(summaries),
                   "windows": sum(m["windows"] for m in summaries), "target_dimension": int(train_targets.shape[-1]),
                   "audits": audits, "max_absolute_composition_error": max(a["max_absolute_error"] for a in audits),
                   "elapsed_seconds": time.monotonic() - began,
                   "interpretation": "Causality/algebra target infrastructure only; no adequacy, headroom, ranking or control result."})
    write_json(out / "result.json", result)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("feature processing must run through sbatch on a compute node")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "targets").mkdir(exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("use a fresh run directory; do not overwrite a prior audit")
    result = {"verdict": "RUNNING", "job_id": os.environ["SLURM_JOB_ID"], "node": platform.node(),
              "schema": TARGET_SCHEMA, "temporal_convention": TEMPORAL_CONVENTION, "config": config}
    write_json(args.output / "result.json", result)
    try:
        run(config, args.output, result)
    except Exception:
        result.update({"verdict": "FAILED", "error": traceback.format_exc()})
        write_json(args.output / "result.json", result)
        raise


if __name__ == "__main__":
    main()
