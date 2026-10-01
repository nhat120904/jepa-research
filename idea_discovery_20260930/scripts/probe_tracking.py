#!/usr/bin/env python3
"""Exploratory, frozen-model TAP-Vid-DAVIS tracking pilot; run via Slurm only.

The sparse arms consume identical observed RGB frames. Query-frame observations
are added to the fixed stride schedule so query prototypes come from an observed
image. Every prototype is fixed when its first-visible benchmark query arrives.
No later positions or visibility labels enter the tracking adapter.

The 16x16 prototype reader is deliberately coarse and has no visibility head.
Its predictions are always visible after the query. It is an exploratory adapter,
not a comparison of equally capable trackers or a claim about tracking SOTA.

Official APIs inspected when writing this source:
  https://github.com/facebookresearch/co-tracker/blob/main/cotracker/predictor.py
  https://github.com/facebookresearch/co-tracker/blob/main/cotracker/datasets/tap_vid_datasets.py
  https://github.com/facebookresearch/co-tracker/blob/main/cotracker/evaluation/core/eval_utils.py

This file does not download assets, train models, or submit jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from common_model import FrozenVJEPA


ARMS = ("repeat_features", "predicted_features", "copy_last_position")
SEARCH_RADIUS_PATCHES = 4.0
MAX_CONTEXT_ANCHORS = 7
MAX_TIMELINE_TUBELETS = 32
IMAGE_SIZE = 256
PATCH_SIZE = 16


class RuntimeBudgetExceeded(RuntimeError):
    pass


def check_deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise RuntimeBudgetExceeded("Pilot runtime budget reached; partial videos are unscored.")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(function: Callable[[], Any]) -> tuple[Any, float]:
    synchronize()
    started = time.perf_counter()
    result = function()
    synchronize()
    return result, time.perf_counter() - started


def git_revision(source: Path) -> str:
    # rsync intentionally omits upstream .git. Do not accidentally report the
    # enclosing research repository's revision as an upstream source revision.
    declared = source / ".upstream-revision"
    if declared.is_file():
        return declared.read_text().strip()
    try:
        checkout_root = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        ).strip()
        if Path(checkout_root).resolve() != source.resolve():
            return "unknown"
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_counters(model: Any) -> dict[str, int | float]:
    counters = getattr(model, "counters", None)
    values = counters() if callable(counters) else getattr(model, "stats", {})
    if not isinstance(values, dict):
        return {}
    return {key: value for key, value in values.items() if isinstance(value, (int, float))}


def find_davis_pickle(data_root: Path) -> Path:
    directory = data_root / "tapvid_davis"
    candidates = sorted(directory.glob("**/*.pkl")) if directory.exists() else []
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one official DAVIS pickle under {directory}; found {candidates}"
        )
    return candidates[0]


def official_modules(source: Path) -> tuple[Any, Any, Any]:
    if not (source / "cotracker" / "predictor.py").is_file():
        raise FileNotFoundError(f"Not a CoTracker source checkout: {source}")
    sys.path.insert(0, str(source.resolve()))
    predictor = importlib.import_module("cotracker.predictor")
    loader = importlib.import_module("cotracker.datasets.tap_vid_datasets")
    metrics = importlib.import_module("cotracker.evaluation.core.eval_utils")
    return predictor.CoTrackerPredictor, loader, metrics.compute_tapvid_metrics


def prepare_video(item: dict[str, Any], loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the official loader's resize/query rules without its in-place labels edit."""
    frames = item["video"]
    if isinstance(frames[0], bytes):
        import io
        from PIL import Image

        frames = np.stack(
            [np.asarray(Image.open(io.BytesIO(frame)).convert("RGB")) for frame in frames]
        )
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB video, got {frames.shape}, {frames.dtype}")
    frames = loader.resize_video(frames, (IMAGE_SIZE, IMAGE_SIZE))
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    # Exact current CoTracker TAP-Vid loader convention: normalized 1 maps to 255.
    points = np.array(item["points"], dtype=np.float32, copy=True) * (IMAGE_SIZE - 1)
    occluded = np.asarray(item["occluded"], dtype=bool)
    if not np.any(~occluded):
        raise ValueError("Official video has no visible query points")
    converted = loader.sample_queries_first(occluded, points, frames)
    queries = np.asarray(converted["query_points"][0], dtype=np.float32)
    gt_tracks = np.asarray(converted["target_points"][0], dtype=np.float32)
    gt_occluded = np.asarray(converted["occluded"][0], dtype=bool)
    if not np.all(np.isfinite(queries)):
        raise ValueError("Non-finite benchmark query")
    return frames, queries, gt_tracks, gt_occluded


def normalized_map(features: torch.Tensor, expected_count: int, *, observed: bool = False) -> torch.Tensor:
    expected_shape = (expected_count, 16, 16, 1408)
    if tuple(features.shape) != expected_shape:
        raise ValueError(f"FrozenVJEPA returned {tuple(features.shape)}, expected {expected_shape}")
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite frozen-model features")
    features = features.float()
    if observed:
        # Official V-JEPA2 loss predicts non-affine layer-normalized EMA targets.
        # Match that target space before cosine correspondence; predictions
        # already live in this space and must not be normalized a second time.
        features = F.layer_norm(features, (features.shape[-1],))
    return F.normalize(features, dim=-1)


def read_positions(
    feature_map: torch.Tensor,
    prototypes: torch.Tensor,
    offsets: torch.Tensor,
    previous: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Cosine nearest patch inside the same past-position radius for every arm."""
    similarities = prototypes @ feature_map.reshape(256, 1408).T
    previous_patch = torch.floor(previous / PATCH_SIZE).clamp(0, 15)
    candidate_patch = (centers - PATCH_SIZE / 2) / PATCH_SIZE
    distances = (candidate_patch[None] - previous_patch[:, None]).square().sum(-1)
    allowed = distances <= SEARCH_RADIUS_PATCHES**2
    similarities = similarities.masked_fill(~allowed, -torch.inf)
    best_patch = similarities.argmax(dim=-1)
    return (centers[best_patch] + offsets).clamp(0, IMAGE_SIZE - 1)


@torch.inference_mode()
def sparse_tracks(
    model: FrozenVJEPA,
    frames: np.ndarray,
    queries: np.ndarray,
    stride: int,
    deadline: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Track using query prompts and observed images only, never scoring labels."""
    n_frames = len(frames)
    n_points = len(queries)
    query_times = np.rint(queries[:, 0]).astype(np.int64)
    if np.any(query_times < 0) or np.any(query_times >= n_frames):
        raise ValueError("Query time outside video")
    stride_anchors = set(range(0, n_frames, stride))
    observed_times = sorted(stride_anchors | set(query_times.tolist()))
    observed_set = set(observed_times)
    next_anchor = {
        t: observed_times[index + 1] if index + 1 < len(observed_times) else n_frames
        for index, t in enumerate(observed_times)
    }
    query_xy = torch.as_tensor(queries[:, [2, 1]], dtype=torch.float32, device="cuda")
    prototypes = torch.zeros((n_points, 1408), dtype=torch.float32, device="cuda")
    offsets = torch.zeros((n_points, 2), dtype=torch.float32, device="cuda")
    active = torch.zeros(n_points, dtype=torch.bool, device="cuda")
    positions = {arm: torch.zeros((n_points, 2), device="cuda") for arm in ARMS}
    trajectories = {
        arm: torch.zeros((n_points, n_frames, 2), device="cuda") for arm in ARMS
    }
    y, x = torch.meshgrid(
        torch.arange(16, device="cuda"), torch.arange(16, device="cuda"), indexing="ij"
    )
    centers = torch.stack((x, y), dim=-1).reshape(-1, 2).float() * PATCH_SIZE + PATCH_SIZE / 2
    history: list[int] = []
    latest_map: torch.Tensor | None = None
    forecasts: dict[int, torch.Tensor] = {}
    stats: dict[str, Any] = {
        "observed_frames": len(observed_times),
        "total_frames": n_frames,
        "observed_fraction": len(observed_times) / n_frames,
        "observed_times": observed_times,
        "fixed_stride_frames": len(stride_anchors),
        "extra_query_frames": sorted(set(query_times.tolist()) - stride_anchors),
        "wrapper_encode_calls_shared": 0,
        "wrapper_predict_calls": 0,
        "predicted_frame_maps": 0,
        "encoder_seconds_shared": 0.0,
        "predictor_wrapper_seconds": 0.0,
        "reader_seconds_by_arm": {arm: 0.0 for arm in ARMS},
        "context_anchor_counts": [],
        "causal_observation_check": True,
    }
    counters_before = model_counters(model)
    started = time.perf_counter()
    for t in range(n_frames):
        check_deadline(deadline)
        is_anchor = t in observed_set
        if is_anchor:
            history.append(t)
            target_times = list(range(t + 1, next_anchor[t]))
            # Use the same retained context for encode and prediction. The maximum
            # target index is <32 after normalization; no future RGB is supplied.
            last_target = target_times[-1] if target_times else t
            history = [s for s in history[-MAX_CONTEXT_ANCHORS:] if last_target - s < MAX_TIMELINE_TUBELETS]
            if not history or max(history) > t or last_target - min(history) >= MAX_TIMELINE_TUBELETS:
                raise AssertionError("Invalid causal sparse context")
            context_images = [frames[s] for s in history]
            encoded, elapsed = timed_call(lambda: model.encode_sparse(history, context_images))
            stats["wrapper_encode_calls_shared"] += 1
            stats["encoder_seconds_shared"] += elapsed
            stats["context_anchor_counts"].append(len(history))
            latest_map = normalized_map(encoded, len(history), observed=True)[-1]
            forecasts = {}
            if target_times:
                # Joint masked targets contain positional mask tokens, not future
                # observations. Cache their maps until the next real observation.
                predicted, elapsed = timed_call(
                    lambda: model.predict_sparse(history, context_images, target_times)
                )
                stats["wrapper_predict_calls"] += 1
                stats["predictor_wrapper_seconds"] += elapsed
                stats["predicted_frame_maps"] += len(target_times)
                predicted = normalized_map(predicted, len(target_times))
                forecasts = {target: predicted[index] for index, target in enumerate(target_times)}
        if latest_map is None:
            raise AssertionError("Missing observed map at stream start")
        newly_queried = torch.as_tensor(query_times == t, device="cuda")
        if newly_queried.any():
            if not is_anchor:
                raise AssertionError("Query prototype must use an observed query frame")
            patch_xy = torch.floor(query_xy[newly_queried] / PATCH_SIZE).long().clamp(0, 15)
            prototypes[newly_queried] = latest_map[patch_xy[:, 1], patch_xy[:, 0]]
            query_centers = patch_xy.float() * PATCH_SIZE + PATCH_SIZE / 2
            offsets[newly_queried] = query_xy[newly_queried] - query_centers
            for arm in ARMS:
                positions[arm][newly_queried] = query_xy[newly_queried]
            active |= newly_queried
        continuing = active & ~newly_queried
        if continuing.any():
            for arm in ARMS:
                if arm == "copy_last_position" and not is_anchor:
                    continue
                feature_map = (
                    forecasts[t] if arm == "predicted_features" and not is_anchor else latest_map
                )
                updated, elapsed = timed_call(
                    lambda: read_positions(
                        feature_map,
                        prototypes[continuing],
                        offsets[continuing],
                        positions[arm][continuing],
                        centers,
                    )
                )
                positions[arm][continuing] = updated
                stats["reader_seconds_by_arm"][arm] += elapsed
        for arm in ARMS:
            trajectories[arm][:, t] = positions[arm]
    synchronize()
    stats["sparse_combined_seconds"] = time.perf_counter() - started
    counters_after = model_counters(model)
    stats["actual_frozen_model_counter_deltas"] = {
        key: value - counters_before.get(key, 0) for key, value in counters_after.items()
    }
    # Before-query predictions are ignored by query_mode='first'. No visibility
    # labels or feature thresholds are used to create a visibility prediction.
    predicted_occluded = np.arange(n_frames)[None, :] < query_times[:, None]
    results = {
        arm: (trajectories[arm].cpu().numpy(), predicted_occluded.copy()) for arm in ARMS
    }
    for arm in ARMS:
        stats.setdefault("allocated_wrapper_calls_by_arm", {})[arm] = {
            "encode": stats["wrapper_encode_calls_shared"],
            "predict": stats["wrapper_predict_calls"] if arm == "predicted_features" else 0,
        }
    return results, stats


@torch.inference_mode()
def cotracker_tracks(
    tracker: Any, frames: np.ndarray, queries: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    video = torch.as_tensor(frames, device="cuda").permute(0, 3, 1, 2).float()[None]
    # Official reader query order is t,y,x; CoTracker predictor expects t,x,y.
    query_tensor = torch.as_tensor(queries[:, [0, 2, 1]], device="cuda")[None].float()
    (tracks, visible), seconds = timed_call(
        lambda: tracker(video=video, queries=query_tensor, backward_tracking=False)
    )
    expected = (1, len(frames), len(queries), 2)
    if tuple(tracks.shape) != expected or tuple(visible.shape) != expected[:-1]:
        raise ValueError(f"Unexpected CoTracker shapes: {tracks.shape}, {visible.shape}")
    return (
        tracks[0].permute(1, 0, 2).float().cpu().numpy(),
        (~visible[0].bool()).permute(1, 0).cpu().numpy(),
        seconds,
    )


def score_tracks(
    metrics_function: Callable[..., Any],
    queries: np.ndarray,
    gt_tracks: np.ndarray,
    gt_occluded: np.ndarray,
    predicted_tracks: np.ndarray,
    predicted_occluded: np.ndarray,
) -> dict[str, float]:
    if not np.isfinite(predicted_tracks).all():
        raise ValueError("Non-finite predicted positions")
    metrics = metrics_function(
        query_points=queries[None],
        gt_occluded=gt_occluded[None],
        gt_tracks=gt_tracks[None],
        pred_occluded=predicted_occluded[None],
        pred_tracks=predicted_tracks[None],
        query_mode="first",
    )
    scores = {key: float(np.asarray(value).reshape(-1)[0]) * 100 for key, value in metrics.items()}
    evaluated_visible = (~gt_occluded) & (
        np.arange(gt_tracks.shape[1])[None] > np.rint(queries[:, 0]).astype(int)[:, None]
    )
    distances = np.linalg.norm(predicted_tracks - gt_tracks, axis=-1)
    scores["visible_epe_pixels"] = float(distances[evaluated_visible].mean())
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Official metric has a zero denominator or non-finite result")
    return scores


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    completed = summary["videos"]
    summary["completed_videos"] = len(completed)
    summary["mean_metrics"] = {}
    if completed:
        for arm in completed[0]["metrics"]:
            summary["mean_metrics"][arm] = {
                metric: float(np.mean([entry["metrics"][arm][metric] for entry in completed]))
                for metric in completed[0]["metrics"][arm]
            }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vjepa-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tracker-source", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-runtime-seconds", type=int, default=2400)
    args = parser.parse_args()
    if args.max_videos <= 0 or not 1 <= args.stride < MAX_TIMELINE_TUBELETS:
        parser.error("max-videos must be positive and stride must lie in [1,31]")
    if not 1 <= args.max_runtime_seconds <= 2400:
        parser.error("runtime budget must lie in [1,2400] seconds, leaving headroom in a 45min job")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this pilot in an allocated single-GPU batch job")
    started = time.perf_counter()
    deadline = started + args.max_runtime_seconds
    torch.manual_seed(0)
    np.random.seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite pilot results: {summary_path}")
    prediction_directory = args.out / "predictions"
    prediction_directory.mkdir(exist_ok=True)
    tracker_checkpoint = args.data_root / "scaled_online.pth"
    dataset_path = find_davis_pickle(args.data_root)
    predictor_class, loader, metrics_function = official_modules(args.tracker_source)
    # Trusted public benchmark pickle only; this load occurs on the compute node.
    with dataset_path.open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, dict):
        raise ValueError("Expected official TAP-Vid-DAVIS keyed video dictionary")
    selected = sorted(
        dataset, key=lambda key: (hashlib.sha256(str(key).encode("utf-8")).hexdigest(), str(key))
    )[: args.max_videos]
    summary: dict[str, Any] = {
        "status": "running",
        "protocol": "TAP-Vid-DAVIS first-query; fixed SHA256 key subset; 256x256",
        "selected_keys": [str(key) for key in selected],
        "selection_rule": "ascending SHA256(UTF8(str(video_key))), no label filtering",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sources": {
            "vjepa_git_revision": git_revision(args.vjepa_source),
            "cotracker_git_revision": git_revision(args.tracker_source),
            "vjepa_checkpoint": checkpoint_metadata(args.checkpoint),
            "cotracker_checkpoint": checkpoint_metadata(tracker_checkpoint),
            "dataset": str(dataset_path.resolve()),
        },
        "limitations": [
            "Exploratory frozen-model adapter; no novelty or SOTA claim.",
            "Sparse reader uses a 16x16 grid, fixed query prototype/patch offset and a radius4 patch prior.",
            "Sparse arms are always visible after the query; no learned or fitted occlusion head.",
            "Copy-last-position uses the same reader at observed anchors and holds positions on skipped frames.",
            "Query-frame images augment stride anchors equally for all sparse arms.",
            "CoTracker online checkpoint uses full-video forward sliding windows; not strictly per-frame causal.",
            "CoTracker is a full-observation reference with a different reader and compute budget.",
            "Predictor target tokens contain positions only; recursive latent rollout is not used.",
            "Wrapper encode-call counts do not establish matched encoder FLOPs if predict_sparse re-encodes context.",
            "Metrics are mean across complete selected videos; a partial pilot is not the full benchmark.",
        ],
        "metric_units": "TAP-Vid metrics percent; visible_epe_pixels pixels; timing seconds",
        "feature_space": "Observed EMA maps use non-affine layer_norm to match official predictor targets; both maps L2-normalized for cosine reader",
        "videos": [],
    }
    write_summary(summary_path, summary)
    model = None
    try:
        model, model_load_seconds = timed_call(
            lambda: FrozenVJEPA(source=args.vjepa_source, checkpoint=args.checkpoint, device="cuda")
        )
        tracker, tracker_load_seconds = timed_call(
            lambda: predictor_class(
                checkpoint=str(tracker_checkpoint), offline=False, v2=False, window_len=16
            ).eval().cuda()
        )
        summary["load_seconds"] = {"vjepa": model_load_seconds, "cotracker": tracker_load_seconds}
        summary["frozen_model_load_report"] = getattr(model, "load_report", None)
        for key in selected:
            check_deadline(deadline)
            frames, queries, gt_tracks, gt_occluded = prepare_video(dataset[key], loader)
            print(f"Tracking {key}: {len(frames)} frames, {len(queries)} first queries", flush=True)
            sparse_results, sparse_stats = sparse_tracks(model, frames, queries, args.stride, deadline)
            check_deadline(deadline)
            tracker_tracks, tracker_occluded, tracker_seconds = cotracker_tracks(tracker, frames, queries)
            results = {**sparse_results, "cotracker3_full_observations": (tracker_tracks, tracker_occluded)}
            metrics = {
                arm: score_tracks(metrics_function, queries, gt_tracks, gt_occluded, tracks, occluded)
                for arm, (tracks, occluded) in results.items()
            }
            identifier = hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]
            prediction_file = prediction_directory / f"{identifier}.npz"
            if prediction_file.exists():
                raise FileExistsError(f"Refusing to overwrite predictions: {prediction_file}")
            arrays: dict[str, np.ndarray] = {"queries_tyx": queries}
            for arm, (tracks, occluded) in results.items():
                arrays[f"{arm}_tracks_xy"] = tracks
                arrays[f"{arm}_occluded"] = occluded
            np.savez_compressed(prediction_file, **arrays)
            summary["videos"].append({
                "key": str(key),
                "frames": len(frames),
                "query_points": len(queries),
                "metrics": metrics,
                "sparse_budget": sparse_stats,
                "cotracker_budget": {
                    "observed_frames": len(frames),
                    "observed_fraction": 1.0,
                    "predictor_forward_calls": 1,
                    "seconds": tracker_seconds,
                },
                "predictions": str(prediction_file.resolve()),
            })
            summary["elapsed_seconds"] = time.perf_counter() - started
            write_summary(summary_path, summary)
            print(json.dumps({"video": str(key), "metrics": metrics}), flush=True)
        summary["status"] = "complete"
    except RuntimeBudgetExceeded as error:
        summary["status"] = "partial_runtime_budget"
        summary["stop_reason"] = str(error)
    except Exception as error:
        summary["status"] = "failed"
        summary["stop_reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        # If common_model exposes actual cache/pass counters, preserve them. An
        # absent interface is explicit; wrapper counters are not encoder FLOPs.
        summary["frozen_model_counters"] = model_counters(model)
        summary["actual_encoder_counts_available"] = bool(summary["frozen_model_counters"])
        summary["matched_sparse_observations"] = True
        summary["matched_encoder_compute"] = "Not established by shared RGB observations; inspect actual encoder-pass counters."
        write_summary(summary_path, summary)
    print(json.dumps({"status": summary["status"], "mean_metrics": summary["mean_metrics"]}), flush=True)


if __name__ == "__main__":
    main()
