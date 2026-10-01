#!/usr/bin/env python3
"""Frozen, causal PushChair outcome probe; run on a compute node.

The prediction residual is available only when its target frame arrives. This
script evaluates rollout outcomes from observed prefixes, not failure onset.
Sentinel labels are final episode outcomes; no onset annotation is assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


STAC_EXP = "pred_horizon_16_sample_size_256_error_fn_kde_kl_all_rev_eig"
STAC_QUANTILE = STAC_EXP + "_quantile_0.95"
SCORE_NAMES = (
    "wm_residual_mean",
    "wm_residual_motion_weighted",
    "feature_change_mean",
    "feature_change_motion_weighted",
    "rgb_change_mean",
)


def scalar(value):
    if isinstance(value, dict):
        return {str(k): scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar(v) for v in value]
    if isinstance(value, np.ndarray):
        return scalar(value.tolist())
    if isinstance(value, np.generic):
        return scalar(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(scalar(payload), indent=2, allow_nan=False))
    temporary.replace(path)


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def normalize_encoded(features):
    tensor = torch.as_tensor(features)
    return as_numpy(F.layer_norm(tensor.float(), (tensor.shape[-1],)))


def frame_rgb(row):
    # Match Sentinel get_rgb: the final entry of 4-D RGB is the latest image.
    observation = row.get("obs", row)
    if not isinstance(observation, dict) or "rgb" not in observation:
        raise ValueError("Episode row has no explicit RGB observation")
    image = np.asarray(observation["rgb"])
    original_shape = list(image.shape)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3:
        raise ValueError(f"Unsupported RGB shape: {original_shape}")
    if image.shape[-1] not in (3, 4) and image.shape[0] in (3, 4):
        image = image.transpose(1, 2, 0)
    if image.shape[-1] not in (3, 4):
        raise ValueError(f"RGB channel axis is ambiguous: {original_shape}")
    image = image[..., :3]
    if not np.isfinite(image).all():
        raise ValueError("Non-finite RGB values")
    if image.dtype != np.uint8:
        if image.min() < 0 or image.max() > 255:
            raise ValueError("RGB values are outside [0,255]")
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
            image = image * 255
        image = np.rint(image).astype(np.uint8)
    image = np.asarray(Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(image), original_shape


def load_episode(path, split, max_native_timestep):
    with path.open("rb") as handle:
        frame = pickle.load(handle)
    if not hasattr(frame, "to_dict"):
        raise ValueError(f"Expected released pandas episode: {path}")
    rows = frame.to_dict("records")
    if not rows or any("success" not in r for r in rows):
        raise ValueError(f"Missing explicit success labels: {path}")
    labels = {bool(r["success"]) for r in rows}
    identifiers = {int(r["episode"]) for r in rows}
    if len(labels) != 1 or len(identifiers) != 1:
        raise ValueError(f"Episode ID or outcome changes within {path}")
    expected_id = int(path.stem.removeprefix("ep"))
    if identifiers != {expected_id}:
        raise ValueError(f"Filename and native episode ID disagree: {path}")
    selected = []
    times = []
    shapes = set()
    for row in rows:
        time = int(row["timestep"])
        if float(row["timestep"]) != time or time < 0:
            raise ValueError(f"Non-integer/negative native time in {path}")
        if time >= max_native_timestep:
            continue
        image, shape = frame_rgb(row)
        selected.append(image)
        times.append(time)
        shapes.add(tuple(shape))
    if len(times) < 2 or times != sorted(set(times)):
        raise ValueError(f"Need >=2 strictly increasing observations: {path}")
    return {
        "split": split, "episode": expected_id, "success": labels.pop(),
        "file": str(path), "native_timesteps": times,
        "rgb_original_shapes": [list(s) for s in sorted(shapes)],
        "images": selected,
    }


def balanced_cap(episodes, limit):
    if len(episodes) <= limit:
        return episodes
    groups = [[e for e in episodes if e["success"] == label] for label in (False, True)]
    order = []
    for i in range(max(map(len, groups))):
        for group in groups:
            if i < len(group):
                order.append(group[i])
    return sorted(order[:limit], key=lambda e: e["episode"])


def discover(data_root, max_episodes, max_native_timestep):
    calib_dirs = sorted(data_root.rglob("0914_push_chair_sim3_dp_ddim_calib/episodes"))
    test_dirs = sorted(data_root.rglob("0914_push_chair_sim3_dp_ddim_test/episodes"))
    if len(calib_dirs) != 1 or len(test_dirs) != 1:
        raise ValueError(f"Expected one official calib/test pair, found {calib_dirs}, {test_dirs}")
    calib = [load_episode(p, "calibration", max_native_timestep)
             for p in sorted(calib_dirs[0].glob("ep*.pkl"))]
    test = [load_episode(p, "test", max_native_timestep)
            for p in sorted(test_dirs[0].glob("ep*.pkl"))]
    if len(calib) != 10 or len(test) != 20 or not all(e["success"] for e in calib):
        raise ValueError("Expected official 10-success calibration and 20-test release")
    if sum(e["success"] for e in test) != 10:
        raise ValueError("Expected official test release: 10 successes, 10 failures")
    if max_episodes < len(calib) + 2:
        raise ValueError("--max-episodes must allow all 10 calibration plus >=2 tests")
    selected_test = balanced_cap(test, max_episodes - len(calib))
    result_files = sorted(test_dirs[0].parent.rglob("temporal_consistency_results.pkl"))
    official = [p for p in result_files if p.parent.name == "0914_results_calib_on_light_1"]
    if len(official) != 1:
        raise ValueError(f"Expected one released official STAC pickle, found {official}")
    return calib, selected_test, test, official[0]


def auc_outcome(records, score_name, cutoff=None):
    scores = []
    for episode in records:
        available = [r[score_name] for r in episode["scores"]
                     if cutoff is None or r["elapsed_seconds"] <= cutoff + 1e-9]
        if not available:
            return {"status": "unavailable", "reason": "a trajectory has no scored observation before cutoff"}
        scores.append((not episode["success"], max(available)))
    failure = np.array([s for y, s in scores if y])
    success = np.array([s for y, s in scores if not y])
    if not len(failure) or not len(success):
        return {"status": "unavailable", "reason": "both rollout outcomes are required"}
    # Mann-Whitney AUROC with exact half credit for tied scores.
    difference = failure[:, None] - success[None, :]
    return {"status": "ok", "auroc": float(np.mean((difference > 0) + .5 * (difference == 0))),
            "failure_count": len(failure), "success_count": len(success)}


def detection_metrics(records, score_name, threshold):
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    detection = []
    first_alarm = {}
    for episode in records:
        hits = [r for r in episode["scores"] if r[score_name] >= threshold]
        alarm = bool(hits)
        failure = not episode["success"]
        counts["TP" if failure and alarm else "FN" if failure else "FP" if alarm else "TN"] += 1
        first_alarm[str(episode["episode"])] = hits[0]["elapsed_seconds"] if hits else None
        if failure and alarm:
            detection.append(hits[0]["elapsed_seconds"])
    tpr = counts["TP"] / (counts["TP"] + counts["FN"])
    tnr = counts["TN"] / (counts["TN"] + counts["FP"])
    return {**counts, "TPR": tpr, "TNR": tnr, "balanced_accuracy": (tpr + tnr) / 2,
            "accuracy": (counts["TP"] + counts["TN"]) / len(records),
            "threshold": threshold, "true_positive_first_alarm_seconds_mean":
            float(np.mean(detection)) if detection else None, "first_alarm_seconds": first_alarm}


def aggregate(calibration, test, score_names, prefix_seconds):
    output = {}
    shared_end = min(e["scores"][-1]["elapsed_seconds"] for e in test)
    for name in score_names:
        calibration_maxima = [max(r[name] for r in e["scores"]) for e in calibration]
        threshold = float(np.quantile(calibration_maxima, .95))
        prefixes = {}
        for cutoff in prefix_seconds:
            prefixes[str(cutoff)] = (
                auc_outcome(test, name, cutoff) if cutoff <= shared_end + 1e-9 else
                {"status": "unavailable", "reason": "cutoff exceeds shared observed coverage",
                 "shared_coverage_seconds": shared_end})
        output[name] = {"calibration_episode_maxima": calibration_maxima,
                        "whole_episode_detection": detection_metrics(test, name, threshold),
                        "whole_episode_outcome_auroc": auc_outcome(test, name),
                        "fixed_elapsed_prefix_outcome_auroc": prefixes}
    return output


def released_stac(path, native_calib, native_test, selected_test, max_native_timestep, prefix_seconds):
    with path.open("rb") as handle:
        release = pickle.load(handle)
    original = release["results_dict"][STAC_QUANTILE]["ep_iid_cum"]["episode"]
    frames = {"calibration": release["demo_results_frame"], "test": release["test_results_frame"]}
    converted = {}
    for split, episodes in (("calibration", native_calib), ("test", native_test)):
        frame = frames[split]
        if set(map(int, frame["episode"].unique())) != {e["episode"] for e in episodes}:
            raise ValueError(f"Precomputed STAC {split} IDs do not match native episodes")
        converted[split] = []
        for episode in episodes:
            subset = frame[frame["episode"] == episode["episode"]].sort_values("timestep")
            if set(map(bool, subset["success"])) != {episode["success"]}:
                raise ValueError("Precomputed STAC outcome disagrees with native label")
            observed_times = set(episode["native_timesteps"])
            subset = subset[subset["timestep"] < max_native_timestep]
            times = list(map(int, subset["timestep"]))
            if not set(times).issubset(observed_times):
                raise ValueError("STAC timestamps do not match observed RGB timestamps")
            scores = [{"native_timestep": int(r["timestep"]),
                       "elapsed_seconds": int(r["timestep"]) / 3,
                       "stac_step": float(r[STAC_EXP + "_score"]),
                       "stac_cumulative": float(r[STAC_EXP + "_cum_score"])}
                      for r in subset.to_dict("records")]
            if not scores:
                raise ValueError("No precomputed STAC scores in selected window")
            converted[split].append({"split": split, "episode": episode["episode"],
                                     "success": episode["success"], "scores": scores})
    chosen = {e["episode"] for e in selected_test}
    test = [e for e in converted["test"] if e["episode"] in chosen]
    metrics = scalar(original["metrics"])
    native_mean = metrics.get("TP Time Mean")
    return {"file": str(path), "experiment_key": STAC_EXP,
            "quantile_key": STAC_QUANTILE, "calibration_key": "ep_iid_cum",
            "precomputed_official_episode_metrics_native_units": metrics,
            "precomputed_true_positive_first_alarm_seconds_mean":
            native_mean / 3 if native_mean is not None else None,
            "recomputed_selected_episode_metrics": aggregate(
                converted["calibration"], test, ["stac_cumulative"], prefix_seconds),
            "calibration_episodes": converted["calibration"], "test_episodes": test,
            "note": "Released episode AUROC uses first-alarm score; recomputed AUROC uses prefix max cumulative score."}


def score_episode(model, episode, context_observations):
    times, images = episode["native_timesteps"], episode["images"]
    previous_feature = normalize_encoded(model.encode_sparse([times[0]], [images[0]]))[0]
    scores = []
    for index in range(1, len(times)):
        target_time = times[index]
        first = max(0, index - context_observations)
        while first < index and target_time - times[first] > 31:
            first += 1
        if first == index:
            raise ValueError("Observation gap exceeds model sparse time range (31 native steps)")
        origin = times[first]
        context_times = [t - origin for t in times[first:index]]
        context_images = images[first:index]
        # No target image is passed to the predictor. Residual becomes available now.
        prediction = as_numpy(model.predict_sparse(context_times, context_images, [target_time - origin]))[0]
        observed_times = context_times + [target_time - origin]
        observed = normalize_encoded(model.encode_sparse(observed_times, context_images + [images[index]]))[-1]
        if observed.shape != prediction.shape or observed.shape != previous_feature.shape:
            raise ValueError(f"Feature shapes disagree: {observed.shape}, {prediction.shape}, {previous_feature.shape}")
        if observed.ndim != 3 or observed.shape[:2] != (16, 16):
            raise ValueError(f"Expected [16,16,D] observed features, got {observed.shape}")
        residual = np.sqrt(np.mean(np.square(observed - prediction), axis=-1))
        feature_change = np.sqrt(np.mean(np.square(observed - previous_feature), axis=-1))
        pixel_change = np.abs(images[index].astype(np.float32) - images[index - 1].astype(np.float32)) / 255
        motion = pixel_change.mean(axis=-1).reshape(16, 16, 16, 16).mean(axis=(1, 3))
        weights = motion + 1 / 255  # Fixed uniform floor; no data-dependent choice.
        record = {"native_timestep": target_time, "elapsed_seconds": target_time / 3,
                  "prediction_context_end_native_timestep": times[index - 1],
                  "context_native_timesteps": times[first:index],
                  "prediction_target_native_timestep": target_time,
                  "wm_residual_mean": float(residual.mean()),
                  "wm_residual_motion_weighted": float(np.sum(residual * weights) / weights.sum()),
                  "feature_change_mean": float(feature_change.mean()),
                  "feature_change_motion_weighted": float(np.sum(feature_change * weights) / weights.sum()),
                  "rgb_change_mean": float(pixel_change.mean())}
        if not all(math.isfinite(record[name]) for name in SCORE_NAMES):
            raise ValueError("Non-finite frozen model score")
        scores.append(record)
        previous_feature = observed
    return {k: v for k, v in episode.items() if k != "images"} | {"scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vjepa-source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=30)
    parser.add_argument("--max-native-timestep", type=int, default=40)
    parser.add_argument("--context-observations", type=int, default=7)
    parser.add_argument("--prefix-seconds", type=float, nargs="+", default=[3, 6, 9, 12])
    args = parser.parse_args()
    if args.context_observations < 1 or any(t <= 0 for t in args.prefix_seconds):
        parser.error("Need positive context observations and prefix cutoffs")
    output_file = args.out / "failure_probe.json"
    payload = {"status": "initializing", "arguments": vars(args),
               "protocol": {
                   "task": "causal observed-prefix rollout outcome discrimination; no onset labels",
                   "primary_score": "wm_residual_mean",
                   "secondary_scores": list(SCORE_NAMES[1:]),
                   "calibration": "95th numpy quantile of max observed score in 10 successful calibration episodes",
                   "motion_weighting": "16x16 mean absolute RGB-change cells plus fixed 1/255 floor",
                   "feature_space": "EMA observed features get F.layer_norm over D; prediction is compared in its trained target space",
                   "clock": "absolute native timestep / 3 Hz; native timestep < 40 by default",
                   "action_conditioning": "none; PushChair actions are not portable to V-JEPA2-AC",
                   "future_access": "predict_sparse sees previous images only; encoder sees only arrived images",
                   "prefix_coverage": "report fixed elapsed-time cutoff only if every test episode reaches it",
                   "training": "none; no target readout, policy, or world-model training"},
               "episodes": []}
    payload["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    try:
        calib, test, full_test, stac_file = discover(args.data_root, args.max_episodes, args.max_native_timestep)
        payload["released_stac"] = released_stac(stac_file, calib, full_test, test,
                                                 args.max_native_timestep, args.prefix_seconds)
        payload["counts"] = {"calibration": len(calib), "test": len(test), "released_test": len(full_test)}
        payload["status"] = "scoring"
        save_json(output_file, payload)
        from common_model import FrozenVJEPA
        model = FrozenVJEPA(args.vjepa_source, args.checkpoint, device=args.device)
        payload["checkpoint_load_report"] = getattr(model, "load_report", None)
        for episode in calib + test:
            scored = score_episode(model, episode, args.context_observations)
            payload["episodes"].append(scored)
            save_json(output_file, payload)
            print(f"Scored {episode['split']} ep{episode['episode']:04d}: {len(scored['scores'])} causal observations", flush=True)
        scored_calib = [e for e in payload["episodes"] if e["split"] == "calibration"]
        scored_test = [e for e in payload["episodes"] if e["split"] == "test"]
        payload["metrics"] = aggregate(scored_calib, scored_test, SCORE_NAMES, args.prefix_seconds)
        payload["model_counters"] = model.counters()
        payload["status"] = "complete"
        save_json(output_file, payload)
        print(f"Saved {output_file}", flush=True)
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        save_json(output_file, payload)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
