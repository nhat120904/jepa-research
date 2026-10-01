"""Compute-node-only summary and episode bootstrap for completed probes."""
import argparse
import json
from pathlib import Path
import numpy as np


def auc(labels, scores):
    good = np.asarray(scores)[~np.asarray(labels, bool)]
    bad = np.asarray(scores)[np.asarray(labels, bool)]
    delta = bad[:, None] - good[None, :]
    return float(((delta > 0) + .5 * (delta == 0)).mean())


p = argparse.ArgumentParser()
p.add_argument("--runs", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
result = {}
for path in sorted(args.runs.glob("physics_*/result.json")):
    data = json.loads(path.read_text())
    result[path.parent.name] = {k: data[k] for k in ("videos", "metrics", "compute", "load_report", "limits")}
for path in sorted(args.runs.glob("tracking_*/summary.json")):
    data = json.loads(path.read_text())
    compact = {k: data.get(k) for k in ("status", "completed_videos", "selected_keys", "mean_metrics", "limitations", "load_seconds", "metric_units", "feature_space", "sources", "frozen_model_load_report")}
    compact["eligible_for_selection"] = data.get("status") == "complete" and bool(data.get("feature_space"))
    result[path.parent.name] = compact
    if data.get("videos"):
        differences = [v["metrics"]["predicted_features"]["average_jaccard"] - v["metrics"]["repeat_features"]["average_jaccard"] for v in data["videos"]]
        rng = np.random.default_rng(20260930)
        boot = rng.choice(differences, (2000, len(differences)), replace=True).mean(1)
        compact["pred_minus_repeat_AJ_pp"] = {"mean": float(np.mean(differences)), "video_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(), "positive_videos": int(np.sum(np.asarray(differences) > 0)), "total_videos": len(differences)}
        compact["budget_totals"] = {
            "frames": sum(v["frames"] for v in data["videos"]),
            "observed_frames": sum(v["sparse_budget"]["observed_frames"] for v in data["videos"]),
            "encoder_seconds_shared": sum(v["sparse_budget"]["encoder_seconds_shared"] for v in data["videos"]),
            "predictor_wrapper_seconds_added": sum(v["sparse_budget"]["predictor_wrapper_seconds"] for v in data["videos"]),
            "cotracker_seconds": sum(v["cotracker_budget"]["seconds"] for v in data["videos"]),
            "reader_seconds": {arm: sum(v["sparse_budget"]["reader_seconds_by_arm"][arm] for v in data["videos"]) for arm in ("repeat_features", "predicted_features", "copy_last_position")},
        }
for path in sorted(args.runs.glob("failure_*/failure_probe.json")):
    data = json.loads(path.read_text())
    compact = {k: data.get(k) for k in ("status", "counts", "metrics", "model_counters", "checkpoint_load_report", "error")}
    stac = data.get("released_stac", {})
    compact["released_stac"] = stac
    test = [e for e in data.get("episodes", []) if e["split"] == "test"]
    if len(test) == 20 and data.get("status") == "complete":
        labels = np.array([not e["success"] for e in test])
        scores = {name: np.array([max(r[name] for r in e["scores"]) for e in test]) for name in data["metrics"]}
        differences = {}
        rng = np.random.default_rng(20260930)
        success = np.flatnonzero(~labels)
        failure = np.flatnonzero(labels)
        for control in ("feature_change_mean", "rgb_change_mean"):
            boot = []
            for _ in range(2000):
                idx = np.r_[rng.choice(success, len(success)), rng.choice(failure, len(failure))]
                boot.append(auc(labels[idx], scores["wm_residual_mean"][idx]) - auc(labels[idx], scores[control][idx]))
            differences[control] = {"WM_minus_control_AUC": auc(labels, scores["wm_residual_mean"]) - auc(labels, scores[control]), "stratified_episode_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist()}
        compact["paired_AUC_differences"] = differences
    result[path.parent.name] = compact
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
