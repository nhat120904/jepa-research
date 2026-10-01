"""Exploratory no-training IntPhys2 task probe. sbatch ONLY."""
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from common_model import FrozenVJEPA


def read_frames(path, step=8):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {path}")
    frames = []
    index = 0
    while True:
        good, image = capture.read()
        if not good:
            break
        if index % step == 0:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"Empty video {path}")
    return frames


def task_metrics(rows, scores):
    groups = defaultdict(dict)
    for row in rows:
        kind, possible = row["type"].split("_", 1)
        groups[(row["SceneIndex"], kind)][possible] = row
    result = {}
    for name in scores:
        pairs = []
        strata = defaultdict(list)
        for key, pair in groups.items():
            if set(pair) != {"Possible", "Impossible"}:
                raise RuntimeError(f"Incomplete standard pair {key}")
            a, b = pair["Possible"], pair["Impossible"]
            delta = b["scores"][name] - a["scores"][name]
            hit = float(delta > 0) + 0.5 * float(delta == 0)
            pairs.append((key[0], hit))
            strata[a["Camera"]].append(hit)
            strata[a["condition"]].append(hit)
        by_scene = defaultdict(list)
        for scene, hit in pairs:
            by_scene[scene].append(hit)
        values = np.array([np.mean(v) for v in by_scene.values()])
        rng = np.random.default_rng(20260930)
        bootstrap = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        labels = [int("Impossible" in row["type"]) for row in rows]
        result[name] = {"relative_accuracy": float(np.mean([hit for _, hit in pairs])),
                        "scene_bootstrap_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                        "auc": float(roc_auc_score(labels, [r["scores"][name] for r in rows])),
                        "by_stratum": {k: float(np.mean(v)) for k, v in strata.items()}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vjepa-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=24)
    parser.add_argument("--frame-step", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((args.data_root / "intphys2" / "metadata.csv").open()))
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["SceneIndex"]].append(row)
    strata = defaultdict(list)
    for sid, bank in scenes.items():
        r = bank[0]
        strata[(r["condition"], r["Difficulty"], r["Camera"])].append(sid)
    ordered = []
    for key, ids in sorted(strata.items()):
        ids.sort(key=lambda sid: hashlib.sha256(("probe-" + sid).encode()).hexdigest())
    while any(strata.values()):
        for key in sorted(strata):
            if strata[key]:
                ordered.append(strata[key].pop(0))
    selected = ordered[:args.max_scenes]
    model = FrozenVJEPA(args.vjepa_source, args.checkpoint)
    records = []
    start = time.perf_counter()
    for sid in selected:
        for row in scenes[sid]:
            path = args.out / (row["name"] + ".json")
            if path.exists():
                records.append(json.loads(path.read_text()))
                continue
            frames = read_frames(args.data_root / "intphys2" / row["file_name"], args.frame_step)
            valid = len(frames)
            if len(frames) < 16:
                frames.extend([frames[-1]] * (16 - len(frames)))
            starts = list(range(0, len(frames) - 15, 2))
            if len(starts) > args.max_windows:
                starts = [starts[i] for i in np.linspace(0, len(starts)-1, args.max_windows, dtype=int)]
            windows = []
            for offset in starts:
                pred, target = model.score_clip(frames[offset:offset+16], context_frames=12)
                residual = (pred - target).float()
                centered = residual - residual.mean(dim=1, keepdim=True)
                patch_losses = residual.abs().mean(dim=-1)
                windows.append({"global": float(residual.abs().mean()),
                                "centered": float(centered.abs().mean()),
                                "top10": float(patch_losses.flatten().topk(max(1, int(patch_losses.numel()*.1))).values.mean())})
            scores = {name + "_max": max(w[name] for w in windows) for name in windows[0]}
            scores.update({name + "_mean": float(np.mean([w[name] for w in windows])) for name in windows[0]})
            record = dict(row, scores=scores, windows=windows, sampled_frames=valid, padded=valid < 16)
            path.write_text(json.dumps(record, indent=2))
            records.append(record)
        print(f"PHYSICS scene={sid} videos={len(records)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        (args.out / "partial_metrics.json").write_text(json.dumps(task_metrics(records, records[0]["scores"]), indent=2))
    report = {"training": False, "publication_reproduction": False,
              "protocol": "Released Giant256;16 actual sampled frames;12 context; frame-step8;max8 windows; future removed before context attention; full-window EMA only for ground-truth scoring",
              "primary_score": "global_max", "adaptations": ["centered_max", "top10_max"],
              "subset_scenes": selected, "videos": len(records), "metrics": task_metrics(records, records[0]["scores"]),
              "load_report": model.load_report, "compute": model.counters(),
              "wall_seconds": time.perf_counter()-start,
              "limits": ["Small discovery subset; not full benchmark or SOTA comparison", "Modified clip/window protocol; published accuracy is not directly comparable", "Adaptations are probes, not learned methods; inspect existing geometry/surprise prior art"]}
    (args.out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
