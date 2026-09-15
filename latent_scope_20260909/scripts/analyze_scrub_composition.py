#!/usr/bin/env python3
"""Scrub label and composition audit on all released demonstrations (CPU only).

Part A: download all non-video files plus videos; verify every video decodes with one
frame per parquet row; label every episode by recorded-state playback with the native
ScrubCuttingBoard monitor (as in check_scrub_demo_data.py), also saving board pose.

Part B: for two candidate targets, measure how segment summaries compose:
  * native_count: the task's ordered greedy distinct-contact counter (2 cm rule);
  * coverage_area: union of 1 cm discs around grasped sponge-board contact positions,
    which is order independent and composes exactly by set union.
For windows of 16/32/64/128 native steps, compare
  * empty-history value vs increment given the real history (<, =, >);
  * sum of part values vs whole value for 2- and 3-part splits (<, =, >);
  * for native_count, re-filtering the concatenated per-part accepted points vs whole;
  * robustness of each target to halving the frame rate.
Counts are reported per window (overlapping stride 4 and non-overlapping blocks) and as
numbers of distinct episodes, separately for episodes whose playback success agrees with
the recorded success and those that do not. No policy, rendering or training.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CONTACT_RADIUS = 0.02
DISC_RADIUS = 0.01
GRID = 0.0025
WINDOW_LENGTHS = (16, 32, 64, 128)
SPLITS = {"2@1/4": (0.25,), "2@1/2": (0.5,), "2@3/4": (0.75,), "3@1/3,2/3": (1 / 3, 2 / 3)}


def require_slurm() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Scrub composition audit must run inside an sbatch job")
    return job_id


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Part A: data and labels


def download_all(config: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    source = config["dataset_source"]
    base = source["path_in_repo"]
    start = time.time()
    snapshot_download(
        repo_id=source["repo_id"],
        repo_type="dataset",
        revision=source["revision"],
        allow_patterns=[f"{base}/meta/*", f"{base}/data/*", f"{base}/extras/*", f"{base}/videos/*"],
        local_dir=config["mirror_dir"],
        max_workers=8,
    )
    return {"seconds": time.time() - start}


def check_video(task: tuple[str, int, list[str]]) -> dict[str, Any]:
    import av
    import pandas as pd

    dataset_dir, episode, cameras = task
    root = Path(dataset_dir)
    rows = len(pd.read_parquet(root / f"data/chunk-000/episode_{episode:06d}.parquet"))
    row: dict[str, Any] = {"episode": episode, "rows": rows, "ok": True, "cameras": {}}
    for camera in cameras:
        path = root / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
        try:
            with av.open(str(path)) as container:
                frames = sum(1 for _ in container.decode(video=0))
            row["cameras"][camera] = frames
            row["ok"] = row["ok"] and frames == rows
        except Exception as error:
            row["cameras"][camera] = repr(error)[:200]
            row["ok"] = False
    return row


_ENV = None


def _init_worker(dataset_dir: str) -> None:
    global _ENV
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_scrub_demo_data import make_env

    _ENV = make_env(Path(dataset_dir))


def label_episode(task: tuple[str, int, str]) -> dict[str, Any]:
    import pandas as pd
    import robocasa.utils.lerobot_utils as lerobot_utils
    import robocasa.utils.object_utils as OU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    dataset_dir, episode, label_dir = task
    root = Path(dataset_dir)
    out = Path(label_dir) / f"episode_{episode:06d}.npz"
    env = _ENV
    try:
        states = lerobot_utils.get_episode_states(root, episode)
        recorded = pd.read_parquet(root / f"data/chunk-000/episode_{episode:06d}.parquet")["next.reward"].to_numpy()
        reset_to(
            env,
            {
                "states": states[0],
                "model": lerobot_utils.get_episode_model_xml(root, episode),
                "ep_meta": json.dumps(lerobot_utils.get_episode_meta(root, episode)),
            },
        )
        env.board_contact_positions = []
        env.board_contact_timer = 0
        model = env.sim.model
        sponge = env.obj_body_id["sponge"]
        board = env.obj_body_id["cutting_board"]
        sponge_geoms = [model.geom_id2name(g) for g in range(model.ngeom) if model.geom_bodyid[g] == sponge]
        board_geoms = [model.geom_id2name(g) for g in range(model.ngeom) if model.geom_bodyid[g] == board]
        n = len(states)
        arrays = {
            "count": np.zeros(n, np.int32),
            "contact": np.zeros(n, bool),
            "grasped": np.zeros(n, bool),
            "far": np.zeros(n, bool),
            "success": np.zeros(n, bool),
            "sponge_xy": np.zeros((n, 2)),
            "board_pos": np.zeros((n, 3)),
            "board_quat": np.zeros((n, 4)),
        }
        max_error = 0.0
        for t in range(n):
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()
            max_error = max(max_error, float(np.max(np.abs(np.asarray(env.sim.get_state().flatten()) - states[t]))))
            arrays["grasped"][t] = bool(OU.check_obj_grasped(env, "sponge"))
            arrays["contact"][t] = bool(env.check_contact(sponge_geoms, board_geoms))
            arrays["sponge_xy"][t] = np.asarray(env.sim.data.body_xpos[sponge])[:2]
            arrays["board_pos"][t] = np.asarray(env.sim.data.body_xpos[board])
            arrays["board_quat"][t] = np.asarray(env.sim.data.body_xquat[board])
            arrays["far"][t] = bool(OU.gripper_obj_far(env, "sponge", th=0.15))
            env.update_state()
            arrays["count"][t] = int(env.board_contact_timer)
            arrays["success"][t] = bool(env._check_success())
        arrays["recorded_reward"] = recorded
        np.savez_compressed(out, **arrays)
        recorded_first = int(np.argmax(recorded > 0)) if (recorded > 0).any() else None
        playback_first = int(np.argmax(arrays["success"])) if arrays["success"].any() else None
        return {
            "episode": episode,
            "frames": n,
            "max_state_error": max_error,
            "recorded_first_success": recorded_first,
            "playback_first_success": playback_first,
            "agree": (recorded_first is None) == (playback_first is None)
            and (recorded_first is None or recorded_first == playback_first),
            "final_count": int(arrays["count"][-1]),
        }
    except Exception as error:
        return {"episode": episode, "error": repr(error)[:500]}


# ---------------------------------------------------------------------------
# Part B: composition statistics


def greedy_accept(points: list[np.ndarray], history: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """Native ordered filter: accept a point if it is >= 2 cm from every accepted point."""
    accepted = np.empty((0, 2)) if not history else np.asarray(history, dtype=float).reshape(-1, 2)
    new = []
    limit = CONTACT_RADIUS * CONTACT_RADIUS
    for p in points:
        if accepted.shape[0] == 0 or float(np.min(np.sum((accepted - p) ** 2, axis=1))) >= limit:
            accepted = np.vstack([accepted, p])
            new.append(p)
    return new


_OFFSETS = None


def disc_cells(point: np.ndarray) -> set[int]:
    global _OFFSETS
    if _OFFSETS is None:
        r = int(round(DISC_RADIUS / GRID))
        _OFFSETS = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r]
    ix = int(np.floor(point[0] / GRID)) + (1 << 20)
    iy = int(np.floor(point[1] / GRID)) + (1 << 20)
    return {((ix + dx) << 21) | (iy + dy) for dx, dy in _OFFSETS}


def cells_of(point_cells: list[set[int]]) -> set[int]:
    union: set[int] = set()
    for cells in point_cells:
        union |= cells
    return union


def sign(a: float, b: float, tol: float = 0.0) -> str:
    if a > b + tol:
        return ">"
    if a < b - tol:
        return "<"
    return "="


class Tally:
    """Counts of <,=,> per comparison, with the set of contributing episodes."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def add(self, key: str, outcome: str, episode: int, excess: float = 0.0) -> None:
        row = self.rows.setdefault(key, {"<": 0, "=": 0, ">": 0, "episodes": set(), "noneq_episodes": set(), "abs_excess": 0.0})
        row[outcome] += 1
        row["episodes"].add(episode)
        if outcome != "=":
            row["noneq_episodes"].add(episode)
            row["abs_excess"] += abs(excess)

    def export(self) -> dict[str, Any]:
        out = {}
        for key, row in sorted(self.rows.items()):
            total = row["<"] + row["="] + row[">"]
            out[key] = {
                "windows": total,
                "lt": row["<"],
                "eq": row["="],
                "gt": row[">"],
                "fraction_not_equal": (row["<"] + row[">"]) / max(total, 1),
                "episodes": len(row["episodes"]),
                "episodes_with_any_not_equal": len(row["noneq_episodes"]),
                "mean_abs_error_when_not_equal": row["abs_excess"] / max(row["<"] + row[">"], 1),
            }
        return out


def analyze_episode(episode: int, arrays: dict[str, np.ndarray], tally_sets: list[dict[str, Tally]]) -> dict[str, Any]:
    def add(kind: str, key: str, outcome: str, excess: float = 0.0) -> None:
        for tallies in tally_sets:
            tallies[kind].add(key, outcome, episode, excess)

    live = arrays["contact"] & arrays["grasped"]
    xy = arrays["sponge_xy"]
    n = len(live)
    idx = np.flatnonzero(live)
    points = [xy[i] for i in idx]
    point_cells = [disc_cells(p) for p in points]

    # Sequential filter: history-accepted set for any prefix is a prefix of the full run.
    accepted_positions: list[int] = []
    acc = np.empty((0, 2))
    for k, p in enumerate(points):
        if acc.shape[0] == 0 or float(np.min(np.sum((acc - p) ** 2, axis=1))) >= CONTACT_RADIUS ** 2:
            acc = np.vstack([acc, p])
            accepted_positions.append(k)
    first_cover: dict[int, int] = {}
    for k, cells in enumerate(point_cells):
        for c in cells:
            first_cover.setdefault(c, k)

    def pts(a: int, b: int) -> tuple[int, int]:
        return int(np.searchsorted(idx, a)), int(np.searchsorted(idx, b))

    native_total = len(accepted_positions)
    matches_native = native_total == int(arrays["count"][-1])
    half_mask = [k for k, i in enumerate(idx) if i % 2 == 0]
    robustness = {
        "native_full": native_total,
        "native_half": len(greedy_accept([points[k] for k in half_mask])),
        "area_full_cm2": len(first_cover) * GRID * GRID * 1e4,
        "area_half_cm2": len(cells_of([point_cells[k] for k in half_mask])) * GRID * GRID * 1e4,
    }
    cell_area = GRID * GRID * 1e4
    for length in WINDOW_LENGTHS:
        for mode, stride in (("overlap", 4), ("block", length)):
            for t in range(0, n - length + 1, stride):
                u = t + length
                a, b = pts(t, u)
                if a == b:
                    continue  # no grasped contact: every target is zero on both sides
                prefix = f"L{length}/{mode}"
                window_pts = points[a:b]
                whole = len(greedy_accept(window_pts))
                history_accepted = [points[k] for k in accepted_positions if k < a]
                increment = len(greedy_accept(window_pts, history_accepted))
                add("native", f"{prefix}/empty_history_vs_increment", sign(whole, increment), whole - increment)
                window_cells = cells_of(point_cells[a:b])
                area_whole = len(window_cells) * cell_area
                area_increment = sum(1 for c in window_cells if first_cover[c] >= a) * cell_area
                add("coverage", f"{prefix}/empty_history_vs_increment", sign(area_whole, area_increment), area_whole - area_increment)
                for split_name, fractions in SPLITS.items():
                    cuts = [t] + [t + int(round(f * length)) for f in fractions] + [u]
                    parts = [pts(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
                    part_accepted = [greedy_accept(points[p:q]) for p, q in parts]
                    count_sum = sum(len(x) for x in part_accepted)
                    add("native", f"{prefix}/split{split_name}/sum_counts_vs_whole", sign(count_sum, whole), count_sum - whole)
                    merged = len(greedy_accept([p for x in part_accepted for p in x]))
                    add("native", f"{prefix}/split{split_name}/refilter_part_accepted_vs_whole", sign(merged, whole), merged - whole)
                    part_cells = [cells_of(point_cells[p:q]) for p, q in parts]
                    area_sum = sum(len(c) for c in part_cells) * cell_area
                    add("coverage", f"{prefix}/split{split_name}/sum_areas_vs_whole", sign(area_sum, area_whole), area_sum - area_whole)
                    union_equal = cells_of(part_cells) == window_cells
                    add("coverage", f"{prefix}/split{split_name}/union_of_part_masks_vs_whole", "=" if union_equal else ">")
    return {"episode": episode, "matches_native": matches_native, "robustness": robustness}


def self_test() -> dict[str, Any]:
    """Invariants on synthetic trajectories, run on the compute node before any download."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = 300
        live = rng.random(n) < 0.3
        xy = np.cumsum(rng.normal(0, 0.006, size=(n, 2)), axis=0)
        arrays = {"contact": live, "grasped": np.ones(n, bool), "sponge_xy": xy, "count": np.zeros(n, np.int32)}
        pts_all = [xy[i] for i in np.flatnonzero(live)]
        arrays["count"][-1] = len(greedy_accept(pts_all))
        # greedy_accept against a naive reference
        naive: list[np.ndarray] = []
        for p in pts_all:
            if all(np.linalg.norm(p - q) >= CONTACT_RADIUS for q in naive):
                naive.append(p)
        assert len(naive) == int(arrays["count"][-1])
        groups = {"native": Tally(), "coverage": Tally()}
        result = analyze_episode(trial, arrays, [groups])
        assert result["matches_native"]
        cov = groups["coverage"].export()
        for key, row in cov.items():
            if key.endswith("union_of_part_masks_vs_whole"):
                assert row["gt"] == 0 and row["lt"] == 0, key
            if key.endswith("empty_history_vs_increment") or key.endswith("sum_areas_vs_whole"):
                assert row["lt"] == 0, key  # coverage is subadditive
    return {"passed": True, "trials": 20}


def composition_report(label_dir: Path, labels: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"all": {"native": Tally(), "coverage": Tally()}, "label_agree": {"native": Tally(), "coverage": Tally()}, "label_disagree": {"native": Tally(), "coverage": Tally()}}
    per_episode = []
    for row in labels:
        if "error" in row:
            continue
        z = np.load(label_dir / f"episode_{row['episode']:06d}.npz")
        arrays = {k: z[k] for k in z.files}
        group = "label_agree" if row["agree"] else "label_disagree"
        result = analyze_episode(row["episode"], arrays, [groups["all"], groups[group]])
        per_episode.append({**result, "agree": row["agree"]})
    robust = [r["robustness"] for r in per_episode]
    native_changed = sum(r["native_full"] != r["native_half"] for r in robust)
    area_rel = [abs(r["area_half_cm2"] - r["area_full_cm2"]) / r["area_full_cm2"] for r in robust if r["area_full_cm2"] > 0]
    native_rel = [abs(r["native_half"] - r["native_full"]) / r["native_full"] for r in robust if r["native_full"] > 0]
    return {
        "episodes_analyzed": len(per_episode),
        "offline_filter_matches_native": sum(r["matches_native"] for r in per_episode),
        "frame_rate_robustness": {
            "native_final_changed_episodes": native_changed,
            "native_median_relative_change": float(np.median(native_rel)) if native_rel else None,
            "native_p90_relative_change": float(np.quantile(native_rel, 0.9)) if native_rel else None,
            "area_median_relative_change": float(np.median(area_rel)) if area_rel else None,
            "area_p90_relative_change": float(np.quantile(area_rel, 0.9)) if area_rel else None,
            "median_final_area_cm2": float(np.median([r["area_full_cm2"] for r in robust])),
            "median_final_native_count": float(np.median([r["native_full"] for r in robust])),
        },
        "tallies": {g: {k: t.export() for k, t in v.items()} for g, v in groups.items()},
        "per_episode": per_episode,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    job_id = require_slurm()
    config = json.loads(args.config.read_text())
    out = args.output_dir
    label_dir = out / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(config["mirror_dir"]) / config["dataset_source"]["path_in_repo"]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    import importlib.metadata as metadata

    report: dict[str, Any] = {
        "job_id": job_id,
        "node": platform.node(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "workers": workers,
        "packages": {name: metadata.version(name) for name in ("robocasa", "robosuite", "mujoco", "huggingface_hub", "av")},
        "constants": {"contact_radius_m": CONTACT_RADIUS, "disc_radius_m": DISC_RADIUS, "grid_m": GRID, "window_lengths": WINDOW_LENGTHS, "splits": SPLITS},
    }

    def save() -> None:
        (out / "composition_audit_result.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n")

    report["self_test"] = self_test()
    save()
    report["download"] = download_all(config)
    episodes = sorted(json.loads(line)["episode_index"] for line in (dataset_dir / "meta/episodes.jsonl").read_text().splitlines() if line.strip())
    report["episodes_listed"] = len(episodes)
    save()

    ctx = mp.get_context("spawn")
    start = time.time()
    with ctx.Pool(workers) as pool:
        videos = pool.map(check_video, [(str(dataset_dir), e, config["camera_names"]) for e in episodes], chunksize=8)
    report["videos"] = {
        "seconds": time.time() - start,
        "episodes_ok": sum(v["ok"] for v in videos),
        "episodes_checked": len(videos),
        "failures": [v for v in videos if not v["ok"]],
    }
    save()

    start = time.time()
    labels = []
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(dataset_dir),)) as pool:
        for row in pool.imap_unordered(label_episode, [(str(dataset_dir), e, str(label_dir)) for e in episodes]):
            labels.append(row)
            if len(labels) % 50 == 0:
                print(f"labelled {len(labels)}/{len(episodes)}", flush=True)
    labels.sort(key=lambda r: r["episode"])
    ok = [r for r in labels if "error" not in r]
    report["labels"] = {
        "seconds": time.time() - start,
        "episodes": len(labels),
        "errors": [r for r in labels if "error" in r],
        "max_state_error": max((r["max_state_error"] for r in ok), default=None),
        "recorded_success": sum(r["recorded_first_success"] is not None for r in ok),
        "playback_success": sum(r["playback_first_success"] is not None for r in ok),
        "agree": sum(r["agree"] for r in ok),
        "disagree_episodes": [r["episode"] for r in ok if not r["agree"]],
        "per_episode": labels,
    }
    save()
    print(f"labels done: {report['labels']['agree']}/{len(ok)} agree", flush=True)

    start = time.time()
    report["composition"] = composition_report(label_dir, ok)
    report["composition"]["seconds"] = time.time() - start
    save()
    print(f"WROTE {out / 'composition_audit_result.json'}", flush=True)


if __name__ == "__main__":
    main()
