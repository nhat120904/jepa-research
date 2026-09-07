"""Materialise one OGBench-Scene play split into a training cache.

Stage 0 (job 49473) showed the released visual frames are not reproducible by this
renderer, so every frame in this program is drawn here from the stored ``qpos``.  Job
49475 then showed a single 64x64 render costs ~164 ms on these pods -- the NVIDIA
graphics userspace is absent, so EGL falls back to a software rasteriser -- while state
restoration costs 0.2 ms.  The build is therefore sharded across Slurm tasks and renders
only rows on the action-block grid, which is the only grid the model or the planner ever
sees.

Stages
------
``prep``      write the small members, the offline state track and an empty pixel memmap
``render``    fill one shard of the grid rows (many tasks, disjoint ranges)
``finalize``  verify every grid row was written and stamp the cache complete
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.scene_data import (  # noqa: E402
    EPISODE_LEN,
    MEMBERS,
    OfflineSceneState,
    load_small,
    member_header,
)
from scene_progress_wm.scene_render import (  # noqa: E402
    make_scene_env,
    restore_state,
    restore_state_fast,
)

PROTOCOL = "scene_progress_wm_cache_v2"
STATE_COLUMNS = ("cube_x", "cube_y", "cube_z", "drawer", "window", "drawer_site_y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prep", "render", "finalize"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--render-size", type=int, default=64)
    parser.add_argument("--camera", type=str, default="front_pixels")
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=5,
        help="render every k-th row; must equal the action block",
    )
    parser.add_argument(
        "--render-quality",
        choices=("full", "fast"),
        default="fast",
        help="fast disables shadows and reflections, identically at train and eval time",
    )
    parser.add_argument("--limit-rows", type=int, default=0, help="0 = whole split")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def file_digest(path: Path, limit: int = 1 << 20) -> str:
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read(limit)).hexdigest()


def resolve_total(dataset: Path, limit_rows: int) -> tuple[int, tuple[int, ...]]:
    shape, _dtype = member_header(dataset, "observations")
    total = int(shape[0]) if limit_rows <= 0 else min(int(shape[0]), limit_rows)
    if total % EPISODE_LEN != 0:
        raise RuntimeError(
            f"row count {total} is not a whole number of {EPISODE_LEN}-step episodes"
        )
    return total, tuple(int(v) for v in shape[1:])


def grid_rows(total: int, stride: int) -> np.ndarray:
    return np.arange(0, total, stride, dtype=np.int64)


def stage_prep(args: argparse.Namespace) -> None:
    total, pixel_shape = resolve_total(args.dataset, args.limit_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    small = {}
    for member in MEMBERS:
        if member == "observations":
            continue
        array = load_small(args.dataset, member)[:total]
        np.save(args.out_dir / f"{member}.npy", array)
        small[member] = array
        print(f"saved {member} {array.shape} {array.dtype}", flush=True)

    pixels = np.lib.format.open_memmap(
        args.out_dir / "pixels.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total, *pixel_shape),
    )
    pixels.flush()
    del pixels

    _world, raw, render_info = make_scene_env(
        render_size=args.render_size, quality=args.render_quality, camera=args.camera
    )
    offline = OfflineSceneState(raw)
    track = np.empty((total, len(STATE_COLUMNS)), dtype=np.float32)
    started = time.time()
    for row in range(total):
        readout = offline.read(small["qpos"][row])
        cube = readout["cube_pos"]
        track[row, 0:3] = cube
        track[row, 3] = readout["drawer"]
        track[row, 4] = readout["window"]
        track[row, 5] = readout["drawer_site_y"]
        if (row + 1) % 200000 == 0 or row + 1 == total:
            rate = (row + 1) / max(time.time() - started, 1e-6)
            print(f"state {row + 1}/{total} ({rate:.0f} rows/s)", flush=True)
    np.save(args.out_dir / "state.npy", track)

    meta = {
        "protocol": PROTOCOL,
        "prep_job_id": os.environ.get("SLURM_JOB_ID"),
        "source": str(args.dataset),
        "source_head_sha256": file_digest(args.dataset),
        "num_rows": total,
        "episode_len": EPISODE_LEN,
        "num_episodes": total // EPISODE_LEN,
        "pixel_source": "rerender",
        "pixel_shape": list(pixel_shape),
        "pixel_stride": args.pixel_stride,
        "num_grid_rows": int(grid_rows(total, args.pixel_stride).size),
        "render_size": args.render_size,
        "render_quality": args.render_quality,
        "render_info": render_info,
        "camera": args.camera,
        "state_columns": list(STATE_COLUMNS),
        "joint_index": offline.index.to_json(),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, sort_keys=True))


def stage_render(args: argparse.Namespace) -> None:
    meta = json.loads((args.out_dir / "meta.json").read_text())
    if meta["pixel_stride"] != args.pixel_stride:
        raise RuntimeError("pixel stride disagrees with the prepared cache")
    if meta["render_quality"] != args.render_quality:
        raise RuntimeError("render quality disagrees with the prepared cache")
    total = int(meta["num_rows"])

    qpos = np.load(args.out_dir / "qpos.npy", mmap_mode="r")
    qvel = np.load(args.out_dir / "qvel.npy", mmap_mode="r")
    buttons = np.load(args.out_dir / "button_states.npy", mmap_mode="r")
    pixels = np.load(args.out_dir / "pixels.npy", mmap_mode="r+")

    rows = grid_rows(total, args.pixel_stride)
    shard = rows[args.shard_index :: args.num_shards]
    print(f"shard {args.shard_index}/{args.num_shards}: {shard.size} rows", flush=True)

    _world, raw, render_info = make_scene_env(
        render_size=args.render_size, quality=args.render_quality, camera=args.camera
    )
    if render_info != meta["render_info"]:
        raise RuntimeError(
            f"renderer configuration drifted from the prepared cache: "
            f"{render_info} != {meta['render_info']}"
        )
    print(f"render info {render_info}", flush=True)

    prev_buttons = None
    started = time.time()
    for done, row in enumerate(shard):
        row = int(row)
        # set_state re-applies button colouring and re-runs mj_forward; when the
        # buttons have not changed the cheaper path is identical in effect
        current = (int(buttons[row][0]), int(buttons[row][1]))
        if current == prev_buttons:
            restore_state_fast(raw, qpos[row], qvel[row])
        else:
            restore_state(raw, qpos[row], qvel[row], buttons[row])
            prev_buttons = current
        pixels[row] = np.asarray(raw.render(camera=args.camera), dtype=np.uint8)
        if (done + 1) % 2000 == 0 or done + 1 == shard.size:
            rate = (done + 1) / max(time.time() - started, 1e-6)
            print(f"render {done + 1}/{shard.size} ({rate:.1f} rows/s)", flush=True)
    pixels.flush()
    print(
        json.dumps(
            {
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "rows": int(shard.size),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "seconds": time.time() - started,
            },
            sort_keys=True,
        )
    )


def stage_finalize(args: argparse.Namespace) -> None:
    meta = json.loads((args.out_dir / "meta.json").read_text())
    total = int(meta["num_rows"])
    stride = int(meta["pixel_stride"])
    pixels = np.load(args.out_dir / "pixels.npy", mmap_mode="r")
    rows = grid_rows(total, stride)

    blank = []
    checked = 0
    block = 20000
    for start in range(0, rows.size, block):
        chunk = rows[start : start + block]
        sums = pixels[chunk].reshape(chunk.size, -1).max(axis=1)
        blank.extend(int(r) for r in chunk[sums == 0])
        checked += int(chunk.size)
        print(f"checked {checked}/{rows.size}", flush=True)

    complete = len(blank) == 0
    report = {
        "protocol": PROTOCOL,
        "finalize_job_id": os.environ.get("SLURM_JOB_ID"),
        "num_grid_rows": int(rows.size),
        "num_blank_rows": len(blank),
        "blank_rows_head": blank[:20],
        "complete": complete,
    }
    (args.out_dir / "cache_complete.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))
    if not complete:
        raise RuntimeError(f"{len(blank)} grid rows were never rendered")


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("cache building must run inside a Slurm compute job")
    {"prep": stage_prep, "render": stage_render, "finalize": stage_finalize}[args.stage](args)


if __name__ == "__main__":
    main()
