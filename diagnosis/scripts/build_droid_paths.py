"""Build the DROID episode-paths CSV the dataloader needs, with a length filter.

DROIDVideoDataset.loadvideo_decord samples ``frames_per_clip * ceil(vfps/fps)``
consecutive source frames from each episode. Episodes shorter than that raise
"video too short", which the dataset's internal retry loop turns into an
infinite loop when too many short clips are present. We therefore pre-filter:
only episodes whose ``trajectory.h5`` has >= the required number of timesteps
(plus a small margin) are written to the CSV.

Each kept line is ``<abs episode dir> <index>`` (space-separated), matching
``src/scripts/generate_droid_paths.py`` and ``pd.read_csv(..., delimiter=' ')``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from math import ceil
from pathlib import Path

import h5py
import numpy as np


def episode_length(traj_h5: str) -> int:
    with h5py.File(traj_h5, "r") as f:
        return int(np.asarray(f["observation"]["robot_state"]["cartesian_position"]).shape[0])


def wrist_serial_present(ep_dir: str) -> bool:
    """Confirm the wrist MP4 referenced by metadata is actually on disk."""
    metas = glob.glob(os.path.join(ep_dir, "metadata_*.json"))
    if not metas:
        return False
    try:
        meta = json.load(open(metas[0]))
        wrist = meta.get("wrist_mp4_path", "")
        mp4_name = wrist.split("recordings/MP4/")[-1]
        return bool(mp4_name) and os.path.exists(os.path.join(ep_dir, "recordings", "MP4", mp4_name))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_root", default="data/droid_subset",
                    help="dir mirroring <lab>/success/<date>/<episode>/trajectory.h5")
    ap.add_argument("--output", default="data/droid_subset/droid_paths.csv")
    ap.add_argument("--frames_per_clip", type=int, default=8)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--vfps", type=int, default=60, help="source video fps (DROID wrist = 60)")
    ap.add_argument("--margin", type=int, default=5, help="extra frames required above the minimum")
    args = ap.parse_args()

    need = args.frames_per_clip * ceil(args.vfps / args.fps) + args.margin
    print(f"Required episode length >= {need} timesteps "
          f"(fpc={args.frames_per_clip}, fps={args.fps}, vfps={args.vfps})")

    traj_files = sorted(glob.glob(os.path.join(args.subset_root, "**", "trajectory.h5"), recursive=True))
    print(f"Found {len(traj_files)} episodes with trajectory.h5")

    kept, lengths, grip_min, grip_max = [], [], np.inf, -np.inf
    too_short = no_wrist = bad = 0
    for tf in traj_files:
        ep_dir = os.path.dirname(tf)
        try:
            n = episode_length(tf)
        except Exception:
            bad += 1
            continue
        if n < need:
            too_short += 1
            continue
        if not wrist_serial_present(ep_dir):
            no_wrist += 1
            continue
        # track gripper range for a sanity report (regime threshold is 0.2)
        try:
            with h5py.File(tf, "r") as f:
                g = np.asarray(f["observation"]["robot_state"]["gripper_position"])
            grip_min, grip_max = min(grip_min, float(g.min())), max(grip_max, float(g.max()))
        except Exception:
            pass
        kept.append(os.path.abspath(ep_dir))
        lengths.append(n)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, p in enumerate(kept):
            f.write(f"{p} {i}\n")

    print(f"Kept {len(kept)} episodes  (too_short={too_short}, no_wrist={no_wrist}, unreadable={bad})")
    if lengths:
        a = np.array(lengths)
        print(f"  length: min={a.min()} median={int(np.median(a))} max={a.max()}")
        print(f"  ~transitions (fpc-1 each): {len(kept) * (args.frames_per_clip - 1)}")
        print(f"  gripper raw range across kept eps: [{grip_min:.3f}, {grip_max:.3f}]")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
