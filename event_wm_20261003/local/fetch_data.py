"""Download the OGBench play datasets used by event_wm from the HF mirror (resumable, idempotent).

    python local/fetch_data.py                 # everything in GROUPS
    python local/fetch_data.py --groups state_unified visual_cube_scene
    python local/fetch_data.py --list          # print files and sizes only

Same source as slurm/fetch_state.sh and slurm/fetch_scene.sh (HF dataset ryanhoangt/ogbench_data).
Files land in $EVENT_WM_DATA (default E:\\jepa-data\\ogbench\\data) as <name>.npz and <name>-val.npz,
which is the layout every script expects (--data <dir> --env <name>).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = "ryanhoangt/ogbench_data"
GROUPS = {
    # STATE track of the unified method (cube / puzzle / scene) and the generic front end.
    "state_unified": ["cube-triple-play-v0", "puzzle-4x5-play-v0", "scene-play-v0"],
    # Grid-size curve for the puzzle (README section 5, "not yet done").
    "state_puzzle_sizes": ["puzzle-3x3-play-v0", "puzzle-4x4-play-v0", "puzzle-4x6-play-v0"],
    # Scene memory v1 (sm_*.py) and the pixel front ends.
    "visual_cube_scene": ["visual-cube-triple-play-v0", "visual-scene-play-v0"],
    # Pixel track of the puzzle (encoder / SFA code / reader / skill v3).
    "visual_puzzle": ["visual-puzzle-3x3-play-v0", "visual-puzzle-4x4-play-v0",
                      "visual-puzzle-4x5-play-v0", "visual-puzzle-4x6-play-v0"],
    # Held-out cube environment of the object method (never used in development).
    "visual_cube_single": ["visual-cube-single-play-v0"],
    # The full OGBench pixel cube suite (play) for held-out evaluation (user approval 2026-10-09).
    "visual_cube_all": ["visual-cube-single-play-v0", "visual-cube-double-play-v0",
                        "visual-cube-triple-play-v0", "visual-cube-quadruple-play-v0"],
    # State-based cube and puzzle datasets, play and noisy (small; reference / privileged analysis).
    "state_cube_puzzle_all": [f"{e}-{d}-v0" for e in ("cube-single", "cube-double", "cube-triple", "cube-quadruple",
                                                     "puzzle-3x3", "puzzle-4x4", "puzzle-4x5", "puzzle-4x6")
                              for d in ("play", "noisy")],
    # Pixel noisy datasets (~43 GB); not fetched by default (disk), the method trains on play data.
    "visual_noisy_cube_puzzle": [f"visual-{e}-noisy-v0" for e in ("cube-single", "cube-double", "cube-triple", "cube-quadruple",
                                                                 "puzzle-3x3", "puzzle-4x4", "puzzle-4x5", "puzzle-4x6")],
}
DEFAULT_GROUPS = [g for g in GROUPS if g != "visual_noisy_cube_puzzle"]


def data_dir() -> Path:
    return Path(os.environ.get("EVENT_WM_DATA", r"E:\jepa-data\ogbench\data"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, choices=list(GROUPS))
    ap.add_argument("--list", action="store_true", help="only list files and sizes")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", r"E:\jepa-data\hf_cache")   # keep the small C: drive clear
    from huggingface_hub import HfApi, hf_hub_download

    names = list(dict.fromkeys(f"{env}{suffix}.npz" for g in a.groups for env in GROUPS[g] for suffix in ("", "-val")))
    sizes = {f.path: f.size for f in HfApi().list_repo_tree(REPO, repo_type="dataset") if getattr(f, "size", None)}
    missing = [n for n in names if n not in sizes]
    if missing:
        print("NOT FOUND ON MIRROR:", missing, file=sys.stderr)
        return 2
    total = sum(sizes[n] for n in names)
    out = data_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(names)} files, {total / 1e9:.2f} GB -> {out}", flush=True)
    if a.list:
        for n in names:
            print(f"{sizes[n] / 1e9:7.2f} GB  {n}")
        return 0
    for i, n in enumerate(names, 1):
        dest = out / n
        if dest.exists() and dest.stat().st_size == sizes[n]:
            print(f"[{i}/{len(names)}] have {n}", flush=True)
            continue
        t = time.time()
        print(f"[{i}/{len(names)}] fetching {n} ({sizes[n] / 1e9:.2f} GB)", flush=True)
        hf_hub_download(REPO, n, repo_type="dataset", local_dir=str(out))
        got = dest.stat().st_size
        if got != sizes[n]:
            print(f"SIZE MISMATCH {n}: {got} != {sizes[n]}", file=sys.stderr)
            return 3
        print(f"[{i}/{len(names)}] done {n} in {time.time() - t:.0f}s", flush=True)
    print("ALL_DOWNLOADED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
