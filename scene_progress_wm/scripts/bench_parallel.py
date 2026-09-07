"""Does render throughput scale with worker processes, and does osmesa beat egl?

`bench_render2` showed the 164 ms render is flat in output size (16 px costs 140 ms,
64 px costs 164 ms), so the cost is per-call overhead rather than rasterisation work.
That leaves two levers: a different GL backend, and process-level parallelism. This
measures both.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time


def worker(rank: int, frames: int, gl: str, size: int, queue) -> None:
    os.environ["MUJOCO_GL"] = gl
    os.environ["PYOPENGL_PLATFORM"] = gl
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from scene_progress_wm.scripts.stage0_plumbing import make_env

    try:
        _world, raw = make_env(size)
        raw.render(camera="front_pixels")
        start = time.perf_counter()
        for _ in range(frames):
            raw.render(camera="front_pixels")
        queue.put({"rank": rank, "seconds": time.perf_counter() - start, "frames": frames})
    except Exception as exc:  # noqa: BLE001 - reported, not raised, per worker
        queue.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})


def run(workers: int, frames: int, gl: str, size: int) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(rank, frames, gl, size, queue))
        for rank in range(workers)
    ]
    start = time.perf_counter()
    for proc in procs:
        proc.start()
    results = [queue.get() for _ in procs]
    for proc in procs:
        proc.join()
    wall = time.perf_counter() - start

    errors = [r for r in results if "error" in r]
    if errors:
        return {"workers": workers, "gl": gl, "error": errors[0]["error"]}
    total_frames = sum(r["frames"] for r in results)
    per_worker = [r["frames"] / r["seconds"] for r in results]
    return {
        "workers": workers,
        "gl": gl,
        "wall_seconds": wall,
        "aggregate_fps": total_frames / wall,
        "per_worker_fps_mean": sum(per_worker) / len(per_worker),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--gl", type=str, nargs="+", default=["egl", "osmesa"])
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("bench must run inside a Slurm compute job")

    report = []
    for gl in args.gl:
        for workers in args.workers:
            result = run(workers, args.frames, gl, args.size)
            report.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if "error" in result:
                break  # a backend that cannot start will not start with more workers
    print(json.dumps({"report": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
