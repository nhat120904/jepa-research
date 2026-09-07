"""Is the 164 ms render fixed overhead or rasterisation work?

Sweeps output size and the scene render flags. If time scales with pixel count the
renderer is rasterisation-bound and the quality flags will pay; if it is flat the cost is
context/readback overhead and the only lever is process-level parallelism.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.scripts.stage0_plumbing import make_env  # noqa: E402


def timeit(fn, n: int) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("bench must run inside a Slurm compute job")

    import mujoco

    result = {"gl": os.environ.get("MUJOCO_GL"), "sizes": {}, "flags": {}, "offsamples": {}}

    _world, raw = make_env(64)
    model = raw._model
    print(f"offsamples={model.vis.quality.offsamples} "
          f"offwidth={model.vis.global_.offwidth} offheight={model.vis.global_.offheight}",
          flush=True)
    result["model_offsamples"] = int(model.vis.quality.offsamples)
    result["model_offwidth"] = int(model.vis.global_.offwidth)
    result["model_offheight"] = int(model.vis.global_.offheight)
    result["model_shadowsize"] = int(model.vis.quality.shadowsize)
    result["model_nlight"] = int(model.nlight)
    result["shadowsize"] = {}

    # -- size sweep with independent renderers on the same model
    for size in (16, 32, 64):
        renderer = mujoco.Renderer(model, height=size, width=size)
        renderer.update_scene(data=raw._data, camera="front_pixels")
        result["sizes"][str(size)] = timeit(renderer.render, args.n)
        renderer.close()
        print(f"size {size}: {result['sizes'][str(size)]:.2f} ms", flush=True)

    # -- render-flag sweep at 64
    renderer = mujoco.Renderer(model, height=64, width=64)
    renderer.update_scene(data=raw._data, camera="front_pixels")
    flag_sets = {
        "baseline": [],
        "no_shadow": [mujoco.mjtRndFlag.mjRND_SHADOW],
        "no_shadow_reflection": [
            mujoco.mjtRndFlag.mjRND_SHADOW,
            mujoco.mjtRndFlag.mjRND_REFLECTION,
        ],
        "no_shadow_reflection_skybox": [
            mujoco.mjtRndFlag.mjRND_SHADOW,
            mujoco.mjtRndFlag.mjRND_REFLECTION,
            mujoco.mjtRndFlag.mjRND_SKYBOX,
        ],
    }
    touched = flag_sets["no_shadow_reflection_skybox"]
    for name, flags in flag_sets.items():
        for flag in touched:
            renderer.scene.flags[flag] = 1
        for flag in flags:
            renderer.scene.flags[flag] = 0
        result["flags"][name] = timeit(renderer.render, args.n)
        print(f"flags {name}: {result['flags'][name]:.2f} ms", flush=True)
    renderer.close()

    # -- multisampling sweep (needs a fresh renderer after touching the model)
    for samples in (0, 4):
        model.vis.quality.offsamples = samples
        renderer = mujoco.Renderer(model, height=64, width=64)
        renderer.update_scene(data=raw._data, camera="front_pixels")
        for flag in (
            mujoco.mjtRndFlag.mjRND_SHADOW,
            mujoco.mjtRndFlag.mjRND_REFLECTION,
        ):
            renderer.scene.flags[flag] = 0
        result["offsamples"][str(samples)] = timeit(renderer.render, args.n)
        renderer.close()
        print(f"offsamples {samples}: {result['offsamples'][str(samples)]:.2f} ms", flush=True)

    # -- shadow-map resolution sweep: a shadow pass is independent of output size,
    #    which is exactly the flat-in-pixels signature the first sweep showed
    model.vis.quality.offsamples = 4
    for shadowsize in (4096, 1024, 256):
        model.vis.quality.shadowsize = shadowsize
        renderer = mujoco.Renderer(model, height=64, width=64)
        renderer.update_scene(data=raw._data, camera="front_pixels")
        result["shadowsize"][str(shadowsize)] = timeit(renderer.render, args.n)
        renderer.close()
        print(f"shadowsize {shadowsize}: {result['shadowsize'][str(shadowsize)]:.2f} ms", flush=True)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
