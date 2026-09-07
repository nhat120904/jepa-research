"""Where the 8 rows/s goes: time each part of restore-and-render separately."""

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

from scene_progress_wm.scene_data import PLAY_TRAIN, load_small  # noqa: E402
from scene_progress_wm.scripts.stage0_plumbing import make_env  # noqa: E402


def timeit(fn, n: int) -> float:
    fn()  # warm
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--render-size", type=int, default=64)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("bench must run inside a Slurm compute job")

    import mujoco

    qpos = load_small(PLAY_TRAIN, "qpos")[:1000]
    qvel = load_small(PLAY_TRAIN, "qvel")[:1000]
    buttons = load_small(PLAY_TRAIN, "button_states")[:1000]

    _world, raw = make_env(args.render_size)
    counter = {"i": 0}

    def next_row() -> int:
        counter["i"] = (counter["i"] + 1) % 1000
        return counter["i"]

    def only_qpos():
        row = next_row()
        raw._data.qpos[:] = qpos[row]
        raw._data.qvel[:] = qvel[row]

    def qpos_forward():
        only_qpos()
        mujoco.mj_forward(raw._model, raw._data)

    def apply_buttons():
        raw._cur_button_states = list(buttons[next_row()])
        raw._apply_button_states()

    def full_set_state():
        row = next_row()
        raw.set_state(
            np.asarray(qpos[row], dtype=np.float64),
            np.asarray(qvel[row], dtype=np.float64),
            button_state_0=int(buttons[row][0]),
            button_state_1=int(buttons[row][1]),
        )

    def render_only():
        raw.render(camera="front_pixels")

    def update_scene_only():
        raw._renderer.update_scene(
            data=raw._data, camera="front_pixels", scene_option=raw._scene_option
        )

    def renderer_render_only():
        raw._renderer.render()

    raw.render(camera="front_pixels")  # force renderer init

    result = {
        "render_size": args.render_size,
        "n": args.n,
        "renderer_width": int(getattr(raw._renderer, "width", -1)),
        "renderer_height": int(getattr(raw._renderer, "height", -1)),
        "model_ngeom": int(raw._model.ngeom),
        "model_nbody": int(raw._model.nbody),
        "ms": {
            "set_qpos_only": timeit(only_qpos, args.n),
            "qpos_plus_mj_forward": timeit(qpos_forward, args.n),
            "apply_button_states": timeit(apply_buttons, args.n),
            "env_set_state": timeit(full_set_state, args.n),
            "render_full": timeit(render_only, args.n),
            "renderer_update_scene": timeit(update_scene_only, args.n),
            "renderer_render": timeit(renderer_render_only, args.n),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
