"""The one place that builds a Scene environment and configures its renderer.

Every frame in this program -- the training cache and the closed loop alike -- is drawn
through here, so train-time and plan-time appearance cannot drift apart.

Job 49480 measured, at 64x64 on these pods (software rasteriser, no GPU needed):

===========================  ========
setting                      ms/frame
===========================  ========
baseline                       158.6
shadows off                     54.8
shadows off, MSAA off           22.6
===========================  ========

The ``fast`` profile takes both. Reflections and the skybox cost nothing measurable and
are left on. ``offsamples`` must be set on the model **before** the renderer is built,
which is why this function forces renderer construction itself rather than leaving it to
the first ``render`` call.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

QUALITIES = ("full", "fast")
FAST_DISABLED_FLAGS = ("mjRND_SHADOW",)
FAST_OFFSAMPLES = 0


def make_scene_env(
    render_size: int = 64,
    quality: str = "fast",
    camera: str = "front_pixels",
    reward_task_id: int = 4,
    seed: int = 0,
):
    """Build ``swm/OGBScene-v0`` with a configured renderer.

    Returns ``(world, raw, render_info)``. ``render_info`` records exactly what was
    changed, so it can be stamped into every artifact.
    """
    import mujoco
    import stable_worldmodel as swm

    if quality not in QUALITIES:
        raise ValueError(f"unknown render quality: {quality}")

    world = swm.World(
        "swm/OGBScene-v0",
        num_envs=1,
        max_episode_steps=2000,
        add_pixels=False,
        ob_type="states",
        multiview=False,
        visualize_info=False,
        terminate_at_goal=False,
        mode="task",
        reward_task_id=reward_task_id,
        width=render_size,
        height=render_size,
    )
    raw = world.envs.envs[0].unwrapped
    raw.reset(seed=seed, options={"variation": []})
    raw._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)

    offsamples = int(raw._model.vis.quality.offsamples)
    disabled: list[str] = []
    if quality == "fast":
        raw._model.vis.quality.offsamples = FAST_OFFSAMPLES
        offsamples = FAST_OFFSAMPLES

    raw.render(camera=camera)  # construct the renderer under the chosen offsamples
    if quality == "fast":
        for name in FAST_DISABLED_FLAGS:
            raw._renderer.scene.flags[getattr(mujoco.mjtRndFlag, name)] = 0
            disabled.append(name)

    info = {
        "quality": quality,
        "offsamples": offsamples,
        "disabled_flags": disabled,
        "camera": camera,
        "render_size": render_size,
    }
    return world, raw, info


def restore_state(raw, qpos, qvel, buttons) -> None:
    """Exact reset from a dataset row, buttons included."""
    import mujoco

    raw.set_state(
        np.asarray(qpos, dtype=np.float64),
        np.asarray(qvel, dtype=np.float64),
        button_state_0=int(buttons[0]),
        button_state_1=int(buttons[1]),
    )
    mujoco.mj_forward(raw._model, raw._data)


def restore_state_fast(raw, qpos, qvel) -> None:
    """Reset when the button states are unchanged: identical in effect, 3x cheaper."""
    import mujoco

    raw._data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    raw._data.qvel[:] = np.asarray(qvel, dtype=np.float64)
    mujoco.mj_forward(raw._model, raw._data)


def set_goal_targets(raw, goal_state, goal_buttons) -> None:
    """Point the environment's own success predicate at a dataset goal row."""
    raw.set_cube_target_pos(0, np.asarray(goal_state["cube_pos"], dtype=np.float64))
    raw.set_target_button_state(0, int(goal_buttons[0]))
    raw.set_target_button_state(1, int(goal_buttons[1]))
    raw.set_target_drawer_pos(float(goal_state["drawer"]))
    raw.set_target_window_pos(float(goal_state["window"]))


def env_successes(raw) -> dict[str, bool]:
    cubes, buttons, drawer, window = raw._compute_successes()
    return {
        "cube": bool(cubes[0]),
        "button_0": bool(buttons[0]),
        "button_1": bool(buttons[1]),
        "drawer": bool(drawer),
        "window": bool(window),
        "success": bool(cubes[0] and buttons[0] and buttons[1] and drawer and window),
    }


__all__ = [
    "FAST_DISABLED_FLAGS",
    "FAST_OFFSAMPLES",
    "QUALITIES",
    "env_successes",
    "make_scene_env",
    "restore_state",
    "restore_state_fast",
    "set_goal_targets",
]
