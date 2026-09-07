"""Stage 0 plumbing gate for the Scene progress-WM arena.

Three checks, all of which must pass before any model is trained:

1. **Frame reproduction** -- restoring a dataset row into ``swm/OGBScene-v0``
   and rendering must reproduce the pixels stored in the released visual Scene
   play dataset.  If it does not, the fallback is to re-render our own frames
   from ``qpos``; the report carries the numbers needed to make that call.
2. **Offline state readout** -- cube / drawer / window / drawer-site resolved
   from a stored ``qpos`` by forward kinematics alone must equal what the live
   environment reports after the same row is restored.
3. **Goal-relative success wiring** -- with the four target families set from a
   goal row, ``SceneEnv._compute_successes`` must fire on that row and must not
   fire on the earlier row it is paired with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.scene_data import (  # noqa: E402
    EPISODE_LEN,
    OfflineSceneState,
    PLAY_TRAIN,
    goal_match,
    load_small,
    stream_rows,
)

PROTOCOL = "scene_progress_wm_stage0_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=PLAY_TRAIN)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--num-pairs", type=int, default=12)
    parser.add_argument("--goal-offset", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--render-size", type=int, default=64)
    parser.add_argument("--camera", type=str, default="front_pixels")
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def make_env(render_size: int):
    import mujoco
    import stable_worldmodel as swm

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
        reward_task_id=4,
        width=render_size,
        height=render_size,
    )
    raw = world.envs.envs[0].unwrapped
    raw.reset(seed=0, options={"variation": []})
    raw._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    return world, raw


def restore(raw, qpos, qvel, buttons) -> None:
    import mujoco

    raw.set_state(
        np.asarray(qpos, dtype=np.float64),
        np.asarray(qvel, dtype=np.float64),
        button_state_0=int(buttons[0]),
        button_state_1=int(buttons[1]),
    )
    mujoco.mj_forward(raw._model, raw._data)


def env_state(raw) -> dict[str, float | np.ndarray]:
    return {
        "cube_pos": np.asarray(
            raw._data.joint("object_joint_0").qpos[:3], dtype=np.float64
        ).copy(),
        "drawer": float(raw._data.joint("drawer_slide").qpos[0]),
        "window": float(raw._data.joint("window_slide").qpos[0]),
        "drawer_site_y": float(raw._data.site_xpos[raw._drawer_site_id][1]),
    }


def set_goal_targets(raw, goal_state, goal_buttons) -> None:
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


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("stage 0 must run inside a Slurm compute job")

    rng = np.random.default_rng(args.seed)
    horizon = args.episodes * EPISODE_LEN

    frame_rows = np.sort(
        rng.choice(horizon, size=args.num_frames, replace=False).astype(np.int64)
    )
    starts: list[int] = []
    while len(starts) < args.num_pairs:
        candidate = int(rng.integers(0, horizon))
        if (candidate % EPISODE_LEN) + args.goal_offset >= EPISODE_LEN:
            continue
        if candidate in starts:
            continue
        starts.append(candidate)
    start_rows = np.array(sorted(starts), dtype=np.int64)
    goal_rows = start_rows + args.goal_offset

    needed = np.unique(np.concatenate([frame_rows, start_rows, goal_rows]))

    qpos_all = load_small(args.dataset, "qpos")
    qvel_all = load_small(args.dataset, "qvel")
    buttons_all = load_small(args.dataset, "button_states")
    pixels = stream_rows(args.dataset, "observations", needed)
    row_to_slot = {int(r): i for i, r in enumerate(needed)}

    world, raw = make_env(args.render_size)
    offline = OfflineSceneState(raw)

    # ---------------------------------------------------------------- check 1
    frame_records = []
    for row in frame_rows:
        restore(raw, qpos_all[row], qvel_all[row], buttons_all[row])
        rendered = np.asarray(raw.render(camera=args.camera))
        stored = pixels[row_to_slot[int(row)]]
        record = {
            "row": int(row),
            "rendered_shape": list(rendered.shape),
            "stored_shape": list(stored.shape),
        }
        if rendered.shape == stored.shape:
            diff = np.abs(rendered.astype(np.int16) - stored.astype(np.int16))
            record["max_abs_diff"] = int(diff.max())
            record["mean_abs_diff"] = float(diff.mean())
            record["frac_pixels_exact"] = float((diff == 0).mean())
        else:
            record["max_abs_diff"] = None
            record["mean_abs_diff"] = None
            record["frac_pixels_exact"] = None
        frame_records.append(record)

    shapes_match = all(r["rendered_shape"] == r["stored_shape"] for r in frame_records)
    max_abs = max(r["max_abs_diff"] for r in frame_records) if shapes_match else None
    mean_abs = (
        float(np.mean([r["mean_abs_diff"] for r in frame_records]))
        if shapes_match
        else None
    )

    # ---------------------------------------------------------------- check 2
    state_records = []
    for row in frame_rows:
        restore(raw, qpos_all[row], qvel_all[row], buttons_all[row])
        live = env_state(raw)
        off = offline.read(qpos_all[row])
        state_records.append(
            {
                "row": int(row),
                "cube_max_abs": float(np.abs(live["cube_pos"] - off["cube_pos"]).max()),
                "drawer_abs": abs(live["drawer"] - off["drawer"]),
                "window_abs": abs(live["window"] - off["window"]),
                "drawer_site_y_abs": abs(live["drawer_site_y"] - off["drawer_site_y"]),
                "in_drawer_agree": bool(
                    raw._is_in_drawer(live["cube_pos"])
                    == offline.is_in_drawer(off["cube_pos"], off["drawer_site_y"])
                ),
            }
        )
    state_max = {
        key: max(r[key] for r in state_records)
        for key in ("cube_max_abs", "drawer_abs", "window_abs", "drawer_site_y_abs")
    }
    in_drawer_all_agree = all(r["in_drawer_agree"] for r in state_records)

    # ---------------------------------------------------------------- check 3
    pair_records = []
    for start, goal in zip(start_rows, goal_rows):
        goal_off = offline.read(qpos_all[goal])
        set_goal_targets(raw, goal_off, buttons_all[goal])

        restore(raw, qpos_all[goal], qvel_all[goal], buttons_all[goal])
        at_goal = env_successes(raw)

        restore(raw, qpos_all[start], qvel_all[start], buttons_all[start])
        at_start = env_successes(raw)

        start_off = offline.read(qpos_all[start])
        predicted_start = goal_match(
            start_off, buttons_all[start], goal_off, buttons_all[goal]
        ).to_json()
        predicted_goal = goal_match(
            goal_off, buttons_all[goal], goal_off, buttons_all[goal]
        ).to_json()

        pair_records.append(
            {
                "start": int(start),
                "goal": int(goal),
                "env_at_goal": at_goal,
                "env_at_start": at_start,
                "offline_at_goal": predicted_goal,
                "offline_at_start": predicted_start,
                "goal_agrees": at_goal == predicted_goal,
                "start_agrees": at_start == predicted_start,
            }
        )

    goal_success_rate = float(
        np.mean([p["env_at_goal"]["success"] for p in pair_records])
    )
    start_success_rate = float(
        np.mean([p["env_at_start"]["success"] for p in pair_records])
    )
    offline_agrees = all(p["goal_agrees"] and p["start_agrees"] for p in pair_records)

    # ------------------------------------------------------------------ gates
    frames_reproduce = bool(shapes_match and max_abs is not None and max_abs <= 2)
    state_exact = bool(
        state_max["cube_max_abs"] <= 1e-9
        and state_max["drawer_abs"] <= 1e-9
        and state_max["window_abs"] <= 1e-9
        and state_max["drawer_site_y_abs"] <= 1e-9
        and in_drawer_all_agree
    )
    success_wired = bool(goal_success_rate == 1.0 and offline_agrees)

    if not state_exact:
        verdict = "OFFLINE_STATE_MISMATCH"
    elif not success_wired:
        verdict = "SUCCESS_WIRING_BROKEN"
    elif not frames_reproduce:
        verdict = "RERENDER_REQUIRED"
    else:
        verdict = "STAGE0_PASS"

    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "dataset": str(args.dataset),
        "dataset_head_sha256": hashlib.sha256(
            args.dataset.open("rb").read(1 << 20)
        ).hexdigest(),
        "render": {
            "camera": args.camera,
            "size": args.render_size,
            "shapes_match": shapes_match,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "frames_reproduce": frames_reproduce,
        },
        "offline_state": {
            "max_abs": state_max,
            "in_drawer_all_agree": in_drawer_all_agree,
            "exact": state_exact,
            "joint_index": offline.index.to_json(),
        },
        "goal_success": {
            "goal_offset": args.goal_offset,
            "num_pairs": int(len(pair_records)),
            "env_success_rate_at_goal": goal_success_rate,
            "env_success_rate_at_start": start_success_rate,
            "offline_agrees_with_env": offline_agrees,
            "wired": success_wired,
        },
        "frames": frame_records,
        "states": state_records,
        "pairs": pair_records,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stage0.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "render": summary["render"],
                "offline_state": {"exact": state_exact, "max_abs": state_max},
                "goal_success": summary["goal_success"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
