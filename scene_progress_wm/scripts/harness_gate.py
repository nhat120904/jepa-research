"""Open-loop replay gate: can the arena reach the goal at all?

Executing a trajectory's own recorded actions from its start row must land on its goal
row, because that is what the dataset says those actions do. If it does not, then a low
planning score would be a state-restoration or stepping defect in the arena rather than
a statement about any world model, and no planning number is interpretable.

This uses no model, no encoder and no pixels -- only reset, step, and OGBench's own
success predicate -- so it isolates the arena from everything downstream.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.scene_eval import (  # noqa: E402
    build_episode_specs,
    load_cache,
    state_row,
)
from scene_progress_wm.scene_render import (  # noqa: E402
    env_successes,
    make_scene_env,
    restore_state,
    set_goal_targets,
)

PROTOCOL = "scene_progress_wm_harness_gate_v1"
COMPONENTS = ("cube", "button_0", "button_1", "drawer", "window")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--goal-offsets", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--episode-seed", type=int, default=90100)
    parser.add_argument("--pass-rate", type=float, default=0.95)
    parser.add_argument(
        "--exact-offset",
        type=int,
        default=50,
        help="offsets at or below this must replay near-exactly for the arena to pass",
    )
    parser.add_argument("--screen-trivial", action="store_true")
    return parser.parse_args()


def replay(raw, cache, track, spec) -> dict:
    qpos, qvel, buttons, actions = (
        cache["qpos"],
        cache["qvel"],
        cache["button_states"],
        cache["actions"],
    )
    goal_state = state_row(track, spec.goal_row)
    goal_buttons = buttons[spec.goal_row]

    restore_state(raw, qpos[spec.start_row], qvel[spec.start_row], buttons[spec.start_row])
    set_goal_targets(raw, goal_state, goal_buttons)

    start = env_successes(raw)
    ever = {name: False for name in COMPONENTS}
    horizon = spec.goal_row - spec.start_row
    for offset in range(horizon):
        raw.step(np.asarray(actions[spec.start_row + offset], dtype=np.float32))
        current = env_successes(raw)
        for name in COMPONENTS:
            ever[name] = ever[name] or bool(current[name])
    final = env_successes(raw)

    # where the replay actually ended up, against where the dataset says it should be
    reached = {
        "cube_pos": np.asarray(raw._data.joint("object_joint_0").qpos[:3], dtype=np.float64),
        "drawer": float(raw._data.joint("drawer_slide").qpos[0]),
        "window": float(raw._data.joint("window_slide").qpos[0]),
    }
    return {
        "spec": spec.to_json(),
        "start_success": start["success"],
        "final": final,
        "ever": ever,
        "cube_error_m": float(
            np.linalg.norm(reached["cube_pos"] - np.asarray(goal_state["cube_pos"]))
        ),
        "drawer_error_m": abs(reached["drawer"] - float(goal_state["drawer"])),
        "window_error_m": abs(reached["window"] - float(goal_state["window"])),
    }


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("the harness gate must run inside a Slurm compute job")

    meta = json.loads((args.cache_dir / "meta.json").read_text())
    complete = json.loads((args.cache_dir / "cache_complete.json").read_text())
    if not complete["complete"]:
        raise RuntimeError("cache is incomplete")

    cache, track = load_cache(args.cache_dir)
    world, raw, render_info = make_scene_env(
        render_size=int(meta["render_size"]),
        quality=meta["render_quality"],
        camera=meta["camera"],
    )
    if render_info != meta["render_info"]:
        raise RuntimeError(f"renderer drifted from the cache: {render_info}")

    per_offset = {}
    for offset in args.goal_offsets:
        specs = build_episode_specs(
            num_rows=int(meta["num_rows"]),
            goal_offset=offset,
            num_episodes=args.num_episodes,
            seed=args.episode_seed,
            track=track if args.screen_trivial else None,
            buttons=cache["button_states"] if args.screen_trivial else None,
        )
        records = [replay(raw, cache, track, spec) for spec in specs]
        nontrivial = [r for r in records if not r["start_success"]]
        per_offset[str(offset)] = {
            "num_episodes": len(records),
            "num_trivial": len(records) - len(nontrivial),
            "final_success_rate": float(np.mean([r["final"]["success"] for r in records])),
            "component_final_rate": {
                name: float(np.mean([r["final"][name] for r in records]))
                for name in COMPONENTS
            },
            "component_ever_rate": {
                name: float(np.mean([r["ever"][name] for r in records]))
                for name in COMPONENTS
            },
            "cube_error_m_median": float(
                np.median([r["cube_error_m"] for r in records])
            ),
            "cube_error_m_max": float(max(r["cube_error_m"] for r in records)),
            "drawer_error_m_max": float(max(r["drawer_error_m"] for r in records)),
            "window_error_m_max": float(max(r["window_error_m"] for r in records)),
            "records": records,
        }
        print(
            json.dumps(
                {
                    "goal_offset": offset,
                    "final_success_rate": per_offset[str(offset)]["final_success_rate"],
                    "cube_error_m_median": per_offset[str(offset)]["cube_error_m_median"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # The replay rate is an arena *ceiling*, not a pass/fail line, and it legitimately
    # falls with horizon: the released dataset stores qpos/qvel in float32 while MuJoCo
    # integrates in float64, so a restored state is truncated and contact-rich rollouts
    # amplify the difference. A flat threshold across offsets would call that a defect.
    # A defect shows up at the SHORT offsets, where 25-50 steps leave no room for
    # divergence to matter; those are what the gate tests.
    short = [o for o in args.goal_offsets if o <= args.exact_offset]
    if not short:
        raise ValueError("the gate needs at least one offset at or below --exact-offset")
    worst_short = min(per_offset[str(o)]["final_success_rate"] for o in short)
    worst = min(v["final_success_rate"] for v in per_offset.values())
    verdict = "HARNESS_OK" if worst_short >= args.pass_rate else "HARNESS_BROKEN"

    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cache_dir": str(args.cache_dir),
        "cache_meta": meta,
        "render_info": render_info,
        "episode_seed": args.episode_seed,
        "num_episodes": args.num_episodes,
        "pass_rate": args.pass_rate,
        "worst_offset_success_rate": worst,
        "worst_short_offset_success_rate": worst_short,
        "exact_offset": args.exact_offset,
        "ceiling_by_offset": {
            k: v["final_success_rate"] for k, v in per_offset.items()
        },
        "per_offset": per_offset,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "harness_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "worst_offset_success_rate": worst,
                "worst_short_offset_success_rate": worst_short,
        "worst_short_offset_success_rate": worst_short,
        "exact_offset": args.exact_offset,
        "ceiling_by_offset": {
            k: v["final_success_rate"] for k, v in per_offset.items()
        },
                "per_offset": {
                    k: {
                        "final_success_rate": v["final_success_rate"],
                        "num_trivial": v["num_trivial"],
                        "cube_error_m_median": v["cube_error_m_median"],
                    }
                    for k, v in per_offset.items()
                },
            },
            sort_keys=True,
        )
    )
    world.close() if hasattr(world, "close") else None


if __name__ == "__main__":
    main()
