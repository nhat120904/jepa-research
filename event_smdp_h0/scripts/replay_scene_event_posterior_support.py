#!/usr/bin/env python3
"""Replay persisted task-5 plans and measure true-q posterior support."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.scene_core import SKILLS, initial_milestones  # noqa: E402
from event_smdp_h0.scene_event_perception import EventObserverEvaluator  # noqa: E402
from event_smdp_h0.scene_learning import goal_feature  # noqa: E402
from event_smdp_h0.scripts.collect_scene_h1 import encode_images, resize_render  # noqa: E402
from event_smdp_h0.scripts.run_scene_gate0 import (  # noqa: E402
    SceneSnapshotManager,
    SkillLibrary,
    make_world,
)


PROTOCOL = "scene_event_posterior_support_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-seed", type=int, required=True)
    parser.add_argument("--observer-seed", type=int, default=1)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--observer-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def joint_rank(
    cube_probability: np.ndarray,
    window_probability: np.ndarray,
    stable_probability: float,
    true_cube: int,
    true_window: int,
    true_stable: bool,
) -> tuple[float, int]:
    stable = np.asarray([1.0 - stable_probability, stable_probability])
    joint = (
        cube_probability[:, None, None]
        * window_probability[None, :, None]
        * stable[None, None, :]
    )
    true_probability = float(joint[true_cube, true_window, int(true_stable)])
    rank = 1 + int(np.sum(joint > true_probability))
    return true_probability, rank


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("posterior replay must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("posterior replay requires a GPU allocation")
    if args.observer_seed != 1:
        raise ValueError("locked primary replay uses observer seed 1")
    source = json.loads(args.source_result.read_text())
    if source.get("protocol") != "scene_event_perception_replication_v1":
        raise ValueError("unexpected source-result protocol")
    if int(source["task_id"]) != 5 or int(source["reset_seed"]) != args.reset_seed:
        raise ValueError("source result does not match locked task/reset")
    source_rows = [
        row
        for row in source["results"]
        if row["arm"] == "learned_latent"
        and int(row["observer_seed"]) == args.observer_seed
    ]
    if len(source_rows) != 1:
        raise ValueError("source result lacks one seed-1 learned trajectory")
    source_row = source_rows[0]

    observer = EventObserverEvaluator(args.observer_checkpoint, device="cuda")
    import stable_worldmodel as swm

    visual_model: Any = swm.wm.utils.load_pretrained(args.visual_checkpoint).cuda().eval()
    visual_model.requires_grad_(False)
    if hasattr(visual_model, "interpolate_pos_encoding"):
        visual_model.interpolate_pos_encoding = True

    world, raw = make_world(5, args.reset_seed)
    try:
        snapshots = SceneSnapshotManager(raw)
        library = SkillLibrary(raw, stable_dwell=3)
        root = snapshots.capture()
        snapshots.restore(root)
        goal = goal_feature(raw)
        true_state = initial_milestones(5)
        records: list[dict[str, object]] = []
        for source_replan in source_row["replans"]:
            expected = source_replan["true_state"]
            actual_key = (
                true_state.cube_stage,
                true_state.window_stage,
                true_state.stable_count,
            )
            expected_key = (
                int(expected["cube_stage"]),
                int(expected["window_stage"]),
                int(expected["stable_count"]),
            )
            if actual_key != expected_key:
                raise RuntimeError(
                    f"replay q drift before decision {source_replan['decision']}: "
                    f"{actual_key} != {expected_key}"
                )
            feature = encode_images(visual_model, resize_render(raw)[None], batch_size=1)[0]
            observed_state, details = observer.predict(feature, goal, 5)
            cube_probability = np.asarray(details["cube_probability"], dtype=np.float64)
            window_probability = np.asarray(details["window_probability"], dtype=np.float64)
            stable_probability = float(details["stable_probability"])
            true_probability, rank = joint_rank(
                cube_probability,
                window_probability,
                stable_probability,
                true_state.cube_stage,
                true_state.window_stage,
                true_state.stable_success,
            )
            cube_true_probability = float(cube_probability[true_state.cube_stage])
            window_true_probability = float(window_probability[true_state.window_stage])
            cube_rank = 1 + int(np.sum(cube_probability > cube_true_probability))
            window_rank = 1 + int(np.sum(window_probability > window_true_probability))
            hard_correct = (
                observed_state.cube_stage == true_state.cube_stage
                and observed_state.window_stage == true_state.window_stage
                and observed_state.stable_success == true_state.stable_success
            )
            records.append(
                {
                    "decision": int(source_replan["decision"]),
                    "true_state": asdict(true_state),
                    "observed_state": asdict(observed_state),
                    "hard_correct": hard_correct,
                    "true_joint_probability": true_probability,
                    "true_joint_rank": rank,
                    "true_cube_probability": cube_true_probability,
                    "true_cube_rank": cube_rank,
                    "true_window_probability": window_true_probability,
                    "true_window_rank": window_rank,
                    "stable_probability": stable_probability,
                    "selected_skill": str(source_replan["selected_skill"]),
                }
            )
            skill_index = SKILLS.index(str(source_replan["selected_skill"]))
            true_state, _ = library.execute(skill_index, true_state)
        true_state, _ = library.hold(true_state, 3)
        replay_success = bool(true_state.stable_success)
        if replay_success != bool(source_row["success"]):
            raise RuntimeError(
                f"replay success drift: {replay_success} != {source_row['success']}"
            )
        output = {
            "protocol": PROTOCOL,
            "task_id": 5,
            "reset_seed": args.reset_seed,
            "observer_seed": args.observer_seed,
            "source_result": str(args.source_result),
            "source_success": bool(source_row["success"]),
            "replay_success": replay_success,
            "num_replans": len(records),
            "records": records,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.out_dir / "result.json"
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(output_path), "records": len(records)}))
    finally:
        world.close()


if __name__ == "__main__":
    main()
