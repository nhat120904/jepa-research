#!/usr/bin/env python3
"""Collect exact counterfactual skill transitions for the Scene H1 gate.

Each shard is one fresh reset.  Canonical paths create roots at every task
milestone; every one of the seven task-agnostic skills is then executed from
the exact same root snapshot.  Both frozen DINO/LeWM embeddings and privileged
state vectors are saved so representation failure is distinguishable from
head/data failure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from event_smdp_h0.scene_core import (  # noqa: E402
    SKILLS,
    ScenePredicates,
    initial_milestones,
)
from event_smdp_h0.scene_learning import (  # noqa: E402
    goal_feature,
    predicate_vector,
    raw_state_feature,
)
from event_smdp_h0.scripts.run_scene_gate0 import (  # noqa: E402
    SceneSnapshotManager,
    SkillLibrary,
    known_solution,
    make_world,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=(4, 5), required=True)
    parser.add_argument("--reset-seed", type=int, required=True)
    parser.add_argument("--split", choices=("smoke", "train", "val"), required=True)
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_paths(task_id: int) -> tuple[tuple[int, ...], ...]:
    primary = known_solution(task_id)
    if task_id == 4:
        return (primary,)
    by_name = {name: index for index, name in enumerate(SKILLS)}
    window_first = (
        by_name["toggle_button_1"],
        by_name["window_open"],
        by_name["toggle_button_1"],
        by_name["toggle_button_0"],
        by_name["drawer_open"],
        by_name["place_cube_in_drawer"],
        by_name["drawer_close"],
        by_name["toggle_button_0"],
    )
    return (primary, window_first)


def resize_render(raw_env: Any, size: int = 224) -> np.ndarray:
    from PIL import Image

    frame = np.asarray(raw_env.render())
    if frame.shape[:2] != (size, size):
        frame = np.asarray(Image.fromarray(frame).resize((size, size), Image.BILINEAR))
    return np.ascontiguousarray(frame, dtype=np.uint8)


def make_transform(size: int = 224):
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=size),
        ]
    )


@torch.inference_mode()
def encode_images(model: Any, frames: np.ndarray, batch_size: int) -> np.ndarray:
    from torchvision import tv_tensors

    transform = make_transform(224)
    outputs: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch = np.asarray(frames[start : start + batch_size])
        chw = np.transpose(batch, (0, 3, 1, 2))
        pixels = torch.stack(
            [transform(tv_tensors.Image(image)) for image in chw]
        )[:, None].cuda(non_blocking=True)
        encoded = model.encode({"pixels": pixels})["emb"][:, -1].float()
        outputs.append(encoded.reshape(len(encoded), -1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def q_fields(prefix: str, state: Any, row: dict[str, Any]) -> None:
    row[f"{prefix}_cube_stage"] = int(state.cube_stage)
    row[f"{prefix}_window_stage"] = int(state.window_stage)
    row[f"{prefix}_stable_count"] = int(min(state.stable_count, 3))


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Scene collection must run inside a Slurm compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("frozen visual encoding requires a GPU allocation")

    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    world, raw = make_world(args.task_id, args.reset_seed)
    rows: list[dict[str, Any]] = []
    before_frames: list[np.ndarray] = []
    after_frames: list[np.ndarray] = []
    before_raw: list[np.ndarray] = []
    after_raw: list[np.ndarray] = []
    repeat_check: dict[str, Any] = {}
    try:
        snapshots = SceneSnapshotManager(raw)
        library = SkillLibrary(raw, stable_dwell=3)
        initial_snapshot = snapshots.capture()
        initial_signature = snapshots.signature()
        goal = goal_feature(raw)

        for path_id, path in enumerate(canonical_paths(args.task_id)):
            snapshots.restore(initial_snapshot)
            state = initial_milestones(args.task_id)
            for root_index in range(len(path) + 1):
                root_snapshot = snapshots.capture()
                root_signature = snapshots.signature()
                root_predicates = library.predicates()
                root_frame = resize_render(raw)
                root_raw = raw_state_feature(raw)

                for skill_index, skill_name in enumerate(SKILLS):
                    snapshots.restore(root_snapshot)
                    next_state, record = library.execute(skill_index, state)
                    endpoint_signature = snapshots.signature()
                    endpoint_frame = resize_render(raw)
                    endpoint_raw = raw_state_feature(raw)
                    endpoint_predicates = library.predicates()
                    new_events = tuple(record["new_events"])
                    row: dict[str, Any] = {
                        "task_id": args.task_id,
                        "reset_seed": args.reset_seed,
                        "path_id": path_id,
                        "root_index": root_index,
                        "skill": skill_index,
                        "duration": int(record["env_steps"]),
                        "no_effect": int(
                            not new_events
                            and not endpoint_predicates.native_success
                            and next_state.cube_stage == state.cube_stage
                            and next_state.window_stage == state.window_stage
                        ),
                        "native_success_after": int(endpoint_predicates.native_success),
                        "stable_success_after": int(next_state.stable_success),
                        "root_signature": root_signature,
                        "endpoint_signature": endpoint_signature,
                        "new_events": list(new_events),
                        "before_predicates": asdict(root_predicates),
                        "after_predicates": asdict(endpoint_predicates),
                        "skill_name": skill_name,
                    }
                    q_fields("before", state, row)
                    q_fields("after", next_state, row)
                    rows.append(row)
                    before_frames.append(root_frame)
                    after_frames.append(endpoint_frame)
                    before_raw.append(root_raw)
                    after_raw.append(endpoint_raw)

                    if not repeat_check:
                        first_duration = int(record["env_steps"])
                        first_q = (
                            next_state.cube_stage,
                            next_state.window_stage,
                            next_state.stable_count,
                        )
                        snapshots.restore(root_snapshot)
                        repeat_state, repeat_record = library.execute(skill_index, state)
                        repeat_frame = resize_render(raw)
                        repeat_check = {
                            "duration_equal": first_duration == int(repeat_record["env_steps"]),
                            "milestone_equal": first_q
                            == (
                                repeat_state.cube_stage,
                                repeat_state.window_stage,
                                repeat_state.stable_count,
                            ),
                            "endpoint_signature_equal": endpoint_signature
                            == snapshots.signature(),
                            "pixel_max_abs": int(
                                np.abs(
                                    endpoint_frame.astype(np.int16)
                                    - repeat_frame.astype(np.int16)
                                ).max()
                            ),
                        }

                if root_index < len(path):
                    snapshots.restore(root_snapshot)
                    state, _ = library.execute(path[root_index], state)
                else:
                    # Counterfactual skill enumeration leaves the simulator at
                    # the last tested endpoint; restore the canonical full-path
                    # root before the final stable-success support hold.
                    snapshots.restore(root_snapshot)

            state, _ = library.hold(state, 3)
            if not state.stable_success:
                raise RuntimeError(f"canonical path {path_id} failed its support check")

        if not all(
            repeat_check[key]
            for key in ("duration_equal", "milestone_equal", "endpoint_signature_equal")
        ) or repeat_check["pixel_max_abs"] != 0:
            raise RuntimeError(f"counterfactual repeatability failed: {repeat_check}")

        before_images = np.stack(before_frames)
        after_images = np.stack(after_frames)
        latent = encode_images(
            model, np.concatenate([before_images, after_images]), args.encode_batch
        )
        n = len(rows)
        before_latent, after_latent = latent[:n], latent[n:]

        arrays = {
            "task_id": np.asarray([row["task_id"] for row in rows], dtype=np.int64),
            "reset_seed": np.asarray([row["reset_seed"] for row in rows], dtype=np.int64),
            "path_id": np.asarray([row["path_id"] for row in rows], dtype=np.int64),
            "root_index": np.asarray([row["root_index"] for row in rows], dtype=np.int64),
            "skill": np.asarray([row["skill"] for row in rows], dtype=np.int64),
            "duration": np.asarray([row["duration"] for row in rows], dtype=np.float32),
            "no_effect": np.asarray([row["no_effect"] for row in rows], dtype=np.float32),
            "native_success_after": np.asarray(
                [row["native_success_after"] for row in rows], dtype=np.float32
            ),
            "stable_success_after": np.asarray(
                [row["stable_success_after"] for row in rows], dtype=np.float32
            ),
            "before_cube_stage": np.asarray(
                [row["before_cube_stage"] for row in rows], dtype=np.int64
            ),
            "before_window_stage": np.asarray(
                [row["before_window_stage"] for row in rows], dtype=np.int64
            ),
            "before_stable_count": np.asarray(
                [row["before_stable_count"] for row in rows], dtype=np.int64
            ),
            "after_cube_stage": np.asarray(
                [row["after_cube_stage"] for row in rows], dtype=np.int64
            ),
            "after_window_stage": np.asarray(
                [row["after_window_stage"] for row in rows], dtype=np.int64
            ),
            "after_stable_count": np.asarray(
                [row["after_stable_count"] for row in rows], dtype=np.int64
            ),
            "after_predicates": np.stack(
                [
                    predicate_vector(ScenePredicates(**row["after_predicates"]))
                    for row in rows
                ]
            ),
            "goal": np.repeat(goal[None], n, axis=0).astype(np.float32),
            "before_privileged": np.stack(before_raw).astype(np.float32),
            "after_privileged": np.stack(after_raw).astype(np.float32),
            "before_latent": before_latent,
            "after_latent": after_latent,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_dir / "transitions.npz", **arrays)
        metadata = {
            "protocol": "scene_h1_counterfactual_v1",
            "split": args.split,
            "task_id": args.task_id,
            "reset_seed": args.reset_seed,
            "num_rows": n,
            "num_paths": len(canonical_paths(args.task_id)),
            "skills": list(SKILLS),
            "checkpoint": args.checkpoint,
            "latent_dim": int(before_latent.shape[1]),
            "privileged_dim": int(arrays["before_privileged"].shape[1]),
            "goal_dim": int(goal.size),
            "initial_signature": initial_signature,
            "repeatability": repeat_check,
            "npz_sha256": hashlib.sha256(
                (args.out_dir / "transitions.npz").read_bytes()
            ).hexdigest(),
        }
        (args.out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(metadata, sort_keys=True))
    finally:
        world.close()


if __name__ == "__main__":
    main()
