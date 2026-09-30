"""Released LeWM evaluation settings (le-wm/config/eval/*.yaml) for the four tasks.

Everything that touches the model, simulator or dataset imports lazily and must
run under Slurm.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

STAGE0 = Path(os.environ.get("STABLEWM_HOME", "/mnt/data/nhatnc129/jepa/lewm_stage0"))
BLOCK = 5
HORIZON = 5
GOAL_OFFSET = 25
BUDGET = 50
IMAGE = 224

TASKS = {
    "reacher": {
        "env": "swm/ReacherDMControl-v0",
        "env_kw": {"task": "qpos_match"},
        "repo": "quentinll/lewm-reacher",
        "data": str(STAGE0 / "downloads/lewm/lewm-reacher/extracted/reacher.h5"),
        "cache": ["action"],
        "callables": [
            {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}},
        ],
    },
    "pusht": {
        "env": "swm/PushT-v1",
        "env_kw": {},
        "repo": "quentinll/lewm-pusht",
        "data": "pusht_expert_train.h5",
        "cache": ["action", "proprio", "state"],
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    },
    "cube": {
        "env": "swm/OGBCube-v0",
        "env_kw": {"env_type": "single", "ob_type": "states", "multiview": False, "width": 224,
                   "height": 224, "visualize_info": False, "terminate_at_goal": True},
        "repo": "quentinll/lewm-cube",
        "data": "ogbench/cube_single_expert.h5",
        "cache": ["action"],
        "callables": [
            {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_pos", "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"}}},
        ],
    },
    "tworoom": {
        "env": "swm/TwoRoom-v1",
        "env_kw": {},
        "repo": "quentinll/lewm-tworooms",
        "data": "tworoom.h5",
        "cache": ["action", "proprio"],
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
    },
}


def image_transform():
    import stable_pretraining as spt
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=IMAGE),
    ])


def load_dataset(swm, task: str):
    return swm.data.load_dataset(TASKS[task]["data"], keys_to_cache=TASKS[task]["cache"])


def episode_column(dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def fit_processors(dataset, task: str) -> dict:
    """StandardScaler per cached column, exactly as le-wm/eval.py."""
    from sklearn import preprocessing

    process = {}
    for col in TASKS[task]["cache"]:
        data = np.asarray(dataset.get_col_data(col))
        data = data[~np.isnan(data).any(axis=1)]
        process[col] = preprocessing.StandardScaler().fit(data)
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def sample_episodes(dataset, seed: int, num_eval: int, min_start: int = 0):
    """le-wm/eval.py episode sampling; ``min_start`` > 0 additionally requires
    that many executed steps before the start (for history prefill)."""
    col = episode_column(dataset)
    ep_col = np.asarray(dataset.get_col_data(col))
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    _, inv = np.unique(ep_col, return_inverse=True)
    last = np.zeros(inv.max() + 1, dtype=np.int64)
    np.maximum.at(last, inv, step_idx)
    max_start = last[inv] + 1 - GOAL_OFFSET - 1
    valid = (step_idx <= max_start) & (step_idx >= min_start)
    valid = np.nonzero(valid)[0]
    g = np.random.default_rng(seed)
    pick = np.sort(valid[g.choice(len(valid) - 1, size=num_eval, replace=False)])
    rows = dataset.get_row_data(pick)
    return [int(e) for e in rows[col]], [int(s) for s in rows["step_idx"]]
