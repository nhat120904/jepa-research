"""Dataset trajectory iterators — wrappers over the real upstream datasets.

These wrap the genuine classes in ``app.plan_common.datasets`` (verified
against source), NOT the imagined ``src.datasets.*`` of the first draft:

* ``MetaworldHFDataset``  → ``(obs, act, state, reward, {})``
* ``DROIDVideoDataset``   → ``(obs, actions, states, reward)``  (4-tuple)
* ``RoboCasaDataset``     → ``(obs, act, state, reward, info)``

In every case ``obs = {"visual": (T,C,H,W), "proprio": (T,P)}``. We load with
``transform=None`` and ``normalize_action=False`` so we get *raw* frames /
actions / proprio / state, then the adapter applies the model's own transform
and ``normalize_actions`` (single source of truth).

Transition convention: for ``T`` frames we keep ``T-1`` actions (action[t] maps
frame t → t+1; the final action has no observed target and is dropped).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from einops import rearrange

# Gripper index within the proprio vector, per env (see DATA_STATS).
GRIPPER_IDX = {
    "metaworld": 3,
    "droid": 6,
    "robocasa": 6,
    "franka_custom": 6,
    "pusht": None,
    "point_maze": None,
    "pointmaze": None,
    "wall": None,
}


@dataclass
class TransitionBatch:
    """One trajectory (or clip): frames + actions + per-step metadata.

    Shapes (T = number of frames):
        obs_visual: (T, C, H, W) RGB in **[0, 255]** (model transform applied later)
        action:     (T-1, A) raw
        proprio:    (T, P) raw
        state:      (T, S) raw   — full env state (Metaworld 39-dim) for stratification
        gripper:    (T,) raw     — extracted from proprio[:, gripper_idx]
    """
    obs_visual: torch.Tensor
    action: torch.Tensor
    proprio: torch.Tensor
    state: torch.Tensor
    traj_id: str
    task: str
    gripper: Optional[torch.Tensor] = None
    extras: dict = field(default_factory=dict)


def add_upstream_to_path(external_root: str | Path = "external/jepa-wms") -> None:
    """Put the cloned upstream repo on sys.path so its packages import."""
    root = Path(external_root).resolve()
    if not root.exists():
        raise ImportError(
            f"Upstream repo not found at {root}. Run scripts/01_setup_environment.sh "
            "to clone facebookresearch/jepa-wms into external/jepa-wms."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# ---------------------------------------------------------------------------
# Tuple + visual normalization helpers
# ---------------------------------------------------------------------------

def _unpack_item(item):
    """Datasets return 4- or 5-tuples; normalize to (obs, act, state)."""
    if len(item) == 5:
        obs, act, state, _reward, _info = item
    elif len(item) == 4:
        obs, act, state, _reward = item
    else:
        raise ValueError(f"Unexpected dataset item arity: {len(item)}")
    return obs, act, state


def _to_255(visual: torch.Tensor) -> torch.Tensor:
    """Adapter.encode expects [0,255]. Datasets with transform=None typically
    return [0,1] (they divide by 255). Detect and rescale defensively."""
    visual = torch.as_tensor(visual, dtype=torch.float32)
    if visual.ndim == 4 and visual.shape[-1] in (1, 3) and visual.shape[1] not in (1, 3):
        visual = rearrange(visual, "T H W C -> T C H W")
    if float(visual.max()) <= 1.5:
        visual = visual * 255.0
    return visual


def _build_transition(obs, act, state, *, env: str, traj_id: str, task: str) -> TransitionBatch:
    visual = _to_255(obs["visual"])              # (T, C, H, W)
    proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32)  # (T, P)
    state = proprio.clone() if state is None else torch.as_tensor(state, dtype=torch.float32)  # (T, S)
    act = torch.as_tensor(act, dtype=torch.float32)                 # (T or T-1, A)

    T = visual.shape[0]
    action = act[: T - 1]                        # keep T-1 (drop the unobserved last)
    proprio = proprio[:T]
    state = state[:T]

    gidx = GRIPPER_IDX.get(env)
    gripper = proprio[:, gidx] if (gidx is not None and proprio.shape[-1] > gidx) else None

    return TransitionBatch(
        obs_visual=visual, action=action, proprio=proprio, state=state,
        gripper=gripper, traj_id=traj_id, task=task,
    )


# ---------------------------------------------------------------------------
# Metaworld  (primary diagnostic target)
# ---------------------------------------------------------------------------

def iterate_metaworld_trajectories(
    root: str | Path,
    tasks: Sequence[str],
    max_trajectories_per_task: int = 1000,
    external_root: str | Path = "external/jepa-wms",
) -> Iterator[TransitionBatch]:
    """Yield Metaworld trajectories from ``MetaworldHFDataset``.

    ``root`` is the HF parquet dir (e.g. ``data/Metaworld/data``).
    """
    add_upstream_to_path(external_root)
    from app.plan_common.datasets.metaworld_hf_dset import MetaworldHFDataset  # type: ignore

    # One dataset filtered per task → keeps traj_id = "<task>/<idx>" and lets us
    # cap trajectories per task.
    for task in tasks:
        ds = MetaworldHFDataset(
            data_path=str(root),
            transform=None,            # raw frames; model transform applied in adapter
            normalize_action=False,    # raw actions; adapter applies normalize_actions
            filter_tasks=[task],
            with_reward=False,
            n_rollout=max_trajectories_per_task,
        )
        n = min(len(ds), max_trajectories_per_task)
        for i in range(n):
            obs, act, state = _unpack_item(ds[i])
            yield _build_transition(
                obs, act, state, env="metaworld",
                traj_id=f"{task}/{i:05d}", task=task,
            )


# ---------------------------------------------------------------------------
# DROID  (secondary; clip-based)
# ---------------------------------------------------------------------------

def iterate_droid_trajectories(
    root: str | Path,
    max_transitions: int = 50000,
    external_root: str | Path = "external/jepa-wms",
    dataset_kwargs: Optional[dict] = None,
) -> Iterator[TransitionBatch]:
    """Yield DROID clips from ``DROIDVideoDataset``.

    DROID is clip-based: each item is ``frames_per_clip`` frames. ``dataset_kwargs``
    forwards loader specifics (camera_views, frames_per_clip, fps, mpk paths) that
    depend on how DROID was downloaded on the server.
    """
    add_upstream_to_path(external_root)
    from app.plan_common.datasets.droid_dset import DROIDVideoDataset  # type: ignore

    kwargs = dict(transform=None, normalize_action=False)
    kwargs.update(dataset_kwargs or {})
    ds = DROIDVideoDataset(data_path=str(root), **kwargs)

    total = 0
    for i in range(len(ds)):
        if total >= max_transitions:
            return
        obs, act, state = _unpack_item(ds[i])
        tb = _build_transition(
            obs, act, state, env="droid",
            traj_id=f"droid/{i:06d}", task="droid",
        )
        total += int(tb.action.shape[0])
        yield tb


def iterate_franka_custom_trajectories(
    root: str | Path,
    max_transitions: int = 50000,
    external_root: str | Path = "external/jepa-wms",
    dataset_kwargs: Optional[dict] = None,
) -> Iterator[TransitionBatch]:
    """Yield real Franka custom clips via upstream ``DROIDVideoDataset``.

    The upstream loader treats MPK/HF ``franka_custom`` data as the same
    7-dim pose+gripper format as DROID when ``mpk_dset=True``.
    """
    kwargs = {"mpk_dset": True, "mpk_manifest_patterns": ["**/*.mp4"]}
    kwargs.update(dataset_kwargs or {})
    for tb in iterate_droid_trajectories(
        root=root,
        max_transitions=max_transitions,
        external_root=external_root,
        dataset_kwargs=kwargs,
    ):
        yield TransitionBatch(
            obs_visual=tb.obs_visual,
            action=tb.action,
            proprio=tb.proprio,
            state=tb.state,
            traj_id=tb.traj_id.replace("droid/", "franka_custom/"),
            task="franka_custom",
            gripper=tb.gripper,
            extras=tb.extras,
        )


def iterate_pusht_trajectories(
    root: str | Path,
    max_transitions: int = 50000,
    external_root: str | Path = "external/jepa-wms",
    dataset_kwargs: Optional[dict] = None,
) -> Iterator[TransitionBatch]:
    """Yield Push-T trajectories from ``PushTDataset``."""
    add_upstream_to_path(external_root)
    from app.plan_common.datasets.pusht_dset import PushTDataset  # type: ignore

    kwargs = dict(transform=None, normalize_action=False, with_velocity=True)
    kwargs.update(dataset_kwargs or {})
    split = kwargs.pop("split", None)
    data_path = Path(root)
    if split:
        data_path = data_path / split
    ds = PushTDataset(data_path=str(data_path), **kwargs)

    total = 0
    for i in range(len(ds)):
        if total >= max_transitions:
            return
        obs, act, state = _unpack_item(ds[i])
        tb = _build_transition(
            obs, act, state, env="pusht",
            traj_id=f"pusht/{i:06d}", task="pusht",
        )
        total += int(tb.action.shape[0])
        yield tb


def iterate_point_maze_trajectories(
    root: str | Path,
    max_transitions: int = 50000,
    external_root: str | Path = "external/jepa-wms",
    dataset_kwargs: Optional[dict] = None,
) -> Iterator[TransitionBatch]:
    """Yield PointMaze trajectories from ``PointMazeDataset``."""
    add_upstream_to_path(external_root)
    from app.plan_common.datasets.point_maze_dset import PointMazeDataset  # type: ignore

    kwargs = dict(transform=None, normalize_action=False)
    kwargs.update(dataset_kwargs or {})
    ds = PointMazeDataset(data_path=str(root), **kwargs)

    total = 0
    for i in range(len(ds)):
        if total >= max_transitions:
            return
        obs, act, state = _unpack_item(ds[i])
        tb = _build_transition(
            obs, act, state, env="point_maze",
            traj_id=f"point_maze/{i:06d}", task="point_maze",
        )
        total += int(tb.action.shape[0])
        yield tb


def iterate_wall_trajectories(
    root: str | Path,
    max_transitions: int = 50000,
    external_root: str | Path = "external/jepa-wms",
    dataset_kwargs: Optional[dict] = None,
) -> Iterator[TransitionBatch]:
    """Yield Wall trajectories from ``WallDataset``."""
    add_upstream_to_path(external_root)
    from app.plan_common.datasets.wall_dset import WallDataset  # type: ignore

    kwargs = dict(transform=None, normalize_action=False)
    kwargs.update(dataset_kwargs or {})
    ds = WallDataset(data_path=str(root), **kwargs)

    total = 0
    for i in range(len(ds)):
        if total >= max_transitions:
            return
        obs, act, state = _unpack_item(ds[i])
        tb = _build_transition(
            obs, act, state, env="wall",
            traj_id=f"wall/{i:06d}", task="wall",
        )
        total += int(tb.action.shape[0])
        yield tb


# ---------------------------------------------------------------------------
# RoboCasa  (tertiary)
# ---------------------------------------------------------------------------

def iterate_robocasa_trajectories(
    root: str | Path,
    max_transitions: int = 20000,
    external_root: str | Path = "external/jepa-wms",
    dataset_kwargs: Optional[dict] = None,
) -> Iterator[TransitionBatch]:
    """Yield RoboCasa trajectories from ``RoboCasaDataset``."""
    add_upstream_to_path(external_root)
    from app.plan_common.datasets.robocasa_dset import RoboCasaDataset  # type: ignore

    kwargs = dict(transform=None, normalize_action=False, with_reward=False,
                  rcasa_to_droid_action_format=True)
    kwargs.update(dataset_kwargs or {})
    ds = RoboCasaDataset(data_path=str(root), **kwargs)

    total = 0
    for i in range(len(ds)):
        if total >= max_transitions:
            return
        obs, act, state = _unpack_item(ds[i])
        tb = _build_transition(
            obs, act, state, env="robocasa",
            traj_id=f"robocasa/{i:06d}", task="robocasa",
        )
        total += int(tb.action.shape[0])
        yield tb
