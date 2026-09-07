"""OGBench-Scene play data: streaming access and offline goal-relative state.

The dataset is the released OGBench v0 visual Scene play split.  It carries
pixels *and* the full ``qpos``/``qvel``/``button_states`` for every row, so an
exact reset and an offline recomputation of the environment's own success
predicate are both possible without re-running the scripted collector.

Nothing in this module steps physics.  ``mj_kinematics`` is called to resolve
site positions from ``qpos``, which is the same read the environment performs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import numpy.lib.format as npy_format


DATASET_ROOT = Path("/mnt/data/vhoangth2/datasets/ogbench_data")
PLAY_TRAIN = DATASET_ROOT / "visual-scene-play-v0.npz"
PLAY_VAL = DATASET_ROOT / "visual-scene-play-v0-val.npz"
STATE_TRAIN = DATASET_ROOT / "scene-play-v0.npz"

EPISODE_LEN = 1001
NUM_EPISODES_TRAIN = 1000
NUM_EPISODES_VAL = 100

# OGBench's own tolerances, read from SceneEnv._compute_successes.
CUBE_TOL = 0.04
DRAWER_TOL = 0.04
WINDOW_TOL = 0.04

MEMBERS = (
    "observations",
    "actions",
    "terminals",
    "qpos",
    "qvel",
    "button_states",
)


def episode_bounds(row: int) -> tuple[int, int]:
    """Return ``(start, stop)`` row indices of the episode containing ``row``."""
    ep = row // EPISODE_LEN
    return ep * EPISODE_LEN, (ep + 1) * EPISODE_LEN


def same_episode(a: int, b: int) -> bool:
    return a // EPISODE_LEN == b // EPISODE_LEN


def member_header(path: Path, member: str) -> tuple[tuple[int, ...], np.dtype]:
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{member}.npy") as handle:
            version = npy_format.read_magic(handle)
            shape, _, dtype = npy_format._read_array_header(handle, version)
    return shape, dtype


def stream_rows(path: Path, member: str, rows: np.ndarray) -> np.ndarray:
    """Read selected rows of one npz member without materialising the whole array.

    The members are deflate streams, so decompression is sequential: this reads
    forward exactly as far as the largest requested row and discards the rest.
    Cheap when the requested rows sit near the front, which is how the plumbing
    gate samples them.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("rows must be a non-empty 1-D index array")
    shape, dtype = member_header(path, member)
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize if len(shape) > 1 else dtype.itemsize
    order = np.argsort(rows)
    wanted = rows[order]
    out = np.empty((rows.size, *shape[1:]), dtype=dtype)

    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{member}.npy") as handle:
            version = npy_format.read_magic(handle)
            npy_format._read_array_header(handle, version)
            cursor = 0
            for slot, target in zip(order, wanted):
                if target < cursor:
                    raise RuntimeError("row indices must be strictly increasing after sort")
                skip = int(target - cursor) * row_bytes
                while skip > 0:
                    chunk = handle.read(min(skip, 1 << 22))
                    if not chunk:
                        raise RuntimeError("unexpected end of npy stream while skipping")
                    skip -= len(chunk)
                buf = bytearray()
                while len(buf) < row_bytes:
                    chunk = handle.read(row_bytes - len(buf))
                    if not chunk:
                        raise RuntimeError("unexpected end of npy stream while reading")
                    buf.extend(chunk)
                out[slot] = np.frombuffer(bytes(buf), dtype=dtype).reshape(shape[1:])
                cursor = int(target) + 1
    return out


def load_small(path: Path, member: str) -> np.ndarray:
    """Fully load one of the small (non-pixel) members."""
    if member == "observations":
        raise ValueError("observations is the pixel member; use stream_rows or the cache")
    with np.load(path) as archive:
        return np.asarray(archive[member])


# --------------------------------------------------------------------------
# Offline state readout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneJointIndex:
    """qpos addresses of the three joints the success predicate reads."""

    cube_qposadr: int
    drawer_qposadr: int
    window_qposadr: int
    drawer_site_id: int

    def to_json(self) -> dict[str, int]:
        return {
            "cube_qposadr": int(self.cube_qposadr),
            "drawer_qposadr": int(self.drawer_qposadr),
            "window_qposadr": int(self.window_qposadr),
            "drawer_site_id": int(self.drawer_site_id),
        }


def joint_index(raw_env) -> SceneJointIndex:
    """Resolve qpos addresses from the live model, never hardcoded."""
    model = raw_env._model
    return SceneJointIndex(
        cube_qposadr=int(model.joint("object_joint_0").qposadr[0]),
        drawer_qposadr=int(model.joint("drawer_slide").qposadr[0]),
        window_qposadr=int(model.joint("window_slide").qposadr[0]),
        drawer_site_id=int(raw_env._drawer_site_id),
    )


class OfflineSceneState:
    """Resolve cube/drawer/window/drawer-site from a stored ``qpos`` row.

    Holds its own ``mjData`` so it never disturbs the environment being
    evaluated.  Only forward kinematics is run; no integration, no stepping.
    """

    def __init__(self, raw_env) -> None:
        import mujoco

        self._mujoco = mujoco
        self._model = raw_env._model
        self._data = mujoco.MjData(self._model)
        self._index = joint_index(raw_env)

    @property
    def index(self) -> SceneJointIndex:
        return self._index

    def read(self, qpos: np.ndarray) -> dict[str, np.ndarray | float]:
        self._data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self._data.qvel[:] = 0.0
        self._mujoco.mj_kinematics(self._model, self._data)
        idx = self._index
        return {
            "cube_pos": self._data.qpos[idx.cube_qposadr : idx.cube_qposadr + 3].copy(),
            "drawer": float(self._data.qpos[idx.drawer_qposadr]),
            "window": float(self._data.qpos[idx.window_qposadr]),
            "drawer_site_y": float(self._data.site_xpos[idx.drawer_site_id][1]),
        }

    def is_in_drawer(self, cube_pos: np.ndarray, drawer_site_y: float) -> bool:
        low = np.array([0.21, drawer_site_y - 0.27, 0.0])
        high = np.array([0.45, drawer_site_y - 0.07, 0.15])
        cube_pos = np.asarray(cube_pos, dtype=np.float64)
        return bool(np.all(low <= cube_pos) and np.all(cube_pos <= high))


# --------------------------------------------------------------------------
# Goal-relative success, mirroring SceneEnv._compute_successes exactly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalMatch:
    """Per-component match against a goal row, in OGBench's own tolerances."""

    cube: bool
    button_0: bool
    button_1: bool
    drawer: bool
    window: bool

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [self.cube, self.button_0, self.button_1, self.drawer, self.window],
            dtype=np.float32,
        )

    @property
    def success(self) -> bool:
        return bool(
            self.cube and self.button_0 and self.button_1 and self.drawer and self.window
        )

    def to_json(self) -> dict[str, bool]:
        return {
            "cube": bool(self.cube),
            "button_0": bool(self.button_0),
            "button_1": bool(self.button_1),
            "drawer": bool(self.drawer),
            "window": bool(self.window),
            "success": self.success,
        }


def goal_match(
    state: dict[str, np.ndarray | float],
    buttons: np.ndarray,
    goal_state: dict[str, np.ndarray | float],
    goal_buttons: np.ndarray,
) -> GoalMatch:
    """Component matches, using exactly the comparisons SceneEnv makes.

    ``SceneEnv._compute_successes`` compares the cube against a mocap target
    (tolerance 0.04), each button against its target state by equality, and the
    drawer/window slide joints against target positions (tolerance 0.04).  Here
    the targets come from a dataset goal row instead of a task definition.
    """
    cube = float(np.linalg.norm(np.asarray(state["cube_pos"]) - np.asarray(goal_state["cube_pos"])))
    return GoalMatch(
        cube=cube <= CUBE_TOL,
        button_0=int(buttons[0]) == int(goal_buttons[0]),
        button_1=int(buttons[1]) == int(goal_buttons[1]),
        drawer=abs(float(state["drawer"]) - float(goal_state["drawer"])) <= DRAWER_TOL,
        window=abs(float(state["window"]) - float(goal_state["window"])) <= WINDOW_TOL,
    )


__all__ = [
    "CUBE_TOL",
    "DRAWER_TOL",
    "EPISODE_LEN",
    "GoalMatch",
    "MEMBERS",
    "NUM_EPISODES_TRAIN",
    "NUM_EPISODES_VAL",
    "PLAY_TRAIN",
    "PLAY_VAL",
    "STATE_TRAIN",
    "WINDOW_TOL",
    "OfflineSceneState",
    "SceneJointIndex",
    "episode_bounds",
    "goal_match",
    "joint_index",
    "load_small",
    "member_header",
    "same_episode",
    "stream_rows",
]
