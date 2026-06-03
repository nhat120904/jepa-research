"""HDF5 cache for precomputed encoder outputs (Phase 1, Task 1.3).

Schema (one file per (model, dataset) pair):

    /trajectories/{traj_id}/z          (T, V, H, W, D)  visual latent per frame
    /trajectories/{traj_id}/action     (T-1, A)     a_t (raw)
    /trajectories/{traj_id}/proprio    (T, P)       raw proprio (for predict conditioning)
    /trajectories/{traj_id}/state      (T, S)       raw env state (for stratification)
    /trajectories/{traj_id}/gripper    (T,)         gripper state
    /trajectories/{traj_id}/regime     (T-1,) int8  filled by 04_classify_regimes
    /trajectories/{traj_id}/extras     subgroup     env-specific metadata
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np
import torch


REGIME_TO_ID = {
    "free_space": 0,
    "pre_grasp": 1,
    "gripper_actuation": 2,
    "contact_manipulation": 3,
}
ID_TO_REGIME = {v: k for k, v in REGIME_TO_ID.items()}


def latent_cache_path(root: str | Path, model_name: str, dataset: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{dataset}__{model_name}.h5"


# --- Regime sidecar --------------------------------------------------------
# Regimes are written to a JSON sidecar next to the latent cache instead of
# re-opening the large .h5 in append mode. Appending to the big cache risks a
# truncated/corrupt file if the process is killed mid-write; the sidecar is
# written via os.replace (atomic) so a kill never corrupts an existing file,
# and the expensive latent cache is opened read-only after 03_extract_latents.

def regime_sidecar_path(cache_path: str | Path) -> Path:
    """JSON sidecar holding per-trajectory regime id arrays for a latent cache."""
    cache_path = Path(cache_path)
    return cache_path.with_name(cache_path.name + ".regimes.json")


def write_regimes(cache_path: str | Path, regime_by_traj: dict) -> Path:
    """Atomically write {traj_id: regime_ids} to the sidecar (tmp + os.replace)."""
    out_path = regime_sidecar_path(cache_path)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    payload = {
        str(tid): np.asarray(ids, dtype=np.int8).tolist()
        for tid, ids in regime_by_traj.items()
    }
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)
    return out_path


def read_regimes(cache_path: str | Path) -> Optional[dict]:
    """Read the regime sidecar as {traj_id: np.int8 array}, or None if absent."""
    path = regime_sidecar_path(cache_path)
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {tid: np.asarray(ids, dtype=np.int8) for tid, ids in raw.items()}


class LatentCache:
    """Read/write HDF5 wrapper for precomputed latents."""

    def __init__(self, path: str | Path, mode: str = "r", compression: str = "gzip"):
        self.path = Path(path)
        self.mode = mode
        self.compression = compression
        self.h5: Optional[h5py.File] = None

    # h5py treats "/" in a group name as a nested path, which would split a
    # "task/idx" traj_id into sub-groups and break trajectory_ids(). We store
    # under a flat sanitized key and keep the original id as a group attribute.
    _SLASH = "--SLASH--"

    @classmethod
    def _safe_key(cls, traj_id: str) -> str:
        return traj_id.replace("/", cls._SLASH)

    def __enter__(self):
        self.h5 = h5py.File(self.path, self.mode)
        if "trajectories" not in self.h5 and self.mode in ("w", "a"):
            self.h5.create_group("trajectories")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    # ---- write ----------------------------------------------------------------
    def write_trajectory(
        self,
        traj_id: str,
        z: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray,
        gripper: Optional[torch.Tensor | np.ndarray] = None,
        proprio: Optional[torch.Tensor | np.ndarray] = None,
        state: Optional[torch.Tensor | np.ndarray] = None,
        extras: Optional[dict] = None,
    ) -> None:
        assert self.h5 is not None and self.mode in ("w", "a")
        grp = self.h5["trajectories"].require_group(self._safe_key(traj_id))
        grp.attrs["traj_id"] = traj_id
        if "z" in grp:
            for k in ("z", "action", "gripper", "proprio", "state", "extras", "regime"):
                if k in grp:
                    del grp[k]

        grp.create_dataset("z", data=_to_np(z), compression=self.compression)
        grp.create_dataset("action", data=_to_np(action), compression=self.compression)
        if gripper is not None:
            grp.create_dataset("gripper", data=_to_np(gripper), compression=self.compression)
        if proprio is not None:
            grp.create_dataset("proprio", data=_to_np(proprio), compression=self.compression)
        if state is not None:
            grp.create_dataset("state", data=_to_np(state), compression=self.compression)
        if extras:
            extras_grp = grp.create_group("extras")
            for k, v in extras.items():
                extras_grp.create_dataset(k, data=_to_np(v), compression=self.compression)

    def write_regime(self, traj_id: str, regime_ids: np.ndarray) -> None:
        assert self.h5 is not None and self.mode in ("w", "a")
        grp = self.h5["trajectories"][self._safe_key(traj_id)]
        if "regime" in grp:
            del grp["regime"]
        grp.create_dataset("regime", data=regime_ids.astype(np.int8),
                           compression=self.compression)

    # ---- read -----------------------------------------------------------------
    def trajectory_ids(self):
        assert self.h5 is not None
        grp = self.h5["trajectories"]
        return [grp[k].attrs.get("traj_id", k) for k in grp.keys()]

    def read_trajectory(self, traj_id: str) -> dict:
        grp = self.h5["trajectories"][self._safe_key(traj_id)]
        out = {"z": grp["z"][:], "action": grp["action"][:]}
        for key in ("gripper", "proprio", "state", "regime"):
            if key in grp:
                out[key] = grp[key][:]
        if "extras" in grp:
            out["extras"] = {k: grp["extras"][k][:] for k in grp["extras"].keys()}
        return out

    def iter_transitions(self) -> Iterator[dict]:
        """Yield one transition at a time across all trajectories."""
        for tid in self.trajectory_ids():
            traj = self.read_trajectory(tid)
            T = len(traj["action"])
            for t in range(T):
                yield {
                    "traj_id": tid,
                    "t": t,
                    "z_t": traj["z"][t],
                    "z_t1": traj["z"][t + 1],
                    "a_t": traj["action"][t],
                    "regime": int(traj["regime"][t]) if "regime" in traj else -1,
                    "gripper_t": (float(traj["gripper"][t])
                                  if "gripper" in traj else None),
                    "gripper_t1": (float(traj["gripper"][t + 1])
                                    if "gripper" in traj else None),
                }


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
