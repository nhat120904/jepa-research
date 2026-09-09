"""Compute-node-only OGBench Ant helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def require_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("MuJoCo/policy execution must run inside a Slurm compute job")


def make_env(dataset_name: str, dataset_dir: str):
    require_slurm()
    import ogbench

    return ogbench.make_env_and_datasets(dataset_name, dataset_dir=dataset_dir, env_only=True)


@dataclass(frozen=True)
class PhysicsState:
    qpos: np.ndarray
    qvel: np.ndarray


def snapshot(raw_env) -> PhysicsState:
    return PhysicsState(raw_env.data.qpos.copy(), raw_env.data.qvel.copy())


def restore(raw_env, state: PhysicsState) -> None:
    raw_env.set_state(state.qpos.copy(), state.qvel.copy())


def set_from_observation(raw_env, observation: np.ndarray) -> None:
    observation = np.asarray(observation)
    nq, nv = int(raw_env.model.nq), int(raw_env.model.nv)
    if observation.shape[-1] != nq + nv:
        raise ValueError(f"observation dim {observation.shape[-1]} != nq+nv {nq + nv}")
    raw_env.set_state(observation[:nq].copy(), observation[nq : nq + nv].copy())


def paired_reset(env, task_id: int, seed: int):
    """Reset one arm reproducibly despite OGBench 1.2.1 using global numpy goal noise."""
    np.random.seed(seed)
    observation, info = env.reset(seed=seed, options={"task_id": task_id})
    return observation, info
