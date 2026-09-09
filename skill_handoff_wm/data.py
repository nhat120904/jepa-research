"""Trajectory-safe sampling for OGBench goal-conditioned behavior cloning.

The raw OGBench NPZ contains the final observation of every trajectory.  Its action is
not a valid transition.  This module keeps those boundary states as possible goals but
never samples their actions as policy targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpisodeIndex:
    starts: np.ndarray
    ends: np.ndarray
    episode_of: np.ndarray
    valid_action_indices: np.ndarray

    @classmethod
    def from_terminals(cls, terminals: np.ndarray) -> "EpisodeIndex":
        terminals = np.asarray(terminals).reshape(-1)
        ends = np.flatnonzero(terminals > 0)
        if len(ends) == 0 or ends[-1] != len(terminals) - 1:
            raise ValueError("terminals must mark the final row")
        starts = np.concatenate([np.array([0]), ends[:-1] + 1])
        episode_of = np.empty(len(terminals), dtype=np.int64)
        valid = []
        for episode, (start, end) in enumerate(zip(starts, ends)):
            episode_of[start : end + 1] = episode
            if end > start:
                valid.append(np.arange(start, end, dtype=np.int64))
        valid_action_indices = np.concatenate(valid) if valid else np.empty(0, dtype=np.int64)
        return cls(starts, ends, episode_of, valid_action_indices)

    def future_limits(self, indices: np.ndarray, max_future: int) -> np.ndarray:
        episode_ends = self.ends[self.episode_of[indices]]
        return np.minimum(indices + int(max_future), episode_ends)


class GoalConditionedDataset:
    """In-memory state/action arrays and trajectory-safe future-goal sampler."""

    def __init__(self, path: str | Path):
        path = Path(path)
        with np.load(path) as source:
            self.observations = source["observations"].astype(np.float32, copy=False)
            self.actions = source["actions"].astype(np.float32, copy=False)
            terminals = source["terminals"].astype(np.float32, copy=False)
        self.index = EpisodeIndex.from_terminals(terminals)
        if len(self.observations) != len(self.actions):
            raise ValueError("observations/actions length mismatch")
        if self.observations.ndim != 2 or self.observations.shape[1] < 2:
            raise ValueError("state observations with XY in columns 0:2 are required")

    def normalization(self) -> tuple[np.ndarray, np.ndarray]:
        states = self.observations[self.index.valid_action_indices]
        mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = states.std(axis=0, dtype=np.float64).astype(np.float32)
        return mean, np.maximum(std, 1e-3)

    def sample(
        self,
        rng: np.random.Generator,
        batch_size: int,
        min_future: int,
        max_future: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if min_future < 1 or max_future < min_future:
            raise ValueError("require 1 <= min_future <= max_future")
        candidates = self.index.valid_action_indices
        starts = candidates[rng.integers(0, len(candidates), size=batch_size)]
        limits = self.index.future_limits(starts, max_future)
        lows = np.minimum(starts + min_future, limits)
        # numpy does not support vector-valued high in Generator.integers on all deployed versions.
        uniforms = rng.random(batch_size)
        futures = lows + np.floor(uniforms * (limits - lows + 1)).astype(np.int64)
        return (
            self.observations[starts],
            self.observations[futures, :2],
            self.actions[starts],
            futures - starts,
        )

    def short_goal_cases(
        self,
        rng: np.random.Generator,
        horizons: list[int],
        cases_per_horizon: int,
    ) -> list[dict[str, object]]:
        cases: list[dict[str, object]] = []
        candidates = self.index.valid_action_indices
        for horizon in horizons:
            valid = candidates[self.index.future_limits(candidates, horizon) >= candidates + horizon]
            if len(valid) == 0:
                raise ValueError(f"no validation cases support horizon {horizon}")
            replace = len(valid) < cases_per_horizon
            starts = rng.choice(valid, size=cases_per_horizon, replace=replace)
            for start in starts:
                future = int(start + horizon)
                cases.append(
                    {
                        "start_index": int(start),
                        "future_index": future,
                        "horizon": int(horizon),
                        "start_state": self.observations[start],
                        "goal_xy": self.observations[future, :2],
                    }
                )
        return cases
