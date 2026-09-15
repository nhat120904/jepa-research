"""Segment targets from playback-monitor labels (evaluation and supervised-arm targets only)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

CONTACT_RADIUS = 0.02  # native ScrubCuttingBoard distinct-contact rule
DISC_RADIUS = 0.01
GRID = 0.0025
MAP_CELLS = 12
MAP_CELL = 0.025
MAP_HALF = MAP_CELLS * MAP_CELL / 2

_R = int(round(DISC_RADIUS / GRID))
_OFFSETS = np.array([(dx, dy) for dx in range(-_R, _R + 1) for dy in range(-_R, _R + 1) if dx * dx + dy * dy <= _R * _R])


def greedy_count(points: np.ndarray, history: np.ndarray | None = None) -> int:
    accepted = np.empty((0, 2)) if history is None else np.asarray(history, float).reshape(-1, 2)
    added = 0
    for p in points:
        if accepted.shape[0] == 0 or float(np.min(np.sum((accepted - p) ** 2, axis=1))) >= CONTACT_RADIUS**2:
            accepted = np.vstack([accepted, p])
            added += 1
    return added


class EpisodeLabels:
    """Per-episode live contact points (frame index, world xy, board-frame map cell, disc cells)."""

    def __init__(self, path: Path) -> None:
        z = np.load(path)
        live = z["contact"] & z["grasped"]
        self.count = z["count"].astype(np.int64)
        self.frames = len(live)
        self.idx = np.flatnonzero(live)
        self.xy = z["sponge_xy"][self.idx]
        board = z["board_pos"][self.idx, :2]
        quat = z["board_quat"][self.idx]  # MuJoCo order w, x, y, z
        yaw = np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
        rel = self.xy - board
        c, s = np.cos(-yaw), np.sin(-yaw)
        local = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], axis=1)
        cells = np.floor((local + MAP_HALF) / MAP_CELL).astype(int)
        inside = (cells >= 0).all(1) & (cells < MAP_CELLS).all(1)
        self.map_index = np.where(inside, cells[:, 0] * MAP_CELLS + cells[:, 1], -1)
        base = np.floor(self.xy / GRID).astype(np.int64)
        self.disc = [set(((b[0] + _OFFSETS[:, 0]) * 1_000_003 + (b[1] + _OFFSETS[:, 1])).tolist()) for b in base]
        # sequential accepted set: history for any prefix is a prefix of the full run
        self.accepted = []
        acc = np.empty((0, 2))
        for k, p in enumerate(self.xy):
            if acc.shape[0] == 0 or float(np.min(np.sum((acc - p) ** 2, axis=1))) >= CONTACT_RADIUS**2:
                acc = np.vstack([acc, p])
                self.accepted.append(k)
        self.accepted = np.asarray(self.accepted, dtype=np.int64)

    def _points(self, a: int, b: int) -> tuple[int, int]:
        """Live points on frames (a, b]."""
        return int(np.searchsorted(self.idx, a, side="right")), int(np.searchsorted(self.idx, b, side="right"))

    def window(self, a: int, b: int) -> dict:
        """Targets for frames (a, b]."""
        p, q = self._points(a, b)
        empty = greedy_count(self.xy[p:q]) if q > p else 0
        cells: set[int] = set()
        for k in range(p, q):
            cells |= self.disc[k]
        occupancy = np.zeros(MAP_CELLS * MAP_CELLS, dtype=np.float32)
        valid = self.map_index[p:q]
        occupancy[valid[valid >= 0]] = 1.0
        return {
            "count_increment": float(self.count[b] - self.count[a]),
            "count_empty": float(empty),
            "area_empty": len(cells) * GRID * GRID * 1e4,
            "map": occupancy,
            "has_contact": float(q > p),
        }

    def empty_count(self, a: int, b: int) -> int:
        p, q = self._points(a, b)
        return greedy_count(self.xy[p:q]) if q > p else 0


def split_is_hard(labels: EpisodeLabels, t: int, parts: list[int]) -> bool:
    """True when the parts' empty-history counts do not sum to the whole window's count."""
    cuts = [t]
    for length in parts:
        cuts.append(cuts[-1] + length)
    total = sum(labels.empty_count(cuts[i], cuts[i + 1]) for i in range(len(parts)))
    return total != labels.empty_count(cuts[0], cuts[-1])


def sinusoid(positions: np.ndarray, dim: int) -> np.ndarray:
    half = dim // 2
    freq = np.exp(-math.log(10000.0) * np.arange(half) / half)
    angles = positions[:, None] * freq[None]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
