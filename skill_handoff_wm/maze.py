"""Deterministic grid paths used as a shared skill sequence across switching arms."""

from __future__ import annotations

from collections import deque

import numpy as np


NEIGHBORS = ((-1, 0), (0, -1), (1, 0), (0, 1))


def shortest_cell_path(maze_map: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Return an inclusive BFS path over free maze cells."""
    maze_map = np.asarray(maze_map)
    if maze_map[start] != 0 or maze_map[goal] != 0:
        raise ValueError("start and goal must be free cells")
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for di, dj in NEIGHBORS:
            nxt = (cell[0] + di, cell[1] + dj)
            if (
                0 <= nxt[0] < maze_map.shape[0]
                and 0 <= nxt[1] < maze_map.shape[1]
                and maze_map[nxt] == 0
                and nxt not in parent
            ):
                parent[nxt] = cell
                queue.append(nxt)
    if goal not in parent:
        raise ValueError(f"no path from {start} to {goal}")
    path = []
    current: tuple[int, int] | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def path_waypoints(raw_env, final_goal_xy: np.ndarray, stride: int = 1) -> list[np.ndarray]:
    if stride < 1:
        raise ValueError("stride must be positive")
    start = raw_env.xy_to_ij(raw_env.get_xy())
    goal = raw_env.xy_to_ij(final_goal_xy)
    cells = shortest_cell_path(raw_env.maze_map, start, goal)
    selected = cells[stride::stride]
    if not selected or selected[-1] != cells[-1]:
        selected.append(cells[-1])
    waypoints = [np.asarray(raw_env.ij_to_xy(cell), dtype=np.float32) for cell in selected]
    waypoints[-1] = np.asarray(final_goal_xy, dtype=np.float32)
    return waypoints
