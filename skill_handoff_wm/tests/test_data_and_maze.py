import numpy as np

from skill_handoff_wm.data import EpisodeIndex
from skill_handoff_wm.maze import shortest_cell_path


def test_episode_index_never_uses_terminal_action():
    index = EpisodeIndex.from_terminals(np.array([0, 0, 1, 0, 1], dtype=np.float32))
    assert index.starts.tolist() == [0, 3]
    assert index.ends.tolist() == [2, 4]
    assert index.valid_action_indices.tolist() == [0, 1, 3]
    assert index.future_limits(np.array([0, 1, 3]), 10).tolist() == [2, 2, 4]


def test_shortest_cell_path_is_valid_and_inclusive():
    maze = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])
    path = shortest_cell_path(maze, (1, 1), (2, 2))
    assert path[0] == (1, 1)
    assert path[-1] == (2, 2)
    assert len(path) == 3
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
