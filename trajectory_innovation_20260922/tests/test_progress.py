import unittest

import numpy as np
import torch

from ti_wm.progress import (
    ProgressReader, ProgressScorer, episode_bounds, pool_tokens, proprio_vector, sample_pairs,
)


class _StubVisual:
    device = torch.device("cpu")

    def features(self, frames):
        n = len(frames)
        base = torch.from_numpy(np.asarray(frames, dtype=np.float32).mean(axis=(1, 2, 3)))
        return base.view(n, 1, 1).expand(n, 256, 384).contiguous()


class _Branch:
    def __init__(self, value):
        frame = np.full((96, 96, 3), value, dtype=np.uint8)
        self.hist = [{"pixels": frame, "agent_pos": np.array([1.0, 2.0])},
                     {"pixels": frame, "agent_pos": np.array([3.0, 4.0])}]


class ProgressTests(unittest.TestCase):
    def test_pooling_averages_2x2_blocks(self):
        grid = torch.arange(256, dtype=torch.float32).view(1, 256, 1).expand(1, 256, 384)
        pooled = pool_tokens(grid)
        self.assertEqual(tuple(pooled.shape), (1, 64, 384))
        self.assertAlmostEqual(float(pooled[0, 0, 0]), (0 + 1 + 16 + 17) / 4)

    def test_episode_bounds_and_hindsight_pairs(self):
        episode = np.array([7, 7, 7, 9, 9])
        times = np.array([0, 8, 13, 0, 8])
        end_of = episode_bounds(episode)
        self.assertEqual(end_of.tolist(), [3, 3, 3, 5, 5])
        i, j, y = sample_pairs(times, end_of, np.random.default_rng(0), 5000)
        self.assertTrue(np.all(j >= i) and np.all(j < end_of[i]))
        self.assertTrue(np.all(episode[i] == episode[j]))
        np.testing.assert_allclose(y, np.log1p(times[j] - times[i]))
        self.assertTrue(set(zip(i.tolist(), j.tolist())) >= {(0, 2), (2, 2), (3, 4)})

    def test_reader_and_scorer_shapes(self):
        reader = ProgressReader(width=32, layers=1, heads=2)
        x = torch.zeros(3, 64, 384)
        self.assertEqual(tuple(reader(x, x, x, torch.zeros(3, 4)).shape), (3,))
        goals = np.zeros((2, 96, 96, 3), dtype=np.uint8)
        scores = ProgressScorer(reader, _StubVisual(), goals).scores([_Branch(0), _Branch(200), _Branch(90)])
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(np.isfinite(scores)))
        np.testing.assert_allclose(proprio_vector([512, 0], [0, 512]), [1, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
