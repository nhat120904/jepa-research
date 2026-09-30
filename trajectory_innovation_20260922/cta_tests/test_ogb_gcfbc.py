"""scripts/ogbench/gcfbc.py: goal relabeling, chunk validity, shapes and seeded sampling (CPU)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_p = Path(__file__).resolve().parents[1] / "scripts" / "ogbench" / "gcfbc.py"
_s = importlib.util.spec_from_file_location("gcfbc", _p)
g = importlib.util.module_from_spec(_s)
_s.loader.exec_module(g)


class GCFBC(unittest.TestCase):
    def _data(self, d):
        n = 30                                   # three trajectories of 10
        term = np.zeros(n, bool)
        term[[9, 19, 29]] = True
        obs = np.zeros((n, 64, 64, 3), np.uint8)
        obs[:, 0, 0, 0] = np.arange(n)           # frame id in a pixel
        act = np.repeat(np.arange(n, dtype=np.float32)[:, None], 5, 1) / 100
        np.savez(Path(d) / "x.npz", observations=obs, actions=act, terminals=term)
        return g.Data(Path(d) / "x.npz")

    def test_goals_and_chunks_stay_in_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            data = self._data(d)
            self.assertTrue(all(i % 10 <= 10 - g.H for i in data.valid))
            ob, goal, chunk = data.batch(np.random.default_rng(0), 512, "cpu", p_aug=0)
            i, j = ob[:, 0, 0, 0].long(), goal[:, 0, 0, 0].long()
            self.assertTrue(((j > i) | ((i % 10) == 9)).all())
            self.assertTrue((j // 10 == i // 10).all())
            self.assertTrue(torch.allclose(chunk[:, :, 0] * 100, (i[:, None] + torch.arange(g.H)).float()))

    def test_policy_shapes_and_seeded_sampling(self):
        pol = g.GCFlowPolicy()
        ob = torch.zeros(2, 64, 64, 3, dtype=torch.uint8)
        self.assertEqual(pol.features(ob, ob).shape, (2, 512))
        self.assertTrue(torch.isfinite(pol.loss(ob, ob, torch.zeros(2, g.H, g.ACT))))
        a = g.seeded_noise([1, 2], "cpu")
        self.assertTrue(torch.equal(a, g.seeded_noise([1, 2], "cpu")))
        out = pol.sample(pol.features(ob, ob), a)
        self.assertEqual(out.shape, (2, g.H, g.ACT))
        self.assertTrue((out.abs() <= 1).all())


if __name__ == "__main__":
    unittest.main()
