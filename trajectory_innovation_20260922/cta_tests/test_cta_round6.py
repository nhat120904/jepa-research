import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cta_round6 as r6  # noqa: E402


def _cache(path, n=6, shuffle_pert=False):
    rng = np.random.default_rng(0)
    root = np.r_[np.full(n - 2, 31050), np.full(2, r6.DEV_ROOT)]
    rows = {"ctx_index": np.arange(n), "root": root, "decision": np.arange(n), "t": np.arange(n) * 8,
            "chunk": rng.uniform(0, 512, (n, 8, 8, 2)).astype(np.float32), "ctx_pos": rng.uniform(0, 512, (n, 2, 2)).astype(np.float32),
            "end_pos": np.zeros((n, 8, 2)), "end_prev_pos": np.zeros((n, 8, 2)),
            "geom": rng.uniform(-.1, 0, (n, 8)).astype(np.float32), "cov": np.zeros((n, 8)),
            "src": rng.integers(0, 256, (n, 8, 16)).astype(np.int16), "mixed": np.zeros(n, bool)}
    pert = {k: v.copy() for k, v in rows.items()}
    pert["geom"] = pert["geom"] - 1.            # perturbed half clearly worse: labels identify the half
    if shuffle_pert:
        pert["ctx_index"] = pert["ctx_index"][::-1]
    both = {k: np.concatenate([rows[k], pert[k]]) for k in rows}
    both["source"] = np.r_[np.ones(n, np.int8), np.full(n, 2, np.int8)]
    np.savez(path / "banks.npz", **both)
    np.save(path / "cur.npy", rng.normal(size=(n, 4, 3)).astype(np.float16))
    np.save(path / "prev.npy", rng.normal(size=(n, 2, 3)).astype(np.float16))


class Round6(unittest.TestCase):
    def test_anchored_rank_penalizes_only_default_pairs(self):
        labels = torch.tensor([[0., .05, -.05, 0.]])
        good = torch.tensor([[0., 5., -5., 0.]])          # k=1 above default, k=2 below: correct order
        bad = torch.tensor([[0., -5., 5., 0.]])
        self.assertLess(float(r6.anchored_rank(good, labels)), 0.01)
        self.assertGreater(float(r6.anchored_rank(bad, labels)), 4.)
        # a mistake between two non-default candidates with equal default gap does not change the anchored loss
        self.assertAlmostEqual(float(r6.anchored_rank(torch.tensor([[0., 5., -5., 3.]]), labels)),
                               float(r6.anchored_rank(good, labels)), places=6)

    def test_pairs_align_halves_and_split_dev(self):
        codebook = torch.randn(256, 3)
        with tempfile.TemporaryDirectory() as d:
            _cache(Path(d))
            data = r6.Pairs(Path(d), codebook)
            self.assertEqual(data.chunk.shape, (6, 16, 8, 2))
            self.assertTrue((data.geom[:, 8:] < data.geom[:, :8].min() - .5).all())
            self.assertEqual(list(data.dev_idx), [4, 5])
            ctx, act, src, labels = data.batch([2, 0], "cpu")
            self.assertEqual(ctx["cur"].shape[0], 32)
            self.assertEqual(src.shape, (32, 16, 3))
            self.assertTrue(torch.equal(labels, data.geom[[0, 2]]))
        with tempfile.TemporaryDirectory() as d:
            _cache(Path(d), shuffle_pert=True)
            with self.assertRaises(ValueError):
                r6.Pairs(Path(d), codebook)

    def test_choice_stats_counts_worse_than_default(self):
        labels = np.zeros((2, 16), np.float32)
        labels[:, 9] = -.02                     # a perturbed candidate that is clearly worse
        labels[:, 3] = .01
        scores = np.zeros((2, 16), np.float32)
        scores[0, 9] = 1.                       # decision 0: picks the bad perturbed one
        scores[1, 3] = 1.                       # decision 1: picks the good policy one
        out = r6.choice_stats(scores, labels, np.array([1, 2]))
        self.assertAlmostEqual(out["bank16"]["worse_than_default"], .5)
        self.assertAlmostEqual(out["bank16"]["chose_perturbed"], .5)
        self.assertAlmostEqual(out["policy8"]["worse_than_default"], 0.)


if __name__ == "__main__":
    unittest.main()
