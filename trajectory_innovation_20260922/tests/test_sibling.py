import unittest

import numpy as np
import torch

from ti_wm.contract import select_candidate, sibling_seed
from ti_wm.sibling import SiblingReader, contrast_loss, contrast_targets, fit_pca, project, rank_loss, retained_gap


class SiblingTests(unittest.TestCase):
    def test_pca_keeps_leading_direction(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2000, 6, generator=g) * torch.tensor([10.0, 1, 1, 1, 1, 1])
        mean, basis = fit_pca(x, dim=2)
        self.assertEqual(tuple(basis.shape), (6, 2))
        self.assertGreater(abs(float(basis[0, 0])), 0.99)
        self.assertEqual(tuple(project(x, mean, basis).shape), (2000, 2))

    def test_contrast_targets_share_equal_states(self):
        phys = torch.tensor([[[0.0], [0.0], [1.0]], [[0.0], [2.0], [3.0]]])
        t = contrast_targets(phys, torch.tensor([1, 2]))
        np.testing.assert_allclose(t.numpy(), [[0.5, 0.5, 0], [0, 0, 1]])
        good = contrast_loss(torch.tensor([[5.0, 5.0, -5.0]]), t[:1])
        bad = contrast_loss(torch.tensor([[-5.0, -5.0, 5.0]]), t[:1])
        self.assertLess(float(good), float(bad))

    def test_rank_loss_prefers_higher_label(self):
        labels = torch.tensor([[0.1, 0.5, 0.5]])
        right = rank_loss(torch.tensor([[0.0, 3.0, 3.0]]), labels)
        wrong = rank_loss(torch.tensor([[3.0, 0.0, 0.0]]), labels)
        self.assertLess(float(right), float(wrong))
        self.assertEqual(float(rank_loss(torch.zeros(1, 3), torch.zeros(1, 3))), 0.0)

    def test_reader_shape_and_gap(self):
        r = SiblingReader(dim=8, width=16, layers=1, heads=2)
        x = torch.zeros(5, 256, 8)
        self.assertEqual(tuple(r(x, x, x, torch.zeros(5, 4)).shape), (5,))
        self.assertEqual(retained_gap([0.0, 2.0, 1.0], np.array([0.2, 0.5, 0.9]), select_candidate), (0.3, 0.7))
        self.assertEqual(retained_gap([1.0, 1.0, 0.0], np.array([0.2, 0.5, 0.9]), select_candidate), (0.0, 0.7))

    def test_sibling_seed_namespace(self):
        self.assertNotEqual(sibling_seed(1, 2, 3, 0), sibling_seed(1, 2, 3, 1))
        with self.assertRaises(ValueError):
            sibling_seed(-1, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()


class _StubVisual:
    device = torch.device("cpu")

    def features(self, frames):
        f = torch.from_numpy(np.asarray(frames, dtype=np.float32)).mean(dim=(1, 2, 3))
        return f.view(-1, 1, 1).expand(-1, 256, 384).contiguous() / 255.0


class _Hist:
    def __init__(self, value, pos):
        frame = np.full((96, 96, 3), value, dtype=np.uint8)
        self.hist = [{"pixels": frame, "agent_pos": np.array(pos, float)},
                     {"pixels": frame, "agent_pos": np.array(pos, float) + 1}]


class SiblingScorerTests(unittest.TestCase):
    def test_scorer_shapes_and_determinism(self):
        import tempfile
        from pathlib import Path

        from ti_wm.sibling import SiblingScorer

        torch.manual_seed(0)
        blob = {"state_dict": SiblingReader().state_dict(), "pca_mean": torch.zeros(384),
                "pca_basis": torch.eye(384)[:, :128].contiguous()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.pt"
            torch.save(blob, path)
            scorer = SiblingScorer(path, _StubVisual(), np.zeros((2, 96, 96, 3), np.uint8))
            branches = [_Hist(0, [1, 2]), _Hist(120, [3, 4]), _Hist(250, [5, 6])]
            a = scorer.scores(_Hist(10, [0, 0]), branches)
            b = scorer.scores(_Hist(10, [0, 0]), branches)
        self.assertEqual(len(a), 3)
        self.assertEqual(a, b)
        self.assertTrue(all(np.isfinite(a)))
