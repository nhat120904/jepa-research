"""MAP-grid and explicit audit-label alignment; CPU Slurm compute node only."""
import inspect
import unittest

import numpy as np
import torch

from scripts.cta_code_prediction_audit import aligned_labels, map_code
from ti_wm.contract import require_compute
from ti_wm.cta import IndexedFSQ, LEVELS
from ti_wm.cta_parallel import ParallelFSQWM


def setUpModule():
    require_compute()


class CodePredictionAudit(unittest.TestCase):
    def test_coordinate_map_is_exactly_on_source_fsq_grid(self):
        indices = [torch.tensor([[2, 7], [3, 0]]), torch.tensor([[1, 6], [7, 2]]), torch.tensor([[3, 1], [0, 2]])]
        logits, grids = [], []
        for count, digits in zip(LEVELS, indices):
            grid = (torch.arange(count).float() - count // 2) / (count // 2)
            probability = torch.full((*digits.shape, count), -1000.)
            probability.scatter_(-1, digits[..., None], 1000.)
            logits.append(probability)
            grids.append(grid)
        hard = map_code(logits, grids)
        self.assertEqual(tuple(hard.shape), (2, 2, 3))
        for coordinate, (grid, digits) in enumerate(zip(grids, indices)):
            torch.testing.assert_close(hard[..., coordinate], grid[digits], rtol=0, atol=0)
        fsq = IndexedFSQ(LEVELS)
        torch.testing.assert_close(fsq.indices_to_codes(fsq.codes_to_indices(hard)), hard, rtol=0, atol=0)
        # Deterministic first-index tie rule, including the lower grid boundary.
        tie = map_code([torch.zeros(1, 2, count) for count in LEVELS], grids)
        torch.testing.assert_close(tie, torch.full((1, 2, 3), -1.), rtol=0, atol=0)

    def test_map_prediction_does_not_receive_or_depend_on_future(self):
        self.assertEqual(list(inspect.signature(map_code).parameters), ["logits", "values"])
        torch.manual_seed(23)
        wm = ParallelFSQWM(m=2, width=16, layers=1, heads=2, dim=6, chunk=15).eval()
        ctx = {"cur": torch.randn(2, 256, 6), "prev": torch.randn(2, 64, 6), "prop": torch.randn(2, 4)}
        actions = torch.randn(2, 15, 4)
        future = torch.randn(2, 256, 6)
        values = tuple(getattr(wm, f"values_{coordinate}") for coordinate in range(len(LEVELS)))
        with torch.no_grad():
            _, before = wm(ctx, actions)
            first = map_code(before, values)
            future.mul_(1000.).add_(777.)
            _, after = wm(ctx, actions)
            second = map_code(after, values)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_alignment_rejects_reordered_scores_and_duplicate_labels(self):
        root, decision = np.array([2000, 2000, 2001]), np.array([0, 1, 0])
        geometry = np.arange(6).reshape(3, 2).astype(np.float32)
        hits = np.array([[False, True], [False, False], [True, False]])
        output_geom, output_hits = aligned_labels(root, decision, geometry, hits, root.copy(), decision.copy())
        np.testing.assert_array_equal(output_geom, geometry)
        np.testing.assert_array_equal(output_hits, hits)
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            aligned_labels(root, decision, geometry, hits, root[::-1], decision[::-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aligned_labels(np.array([2000, 2000, 2000]), np.array([0, 0, 1]), geometry, hits,
                           np.array([2000, 2000, 2000]), np.array([0, 0, 1]))
        with self.assertRaisesRegex(ValueError, "Label bank shapes"):
            aligned_labels(root, decision, geometry[:-1], hits, root, decision)


if __name__ == "__main__":
    unittest.main()
