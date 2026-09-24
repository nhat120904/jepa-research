import unittest
from pa_wm.runtime import require_slurm
require_slurm()
import numpy as np
import torch
from pa_wm.readout_diagnostic import PairReader, selection, event_diagnostic


class ReadoutDiagnosticTests(unittest.TestCase):
    def test_ties_do_not_inherit_default_advantage(self):
        result = selection([0, 0, 0], [0, 1, .5])
        self.assertEqual(result['regret_first'], 1.)
        self.assertEqual(result['regret_uniform_ties'], .5)
        self.assertEqual(result['ties'], 3)
        self.assertEqual(result['pair_accuracy'], .5)
        self.assertEqual(selection([0, 1, .5], [0, 1, .5])['regret_first'], 0.)

    def test_near_ties_and_uninformative_pairs(self):
        result = selection([0, 1e-10], [0, 1])
        self.assertEqual(result['ties'], 2)
        self.assertEqual(result['regret_first'], 0.)
        self.assertEqual(result['regret_uniform_ties'], .5)
        self.assertIsNone(selection([0, 1], [.5, .51])['pair_accuracy'])

    def test_event_miss_timing_and_false_positive(self):
        truth = np.zeros((2, 4, 2))
        truth[0, 1, 0] = .9
        truth[1, 1, 0] = .9
        predicted = np.zeros_like(truth)
        predicted[0, 2, 0] = .8
        predicted[0, 0, 1] = .8
        result = event_diagnostic(predicted, truth)
        self.assertEqual(result['positive_events'], 2)
        self.assertEqual(result['missed_below_0_1'], 1)
        self.assertEqual(result['detected_above_0_5'], 1)
        self.assertEqual(result['false_positive_above_0_5'], 1)
        self.assertEqual(result['peak_timing_error_sum'], 1.)

    def test_pair_reader_broadcast_and_gradient(self):
        torch.manual_seed(26)
        reader = PairReader(8)
        frame = torch.randn(2, 4, 1, 8)
        anchor = torch.randn(2, 1, 2, 8)
        result = reader(frame, anchor)
        self.assertEqual(tuple(result.shape), (2, 4, 2))
        self.assertTrue(bool(((result > 0) & (result < 1)).all()))
        result.square().mean().backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in reader.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in reader.parameters()), 0.)


if __name__ == '__main__':
    unittest.main()
