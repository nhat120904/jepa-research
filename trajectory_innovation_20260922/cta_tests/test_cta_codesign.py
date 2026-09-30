"""Tests of the Round-2 objective's value and gradient boundaries."""
import unittest
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ti_wm.cta import IndexedFSQ, fsq_predictability_nll


class CoDesignTests(unittest.TestCase):
    def test_matches_categorical_nll_at_every_codeword(self):
        torch.manual_seed(41)
        fsq = IndexedFSQ()
        code = fsq.codebook.clone().requires_grad_(True)
        logits = torch.randn(256, 256, requires_grad=True)
        loss = fsq_predictability_nll(code, logits, fsq)
        self.assertTrue(torch.allclose(loss, F.cross_entropy(logits, torch.arange(256)), atol=1e-6))
        loss.backward()
        self.assertIsNone(logits.grad)
        self.assertTrue(torch.isfinite(code.grad).all())
        self.assertGreater(float(code.grad.abs().sum()), 0)

    def test_bimodal_prior_does_not_pull_to_unlikely_mean(self):
        fsq = IndexedFSQ((5,))
        logits = torch.tensor([[2., -5., -5., -5., 2.]])
        code = torch.tensor([[-1.]], requires_grad=True)
        fsq_predictability_nll(code, logits, fsq).backward()
        # Moving right from the left mode increases NLL. Mean-code MSE instead
        # has a negative gradient here and would pull toward the unlikely center.
        self.assertGreater(float(code.grad), 0)
        mean_loss_grad = 2 * float(code.detach())
        self.assertLess(mean_loss_grad, 0)

    def test_gradient_reaches_encoder_at_grid_boundary(self):
        fsq = IndexedFSQ((4,))
        z = torch.tensor([[1.5]], requires_grad=True)
        code = fsq(z)
        loss = fsq_predictability_nll(code, torch.tensor([[0., 1., 2., 3.]]), fsq)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0)

    def test_paired_gap_uses_shared_oracle_denominator(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        try:
            from cta_round2 import paired_gap
            labels = np.array([[0., 1.], [0., 2.], [0., 4.]])
            roots = np.array([10, 10, 20])
            baseline = np.zeros_like(labels)
            perfect = labels.copy()
            gain = paired_gap(perfect, baseline, labels, roots)
            self.assertAlmostEqual(gain['ratio'], 1.)
            self.assertAlmostEqual(gain['lo'], 1.)
            self.assertAlmostEqual(gain['hi'], 1.)
            identical = paired_gap(perfect, perfect, labels, roots)
            self.assertAlmostEqual(identical['ratio'], 0.)
            self.assertAlmostEqual(identical['lo'], 0.)
            self.assertAlmostEqual(identical['hi'], 0.)
        finally:
            sys.path.pop(0)


if __name__ == '__main__':
    unittest.main()
