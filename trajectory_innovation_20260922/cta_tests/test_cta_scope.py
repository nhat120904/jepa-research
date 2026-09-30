"""Information-access and initialization checks for the scope comparison."""
import unittest
import sys
from pathlib import Path

import torch

from ti_wm.cta_scope import build, future_evidence, shared_initial_hash


class ScopeTests(unittest.TestCase):
    def test_shared_initialization_and_information_access(self):
        opts = dict(m=4, width=16, heads=2, dim=8, encoder_layers=1, reader_layers=1, decoder_layers=1)
        end, path = build(0, False, **opts), build(0, True, **opts)
        self.assertEqual(shared_initial_hash(end), shared_initial_hash(path))
        ctx = {'cur': torch.randn(2, 256, 8), 'prev': torch.randn(2, 64, 8), 'prop': torch.randn(2, 4)}
        fut = {'end': torch.randn(2, 256, 8), 'prop': torch.randn(2, 4),
               'seg': torch.randn(2, 3, 64, 8, requires_grad=True)}
        self.assertEqual(set(future_evidence(fut, False)), {'end', 'prop'})
        end_code, pre = end['enc'](ctx, future_evidence(fut, False), return_pre=True)
        pre.sum().backward()
        self.assertIsNone(fut['seg'].grad)
        _, path_pre = path['enc'](ctx, future_evidence(fut, True), return_pre=True)
        path_pre.sum().backward()
        self.assertGreater(float(fut['seg'].grad.abs().sum()), 0.)
        out = end['dec'](ctx, end_code.detach())
        self.assertEqual(out.shape, fut['end'].shape)
        self.assertEqual(end['dec'].queries.shape[0], 256)

    def test_seed_changes_common_initialization(self):
        opts = dict(m=4, width=16, heads=2, dim=8, encoder_layers=1, reader_layers=1, decoder_layers=1)
        self.assertNotEqual(shared_initial_hash(build(0, False, **opts)),
                            shared_initial_hash(build(1, False, **opts)))

    def test_wide_null_interval_is_not_equivalence(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        try:
            from cta_scope_compare import verdict
            self.assertEqual(verdict({'ratio': .01, 'lo': -.10, 'hi': .12}, [.02, 0.]), 'INCONCLUSIVE')
            self.assertEqual(verdict({'ratio': .01, 'lo': -.02, 'hi': .03}, [.02, 0.]),
                             'DIFFERENCE_WITHIN_PRACTICAL_MARGIN_ON_TESTED_SEEDS')
            self.assertEqual(verdict({'ratio': .08, 'lo': .02, 'hi': .14}, [.07, .09]),
                             'TRAJECTORY_ADVANTAGE_SUPPORTED_ON_ENDPOINT_TARGET')
            self.assertEqual(verdict({'ratio': .08, 'lo': .02, 'hi': .14}, [-.01, .17]), 'INCONCLUSIVE')
        finally:
            sys.path.pop(0)
