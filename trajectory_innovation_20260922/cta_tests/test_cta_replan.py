"""Replan-interval study (docs/CTA_REPLAN_INTERVAL_PROTOCOL.md): executed chunk length is a parameter, 8 stays default."""
import unittest

import numpy as np
import torch

from ti_wm import dinowm_scorer as dw
from ti_wm.cta import CHUNK, Scorer
from ti_wm.cta_parallel import EndpointWM, ParallelFSQWM
from ti_wm.cta_runtime import KEEP, keep_steps
from ti_wm.pusht_runtime import executed_slice


def context(b, dim=6):
    return {'cur': torch.randn(b, 256, dim), 'prev': torch.randn(b, 64, dim), 'prop': torch.randn(b, 4)}


class ReplanInterval(unittest.TestCase):
    def test_keep_steps_quarter_points(self):
        self.assertEqual(keep_steps(8), KEEP)
        self.assertEqual(keep_steps(15), (4, 8, 11))
        with self.assertRaises(ValueError):
            keep_steps(3)

    def test_executed_slice(self):
        self.assertEqual(executed_slice(1, 9, 16), (1, 9))           # native: unchanged
        self.assertEqual(executed_slice(1, 9, 16, 15), (1, 16))      # all future actions of one sample
        for bad in (0, 16):
            with self.assertRaises(ValueError):
                executed_slice(1, 9, 16, bad)

    def test_default_chunk_keeps_checkpoint_shapes(self):
        self.assertEqual(CHUNK, 8)
        self.assertEqual(tuple(Scorer('action', width=16, layers=1, heads=2, dim=6).ev_pos.shape), (8, 16))
        self.assertEqual(tuple(ParallelFSQWM(m=2, width=16, layers=1, heads=2, dim=6).act_pos.shape), (8, 16))

    def test_modules_accept_longer_chunks(self):
        torch.manual_seed(0)
        ctx, act, goal = context(2), torch.randn(2, 15, 4), torch.randn(2, 256, 6)
        direct = Scorer('action', width=16, layers=1, heads=2, dim=6, chunk=15)
        self.assertEqual(tuple(direct(ctx, act, goal).shape), (2,))
        code, _ = ParallelFSQWM(m=2, width=16, layers=1, heads=2, dim=6, chunk=15)(ctx, act)
        self.assertEqual(tuple(code.shape), (2, 2, 3))
        out = EndpointWM(width=16, layers=1, heads=2, dim=6, chunk=15)(ctx, act)
        self.assertEqual(tuple(out['end'].shape), (2, 256, 6))
        with self.assertRaises(RuntimeError):         # an 8-step module must reject 15-step chunks
            Scorer('action', width=16, layers=1, heads=2, dim=6)(ctx, act, goal)

    def test_dinowm_macro_for_fifteen_actions(self):
        pos, vel = np.array([[100., 200.]]), np.zeros((1, 2))
        chunk = np.linspace([120., 210.], [200., 260.], 15)[None]
        a = dw.macro_actions(pos, vel, chunk, macro=3)
        self.assertEqual(tuple(a.shape), (1, 3, 10))
        # exactly the 15 targets, no held padding: the last macro step ends at the last target
        rel = chunk - dw.pd_positions(pos, vel, chunk)
        ref = (torch.as_tensor(rel, dtype=torch.float32) / dw.ACTION_SCALE - dw.ACTION_MEAN) / dw.ACTION_STD
        torch.testing.assert_close(a, ref.reshape(1, 3, 10))
        self.assertEqual(tuple(dw.macro_actions(pos, vel, chunk[:, :8]).shape), (1, dw.MACRO, 10))


if __name__ == "__main__":
    unittest.main()
