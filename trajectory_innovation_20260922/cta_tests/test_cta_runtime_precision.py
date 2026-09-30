"""A caller's scorer autocast must not change the frozen feature contract."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ti_wm.cta_runtime import Planner


class RuntimePrecisionTests(unittest.TestCase):
    def test_features_and_projection_ignore_outer_autocast(self):
        seen = []

        def features(frames):
            seen.append(torch.is_autocast_enabled('cpu'))
            return torch.arange(256 * 6, dtype=torch.float32).view(1, 256, 6).expand(len(frames), -1, -1) / 100

        planner = object.__new__(Planner)
        planner.device = torch.device('cpu')
        planner.visual = SimpleNamespace(features=features)
        planner.mean = torch.zeros(6)
        planner.basis = torch.tensor([[.31, .72], [.22, .43], [.15, .83], [.29, .51], [.45, .64], [.36, .91]])
        frames = np.zeros((2, 3, 8, 8, 3), dtype=np.uint8)
        for side in (None, 8):
            plain = planner.tokens(frames, side)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                nested = planner.tokens(frames, side)
            self.assertTrue(torch.equal(plain, nested))
            self.assertEqual(plain.dtype, torch.float16)
            self.assertEqual(plain.shape[:2], (2, 3))
        self.assertEqual(seen, [False] * 4)
