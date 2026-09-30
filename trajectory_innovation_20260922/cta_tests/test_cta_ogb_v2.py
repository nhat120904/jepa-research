"""scripts/ogbench/ogb_cta_train_v2.py: labels, goal expansion, dropout, and checkpoint compatibility with the v1 eval."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ogbench"))
import ogb_cta_train as v1  # noqa: E402
import ogb_cta_train_v2 as v2  # noqa: E402
from ti_wm.cta import Scorer  # noqa: E402


class V2(unittest.TestCase):
    def test_progress_label(self):
        target = v2.GOAL_XYZ[0]
        cubes = np.stack([target + [0.03, 0, 0], target + [0.05, 0, 0], target + [0, 0.1, 0]]).astype(np.float32)
        lab = v2.progress_label(cubes, target)
        np.testing.assert_allclose(lab, [1.0, -0.05, -0.1], atol=1e-6)
        np.testing.assert_allclose(v2.progress_label(torch.from_numpy(cubes), torch.from_numpy(target)).numpy(), lab,
                                   atol=1e-6)

    def test_expand_order(self):
        b, g, k = 2, 3, v2.K
        x = torch.arange(b * k).float()
        y = v2.expand(x, b, g).view(b, g, k)
        for bi in range(b):
            for gi in range(g):
                self.assertTrue(torch.equal(y[bi, gi], torch.arange(bi * k, (bi + 1) * k).float()))
        d = v2.expand({"cur": x[:, None]}, b, g)
        self.assertEqual(d["cur"].shape, (b * g * k, 1))

    def test_dropout_only_in_train_mode(self):
        torch.manual_seed(0)
        s = Scorer("code", m=4, dropout=0.5, layers=1)
        ctx = {"cur": torch.randn(2, 256, 128), "prev": torch.randn(2, 64, 128), "prop": torch.zeros(2, 4)}
        code, goal = torch.randn(2, 4, 3), torch.randn(2, 256, 128)
        s.eval()
        self.assertTrue(torch.equal(s(ctx, code, goal), s(ctx, code, goal)))
        s.train()
        self.assertFalse(torch.equal(s(ctx, code, goal), s(ctx, code, goal)))

    def test_nested_drop_masks(self):
        from ti_wm.cta import FutureDecoder, nested_drop
        torch.manual_seed(0)
        d = nested_drop(64, 16, "cpu")
        self.assertTrue((~d[:, 0]).all())                         # the first token is always kept
        self.assertTrue(((d.int().diff(dim=1)) >= 0).all())       # hidden tokens form a suffix
        s = Scorer("code", m=16, layers=1).eval()
        dec = FutureDecoder(16, layers=1).eval()
        ctx = {"cur": torch.randn(2, 256, 128), "prev": torch.randn(2, 64, 128), "prop": torch.zeros(2, 4)}
        code, goal = torch.randn(2, 16, 3), torch.randn(2, 256, 128)
        none = torch.zeros(2, 16, dtype=torch.bool)
        self.assertTrue(torch.allclose(s(ctx, code, goal), s(ctx, code, goal, none), atol=1e-4))
        self.assertTrue(torch.allclose(dec(ctx, code)[0], dec(ctx, code, none)[0], atol=1e-4))
        keep1 = torch.ones(2, 16, dtype=torch.bool)
        keep1[:, 0] = False
        code2 = code.clone()
        code2[:, 1:] = torch.randn(2, 15, 3)                      # hidden tokens must not matter
        self.assertTrue(torch.allclose(s(ctx, code, goal, keep1), s(ctx, code2, goal, keep1), atol=1e-4))

    def test_checkpoint_loads_in_v1_builders(self):
        cfg = {"m": 16, "direct_layers": 8, "dropout": 0.1}
        new = v2.build_stage1(cfg, "cpu")
        old = v1.build_stage1({"m": 16, "direct_layers": 8}, "cpu")
        for k in old:
            old[k].load_state_dict(new[k].state_dict(), strict=True)
        wms = v2.build_wms(cfg, "cpu")
        from ti_wm.cta_ogb import ActEndpointWM, ActParallelFSQWM, FrameWM
        for w, cls in ((wms["cta"], ActParallelFSQWM(m=16)), (wms["endpoint"], ActEndpointWM()), (wms["frame"], FrameWM())):
            cls.load_state_dict(w.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
