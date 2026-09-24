import unittest

import torch

from ti_wm.codec import FSQ, LEVELS, CodeDecoder, CodeEncoder, CodeReader, bits_per_token, pool_grid


class CodecTests(unittest.TestCase):
    def test_fsq_levels_and_gradient(self):
        fsq = FSQ(LEVELS)
        z = (torch.randn(4096, len(LEVELS)) * 3).requires_grad_()
        q = fsq(z)
        for i, n in enumerate(LEVELS):
            self.assertLessEqual(len(torch.unique(q[:, i])), n)
        q.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0)
        self.assertAlmostEqual(bits_per_token(), 9 + 3 * torch.log2(torch.tensor(5.0)).item(), places=4)

    def test_pool_grid(self):
        x = torch.arange(256.0).view(1, 256, 1)
        p = pool_grid(x, 4)
        self.assertEqual(tuple(p.shape), (1, 16, 1))
        self.assertAlmostEqual(float(p[0, 0, 0]), float(torch.tensor([r * 16 + c for r in range(4) for c in range(4)]).float().mean()))

    def test_shapes_all_modes(self):
        b, dim = 3, 16
        ctx, end, seg = torch.randn(b, 256, dim), torch.randn(b, 256, dim), torch.randn(b, 3, 64, dim)
        for mode in ("cond", "uncond", "traj", "pool"):
            enc = CodeEncoder(16, mode, dim=dim, width=32, layers=1, heads=2)
            code = enc(ctx, end, seg)
            self.assertEqual(tuple(code.shape), (b, 16, len(LEVELS)), mode)
        reader = CodeReader(len(LEVELS), 16, dim=dim, width=32, layers=1, heads=2)
        self.assertEqual(tuple(reader(ctx, code, ctx, torch.zeros(b, 4)).shape), (b,))
        full = CodeReader(dim, 256, dim=dim, width=32, layers=1, heads=2)
        self.assertEqual(tuple(full(ctx, end, ctx, torch.zeros(b, 4)).shape), (b,))
        dec = CodeDecoder(len(LEVELS), 16, dim=dim, width=32, layers=1, heads=2)
        self.assertEqual(tuple(dec(ctx, code).shape), (b, 256, dim))

    def test_uncond_ignores_context(self):
        torch.manual_seed(0)
        enc = CodeEncoder(4, "uncond", dim=8, width=16, layers=1, heads=2)
        end = torch.randn(2, 256, 8)
        self.assertTrue(torch.equal(enc(torch.randn(2, 256, 8), end), enc(torch.randn(2, 256, 8), end)))


if __name__ == "__main__":
    unittest.main()
