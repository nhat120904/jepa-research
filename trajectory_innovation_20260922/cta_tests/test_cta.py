import unittest

import numpy as np
import torch
from torch import nn

from ti_wm.cta import (
    LEVELS, CodeWM, FutureDecoder, IndexedFSQ, Scorer, SourceEncoder, action_features, goal_scores,
    saturation_penalty, sibling_contrast, vocab_size,
)
from ti_wm.cta_eval import bank_distinct, perplexity, ranking_metrics

SMALL = dict(width=16, heads=2, dim=8)


def inputs(b=3, dim=8):
    ctx = {"cur": torch.randn(b, 256, dim), "prev": torch.randn(b, 64, dim), "prop": torch.rand(b, 4)}
    fut = {"end": torch.randn(b, 256, dim), "seg": torch.randn(b, 3, 64, dim), "prop": torch.rand(b, 4)}
    return ctx, fut, torch.randn(b, 8, 4), torch.randn(b, 256, dim)


class CTATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_fsq_index_bijection(self):
        fsq = IndexedFSQ(LEVELS)
        self.assertEqual(vocab_size(), 256)
        self.assertEqual(tuple(fsq.codebook.shape), (256, len(LEVELS)))
        self.assertTrue(torch.equal(fsq.codes_to_indices(fsq.codebook), torch.arange(256)))
        codes = fsq(torch.randn(2000, len(LEVELS)) * 3)
        idx = fsq.codes_to_indices(codes)
        self.assertTrue(int(idx.min()) >= 0 and int(idx.max()) < 256)
        self.assertTrue(torch.allclose(fsq.indices_to_codes(idx), codes))

    def test_shapes(self):
        ctx, fut, act, goal = inputs()
        for conditional in (True, False):
            for path in (True, False):
                code = SourceEncoder(4, layers=1, conditional=conditional, path=path, **SMALL)(ctx, fut)
                self.assertEqual(tuple(code.shape), (3, 4, len(LEVELS)))
        self.assertEqual(tuple(Scorer("code", layers=1, m=4, **SMALL)(ctx, code, goal).shape), (3,))
        self.assertEqual(tuple(Scorer("future", layers=1, **SMALL)(ctx, fut, goal).shape), (3,))
        self.assertEqual(tuple(Scorer("action", layers=1, **SMALL)(ctx, act, goal).shape), (3,))
        end, seg = FutureDecoder(4, layers=1, **SMALL)(ctx, code)
        self.assertEqual((tuple(end.shape), tuple(seg.shape)), ((3, 256, 8), (3, 3, 64, 8)))

    def test_source_code_gradient(self):
        ctx, fut, _, goal = inputs()
        enc = SourceEncoder(4, layers=1, **SMALL)
        Scorer("code", layers=1, m=4, **SMALL)(ctx, enc(ctx, fut), goal).sum().backward()
        self.assertTrue(enc.queries.grad is not None and float(enc.queries.grad.abs().sum()) > 0)

    def test_wm_is_causal_and_greedy_consistent(self):
        ctx, _, act, _ = inputs()
        wm = CodeWM(4, enc_layers=1, dec_layers=1, **SMALL).eval()
        mem = wm.encode(ctx, act)
        idx = torch.randint(0, 256, (3, 4))
        changed = idx.clone()
        changed[:, 2:] = (changed[:, 2:] + 1) % 256
        self.assertTrue(torch.allclose(wm.logits(mem, idx)[:, :3], wm.logits(mem, changed)[:, :3], atol=1e-5))
        greedy = wm.decode(mem)
        self.assertTrue(torch.equal(wm.logits(mem, greedy).argmax(-1), greedy))
        prior = CodeWM(4, enc_layers=1, dec_layers=1, use_actions=False, **SMALL)
        self.assertEqual(tuple(prior.logits(prior.encode(ctx), idx).shape), (3, 4, 256))

    def test_saturation_penalty_and_pre(self):
        ctx, fut, _, _ = inputs()
        code, pre = SourceEncoder(4, layers=1, **SMALL)(ctx, fut, return_pre=True)
        self.assertEqual(pre.shape, code.shape)
        self.assertEqual(float(saturation_penalty(torch.tensor([0.5, -1.4]))), 0.0)
        self.assertAlmostEqual(float(saturation_penalty(torch.tensor([2.5, -1.5]))), 0.5)

    def test_sibling_contrast(self):
        ctx, _, _, _ = inputs(b=2 * 4)
        act = torch.randn(8, 8, 4)
        wm = CodeWM(3, enc_layers=1, dec_layers=1, **SMALL)
        idx = torch.randint(0, 256, (8, 3))
        idx[1] = idx[0]                                              # bank 0: siblings 0 and 1 share a code
        loss, acc = sibling_contrast(wm, wm.encode(ctx, act), idx, 4)
        self.assertTrue(torch.isfinite(loss) and 0 <= float(acc) <= 1)
        loss.backward()
        self.assertTrue(wm.act.weight.grad.abs().sum() > 0)          # the contrast trains the action pathway

    def test_action_features(self):
        chunk = torch.tensor([[[100.0, 200.0]] * 8])
        f = action_features(chunk, torch.tensor([90.0, 210.0]))
        np.testing.assert_allclose(f[0, 0].numpy(), [100 / 512 - 0.5, 200 / 512 - 0.5, 10 / 64, -10 / 64], rtol=1e-6)
        batched = action_features(chunk[None].expand(2, 3, 8, 2), torch.tensor([[90.0, 210.0]] * 2)[:, None])
        self.assertTrue(torch.allclose(batched[1, 2], f[0]))

    def test_goal_scores_average_over_goals(self):
        class Probe(nn.Module):
            def forward(self, ctx, x, goal):
                return ctx["cur"].mean((1, 2)) * 100 + goal.mean((1, 2))

        ctx, _, act, _ = inputs(b=2)
        goals = torch.randn(5, 256, 8)
        out = goal_scores(Probe(), ctx, act, goals)
        expect = ctx["cur"].mean((1, 2)) * 100 + goals.mean((1, 2)).mean()
        self.assertTrue(torch.allclose(out, expect, atol=1e-4))

    def test_ranking_metrics(self):
        labels = np.array([[0.1, 0.5, 0.3], [0.2, 0.2, 0.9], [0.4, 0.1, 0.2]])
        roots = np.array([1, 2, 2])
        perfect = ranking_metrics(labels, labels, roots, ci=False)
        self.assertAlmostEqual(perfect["within_bank_spearman"]["ratio"], 1.0)
        self.assertAlmostEqual(perfect["retained_gap"]["ratio"], 1.0)
        default = ranking_metrics(np.zeros_like(labels), labels, roots, ci=False)
        self.assertEqual(default["chose_default"], 1.0)
        self.assertAlmostEqual(default["within_bank_spearman"]["ratio"], 0.0)
        self.assertAlmostEqual(default["retained_gap"]["ratio"], 0.0)
        self.assertEqual(bank_distinct(np.zeros((2, 4, 3), int)), 0.0)
        self.assertEqual(bank_distinct(np.arange(12).reshape(1, 4, 3)), 1.0)
        self.assertAlmostEqual(perplexity(np.arange(256)[:, None], 256), 256.0, places=3)


if __name__ == "__main__":
    unittest.main()
