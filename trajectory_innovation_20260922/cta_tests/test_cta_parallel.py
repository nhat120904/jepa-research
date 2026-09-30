import unittest
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from ti_wm.cta import Scorer
from ti_wm.cta_parallel import ParallelFSQWM, EndpointWM, weighted_rank, score_consistency, normalized_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cta_round3


class ParallelTests(unittest.TestCase):
    def test_frozen_reader_transmits_gradient_to_parallel_wm(self):
        torch.manual_seed(7)
        wm = ParallelFSQWM(m=2, width=16, layers=1, heads=2, dim=6)
        reader = Scorer('code', width=16, layers=1, heads=2, m=2, dim=6).eval().requires_grad_(False)
        ctx = {'cur': torch.randn(2, 256, 6), 'prev': torch.randn(2, 64, 6), 'prop': torch.randn(2, 4)}
        code, logits = wm(ctx, torch.randn(2, 8, 4))
        self.assertEqual(code.shape, (2, 2, 3))
        self.assertEqual([p.shape[-1] for p in logits], [8, 8, 4])
        self.assertTrue((code >= -1).all() and (code[..., :2] <= .75).all() and (code[..., 2] <= .5).all())
        scores = reader(ctx, code, torch.randn(2, 256, 6)).view(1, 2)
        weighted_rank(scores, torch.tensor([[0., .02]])).backward()
        self.assertGreater(float(wm.head.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in reader.parameters()))
        self.assertTrue(torch.isfinite(wm.nll(logits, torch.zeros_like(code))))

    def test_endpoint_predicts_frame_and_proprio_without_future(self):
        wm = EndpointWM(width=16, layers=1, heads=2, dim=6)
        ctx = {'cur': torch.randn(2, 256, 6), 'prev': torch.randn(2, 64, 6), 'prop': torch.randn(2, 4)}
        out = wm(ctx, torch.randn(2, 8, 4))
        self.assertEqual(out['end'].shape, (2, 256, 6))
        self.assertEqual(out['prop'].shape, (2, 4))
        self.assertTrue(torch.equal(out['end'], ctx['cur']))
        self.assertTrue(torch.equal(out['prop'], ctx['prop']))

    def test_weighting_consistency_and_score_contract(self):
        scores = torch.tensor([[0., 0.]], requires_grad=True)
        small = weighted_rank(scores, torch.tensor([[0., .002]]))
        large = weighted_rank(scores, torch.tensor([[0., .02]]))
        self.assertAlmostEqual(float(small / large), .2, places=6)
        self.assertEqual(float(weighted_rank(scores, torch.zeros_like(scores))), 0.)
        a = torch.tensor([[1., 2.]])
        self.assertEqual(float(score_consistency(a + 10, a, 1.)), 0.)
        self.assertEqual(normalized_score(.475), .5)
        self.assertEqual(normalized_score(.96), 1.)

    def test_accumulated_rank_gradient_matches_effective_batch(self):
        scores = torch.randn(4, 3, requires_grad=True)
        labels = torch.tensor([[0., 0., 0.], [0., .002, .02], [0., .1, .1], [0., 0., 0.]])
        count = ((labels[:, :, None] - labels[:, None, :]) > .001).sum()
        full = weighted_rank(scores, labels)
        accumulated = sum(weighted_rank(scores[i:i+2], labels[i:i+2], pair_count=count)
                          for i in (0, 2))
        torch.testing.assert_close(full, accumulated)
        torch.testing.assert_close(torch.autograd.grad(full, scores)[0],
                                   torch.autograd.grad(accumulated, scores)[0])

    def test_episode_score_excludes_reset_coverage(self):
        env = SimpleNamespace(success_threshold=.95, close=lambda: None)
        state = SimpleNamespace(env=env, hist=[], max_coverage=.94, coverage=.94, t=0, success=False)
        runner = SimpleNamespace(bank=lambda hist, seeds: [[0.]])

        def step(branch, actions, cloner):
            self.assertEqual(branch.max_coverage, 0.)
            branch.max_coverage = branch.coverage = .475
            branch.t = 1
            return branch

        with patch.object(cta_round3, 'reset_branch', return_value=state), \
             patch.object(cta_round3, 'done', side_effect=lambda b: b.t == 1), \
             patch.object(cta_round3, 'run_prefix', side_effect=step):
            result = cta_round3.episode(2100, 'P0', runner, None, None)
        self.assertEqual(result['score'], .5)
        self.assertTrue(result['reset_excluded'])

    def test_matching_flat_preflight_is_not_a_scientific_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            frames = np.zeros((2, 1, 1, 3), np.uint8)
            np.savez(path / 'raw.npz', root=[2000, 2000], ctx=frames, ctx_prev=frames,
                     ctx_pos=np.zeros((2, 2, 2)), end=np.zeros((2, 8, 1, 1, 3)),
                     end_pos=np.zeros((2, 8, 2)), end_prev_pos=np.zeros((2, 8, 2)),
                     chunk=np.zeros((2, 8, 8, 2)), seg=np.zeros((2, 8, 3, 1, 1, 3)))
            np.savez(path / 'dev_scores.npz', root=[2000, 2000], task=np.zeros((2, 8)))
            planner = SimpleNamespace(scores=lambda *args, **kwargs: [0.] * 8)
            check = cta_round3.closed.preflight
            with self.assertRaises(RuntimeError):
                check(planner, path / 'raw.npz', path, tiers={'CTA3': 'task'})
            report = check(planner, path / 'raw.npz', path, tiers={'CTA3': 'task'}, allow_matching_flat=True)
            self.assertTrue(report['CTA3']['matching_flat_banks_only'])
            planner.scores = lambda *args, **kwargs: [1.] * 8
            with self.assertRaises(RuntimeError):
                check(planner, path / 'raw.npz', path, tiers={'CTA3': 'task'}, allow_matching_flat=True)
