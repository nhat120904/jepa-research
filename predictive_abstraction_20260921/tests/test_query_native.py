import unittest
from pa_wm.runtime import require_slurm
require_slurm()
import torch
from pa_wm.query_native import QueryNative, query_loss


class QueryNativeTests(unittest.TestCase):
    def test_query_independent_representation_and_shapes(self):
        history, actions = torch.randn(2, 4, 4, 8), torch.randn(2, 32, 2)
        for kind, tokens in [('summary', 16), ('frame', 32), ('direct', 36)]:
            model = QueryNative(4, 8, kind, width=32).eval()
            z = model.encode(history, actions)
            self.assertEqual(tuple(z.shape), (2, tokens, 32))
            anchors = torch.randn(2, 2, 64)
            first = model.decode(z, anchors, 32)
            second = model.decode(z, anchors+1, 32)
            self.assertEqual(tuple(first.shape), (2, 32, 2))
            self.assertFalse(torch.allclose(first, second))
            torch.testing.assert_close(z, model.encode(history, actions))

    def test_gradients_from_query_loss_reach_action_encoder(self):
        model = QueryNative(4, 8, 'summary', width=32)
        z = model.encode(torch.randn(4, 4, 4, 8), torch.randn(4, 32, 2))
        trace = model.decode(z, torch.randn(4, 2, 64), 32)
        truth = torch.zeros_like(trace)
        truth[:, 5:10, 0] = 1
        truth[:, 15:20, 1] = 1
        loss = query_loss(trace, truth, 2)
        loss.backward()
        gradient = model.context.action.weight.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_single_stratum_loss_is_finite(self):
        for value in (0., 1.):
            pred = torch.full((4, 8, 2), .5, requires_grad=True)
            loss = query_loss(pred, torch.full_like(pred, value), 2)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertTrue(bool(torch.isfinite(pred.grad).all()))

    def test_zero_action_candidates_tie_for_shared_queries(self):
        model = QueryNative(4, 8, width=32).eval()
        history = torch.randn(1, 4, 4, 8).expand(8, -1, -1, -1)
        anchor = torch.randn(1, 2, 64).expand(8, -1, -1)
        z = model.encode(history, torch.zeros(8, 32, 2))
        pred = model.decode(z, anchor, 32)
        torch.testing.assert_close(pred, pred[:1].expand_as(pred))


if __name__ == '__main__':
    unittest.main()
