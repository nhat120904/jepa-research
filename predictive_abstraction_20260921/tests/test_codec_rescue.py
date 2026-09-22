import unittest
from pa_wm.runtime import require_slurm
require_slurm()
import torch
from pa_wm.codec_rescue import ObservedReader, candidate_difference_loss, aggregates, FrameMetric


class RescueTests(unittest.TestCase):
    def test_pair_loss_ignores_prefix_offset_and_zero_spread(self):
        y=torch.zeros(4,8)
        y[1,6]=1.; y[3,6]=.5
        self.assertEqual(float(candidate_difference_loss(y,y,2)),0.)
        self.assertGreater(float(candidate_difference_loss(torch.zeros_like(y),y,2)),0.)
        # Different prefix biases cancel; never compare candidates across prefixes.
        offset=torch.tensor([1.,1.,5.,5.])[:,None]
        self.assertAlmostEqual(float(candidate_difference_loss(y+offset,y,2)),0.,places=6)
        zero=torch.zeros(4,8,requires_grad=True)
        loss=candidate_difference_loss(zero,torch.zeros_like(zero),2)
        loss.backward()
        self.assertEqual(float(loss),0.)
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_aggregation_strict_order(self):
        trace=torch.tensor([[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]])
        y=aggregates(trace)
        torch.testing.assert_close(y[:,6],torch.tensor([1.,0.]))
        torch.testing.assert_close(y[:,7],torch.tensor([0.,1.]))
        torch.testing.assert_close(y[:,2:4],torch.full((2,2),.5))

    def test_observed_readers_and_gradients(self):
        torch.set_num_threads(1)
        for compressed in (False,True):
            model=ObservedReader(4,8,20,compressed,width=32)
            x=torch.randn(2,7,4,8)
            q=torch.randn(2,8,20)
            out=model(x,q)
            self.assertEqual(out.shape,(2,8))
            out.mean().backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()
                                   if p.grad is not None),0.)

    def test_metric_identity_and_symmetry(self):
        model=FrameMetric(16,torch.zeros(16),torch.ones(16))
        x,y=torch.randn(3,16),torch.randn(3,16)
        torch.testing.assert_close(model(x,x),torch.ones(3))
        torch.testing.assert_close(model(x,y),model(y,x))
        out=model(x,y)
        self.assertTrue(((out>=0)&(out<=1)).all())


if __name__=="__main__": unittest.main()
