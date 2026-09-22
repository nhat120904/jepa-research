import unittest
from pa_wm.runtime import require_slurm
require_slurm()
import torch
from pa_wm.metric_bridge import (MetricCodec, MetricForecast, fixed_compression,
                                metric_trace, query_vectors)


class MetricBridgeTests(unittest.TestCase):
    def test_frozen_codec_decoder_transmits_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        codec=MetricCodec(4,width=32).eval().requires_grad_(False)
        forecast=MetricForecast(4,8,tokens=4,width=32)
        history=torch.randn(2,4,4,8); actions=torch.randn(2,12,2)
        z=forecast(history,actions)
        self.assertEqual(tuple(z.shape),(2,4,32))
        decoded=codec.decoder(z,12)
        self.assertEqual(tuple(decoded.shape),(2,12,64))
        decoded.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in forecast.parameters()
                               if p.grad is not None),0.)

    def test_frame_forecast_action_interface(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        forecast=MetricForecast(4,8,width=32).eval()
        history=torch.randn(2,4,4,8); actions=torch.randn(2,12,2)
        a=forecast(history,actions)
        b=forecast(history,actions.flip(1))
        self.assertEqual(tuple(a.shape),(2,12,64))
        self.assertFalse(torch.allclose(a,b))

    def test_fixed_compression_preserves_constant(self):
        constant=torch.randn(2,1,64).expand(-1,48,-1)
        for k in (4,16):
            torch.testing.assert_close(fixed_compression(constant,k),constant)

    def test_affinity_and_queries(self):
        torch.manual_seed(3)
        anchor=torch.randn(2,2,64)
        mean=torch.randn(64); std=torch.rand(64)+.1
        normalized=(anchor[:,0:1]-mean)/std
        trace=metric_trace(normalized,anchor,mean,std)
        torch.testing.assert_close(trace[:,:,0],torch.ones(2,1))
        q=query_vectors(anchor)
        self.assertEqual(tuple(q.shape),(2,8,132))
        torch.testing.assert_close(q[:,6,:64],anchor[:,0])
        torch.testing.assert_close(q[:,7,:64],anchor[:,1])


if __name__=="__main__": unittest.main()
