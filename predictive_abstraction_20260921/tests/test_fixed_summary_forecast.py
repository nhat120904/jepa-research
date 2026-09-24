import unittest
from pa_wm.runtime import require_slurm
require_slurm()
import torch
from pa_wm.fixed_summary_forecast import AlignedForecast,bins,expand_bins,time_queries
from pa_wm.metric_bridge import fixed_compression


class FixedSummaryForecastTests(unittest.TestCase):
    def test_summary_matches_validated_reference(self):
        torch.manual_seed(7)
        for h in (32,48,64):
            future=torch.randn(2,h,64)
            torch.testing.assert_close(expand_bins(bins(future),h),fixed_compression(future,16))

    def test_decoder_time_is_known_before_execution(self):
        for h in (32,48,64):
            features=time_queries(h,h,"cpu",torch.float32)
            torch.testing.assert_close(features[:,0],torch.arange(1,h+1)/h)
            torch.testing.assert_close(features[:,2],torch.full((h,),h/48.))
            centers=time_queries(16,h,"cpu",torch.float32)[:,0]
            self.assertTrue(bool((centers[1:]>centers[:-1]).all()))

    def test_matched_parameters_shapes_and_gradients(self):
        torch.set_num_threads(1)
        counts=[]
        history=torch.randn(2,4,4,8); actions=torch.randn(2,32,2)
        for compact in (False,True):
            torch.manual_seed(2)
            model=AlignedForecast(4,8,compact,width=32)
            counts.append(sum(p.numel() for p in model.parameters()))
            full,slots=model(history,actions)
            self.assertEqual(tuple(full.shape),(2,32,64))
            self.assertEqual(tuple(slots.shape),(2,16 if compact else 32,64))
            self.assertFalse(torch.allclose(full,model(history,actions.flip(1))[0]))
            full.square().mean().backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()
                                   if p.grad is not None),0.)
        self.assertEqual(counts[0],counts[1])


if __name__=="__main__": unittest.main()
