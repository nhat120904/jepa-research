import unittest
from pa_wm.runtime import require_slurm
require_slurm()
import numpy as np
import torch
from pa_wm.branch_data import make_queries, targets
from pa_wm.branch_models import ReconstructionCodec, SpatialForecast
from pa_wm.models import QueryReader


class BranchTests(unittest.TestCase):
    def test_query_is_prospective_and_spatial(self):
        torch.manual_seed(3)
        anchors = torch.randn(2,4,8)
        projection = torch.randn(8,3)
        query = make_queries(anchors,projection)
        self.assertEqual(tuple(query.shape),(8,28))
        # No future/action argument; preserve patch location, not just global average.
        self.assertFalse(torch.equal(query,make_queries(anchors.flip(1),projection)))
        torch.testing.assert_close(query,make_queries(anchors,projection))
        self.assertFalse(torch.equal(query[6],query[7]))

    def test_targets_exclude_current_frame(self):
        a = np.full((8,8,3),255,np.uint8); a[1,1]=[255,0,0]
        b = np.full((8,8,3),255,np.uint8); b[6,6]=[255,0,0]
        first = targets(np.stack([a,b,b]),np.stack([a,b]))
        second = targets(np.stack([b,a,b]),np.stack([a,b]))
        self.assertAlmostEqual(float(first[6]),0,places=5)
        self.assertAlmostEqual(float(second[6]),1,places=5)
        self.assertAlmostEqual(float(second[7]),0,places=5)

    def test_full_spatial_targets_and_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        batch,horizon,patches,dim,width = 2,7,4,8,32
        history = torch.randn(batch,4,patches,dim)
        future = torch.randn(batch,horizon,patches,dim)
        actions = torch.randn(batch,horizon,2)
        generic = ReconstructionCodec(patches,dim,width,2)
        reconstruction = generic.reconstruct(generic(future),horizon)
        self.assertEqual(tuple(reconstruction.shape),tuple(future.shape))
        reconstruction.square().mean().backward()
        for endpoint in (False,True):
            model = SpatialForecast(patches,dim,width,endpoint)
            tokens = model(history,actions)
            count = 1 if endpoint else horizon
            self.assertEqual(tuple(tokens.shape),(batch,count*patches,width))
            reconstructed = model.reconstruct(tokens)
            self.assertEqual(tuple(reconstructed.shape),(batch,count,patches,dim))
            reader = QueryReader(20,width).requires_grad_(False)
            loss = reader(tokens,torch.randn(batch,8,20)).mean()
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                                for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in reader.parameters()))


if __name__ == "__main__": unittest.main()
