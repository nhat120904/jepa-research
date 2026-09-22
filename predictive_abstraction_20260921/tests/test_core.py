import os
import unittest

from pa_wm.runtime import require_slurm

require_slurm()
import numpy as np

from pa_wm.proposals import red_centroid, smooth_bank, visual_goal_bank
from pa_wm.queries import answer, chroma_features, observed_phase, ordered_score


class QueryTests(unittest.TestCase):
    def test_strict_order(self):
        self.assertEqual(float(ordered_score([1, 0], [0, 1])), 1)
        self.assertEqual(float(ordered_score([0, 1], [1, 0])), 0)
        self.assertEqual(float(ordered_score([1], [1])), 0)
        self.assertEqual(float(ordered_score([0, 1], [0, 1])), 0)
        # B then A then B is allowed; must revisit after A.
        self.assertEqual(float(ordered_score([0, 1, 0], [1, 0, 1])), 1)

    def test_against_bruteforce(self):
        rng = np.random.default_rng(0)
        for length in (2, 3, 32):
            a, b = rng.random((2, 7, length))
            expected = np.array([max(min(x[i], y[j]) for i in range(length)
                                    for j in range(i+1, length)) for x, y in zip(a, b)])
            np.testing.assert_allclose(ordered_score(a, b), expected)

    def test_basic_answers(self):
        features = np.eye(2)[[0, 0, 1]]
        result = answer(features, np.array([1, 0]), np.array([0, 1]))
        self.assertEqual(result["a_then_b"], 1)
        self.assertEqual(result["b_then_a"], 0)
        self.assertAlmostEqual(result["occupancy_a"], 2/3)
        self.assertEqual(result["endpoint_b"], 1)

    def test_chroma_no_state(self):
        white = np.full((2, 8, 8, 3), 255, dtype=np.uint8)
        np.testing.assert_equal(chroma_features(white), 0)
        white[0, 2, 2] = [255, 0, 0]
        self.assertAlmostEqual(float(np.linalg.norm(chroma_features(white)[0])), 1, places=5)

    def test_phase_uses_distinct_observations(self):
        self.assertEqual(observed_phase(0, 1, 1, .5), 1)
        self.assertEqual(observed_phase(1, 1, 1, .5), 2)
        self.assertEqual(observed_phase(2, 0, 0, .5), 2)


class ProposalTests(unittest.TestCase):
    def test_bounds_and_replay(self):
        a = smooth_bank(np.random.default_rng(8), 64, 32)
        b = smooth_bank(np.random.default_rng(8), 64, 32)
        self.assertEqual(a.shape, (64, 32, 2))
        np.testing.assert_array_equal(a, b)
        self.assertLessEqual(float(np.linalg.norm(a, axis=-1).max()), 1.800001)
        np.testing.assert_array_equal(a[0], 0)
        self.assertGreater(float(a[1:].std()), .1)

    def test_warm_start_clipped(self):
        a = smooth_bank(np.random.default_rng(1), 8, 32,
                        warm_start=np.full((32, 2), 20))
        self.assertLessEqual(float(np.linalg.norm(a, axis=-1).max()), 1.800001)

    def test_visual_goal_bank_uses_rgb_contract(self):
        def dot(x, y):
            image = np.full((65, 65, 3), 255, dtype=np.uint8)
            image[y, x] = [255, 0, 0]
            return image
        start, a, b = dot(10, 10), dot(25, 30), dot(50, 45)
        np.testing.assert_allclose(red_centroid(a), [25, 30])
        bank = visual_goal_bank(np.random.default_rng(2), start, a, b, 32, 48)
        self.assertEqual(bank.shape, (32, 48, 2))
        self.assertLessEqual(float(np.linalg.norm(bank, axis=-1).max()), 1.800001)
        # The fixed exact candidate reaches both visual centroids in order in free space.
        positions = np.array([10., 10.]) + 2 * np.cumsum(bank[1], axis=0)
        self.assertLess(np.linalg.norm(positions[23] - [25, 30]), 1e-4)
        self.assertLess(np.linalg.norm(positions[-1] - [50, 45]), 1e-4)


class ModelTests(unittest.TestCase):
    def test_shapes_and_frozen_reader_gradient(self):
        import torch
        from pa_wm.models import ObservedCodec, ActionSummaryPredictor, QueryReader, forecast_loss
        torch.set_num_threads(1)
        torch.manual_seed(1)
        codec = ObservedCodec(24, width=32, tokens=3).eval()
        predictor = ActionSummaryPredictor(24, width=32, tokens=3).eval()
        reader = QueryReader(13, width=32).eval()
        for module in (codec, reader):
            module.requires_grad_(False)
        history, future = torch.randn(2, 3, 24), torch.randn(2, 16, 24)
        actions, queries = torch.randn(2, 16, 2), torch.randn(2, 5, 13)
        summary = predictor(history, actions)
        self.assertEqual(tuple(summary.shape), (2, 3, 32))
        self.assertEqual(tuple(reader(summary, queries).shape), (2, 5))
        # Teacher query target is fixed; changing proposed action order must matter.
        self.assertFalse(torch.allclose(summary, predictor(history, actions.flip(1))))
        loss = forecast_loss(predictor, codec, reader, history, actions, future,
                             queries, torch.rand(2, 5), latent_weight=0.1)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in predictor.parameters()
                               if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in reader.parameters()))
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        # Multiple readouts never mutate/recompute the query-independent summary.
        before = summary.detach().clone()
        reader(summary, torch.randn_like(queries))
        torch.testing.assert_close(summary.detach(), before)

    def test_spatial_model_arms(self):
        import torch
        from pa_wm.models import (DirectQueryPredictor, EndpointPredictor,
                                  FrameSequencePredictor, GenericSummaryCodec,
                                  QueryReader, SpatialActionSummary,
                                  SpatialObservedCodec)
        torch.set_num_threads(1)
        batch, history_len, horizon, patches, dim, width = 2, 4, 7, 4, 8, 32
        history = torch.randn(batch, history_len, patches, dim)
        future = torch.randn(batch, horizon, patches, dim)
        actions = torch.randn(batch, horizon, 2)
        queries = torch.randn(batch, 5, 20)
        codec = SpatialObservedCodec(patches, dim, width, 2)
        predictor = SpatialActionSummary(patches, dim, width=width, tokens=2)
        reader = QueryReader(20, width)
        observed, predicted = codec(future), predictor(history, actions)
        self.assertEqual(tuple(observed.shape), (batch, 2, width))
        self.assertEqual(tuple(predicted.shape), (batch, 2, width))
        self.assertEqual(tuple(reader(predicted, queries).shape), (batch, 5))
        generic = GenericSummaryCodec(patches, dim, width, 2)
        self.assertEqual(tuple(generic.reconstruct(generic(future), horizon).shape),
                         (batch, horizon, dim))
        direct = DirectQueryPredictor(20, patches, dim, width)
        frame = FrameSequencePredictor(patches, dim, width)
        endpoint = EndpointPredictor(patches, dim, width)
        self.assertEqual(tuple(direct(history, actions, queries).shape), (batch, 5))
        self.assertEqual(tuple(frame(history, actions).shape), (batch, horizon, width))
        self.assertEqual(tuple(endpoint(history, actions).shape), (batch, 1, width))
        total = (reader(predicted, queries).mean() + direct(history, actions, queries).mean()
                 + frame(history, actions).mean() + endpoint(history, actions).mean())
        total.backward()
        self.assertTrue(all(any(p.grad is not None for p in module.parameters())
                            for module in (predictor, direct, frame, endpoint)))


class EnvironmentTests(unittest.TestCase):
    def test_rgb_and_exact_replay(self):
        from pa_wm.wall_adapter import WallRGB
        env = WallRGB(os.environ["PA_UPSTREAM_ROOT"], seed=55)
        initial = env.reset([15.0, 25.0])
        self.assertEqual(initial.shape, (65, 65, 3))
        self.assertEqual(initial.dtype, np.uint8)
        snapshot = env.snapshot()
        bank = smooth_bank(np.random.default_rng(55), 4, 32)
        images, states = env.rollout(snapshot, bank[1])
        env.rollout(snapshot, bank[2])
        images_again, states_again = env.rollout(snapshot, bank[1])
        np.testing.assert_array_equal(images, images_again)
        np.testing.assert_array_equal(states, states_again)
        np.testing.assert_array_equal(images[0], initial)
        self.assertEqual(images.shape[0], len(bank[1]) + 1)
        env.restore(snapshot)
        env.goal_image([45.0, 40.0])
        np.testing.assert_array_equal(env.evaluation_state(), snapshot["position"])
        # Test wall contact, zero/axis actions, doorway crossing and borders explicitly.
        for start, action in (([26., 20.], [1., 0.]), ([26., 30.], [1., 0.]),
                              ([12., 8.], [0., -1.]), ([12., 8.], [0., 0.])):
            env.reset(start)
            rgb = env.step(action)
            self.assertTrue(np.isfinite(env.evaluation_state()).all())
            self.assertEqual(rgb.shape, (65, 65, 3))


if __name__ == "__main__":
    unittest.main()
