"""ti_wm.cta_ogb: OGBench action inputs on the PushT CTA modules (shapes, teacher forcing, no future leak)."""
import unittest

import torch

from ti_wm.cta import FutureDecoder, Scorer, SourceEncoder
from ti_wm.cta_ogb import ActEndpointWM, ActParallelFSQWM, FrameWM, action_scorer, fut_from_frames, zeros_prop


def _ctx(b):
    return {"cur": torch.randn(b, 256, 128), "prev": torch.randn(b, 64, 128), "prop": zeros_prop(b, "cpu")}


class CtaOgb(unittest.TestCase):
    def test_shapes(self):
        b = 3
        ctx, act, goal = _ctx(b), torch.randn(b, 5, 5), torch.randn(b, 256, 128)
        frames = torch.randn(b, 4, 256, 128)
        fut = fut_from_frames(frames)
        self.assertEqual(fut["seg"].shape, (b, 3, 64, 128))
        self.assertTrue(torch.equal(fut["end"], frames[:, 3]))
        code = SourceEncoder(16)(ctx, fut)
        self.assertEqual(code.shape, (b, 16, 3))
        self.assertEqual(Scorer("code", m=16)(ctx, code, goal).shape, (b,))
        self.assertEqual(Scorer("future")(ctx, fut, goal).shape, (b,))
        self.assertEqual(action_scorer()(ctx, act, goal).shape, (b,))
        expected, logits = ActParallelFSQWM(16)(ctx, act)
        self.assertEqual(expected.shape, (b, 16, 3))
        self.assertEqual(ActEndpointWM()(ctx, act)["end"].shape, (b, 256, 128))
        end, seg = FutureDecoder(16)(ctx, code)
        self.assertEqual(seg.shape, (b, 3, 64, 128))

    def test_frame_wm_teacher_forcing_and_rollout(self):
        torch.manual_seed(0)
        w = FrameWM()
        for p in w.parameters():                  # non-zero output heads so frames depend on the input frame
            torch.nn.init.normal_(p, std=0.02)
        ctx, act = _ctx(2), torch.randn(2, 5, 5)
        teacher = torch.randn(2, 4, 256, 128)
        tf, roll = w.rollout(ctx, act, teacher), w.rollout(ctx, act)
        self.assertEqual(tf.shape, (2, 4, 256, 128))
        self.assertTrue(torch.allclose(tf[:, 0], roll[:, 0]))       # step 2 only sees frame 0 in both modes
        self.assertFalse(torch.allclose(tf[:, 1], roll[:, 1]))      # later steps differ: teacher vs own prediction


if __name__ == "__main__":
    unittest.main()
