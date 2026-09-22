"""CPU compute-node tests: algebra, gradients, causality, and boundary regression."""

from __future__ import annotations

import argparse
import json
import os
import unittest
from pathlib import Path

import torch

from .signatures import chen, flatten, integrated_lift, interval, inverse, line_signature, signature
from .temporal import causal_window, continue_memory, memory_positions


class SignatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.path = torch.randn(2, 15, 3, dtype=torch.float64).cumsum(-2) / 10

    def same(self, a, b):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)

    def test_analytic_straight_line(self):
        x = torch.tensor([[0., 0.], [2., 3.]], dtype=torch.float64)
        s = signature(x)
        torch.testing.assert_close(s[1], x[-1])
        torch.testing.assert_close(s[2], torch.tensor([2., 3., 3., 4.5], dtype=torch.float64))
        self.same(s, line_signature(x[-1], 3))

    def test_analytic_xy_order(self):
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.]], dtype=torch.float64)
        yx = xy.flip(-1)
        a, b = signature(xy), signature(yx)
        torch.testing.assert_close(a[1], b[1])
        self.assertEqual(float(a[2][1]), 1.)
        self.assertEqual(float(a[2][2]), 0.)
        self.assertEqual(float(b[2][1]), 0.)
        self.assertEqual(float(b[2][2]), 1.)

    def test_all_splits_and_degrees(self):
        for degree in (1, 2, 3):
            whole = signature(self.path, degree)
            for k in range(self.path.shape[-2]):
                self.same(whole, chen(signature(self.path[:, :k+1], degree),
                                      signature(self.path[:, k:], degree)))

    def test_associative_three_parts(self):
        a, b, c = (signature(self.path[:, i:j]) for i, j in ((0, 5), (4, 10), (9, 15)))
        self.same(chen(chen(a, b), c), chen(a, chen(b, c)))
        self.same(signature(self.path), chen(chen(a, b), c))

    def test_identity_and_inverse(self):
        s, unit = signature(self.path), signature(self.path[:, :1])
        self.same(chen(s, unit), s)
        self.same(chen(unit, s), s)
        self.same(chen(s, inverse(s)), unit)
        self.same(chen(inverse(s), s), unit)
        self.same(inverse(s), signature(self.path.flip(-2)))

    def test_prefix_intervals(self):
        prefix = signature(self.path, stream=True)
        for a, b in ((0, 14), (4, 9), (5, 5), (13, 14)):
            self.same(interval(prefix, a, b), signature(self.path[:, a:b+1]))

    def test_translation_and_collinear_refinement(self):
        self.same(signature(self.path), signature(self.path + 47))
        p = torch.tensor([[0., 0.], [1., 2.]], dtype=torch.float64)
        finer = torch.stack([p[0], p[1] / 3, p[1] * 2 / 3, p[1]])
        self.same(signature(p), signature(finer))

    def test_integrated_lift_shared_boundary(self):
        f = torch.randn(15, 4, dtype=torch.float64)
        whole = signature(integrated_lift(f))
        left = signature(integrated_lift(f[:6]))
        right = signature(integrated_lift(f[5:]))
        self.same(whole, chen(left, right))
        # Omitting a shared sample loses the connecting integration interval.
        bad = chen(left, signature(integrated_lift(f[6:])))
        self.assertFalse(torch.allclose(flatten(whole), flatten(bad)))

    def test_autograd(self):
        p = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: flatten(signature(x, 3)), (p,)))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            signature(torch.empty(0, 2))
        with self.assertRaises(ValueError):
            signature(self.path.half())
        with self.assertRaises(ValueError):
            chen(signature(self.path, 2), signature(self.path, 3))


class TinyMemory(torch.nn.Module):
    """Order-sensitive recurrence; no external model/checkpoint."""
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(6, 4, batch_first=True).double()

    def tokens(self, frames, actions):
        return torch.cat([frames, actions], -1)

    def forward(self, tokens, mask, h):
        return self.gru(tokens, h.unsqueeze(0))[1][0]


class TemporalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.obs = torch.arange(40, dtype=torch.float64).unsqueeze(-1)
        self.proprio = self.obs + 100
        self.actions = self.obs + 1000

    def test_prefix_independent_of_candidate(self):
        a = causal_window(self.obs, self.proprio, self.actions, 8, 10, 4, 4)
        altered = self.actions.clone()
        altered[8:] += 300
        b = causal_window(self.obs, self.proprio, altered, 8, 10, 4, 4)
        for key in ("hist_visual", "hist_proprio", "hist_actions", "cur_visual", "cur_proprio"):
            torch.testing.assert_close(a[key], b[key])
        self.assertFalse(torch.equal(a["actions"], b["actions"]))
        self.assertEqual(float(a["hist_actions"][-1]), 1007.)

    def test_reset_masks_and_boundaries(self):
        b = causal_window(self.obs, self.proprio, self.actions, 0, 8, 4, 4)
        self.assertEqual(b["hist_mask"].tolist(), [False, False, False, True])
        self.assertFalse(bool(b["hist_action_mask"].any()))
        self.assertEqual(float(b["hist_actions"].abs().sum()), 0.)
        self.assertEqual(b["path_visual"].shape[0], 9)
        self.assertEqual(b["fut_visual"].flatten().tolist(), list(range(1, 9)))
        left = causal_window(self.obs, self.proprio, self.actions, 0, 3)
        right = causal_window(self.obs, self.proprio, self.actions, 3, 5)
        torch.testing.assert_close(left["path_visual"][-1], right["path_visual"][0])
        torch.testing.assert_close(torch.cat([left["actions"], right["actions"]]), b["actions"])
        with self.assertRaises(ValueError):
            causal_window(self.obs, self.proprio, self.actions, 39, 1)

    def test_memory_inputs_partition_invariant(self):
        frames, actions = torch.randn(2, 19, 3, dtype=torch.float64), torch.randn(2, 19, 3, dtype=torch.float64)
        memory, h0 = TinyMemory(), torch.randn(2, 4, dtype=torch.float64)
        whole = continue_memory(memory, h0, frames, actions)
        for cuts in ((8, 19), (3, 7, 19), (1, 2, 13, 19), (19,)):
            h, start = h0, 0
            for end in cuts:
                h = continue_memory(memory, h, frames[:, start:end], actions[:, start:end], offset=start)
                start = end
            torch.testing.assert_close(h, whole, atol=1e-12, rtol=1e-12)

    def test_no_next_action_needed_at_boundary(self):
        frames, actions = torch.randn(2, 16, 3, dtype=torch.float64), torch.randn(2, 16, 3, dtype=torch.float64)
        memory, h0 = TinyMemory(), torch.randn(2, 4, dtype=torch.float64)
        first = continue_memory(memory, h0, frames[:, :8], actions[:, :8])
        actions[:, 8:] += 100
        second = continue_memory(memory, h0, frames[:, :8], actions[:, :8])
        torch.testing.assert_close(first, second)
        self.assertEqual(memory_positions(8).tolist(), [3, 7])
        # Explicit regression: the historical pos+1/clamp rule changes index at a boundary.
        pos = torch.tensor([3, 7])
        self.assertNotEqual(int((pos + 1).clamp(max=15)[-1]), int((pos + 1).clamp(max=7)[-1]))


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__, fromlist=["_"]))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {"passed": result.wasSuccessful(), "tests": result.testsRun,
            "failures": [(str(t), e) for t, e in result.failures],
            "errors": [(str(t), e) for t, e in result.errors]}


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run tests through sbatch on a compute node")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    result = run_tests()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
