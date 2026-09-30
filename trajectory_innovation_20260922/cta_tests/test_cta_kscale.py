"""K-scaling additions to ti_wm.cta_batch: larger policy draws, GEOM64, sampled-code reader answers."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ti_wm.contract import candidate_seed  # noqa: E402
from ti_wm.cta import Scorer  # noqa: E402
from ti_wm.cta_batch import CODE_SAMPLES, K, BatchScorer, run_arm  # noqa: E402
from ti_wm.cta_parallel import ParallelFSQWM  # noqa: E402


class _Env:
    success_threshold = .95

    def __init__(self, x):
        self.block = SimpleNamespace(position=(256. + x, 256.), angle=np.pi / 4)

    def close(self):
        pass


def _branch(x=0., t=0):
    obs = {"pixels": np.zeros((2, 2, 3), np.uint8), "agent_pos": np.zeros(2)}
    return SimpleNamespace(env=_Env(x), hist=[obs, obs], t=t, success=False, coverage=.5, max_coverage=.9)


class LargerDraw(unittest.TestCase):
    def test_draw_keeps_k_bank_and_geom64_sees_whole_bank(self):
        draw = 4 * K
        seen = []

        class Runner:
            def draw(self, hists, seeds):
                seen.append(list(seeds))
                return np.zeros((len(seeds), 8, 2), np.float32)

        seg = []

        def segment(state, chunk, cloner):
            k = len(seg) % draw
            seg.append(k)
            out = _branch(x=-4. * abs(k - 20) - (0 if k != 3 else -60.), t=state.t + 8)  # 20 best overall, 3 best of 0-7
            out.coverage = .01 * k
            out.max_coverage = max(state.max_coverage, out.coverage)
            return out, np.zeros((3, 2, 2, 3), np.uint8)

        class Scores:
            def scores(self, states, chunks, branches, segs, names):
                assert chunks.shape[1] == draw and len(branches) == len(states) * draw
                return {n: np.tile(np.arange(draw, dtype=np.float32), (len(states), 1)) for n in names}

        reset = lambda root: _branch(t=0)
        done = lambda s: s.t >= 8
        for arm, expect in (("P0", 0), ("GEOM8", 3), ("GEOM64", 20), ("X", draw - 1)):
            seen.clear()
            seg.clear()
            _, log, _ = run_arm(arm, [5, 6], Runner(), None, Scores(), ["X"], reset, done, segment, draw=draw)
            self.assertTrue((log["chosen"] == expect).all(), (arm, log["chosen"]))
            self.assertEqual(log["geom"].shape, (2, draw))
            # candidate k of root r at decision 0 is candidate_seed(r, 0, k): the K-bank is a prefix of the draw
            self.assertEqual(seen[0][:draw], [candidate_seed(5, 0, k) for k in range(draw)])
            self.assertEqual(seen[0][draw:], [candidate_seed(6, 0, k) for k in range(draw)])


class SampledCodes(unittest.TestCase):
    goals = torch.randn(2, 256, 128).half()

    def _scorer(self, wm, reader, stat):
        planner = SimpleNamespace(device=torch.device("cpu"), models={}, goals=self.goals)
        return BatchScorer(planner, {"S": ("parallel_rs", (wm, reader, stat))})

    def test_confident_wm_gives_the_expected_code_answer_and_zero_spread(self):
        torch.manual_seed(0)
        wm, reader = ParallelFSQWM(m=16), Scorer("code", m=16)
        with torch.no_grad():                     # a confident WM: one digit per coordinate dominates
            wm.head.weight.zero_()
            wm.head.bias.copy_(torch.cat([torch.nn.functional.one_hot(torch.tensor(j % n), n) * 60.
                                          for j, n in enumerate(wm.levels)]))
        wm.eval(), reader.eval()
        n = 3
        ctx = {"cur": torch.randn(n, 256, 128).half(), "prev": torch.randn(n, 64, 128).half(),
               "prop": torch.randn(n, 4)}
        act = torch.randn(n, 8, 4)
        mean = self._scorer(wm, reader, "mean")._one("S", ctx, act, None)
        spread = self._scorer(wm, reader, "std")._one("S", ctx, act, None)
        from ti_wm.cta import goal_scores
        expected, _ = wm(ctx, act)
        ref = goal_scores(reader, ctx, expected, self.goals)
        self.assertEqual(mean.shape, (n,))
        self.assertTrue(torch.allclose(mean, ref, atol=1e-4), (mean, ref))
        self.assertTrue(torch.all(spread < 1e-4))
        self.assertGreaterEqual(CODE_SAMPLES, 2)

    def test_mean_and_std_share_one_sampling_pass_per_step(self):
        torch.manual_seed(1)
        wm, reader = ParallelFSQWM(m=16).eval(), Scorer("code", m=16).eval()
        planner = SimpleNamespace(device=torch.device("cpu"), models={}, goals=self.goals)
        sc = BatchScorer(planner, {"S": ("parallel_rs", (wm, reader, "mean")),
                                   "D": ("parallel_rs", (wm, reader, "std"))})
        n = 2
        ctx = {"cur": torch.randn(n, 256, 128).half(), "prev": torch.randn(n, 64, 128).half(),
               "prop": torch.randn(n, 4)}
        act = torch.randn(n, 8, 4)
        sc._step = 1
        mean = sc._one("S", ctx, act, None)
        forward, calls = wm.forward, []
        wm.forward = lambda *a: calls.append(1) or forward(*a)
        spread = sc._one("D", ctx, act, None)                 # same step: cached answers, no second WM pass
        self.assertEqual(calls, [])
        self.assertTrue(torch.all(spread > 0))
        uncached = BatchScorer(planner, {"S": ("parallel_rs", (wm, reader, "mean"))})._one("S", ctx, act, None)
        self.assertTrue(torch.allclose(mean, uncached))       # the cache holds the same seeded samples
        calls.clear()
        sc._step = 2
        sc._one("D", ctx, act, None)                          # a new step recomputes
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
