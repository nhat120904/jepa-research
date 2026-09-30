import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cta_diag_onpolicy as diag  # noqa: E402
import cta_round4 as r4  # noqa: E402
from ti_wm.cta import IndexedFSQ, Scorer, action_features  # noqa: E402
from ti_wm.cta_batch import (K, SIGMAS, chosen_retention, headroom, mixed_bank, oracle_step, perturb,  # noqa: E402
                             run_arm)
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


class PerturbAndMix(unittest.TestCase):
    def test_perturb_is_deterministic_bounded_and_scaled(self):
        chunks = np.full((K, 8, 2), 256., np.float32)
        a, b = perturb(chunks, 31051, 3), perturb(chunks, 31051, 3)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, perturb(chunks, 31051, 4)))
        self.assertTrue(((a >= 0) & (a <= 512)).all())
        offsets = (a - chunks)[:, 0]
        self.assertTrue(np.allclose(a - chunks, offsets[:, None]))           # one constant offset per candidate
        many = np.stack([np.linalg.norm((perturb(chunks, r, 0) - chunks)[:, 0], axis=-1) for r in range(400)])
        self.assertLess(many[:, 0].mean(), many[:, -1].mean())              # scale grows with the candidate index
        # |N(0, s^2 I_2)| has mean s * sqrt(pi / 2)
        self.assertAlmostEqual(many[:, -1].mean() / SIGMAS[-1], np.sqrt(np.pi / 2), delta=.15)
        edge = perturb(np.full((K, 8, 2), 511.), 31051, 1)
        self.assertLessEqual(float(edge.max()), 512.)

    def test_mixed_execution_rule(self):
        self.assertFalse(any(oracle_step(31050, d) for d in range(200)))
        share = np.mean([oracle_step(31051, d) for d in range(400)])
        self.assertTrue(.4 < share < .6)
        self.assertEqual(oracle_step(31051, 7), oracle_step(31051, 7))


class SegmentStates(unittest.TestCase):
    def test_state_recording_segment_matches_run_segment(self):
        import copy
        import cta_collect_plus as cp
        from ti_wm.cta_runtime import run_segment
        from ti_wm.pusht_runtime import Branch

        class FakeEnv:
            def __init__(self):
                self.agent = SimpleNamespace(position=(0., 0.), velocity=(0., 0.))
                self.block = SimpleNamespace(position=(1., 1.), angle=0., velocity=(0., 0.), angular_velocity=0.)
                self.n_contact_points, self.k = 0, 0

            def step(self, action):
                self.k += 1
                self.agent.position = tuple(float(x) for x in action)
                self.n_contact_points = self.k
                cov = .1 * self.k
                obs = {"pixels": np.full((2, 2, 3), self.k, np.uint8), "agent_pos": np.asarray(action, float)}
                return obs, 0., cov > .35, False, {"coverage": cov}

        obs = {"pixels": np.zeros((2, 2, 3), np.uint8), "agent_pos": np.zeros(2)}
        start = Branch(env=FakeEnv(), hist=[obs, obs], t=0)
        actions = np.arange(16, dtype=np.float32).reshape(8, 2)
        ref, ref_frames = run_segment(start, actions, copy.deepcopy)
        out, frames, states, contacts, executed = cp.run_segment_states(start, actions, copy.deepcopy)
        self.assertEqual((out.t, out.success, out.coverage, out.max_coverage),
                         (ref.t, ref.success, ref.coverage, ref.max_coverage))
        self.assertTrue(np.array_equal(frames, ref_frames))
        self.assertEqual(executed, 4)
        self.assertEqual(states.shape, (8, 10))
        self.assertEqual(list(contacts), [1, 2, 3, 4, 0, 0, 0, 0])
        self.assertTrue(np.array_equal(states[4:], np.repeat(states[3:4], 4, 0)))
        self.assertEqual(start.env.k, 0)                     # the source branch is untouched


class LockstepRunner(unittest.TestCase):
    def test_run_arm_lockstep_choice_logging_and_reset_exclusion(self):
        calls = []

        class Runner:
            def draw(self, hists, seeds):
                calls.append(len(seeds))
                return np.zeros((len(seeds), 8, 2), np.float32)

        def reset(root):
            return _branch(t=0)

        def segment(state, chunk, cloner):
            k = len(calls_seg) % K
            calls_seg.append(k)
            out = _branch(x=-4. * abs(k - 5), t=state.t + 8)    # candidate 5 lands on the goal
            out.coverage = .1 * k
            out.max_coverage = max(state.max_coverage, out.coverage)
            return out, np.zeros((3, 2, 2, 3), np.uint8)

        calls_seg = []

        class Scores:
            def scores(self, states, chunks, branches, segs, names):
                self.shapes = (len(states), chunks.shape, len(branches))
                return {n: np.tile(np.arange(K, dtype=np.float32)[::-1], (len(states), 1)) for n in names}

        done = lambda s: s.t >= 16
        scorer = Scores()
        for arm, expect in (("P0", 0), ("PHYS8", 7), ("GEOM8", 5), ("X", 0)):
            calls.clear()
            calls_seg.clear()
            ep, log, _ = run_arm(arm, [1, 2, 3], Runner(), None, scorer, ["X"], reset, done, segment)
            self.assertEqual(calls, [24, 24])
            self.assertEqual(len(log["root"]), 6)
            self.assertTrue((log["chosen"] == expect).all(), arm)
            self.assertEqual(log["cov"].shape, (6, K))
            self.assertEqual(log["score_X"].shape, (6, K))
            self.assertEqual(scorer.shapes, (3, (3, K, 8, 2), 3 * K))
            self.assertAlmostEqual(ep[0]["max_coverage"], .1 * expect if arm != "P0" else 0.)  # reset .9 excluded
        with self.assertRaises(ValueError):
            run_arm("Y", [1], Runner(), None, scorer, ["X"], reset, done, segment)

    def test_mixed_bank_restricted_arms_and_record_hook(self):
        """Round 5: 16-candidate bank (policy + perturbed copies); "8" oracles / "@8" arms see only the policy half."""
        chunks = np.random.default_rng(0).uniform(100, 400, (2, K, 8, 2)).astype(np.float32)
        bank = mixed_bank(chunks, [7, 9], 3)
        self.assertEqual(bank.shape, (2, 2 * K, 8, 2))
        np.testing.assert_array_equal(bank[:, :K], chunks)
        np.testing.assert_array_equal(bank[0, K:], perturb(chunks[0], 7, 3))

        class Runner:
            def draw(self, hists, seeds):
                return np.full((len(seeds), 8, 2), 256., np.float32)

        seg_calls = []

        def segment(state, chunk, cloner):
            k = len(seg_calls) % (2 * K)
            seg_calls.append(k)
            out = _branch(x=-4. * abs(k - 12) - (0 if k != 5 else -30.), t=state.t + 8)   # 12 best, 5 best of 0-7
            out.coverage = .01 * k
            out.max_coverage = max(state.max_coverage, out.coverage)
            return out, np.zeros((3, 2, 2, 3), np.uint8)

        class Scores:
            def scores(self, states, chunks, branches, segs, names):
                assert chunks.shape[1] == 2 * K and len(branches) == len(states) * 2 * K
                return {n: np.tile(np.r_[np.zeros(K - 1), 1., np.zeros(K - 1), 2.].astype(np.float32),
                                   (len(states), 1)) for n in names}

        seen = []
        record = lambda root, d, state, ch, sc, c: seen.append((root, d, ch.shape, sc["X"].shape, c))
        done = lambda s: s.t >= 8
        reset = lambda root: _branch(t=0)
        for arm, expect in (("P0", 0), ("GEOM16", 12), ("PHYS16", 15), ("GEOM8", 5), ("X", 15), ("X@8", 7)):
            seg_calls.clear()
            seen.clear()
            _, log, _ = run_arm(arm, [1, 2], Runner(), None, Scores(), ["X"], reset, done, segment,
                                bank=mixed_bank, record=record)
            self.assertTrue((log["chosen"] == expect).all(), (arm, log["chosen"]))
            self.assertEqual(log["geom"].shape, (2, 2 * K))
            self.assertEqual([s[:2] for s in seen], [(1, 0), (2, 0)])
            self.assertEqual(seen[0][2:], ((2 * K, 8, 2), (2 * K,), expect))
        with self.assertRaises(ValueError):
            run_arm("Y@8", [1], Runner(), None, Scores(), ["X"], reset, done, segment, bank=mixed_bank)
        # deviation penalty: X scores index 15 (sigma 32) at 2 and index 7 (policy) at 1 -> 15 wins at lam <= 1/32
        for arm, expect in (("X~0.01", 15), ("X~0.1", 7)):
            seg_calls.clear()
            _, log, _ = run_arm(arm, [1], Runner(), None, Scores(), ["X"], reset, done, segment, bank=mixed_bank)
            self.assertTrue((log["chosen"] == expect).all(), (arm, log["chosen"]))

    def test_retention_helpers(self):
        labels = np.array([[0., 1., 2.], [1., 1., 1.], [0., 3., 1.]])
        rep = chosen_retention(labels, np.array([2, 0, 2]), np.array([7, 7, 8]))
        self.assertAlmostEqual(rep["ratio"], (2 + 0 + 1) / (2 + 0 + 3))
        self.assertAlmostEqual(headroom(labels)["share_with_gain"], 2 / 3)

    def test_batched_action_features_match_single_state(self):
        chunks = torch.randn(3, K, 8, 2) * 50 + 256
        agent = torch.randn(3, 2) * 50 + 256
        batched = action_features(chunks, agent[:, None]).flatten(0, 1)
        single = torch.cat([action_features(chunks[j], agent[j]) for j in range(3)])
        self.assertTrue(torch.allclose(batched, single))


class Round4Data(unittest.TestCase):
    def test_plus_loader_and_microbatch_combination(self):
        torch.manual_seed(0)
        fsq = IndexedFSQ()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            d, dim = 3, 6
            np.save(path / "cur.npy", np.random.randn(d, 256, dim).astype(np.float16))
            np.save(path / "prev.npy", np.random.randn(d, 64, dim).astype(np.float16))
            nb = 4
            np.savez(path / "banks.npz", ctx_index=np.array([0, 1, 2, 2]), root=np.arange(nb), decision=np.zeros(nb),
                     t=np.zeros(nb), source=np.array([0, 1, 2, 1], np.int8), mixed=np.zeros(nb, bool),
                     chunk=(np.random.rand(nb, K, 8, 2) * 512).astype(np.float32),
                     ctx_pos=(np.random.rand(nb, 2, 2) * 512).astype(np.float32),
                     end_pos=np.zeros((nb, K, 2), np.float32), end_prev_pos=np.zeros((nb, K, 2), np.float32),
                     geom=np.random.rand(nb, K).astype(np.float32) * .05, cov=np.zeros((nb, K), np.float32),
                     src=np.random.randint(0, 256, (nb, K, 2)).astype(np.int16))
            data = r4.Plus(path, fsq.codebook)
            self.assertEqual(list(data.pools["std"]), [0, 1, 3])
            ctx, act, src, labels = data.batch([3, 1], "cpu")
            self.assertTrue(torch.equal(ctx["cur"][:K], torch.from_numpy(np.load(path / "cur.npy")[2]).repeat(K, 1, 1)))
            self.assertEqual(act.shape, (2 * K, 8, 4))
            self.assertEqual(src.shape, (2 * K, 2, 3))
            self.assertTrue(torch.equal(src[0], fsq.codebook[torch.from_numpy(data.b["src"][3, 0]).long()]))
            # microbatch combination == one full batch (means averaged, ranking summed over a global pair count)
            wm = ParallelFSQWM(m=2, width=16, layers=1, heads=2, dim=dim)
            reader = Scorer("code", width=16, layers=1, heads=2, m=2, dim=dim).eval().requires_grad_(False)
            goal = torch.randn(4 * K, 256, dim)
            idx = [0, 1, 2, 3]
            ctx, act, src, labels = data.batch(idx, "cpu")
            with torch.no_grad():
                teacher = reader(ctx, src, goal).view(-1, K)
            pairs = ((labels[:, :, None] - labels[:, None, :]) > 1e-3).sum()
            _, full = r4.net_loss("task", wm, reader, ctx, act, src, labels, goal, teacher, 1., pairs, (.01, 1e-3))
            whole = full["nll"] + full["consistency"] + full["rank"]
            parts_sum = 0.
            for half in ([0, 1], [2, 3]):
                c, a_, s, l = data.batch(half, "cpu")
                g = goal[half[0] * K:(half[-1] + 1) * K]
                _, p = r4.net_loss("task", wm, reader, c, a_, s, l, g, teacher[half], 1., pairs, (.01, 1e-3))
                parts_sum = parts_sum + (p["nll"] + p["consistency"]) / 2 + p["rank"]
            self.assertTrue(torch.allclose(whole, parts_sum, atol=1e-5))


class Aggregate(unittest.TestCase):
    def test_aggregate_reads_shards_and_builds_onpolicy_matrix(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as folder:
            closed_run = Path(folder) / "closed"
            arms, names = ["P0", "GEOM8", "X"], ["X"]
            for shard, roots in enumerate(([10, 11], [12, 13])):
                sdir = closed_run / f"shard_{shard}"
                sdir.mkdir(parents=True)
                (sdir / "closed_report.json").write_text(json.dumps({
                    "status": "DONE", "arms": arms, "log_scorers": names, "hashes": {"parent": "h"},
                    "timing": {a: {"seconds": 1.} for a in arms}}))
                with (sdir / "episodes.jsonl").open("w") as f:
                    for arm in arms:
                        for r in roots:
                            f.write(json.dumps({"arm": arm, "root": r, "success": bool(rng.random() < .5),
                                                "score": float(rng.random()), "max_coverage": .5}) + "\n")
                for arm in arms:
                    n = 6
                    np.savez(sdir / f"log_{arm}.npz", root=np.repeat(roots, 3), decision=np.tile(np.arange(3), 2),
                             t=np.zeros(n), chosen=rng.integers(0, K, n), cov=rng.random((n, K)),
                             geom=rng.random((n, K)), score_X=rng.random((n, K)))
            out = Path(folder) / "agg"
            out.mkdir()
            diag.aggregate(SimpleNamespace(run=out, closed_run=closed_run, expect_roots=[10, 13],
                                           pairs="GEOM8-X"))
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["n_roots"], 4)
            self.assertIn("GEOM8-X", summary["contrasts"])
            self.assertIn("X", summary["onpolicy"]["GEOM8"]["geom"]["scorers"])
            self.assertTrue((out / "summary.md").read_text().startswith("| arm |"))


if __name__ == "__main__":
    unittest.main()
