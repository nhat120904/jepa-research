"""Meaningful native-boundary and sampling checks; run on a Slurm compute node."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ti_wm.contract import require_compute
from ti_wm.cta_hit_training import HitBank, StratifiedHitPool, native_hit_loss, native_hit_metrics


def setUpModule():
    require_compute()


class FakeBank:
    def __init__(self, name, root, hits, informative, marker=0):
        self.name = name
        self.root = np.array(root, dtype=np.int64)
        self.n = len(root)
        self.geom = np.zeros(hits.shape, np.float32)
        self.spread = np.array(informative, dtype=np.int64)
        self.marker = marker
        self.hits = hits

    def batch(self, indices, device):
        signature = torch.as_tensor(np.asarray(indices) + self.marker, dtype=torch.float32, device=device)
        repeated = signature.repeat_interleave(self.hits.shape[1])[:, None]
        return {"cur": repeated}, {"end": repeated + 1000}, repeated + 2000, signature[:, None].expand(-1, self.hits.shape[1])


class NativeHitLoss(unittest.TestCase):
    def test_gradient_rewards_either_hit_without_ordering_the_hits(self):
        scores = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
        hits = torch.tensor([[True, True, False, False]])
        loss = native_hit_loss(scores, hits, temperature=0.5)
        self.assertAlmostEqual(float(loss), math.log(2), places=12)
        loss.backward()
        self.assertTrue((scores.grad[0, :2] < 0).all())
        self.assertTrue((scores.grad[0, 2:] > 0).all())
        torch.testing.assert_close(scores.grad[0, 0], scores.grad[0, 1])
        torch.testing.assert_close(scores.grad.sum(), torch.tensor(0.0, dtype=torch.float64))

    def test_only_mixed_banks_contribute_and_zero_is_differentiable(self):
        hits = torch.tensor([[True, False], [True, True], [False, False]])
        scores = torch.tensor([[1., -1.], [-5., 10.], [7., 0.]], requires_grad=True)
        loss = native_hit_loss(scores, hits)
        torch.testing.assert_close(loss, native_hit_loss(scores[:1], hits[:1]))
        loss.backward()
        torch.testing.assert_close(scores.grad[1:], torch.zeros_like(scores.grad[1:]))
        scores2 = torch.randn(2, 3, requires_grad=True)
        zero = native_hit_loss(scores2, torch.tensor([[True, True, True], [False, False, False]]))
        self.assertEqual(float(zero), 0.)
        zero.backward()
        torch.testing.assert_close(scores2.grad, torch.zeros_like(scores2))

    def test_extreme_logits_are_finite_and_offset_invariant(self):
        scores = torch.tensor([[-1000., 1000.]], dtype=torch.float64, requires_grad=True)
        hits = torch.tensor([[True, False]])
        loss = native_hit_loss(scores, hits)
        self.assertEqual(float(loss), 2000.)
        torch.testing.assert_close(loss, native_hit_loss(scores + 1e9, hits), atol=1e-7, rtol=0.)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        half = scores.detach().to(torch.bfloat16).requires_grad_()
        self.assertTrue(torch.isfinite(native_hit_loss(half, hits)))

    def test_contract_validation(self):
        scores = torch.zeros(2, 3)
        hits = torch.zeros(2, 3, dtype=torch.bool)
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                native_hit_loss(scores, hits, temperature)
        with self.assertRaises(ValueError):
            native_hit_loss(scores, hits.float())
        with self.assertRaises(ValueError):
            native_hit_loss(scores, hits[:, :2])

    def test_capture_metrics_include_all_success_but_separate_mixed(self):
        scores = np.array([[2, 1], [3, 4], [4, 1], [5, 9]], dtype=np.float32)
        hits = np.array([[False, True], [True, True], [True, False], [False, False]])
        metric = native_hit_metrics(scores, hits)
        self.assertEqual(metric["eligible_banks"], 3)
        self.assertEqual(metric["mixed_banks"], 2)
        self.assertEqual(metric["mixed_hits_selected"], 1)
        self.assertEqual(metric["mixed_misses"], 1)
        self.assertAlmostEqual(metric["eligible_capture"], 2 / 3)
        self.assertEqual(metric["mixed_capture"], .5)
        self.assertEqual(metric, native_hit_metrics(torch.from_numpy(scores), torch.from_numpy(hits)))


class HitSampler(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def bank(self, name, n, standard, mixed_indices=(), roots=None, informative=None, marker=0):
        hits = np.zeros((n, 4), np.bool_)
        hits[np.asarray(mixed_indices, dtype=int), 1] = True
        roots = np.arange(n) + 100 if roots is None else np.asarray(roots)
        informative = np.arange(n // 2) if informative is None else informative
        path = self.directory / name
        path.mkdir()
        np.savez(path / "meta.npz", root=roots, decision=np.zeros(n, np.int64), done8=hits)
        base = FakeBank(name, roots, hits, informative, marker)
        return HitBank(base, path, standard)

    def test_exact_exposure_report_and_deterministic_empirical_sampling(self):
        banks = [self.bank("standard", 100, True, [0, 1, 2]),
                 self.bank("perturbed", 80, False, range(8), roots=np.arange(80) + 500)]
        pool = StratifiedHitPool(banks, standard_mass=.8, mixed_frac=.25, spread_frac=.75)
        report = pool.report(decisions=24)
        self.assertAlmostEqual(report["expected_standard_exposure"], .8, places=12)
        self.assertGreater(report["expected_mixed_exposure"], .25)
        self.assertEqual(sum(x["rows_with_nonzero_probability"] for x in report["banks"].values()), 180)
        self.assertTrue(all(0 < x["expected_distinct_rows_per_batch"] <= x["banks"]
                            for x in report["banks"].values()))
        sampled = pool.sample_rows(np.random.default_rng(57), 100_000)
        same = pool.sample_rows(np.random.default_rng(57), 100_000)
        np.testing.assert_array_equal(sampled, same)
        different = pool.sample_rows(np.random.default_rng(58), 100_000)
        self.assertFalse(np.array_equal(sampled, different))
        self.assertAlmostEqual(float((sampled[:, 0] == 0).mean()), .8, delta=.008)
        mixed = np.array([banks[j].hits[i].any() and not banks[j].hits[i].all() for j, i in sampled])
        self.assertAlmostEqual(float(mixed.mean()), report["expected_mixed_exposure"], delta=.008)

    def test_sample_preserves_alignment_and_keeps_hits_out_of_model_inputs(self):
        banks = [self.bank("standard", 20, True, [0, 2, 4], marker=0),
                 self.bank("perturbed", 20, False, [1, 3], roots=np.arange(20) + 500, marker=100)]
        pool = StratifiedHitPool(banks)
        ctx, fut, act, y, hits, rows = pool.sample(np.random.default_rng(42), 24, "cpu", return_rows=True)
        expected_signature = torch.tensor([banks[j].marker + i for j, i in rows], dtype=torch.float32)
        torch.testing.assert_close(y[:, 0], expected_signature)
        torch.testing.assert_close(ctx["cur"][:, 0], expected_signature.repeat_interleave(4))
        torch.testing.assert_close(fut["end"][:, 0], expected_signature.repeat_interleave(4) + 1000)
        torch.testing.assert_close(act[:, 0], expected_signature.repeat_interleave(4) + 2000)
        torch.testing.assert_close(hits, torch.from_numpy(np.stack([banks[j].hits[i] for j, i in rows])))
        self.assertEqual(set(ctx), {"cur"})
        self.assertEqual(set(fut), {"end"})

    def test_no_mixed_banks_discloses_fallback_and_keeps_full_support(self):
        bank = self.bank("standard", 12, True, informative=[])
        pool = StratifiedHitPool([bank], standard_mass=1.)
        report = pool.report()
        self.assertEqual(report["expected_mixed_exposure"], 0.)
        self.assertTrue(report["components"][0]["fallback_to_all"])
        self.assertTrue(report["components"][1]["fallback_to_all"])
        np.testing.assert_allclose(pool.row_probability[0], np.full(12, 1 / 12))
        with self.assertRaises(ValueError):
            StratifiedHitPool([bank], standard_mass=.8)

    def test_root_split_and_evaluation_exclusion(self):
        train = self.bank("train", 10, True, roots=np.arange(10) + 100)
        selection = self.bank("selection", 5, True, roots=np.arange(5) + 200)
        pool = StratifiedHitPool([train], standard_mass=1.)
        report = pool.assert_disjoint([selection], forbidden_roots=range(2200, 2400))
        self.assertEqual(report, {"train_roots": 10, "selection_roots": 5, "disjoint": True})
        with self.assertRaises(ValueError):
            pool.assert_disjoint([train])
        with self.assertRaises(ValueError):
            pool.assert_disjoint([selection], forbidden_roots=[203])

    def test_authoritative_metadata_validation(self):
        bank = self.bank("standard", 8, True, [0])
        meta = bank.feature_path / "meta.npz"
        np.savez(meta, root=np.arange(8) + 900, decision=np.zeros(8, np.int64), done8=bank.hits)
        with self.assertRaisesRegex(ValueError, "roots do not align"):
            HitBank(bank.base, bank.feature_path)
        flags = bank.hits.astype(np.float32)
        flags[1, 0] = .5
        np.savez(meta, root=bank.root, decision=np.zeros(8, np.int64), done8=flags)
        with self.assertRaisesRegex(ValueError, "nonbinary"):
            HitBank(bank.base, bank.feature_path)
        np.savez(meta, root=bank.root, decision=np.zeros(8, np.int64), native_cov8=np.zeros_like(flags))
        with self.assertRaisesRegex(ValueError, "missing native-label metadata done8"):
            HitBank(bank.base, bank.feature_path)


if __name__ == "__main__":
    unittest.main()
