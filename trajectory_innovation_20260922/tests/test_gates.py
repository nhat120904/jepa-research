import unittest

import numpy as np

from ti_wm.arms import hrep_estimates
from ti_wm.contract import anchor_fraction, candidate_seed, continuation_seed, medoid_index
from ti_wm.gates import (
    cluster_ratio, mcnemar_exact, paired_diff, retention, runtime_blocker, verdict_a, verdict_b, verdict_c, verdict_confirm,
    wilson,
)


class StatisticsTests(unittest.TestCase):
    def test_wilson_known_value(self):
        lo, hi = wilson(65, 100)
        self.assertAlmostEqual(lo, 0.5525, places=3)
        self.assertAlmostEqual(hi, 0.7364, places=3)

    def test_paired_identical_is_zero(self):
        d = paired_diff([1, 0, 1, 1], [1, 0, 1, 1], resamples=200)
        self.assertEqual((d["mean"], d["lo"], d["hi"]), (0.0, 0.0, 0.0))

    def test_mcnemar_counts_discordant_pairs(self):
        m = mcnemar_exact([1, 1, 1, 0, 0], [0, 0, 1, 0, 1])
        self.assertEqual((m["only_a"], m["only_b"]), (2, 1))
        self.assertAlmostEqual(m["p"], 1.0)

    def test_cluster_ratio_and_retention(self):
        r = cluster_ratio([1, 1, 0, 0], [2, 2, 0, 0], resamples=100)
        self.assertAlmostEqual(r["ratio"], 0.5)
        ret = retention(vis=[1, 0, 0, 0], phys=[1, 1, 0, 0], p0=[0, 0, 0, 0], resamples=100)
        self.assertAlmostEqual(ret["ratio"], 0.5)


class VerdictTests(unittest.TestCase):
    def test_gate_a_bound(self):
        self.assertEqual(verdict_a([1] * 56 + [0] * 44)["verdict"], "PASS")
        self.assertEqual(verdict_a([1] * 55 + [0] * 45)["verdict"], "FAIL")
        self.assertEqual(verdict_a([1] * 75 + [0] * 25)["verdict"], "PASS")

    def test_gate_b_rules(self):
        self.assertEqual(verdict_b({"mean": 0.12, "lo": 0.03, "hi": 0.2}), "PASS")
        self.assertEqual(verdict_b({"mean": 0.12, "lo": -0.01, "hi": 0.2}), "EXTEND")
        self.assertEqual(verdict_b({"mean": 0.07, "lo": 0.01, "hi": 0.1}), "EXTEND")
        self.assertEqual(verdict_b({"mean": 0.04, "lo": 0.01, "hi": 0.1}), "FAIL")

    def test_confirm_needs_ci_above_zero(self):
        self.assertEqual(verdict_confirm({"mean": 0.08, "lo": 0.01, "hi": 0.15}), "CONFIRMED")
        self.assertEqual(verdict_confirm({"mean": 0.14, "lo": 0.0, "hi": 0.25}), "NOT_CONFIRMED")
        self.assertEqual(verdict_confirm({"mean": -0.02, "lo": -0.1, "hi": 0.05}), "NOT_CONFIRMED")

    def test_gate_c_needs_b(self):
        self.assertEqual(verdict_c("FAIL", {"ratio": 1.0}), "NOT_INTERPRETED")
        self.assertEqual(verdict_c("PASS", {"ratio": 0.85}), "PASS")
        self.assertEqual(verdict_c("PASS", {"ratio": float("nan")}), "FAIL")

    def test_runtime_blocker(self):
        self.assertTrue(runtime_blocker({"mean": -0.15, "lo": -0.25, "hi": -0.05}))
        self.assertFalse(runtime_blocker({"mean": -0.15, "lo": -0.3, "hi": 0.01}))


class SelectionTests(unittest.TestCase):
    def test_medoid_prefers_central_chunk(self):
        chunks = [[[0, 0]], [[1, 0]], [[10, 0]]]
        self.assertEqual(medoid_index(chunks), 1)

    def test_seeds_are_distinct_and_stable(self):
        cont = {continuation_seed(7, k, r, d) for k in range(8) for r in range(4) for d in range(3)}
        self.assertEqual(len(cont), 96)
        self.assertNotIn(candidate_seed(7, 0, 0), cont)
        self.assertEqual(anchor_fraction(1000), anchor_fraction(1000))
        self.assertTrue(all(0 <= anchor_fraction(r) < 1 for r in range(1000, 1100)))

    def test_hrep_split_removes_winners_curse(self):
        # Candidate 3 looks perfect on selection seeds but fails on evaluation seeds.
        s = np.zeros((8, 4), dtype=bool)
        s[3, :2] = True
        s[0, 2:] = True
        h = hrep_estimates(s)
        self.assertEqual(h["chosen"], 3)
        self.assertAlmostEqual(h["split"], -1.0)
        self.assertAlmostEqual(h["naive"], 0.0)


if __name__ == "__main__":
    unittest.main()
