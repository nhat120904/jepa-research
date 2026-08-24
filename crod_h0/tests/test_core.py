import unittest

import numpy as np

from core import directional_ordinal_score, farthest_point_indices, select_arms


class CRODCoreTest(unittest.TestCase):
    def test_directional_score_only_rewards_native_reject_aux_rescue(self):
        native = np.asarray([2.0, 0.5, 3.0, 4.0])
        auxiliary = np.asarray([0.5, 2.0, 3.0, 0.25])
        score, _, _ = directional_ordinal_score(native, auxiliary, 1.0, 1.0)
        self.assertGreater(score[0], 0)
        self.assertEqual(score[1], 0)
        self.assertEqual(score[2], 0)
        self.assertGreater(score[3], score[0])

    def test_farthest_point_is_unique_and_deterministic(self):
        actions = np.arange(20, dtype=float).reshape(10, 2)
        first = farthest_point_indices(actions, np.zeros(2), 5)
        second = farthest_point_indices(actions, np.zeros(2), 5)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(first.tolist())), 5)

    def test_support_matched_arms_are_native_rejected(self):
        rng = np.random.default_rng(3)
        actions = rng.normal(size=(20, 2, 3))
        native = np.arange(20, dtype=float)
        auxiliary = np.arange(20, dtype=float)[::-1]
        crod = np.linspace(0, 1, 20)
        selections = select_arms(
            actions,
            np.zeros((2, 3)),
            native,
            auxiliary,
            5.5,
            crod,
            crod,
            4,
            7,
        )
        for name, indices in selections.items():
            self.assertEqual(len(set(indices.tolist())), 4, name)
            if name != "action_diverse":
                self.assertTrue(np.all(native[indices] > 5.5), name)


if __name__ == "__main__":
    unittest.main()
