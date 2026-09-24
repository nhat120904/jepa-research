import unittest

from ti_wm.goal_progress import SATISFIED, combine, distance_progress, joint_progress, nearest_endpoint


class GoalProgressTests(unittest.TestCase):
    def test_nearest_endpoint_follows_libero_ranges(self):
        # WoodenCabinet: open once q < -0.14, drawer closed at 0; FlatStove: on once q >= 0.5, off at 0.
        self.assertEqual(nearest_endpoint([-0.16, -0.14], 0.0), -0.14)
        self.assertEqual(nearest_endpoint([0.5, 2.1], 0.0), 0.5)
        self.assertEqual(nearest_endpoint([0.0, 0.005], -0.15), 0.0)   # close from open

    def test_joint_progress_both_directions(self):
        self.assertAlmostEqual(joint_progress(-0.07, 0.0, -0.14), 0.5)
        self.assertAlmostEqual(joint_progress(0.25, 0.0, 0.5), 0.5)
        self.assertEqual(joint_progress(0.02, 0.0, -0.14), 0.0)        # wrong way is clipped
        self.assertEqual(joint_progress(-0.3, 0.0, -0.14), 1.0)
        self.assertEqual(joint_progress(0.3, 0.3, 0.3), 1.0)

    def test_distance_and_satisfied_ordering(self):
        near, far = distance_progress([0, 0, 0.01], [0, 0, 0]), distance_progress([0.3, 0, 0], [0, 0, 0])
        self.assertAlmostEqual(near, -0.01)
        self.assertGreater(near, far)
        self.assertEqual(combine([near], [True]), SATISFIED)
        self.assertGreater(combine([near], [True]), combine([0.99], [False]))
        self.assertAlmostEqual(combine([0.5, -0.2], [False, False]), 0.15)


if __name__ == "__main__":
    unittest.main()
