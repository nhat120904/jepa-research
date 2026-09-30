import unittest
import numpy as np
from ti_wm.cta_geometry import GOAL_POSE, registration_score


class RegistrationTests(unittest.TestCase):
    def test_translation_has_known_distance_and_rotation_wrap_is_invariant(self):
        translated = GOAL_POSE + np.array([3., 4., 0.])
        self.assertAlmostEqual(float(registration_score(translated)), -5 / 512)
        self.assertAlmostEqual(float(registration_score(GOAL_POSE)), 0.)
        self.assertAlmostEqual(float(registration_score(GOAL_POSE + [0, 0, 2*np.pi])), 0.)

    def test_far_candidates_can_differ_but_identical_block_poses_tie(self):
        poses = np.tile(GOAL_POSE, (1, 3, 1))
        poses[0, :, 0] += [180, 190, 190]
        scores = registration_score(poses)
        self.assertEqual(scores.shape, (1, 3))
        self.assertGreater(scores[0, 0], scores[0, 1])
        self.assertEqual(scores[0, 1], scores[0, 2])
