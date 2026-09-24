import os
import unittest
from unittest.mock import patch

from ti_wm.contract import (
    candidate_seed, compatible_config, native_action_slice, require_compute, select_candidate,
)


class ContractTests(unittest.TestCase):
    def test_nested_seeds(self):
        small = [candidate_seed(4, 8, k) for k in range(8)]
        large = [candidate_seed(4, 8, k) for k in range(32)]
        self.assertEqual(small, large[:8])
        self.assertEqual(len(set(large)), 32)
        self.assertNotEqual(large[0], candidate_seed(5, 8, 0))

    def test_native_alignment(self):
        self.assertEqual(native_action_slice(dict(n_obs_steps=2, n_action_steps=8, horizon=16)), (1, 9))
        with self.assertRaises(ValueError):
            native_action_slice(dict(n_obs_steps=2, n_action_steps=16, horizon=16))

    def test_ties_do_not_choose_nonmaximum_default(self):
        self.assertEqual(select_candidate([1, 1, 1]), 0)
        self.assertEqual(select_candidate([0, 1, 1]), 1)
        for scores in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                select_candidate(scores)

    def test_config_only_explicit_runtime_compatibility(self):
        config, removed = compatible_config(dict(type="diffusion", horizon=16, device="cuda"), {"horizon"})
        self.assertEqual(config, {"horizon": 16})
        self.assertEqual(removed, {"device": "cuda"})
        with self.assertRaises(ValueError):
            compatible_config(dict(type="diffusion", surprise=2), set())

    def test_compute_guard(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                require_compute()


if __name__ == "__main__":
    require_compute()
    unittest.main()
