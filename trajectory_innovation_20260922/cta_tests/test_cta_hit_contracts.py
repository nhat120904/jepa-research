"""Source-only contracts: no model, physics, cache or statistical analysis."""

import json
import unittest
from types import SimpleNamespace

from ti_wm.cta_hit_contracts import (format_stat, json_ready, validate_bank_shapes,
                                    validate_decision_domains, validate_train_arguments)


def arguments(**overrides):
    values = dict(steps=2400, decisions=16, eval_every=1200, warmup=100,
                  log_every=200, lr=3e-5, hit_weight=1., standard_mass=.8,
                  mixed_frac=.25, seed=0, selection_limit=None, train_limit=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainContracts(unittest.TestCase):
    def test_rejects_intervals_that_crash_scheduler_or_logging(self):
        for key in ("steps", "decisions", "eval_every", "warmup", "log_every"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_train_arguments(arguments(**{key: 0}))

    def test_rejects_invalid_limits_learning_rate_and_sampling(self):
        for key, value in (("selection_limit", -1), ("train_limit", 0), ("lr", 0),
                           ("lr", float("nan")), ("hit_weight", -1),
                           ("standard_mass", 1.1), ("mixed_frac", -.1), ("seed", -1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_train_arguments(arguments(**{key: value}))

    def test_accepts_smoke_and_zero_hit_weight_controls(self):
        validate_train_arguments(arguments(steps=20, warmup=2, eval_every=10,
                                           selection_limit=8, train_limit=64, hit_weight=0.))

    def test_rejects_cache_candidate_and_horizon_mismatch(self):
        for geometry, hits, actions in (((40, 4), (40, 4), (40, 4, 15, 2)),
                                       ((40, 8), (40, 8), (40, 8, 8, 2)),
                                       ((40, 8), (39, 8), (40, 8, 15, 2)),
                                       ((40, 8), (40, 8), (39, 8, 15, 2))):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                validate_bank_shapes("train", geometry, hits, actions)
        validate_bank_shapes("selection", (40, 8), (40, 8), (40, 8, 15, 2))

    def test_rejects_misaligned_future_feature_caches(self):
        valid = {"cur": (40, 256, 64), "prev": (40, 64, 64),
                 "end": (40, 8, 256, 64), "seg": (40, 8, 3, 64, 64)}
        validate_bank_shapes("train", (40, 8), (40, 8), (40, 8, 15, 2), feature_shapes=valid)
        for key, shape in (("cur", (39, 256, 64)), ("prev", (40, 64, 32)),
                           ("end", (40, 4, 256, 64)), ("seg", (40, 8, 2, 64, 64))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_bank_shapes("train", (40, 8), (40, 8), (40, 8, 15, 2),
                                     feature_shapes={**valid, key: shape})


class ResultContracts(unittest.TestCase):
    def test_rejects_choices_that_silently_index_the_wrong_candidate(self):
        for choice in (-1, 8, 1.5, True):
            with self.subTest(choice=choice), self.assertRaises(ValueError):
                validate_decision_domains([choice], [False, True])
        validate_decision_domains([0, 7], [False, True, 0, 1])

    def test_rejects_fractional_native_hit_labels(self):
        with self.assertRaises(ValueError):
            validate_decision_domains([0], [0.5])

    def test_undefined_metrics_serialize_as_null_in_strict_json(self):
        payload = {"retained_gap": {"ratio": float("nan"), "lo": float("-inf")},
                   "values": (float("inf"), .25), "successes": 2, "valid": True}
        result = json.dumps(json_ready(payload), allow_nan=False)
        self.assertEqual(json.loads(result), {"retained_gap": {"ratio": None, "lo": None},
                                              "values": [None, .25], "successes": 2, "valid": True})

    def test_undefined_statistics_have_readable_markdown(self):
        for value in (None, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertEqual(format_stat(value), "n/a")
        self.assertEqual(format_stat(.125), "0.125")


if __name__ == "__main__":
    unittest.main()
