"""Check category boundaries and reject invalid paired closed-loop inputs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('followup', Path(__file__).resolve().parents[1] / 'scripts/cta_round2_followup.py')
followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)


class FollowupTests(unittest.TestCase):
    def test_disjoint_categories_and_margin(self):
        labels = np.zeros((5, 8), dtype=np.float32)
        labels[1] = .2
        labels[2, 1] = 1e-3
        labels[3, 1] = 1.1e-3
        labels[4, 1] = 1e-6
        result = followup.categories(labels)
        np.testing.assert_array_equal(np.stack(list(result.values())).sum(0), np.ones(5))
        self.assertEqual({k: int(v.sum()) for k, v in result.items()},
                         {'all_zero': 1, 'nonzero_exact_flat': 1, 'near_tie': 2, 'informative': 1})

    def test_pairing_rejects_missing_duplicate_and_mixed_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            path = run / 'roots_test.jsonl'
            a = {'root': 2100, 'checkpoint_sha256': 'x', 'arms': ['P0', 'CODE8', 'CTA8', 'DIRECT8', 'CTA8E']}
            b = {**a, 'root': 2101}
            for rows in ([a], [a, a, b], [a, {**b, 'checkpoint_sha256': 'y'}]):
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with self.assertRaises(ValueError):
                    followup.load_records(run, [2100, 2101])
            path.write_text(''.join(json.dumps(r) + '\n' for r in [b, a]))
            self.assertEqual([r['root'] for r in followup.load_records(run, [2100, 2101])], [2100, 2101])
