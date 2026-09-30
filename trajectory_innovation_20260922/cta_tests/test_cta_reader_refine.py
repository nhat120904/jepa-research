"""Protect matched-bank adaptation and root-paired evaluation contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cta_reader_refine import ARMS, mixed_codes, validate_records, write_json


class ReaderRefineTests(unittest.TestCase):
    def test_evidence_mode_is_shared_by_all_siblings(self):
        src = torch.zeros(24, 2, 4)
        got = mixed_codes(src, src + 1, src + 2, torch.tensor([0, 1, 2]))
        self.assertTrue(torch.equal(got[:, 0, 0], torch.tensor([0.] * 8 + [1.] * 8 + [2.] * 8)))
        with self.assertRaises(ValueError):
            mixed_codes(src, src + 1, src + 2, torch.tensor([0, 1]))

    def test_pairing_rejects_duplicates_and_checkpoint_drift(self):
        first = {'root': 2100, 'arms': list(ARMS), 'checkpoint_hashes': {'base': 'a', 'method': 'b', 'control': 'c'}}
        second = {**first, 'root': 2101}
        for rows in ([first], [first, first], [first, {**second, 'checkpoint_hashes': {'base': 'changed'}}]):
            with self.assertRaises(ValueError):
                validate_records(rows, [2100, 2101])
        self.assertEqual(validate_records([second, first], [2100, 2101]), [first, second])

    def test_json_handles_numpy_and_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            write_json(path, {'x': np.array([1, 2]), 'n': np.int64(2)})
            self.assertEqual(json.loads(path.read_text()), {'x': [1, 2], 'n': 2})
            self.assertFalse(path.with_suffix('.json.tmp').exists())
