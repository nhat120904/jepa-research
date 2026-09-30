"""Regression for the NumPy choice array that stopped job 55018 after training."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cta_geometry_e2e import ranking_summary


class GeometryReportingTests(unittest.TestCase):
    def test_native_diagnostic_serializes_multiple_decisions(self):
        labels = np.array([[0., .2, .1], [.1, .3, .2]])
        report = {'native_coverage_diagnostic': {
            'code': ranking_summary(labels, labels, np.array([2000, 2001]))}}
        restored = json.loads(json.dumps(report, default=float))
        summary = restored['native_coverage_diagnostic']['code']
        self.assertEqual(summary['retained_gap']['ratio'], 1.)
        self.assertNotIn('chosen', summary)
