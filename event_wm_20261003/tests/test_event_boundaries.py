"""Regressions for readable states that change during contact-masked rest gaps."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

assert os.environ.get('SLURM_JOB_ID'), 'Numerical tests run on compute nodes'
import numpy as np


class BoundaryLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        front, cache, thresholds, out = [root / p for p in ('front', 'cache', 'thresholds', 'out')]
        for p in (front, cache, thresholds):
            p.mkdir()
        n, k = 30, 3
        pos = np.broadcast_to(np.array([[10, 10], [10, 20], [10, 30]], np.float32), (n, k, 2)).copy()
        app = np.zeros((n, k, 3), np.float32)
        app[10:, 0] = 1
        app[10:18, 1] = 1
        eff = np.full((n, 2), 50, np.float32)
        eff[8:20] = [10, 20]
        for split in ('train', 'val'):
            np.savez(front / f'entities_{split}.npz', pos=pos, app=app, area=np.ones((n, k)), processed=np.ones(n, bool), effector=eff)
            terminals = np.zeros(n, bool)
            terminals[-1] = True
            np.save(cache / f'{split}_terminals.npy', terminals)
        (front / 'discover.json').write_text(json.dumps({'table': [{'spread_median': 4}] * k, 'identities': []}))
        (thresholds / 'report.json').write_text(json.dumps({'thr_pos': 10, 'r_pos': .1, 'tol_pos': 5, 'r_app': .01, 'thr_app': .5}))
        script = Path(os.environ.get('EVENT_SOURCE', Path(__file__).resolve().parents[1] / 'scripts')) / 'u_events.py'
        subprocess.run([sys.executable, str(script), '--entities', str(front), '--cache', str(cache), '--thresholds-from', str(thresholds), '--per-frame', '--out', str(out)], check=True, stdout=subprocess.DEVNULL)
        cls.events = dict(np.load(out / 'events_train.npz'))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_readable_side_effect_is_not_overwritten_by_rest_interpolation(self):
        i = int(np.flatnonzero(self.events['t_start'] == 10)[0])
        self.assertEqual(self.events['before'][i, 1, 2], 0)
        self.assertEqual(self.events['after'][i, 1, 2], 1,
                         'Visible boundary state must survive a contact-masked rest gap')

    def test_contact_attribution_includes_actor_missing_from_rest_changes(self):
        i = int(np.flatnonzero(self.events['t_start'] == 10)[0])
        self.assertEqual(self.events['e'][i], 1,
                         'The contacted changing object is the actor, even if its rest runs miss the change')


if __name__ == '__main__':
    unittest.main()
