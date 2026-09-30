"""Paired endpoint/trajectory comparison; bootstrap roots, preserve paired seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

from cta_round2 import paired_gap
from ti_wm.contract import require_compute

MARGIN = .05


def verdict(primary, seed_effects):
    if primary['ratio'] >= MARGIN and primary['lo'] > 0 and all(x > 0 for x in seed_effects):
        return 'TRAJECTORY_ADVANTAGE_SUPPORTED_ON_ENDPOINT_TARGET'
    if primary['ratio'] <= -MARGIN and primary['hi'] < 0 and all(x < 0 for x in seed_effects):
        return 'ENDPOINT_ADVANTAGE_SUPPORTED'
    if primary['lo'] > -MARGIN and primary['hi'] < MARGIN:
        return 'DIFFERENCE_WITHIN_PRACTICAL_MARGIN_ON_TESTED_SEEDS'
    return 'INCONCLUSIVE'


def compare(runs, smoke=False):
    records = {}
    for path in runs:
        report = json.loads((path / 'train_report.json').read_text())
        if report['status'] != ('SMOKE_OK' if smoke else 'DONE'):
            raise ValueError('unfinished or wrong-mode training run')
        key = (report['config']['seed'], report['config']['path'])
        if key in records:
            raise ValueError('duplicate seed/arm')
        with np.load(path / 'dev_scores.npz') as data:
            arrays = {k: data[k] for k in ('root', 'decision', 'cov8', 'code')}
        records[key] = (report, arrays, str(path))
    if set(records) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError('both paired seeds and both arms required')
    baseline_cfg = {k: v for k, v in records[0, 0][0]['config'].items() if k not in ('seed', 'path')}
    reference = records[0, 0][1]
    for report, arrays, _ in records.values():
        if {k: v for k, v in report['config'].items() if k not in ('seed', 'path')} != baseline_cfg:
            raise ValueError('unmatched configuration')
        for key in ('root', 'decision', 'cov8'):
            if not np.array_equal(reference[key], arrays[key]):
                raise ValueError('unpaired evaluation rows')
    per_seed = {}
    for seed in (0, 1):
        end_report, end_scores, _ = records[seed, 0]
        path_report, path_scores, _ = records[seed, 1]
        for key in ('shared_initial_sha256', 'batch_goal_sha256', 'endpoint_copy_mse'):
            if end_report[key] != path_report[key]:
                raise ValueError(f'unmatched {key} for seed {seed}')
        per_seed[str(seed)] = paired_gap(path_scores['code'], end_scores['code'], reference['cov8'], reference['root'])
    # Repeated roots intentionally keep seed pairs inside each root cluster.
    primary = paired_gap(np.concatenate([records[s, 1][1]['code'] for s in (0, 1)]),
                         np.concatenate([records[s, 0][1]['code'] for s in (0, 1)]),
                         np.tile(reference['cov8'], (2, 1)), np.tile(reference['root'], 2))
    return {'status': 'SMOKE_OK' if smoke else 'DONE',
            'scope': 'actual-future codec, endpoint-coverage query, offline development only',
            'statistical_scope': 'root-bootstrap conditional on two paired training seeds; not a full estimate of training-seed uncertainty',
            'practical_margin_retained_gap_units': MARGIN,
            'trajectory_minus_endpoint': primary, 'per_seed': per_seed,
            'verdict': 'SMOKE_ONLY' if smoke else verdict(primary, [v['ratio'] for v in per_seed.values()]),
            'runs': {f'seed{seed}_path{path}': {'directory': records[seed, path][2],
                'metrics': records[seed, path][0]['metrics'],
                'parameter_counts': records[seed, path][0]['parameter_counts'],
                'training_seconds': records[seed, path][0]['training_seconds']}
                for seed, path in sorted(records)},
            'limitation': 'No WM, closed-loop success, temporal-query or conditional per-frame claim. A wide interval crossing zero is inconclusive, not equivalence.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs=4, type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    require_compute()
    if a.out.exists():
        raise FileExistsError(a.out)
    result = compare(a.runs, a.smoke)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
