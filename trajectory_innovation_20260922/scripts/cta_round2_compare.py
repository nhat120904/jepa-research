"""Compare the two predeclared Round-2 arms by root, on a CPU compute node."""
import argparse
import json
from pathlib import Path

import numpy as np

from cta_round2 import paired_gap
from ti_wm.contract import require_compute


def main(a):
    require_compute()
    reports = [json.loads((p / 'train_report.json').read_text()) for p in (a.control, a.method)]
    if any(r['status'] != 'DONE' for r in reports):
        raise ValueError('both training runs must complete')
    if reports[0]['config']['lam'] != 0 or reports[1]['config']['lam'] != 0.1:
        raise ValueError('unexpected treatment/control')
    for key in ('parent_sha256', 'updates'):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f'unmatched {key}')
    ignored = {'lam', 'slurm_job'}
    configs = [{k: v for k, v in r['config'].items() if k not in ignored} for r in reports]
    if configs[0] != configs[1]:
        raise ValueError('arm configs differ beyond lambda/job id')
    scores = [np.load(p / 'dev_scores.npz') for p in (a.control, a.method)]
    for key in ('root', 'decision', 'cov8'):
        if not np.array_equal(scores[0][key], scores[1][key]):
            raise ValueError(f'unpaired {key}')
    labels, roots = scores[0]['cov8'], scores[0]['root']
    out = {'status': 'DONE', 'scope': 'offline dev only; no closed-loop claim',
           'control': str(a.control), 'method': str(a.method),
           'method_minus_control': {tier: paired_gap(scores[1][tier], scores[0][tier], labels, roots)
                                    for tier in ('full', 'code', 'pred', 'pred_soft', 'pred_s4', 'direct')},
           'control_ladder': reports[0]['ladder'], 'method_ladder': reports[1]['ladder'],
           'diagnostics': {'control': reports[0]['after'], 'method': reports[1]['after']}}
    # Numeric drift in these independently optimized modules is an audit result,
    # not silently assumed absent because the seeds match.
    out['matched_baseline_max_score_difference'] = {
        tier: float(np.abs(scores[1][tier] - scores[0][tier]).max()) for tier in ('full', 'direct')}
    out['codec_gap_point_drop_vs_parent'] = reports[1]['paired_gap_vs_parent']['code']['ratio']
    out['retention_guardrail_flag'] = out['codec_gap_point_drop_vs_parent'] < -0.05
    out['next_step'] = 'inspect full ladder and then run both locked arms on dev closed-loop roots 2100-2199'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--method', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
