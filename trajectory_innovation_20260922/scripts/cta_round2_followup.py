"""CPU-only progress audit and strictly paired Round-2 closed-loop aggregation."""
import argparse
import json
from pathlib import Path

import numpy as np

from ti_wm.contract import require_compute
from ti_wm.gates import mcnemar_exact, paired_diff


def categories(labels):
    """Disjoint categories, with the same float32/margin contract as training."""
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim != 2 or labels.shape[1] != 8 or not np.isfinite(labels).all():
        raise ValueError('expected finite N x 8 coverage labels')
    spread = np.ptp(labels, axis=1)
    zero = (labels == 0).all(axis=1)
    return {'all_zero': zero, 'nonzero_exact_flat': (spread == 0) & ~zero,
            'near_tie': (spread > 0) & (spread <= 1e-3), 'informative': spread > 1e-3}


def count_categories(labels):
    masks = categories(labels)
    return {'n': len(labels), 'counts': {k: int(v.sum()) for k, v in masks.items()},
            'fractions': {k: float(v.mean()) if len(v) else None for k, v in masks.items()}}


def audit(features, train_runs):
    out = {'scope': 'offline train/dev only; no causal or terminal-value claim', 'splits': {}, 'scores': {}}
    for split, bounds in (('train', (30250, 31049)), ('dev', (2000, 2099))):
        with np.load(features / split / 'meta.npz') as meta:
            labels, roots, time = meta['cov8'].astype(np.float32), meta['root'], meta['t']
        if not ((roots >= bounds[0]) & (roots <= bounds[1])).all():
            raise ValueError(f'unexpected {split} roots')
        masks = categories(labels)
        out['splits'][split] = {**count_categories(labels), 'n_roots': len(np.unique(roots)),
            'root_mean_fractions': {k: float(np.mean([v[roots == r].mean() for r in np.unique(roots)]))
                                    for k, v in masks.items()},
            'by_native_time': {f'{lo}-{hi - 1}': count_categories(labels[(time >= lo) & (time < hi)])
                               for lo, hi in ((0, 64), (64, 128), (128, 192), (192, 256), (256, 301))}}
    with np.load(features / 'dev' / 'meta.npz') as meta:
        dev_roots, labels = meta['root'], meta['cov8'].astype(np.float32)
    masks = categories(labels)
    for run in train_runs:
        report = json.loads((run / 'train_report.json').read_text())
        with np.load(run / 'dev_scores.npz') as scores:
            if not np.array_equal(scores['root'], dev_roots) or not np.array_equal(scores['cov8'], labels):
                raise ValueError('unaligned dev scores')
            tiers = {}
            for tier in ('full', 'code', 'pred', 'pred_soft', 'direct'):
                s = scores[tier]
                tiers[tier] = {name: {'n': int(mask.sum()),
                    'override_default_fraction': float((s[mask].argmax(1) != 0).mean()) if mask.any() else None,
                    'exact_score_tie_fraction': float((np.ptp(s[mask], axis=1) == 0).mean()) if mask.any() else None}
                    for name, mask in masks.items()}
            out['scores'][run.name] = {'by_category': tiers,
                'paired_gap_vs_parent': report.get('paired_gap_vs_parent'),
                'paired_gap_vs_matched_direct': report.get('paired_gap_vs_matched_direct')}
    out['warning'] = ('An override on a coverage-flat bank is not evidence of harm or benefit. '
                      'These are P0-visited offline states, not each selector\'s closed-loop states.')
    with np.load(train_runs[-2] / 'dev_scores.npz') as control, np.load(train_runs[-1] / 'dev_scores.npz') as method:
        out['matched_baseline_drift'] = {}
        for tier in ('full', 'direct'):
            a, b = control[tier].astype(float), method[tier].astype(float)
            centred_a, centred_b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
            out['matched_baseline_drift'][tier] = {
                'choice_agreement': float((a.argmax(1) == b.argmax(1)).mean()),
                'centered_rmse': float(np.sqrt(np.mean((centred_a - centred_b) ** 2))),
                'control_centered_rms': float(np.sqrt(np.mean(centred_a ** 2))),
                'note': 'Matched data/update budgets do not imply bitwise-identical fitted baselines.'}
    return out


def load_records(run, roots):
    records = {}
    for path in sorted(run.glob('roots_*.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec['root'] in records:
                raise ValueError(f'duplicate root {rec["root"]}')
            records[rec['root']] = rec
    if set(records) != set(roots):
        raise ValueError('missing or unexpected roots')
    recs = [records[r] for r in roots]
    if len({r['checkpoint_sha256'] for r in recs}) != 1:
        raise ValueError('mixed checkpoints')
    arms = ['P0', 'CODE8', 'CTA8', 'DIRECT8', 'CTA8E']
    if any(r['arms'] != arms for r in recs):
        raise ValueError('unexpected arms')
    return recs


def visited_stats(recs, arm):
    labels, scores, chosen = [], [], []
    for rec in recs:
        for d in rec[arm]['decisions']:
            if 'phys' in d and 'score' in d:
                labels.append(d['phys'])
                scores.append(d['score'])
                chosen.append(d['chosen'])
    if not labels:
        return {'n': 0}
    labels, scores, chosen = np.asarray(labels), np.asarray(scores), np.asarray(chosen)
    return {**count_categories(labels), 'by_category': {
        name: {'n': int(mask.sum()),
               'override_default_fraction': float((chosen[mask] != 0).mean()) if mask.any() else None,
               'mean_score_spread': float(np.ptp(scores[mask], axis=1).mean()) if mask.any() else None}
        for name, mask in categories(labels).items()}}


def aggregate(run, out):
    from cta_aggregate import main as aggregate_one
    roots = list(range(2100, 2200))
    paired, records = {}, {}
    for name in ('control', 'method'):
        records[name] = load_records(run / name, roots)
        paired[name] = {arm: np.asarray([r[arm]['success'] for r in records[name]], dtype=float)
                        for arm in records[name][0]['arms']}
        aggregate_one([run / name], out / name, 2100, 100)
    same_p0 = bool(np.array_equal(paired['control']['P0'], paired['method']['P0']))
    summary = {'status': 'DONE' if same_p0 else 'P0_REPRODUCIBILITY_FAILURE',
        'scope': '100 reused development roots; seed 0; not sealed confirmation',
        'p0_identical_by_root': same_p0,
        'success_rate': {name: {arm: float(s.mean()) for arm, s in values.items()} for name, values in paired.items()},
        'method_minus_control': {arm: {'paired': paired_diff(paired['method'][arm], paired['control'][arm]),
            'mcnemar': mcnemar_exact(paired['method'][arm], paired['control'][arm])} for arm in paired['control']},
        'coverage_categories_on_own_visited_states': {name: {arm: visited_stats(recs, arm)
             for arm in paired[name] if arm != 'P0'} for name, recs in records.items()}}
    if not same_p0:
        raise RuntimeError('P0 differs between paired runs; inspect per-arm summaries before interpretation')
    return summary


def main(a):
    require_compute()
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.mode == 'audit':
        result = audit(a.features, a.train_runs)
    else:
        result = aggregate(a.closed_run, a.out.parent)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('audit', 'aggregate'), required=True)
    p.add_argument('--features', type=Path)
    p.add_argument('--train-runs', type=Path, nargs='+')
    p.add_argument('--closed-run', type=Path)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
