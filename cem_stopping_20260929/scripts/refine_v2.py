#!/usr/bin/env python3
"""Complete dev-only, episode-grouped comparison; saves deployable policies."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cemstop.continuation import fit_policy, outcomes
from cemstop.evaluate import load_root, evaluate
from cemstop.rules import CHECKPOINTS, grid


def bootstrap(a, b, groups):
    diff = np.asarray(a, float) - np.asarray(b, float)
    group_ids = np.unique(groups)
    sums = np.array([diff[groups == g].sum() for g in group_ids])
    counts = np.array([(groups == g).sum() for g in group_ids])
    rng = np.random.default_rng(0)
    ix = rng.integers(len(group_ids), size=(10000, len(group_ids)))
    samples = sums[ix].sum(1) / counts[ix].sum(1)
    return {'mean': float(diff.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist()}


def fixed_pair(i, j, roots):
    return {'success': np.array([r.leaf_success[i, j] for r in roots]),
            'iterations': np.array([CHECKPOINTS[i] + (0 if r.p1_term[i] else CHECKPOINTS[j]) for r in roots])}


def utility(o, penalty):
    return o['success'].astype(float) - penalty * o['iterations']


def tune_control(family, roots, penalty):
    if family == 'fixed_pair':
        options = [(i, j) for i in range(len(CHECKPOINTS)) for j in range(len(CHECKPOINTS))]
        ev = [fixed_pair(i, j, roots) for i, j in options]
        best = max(range(len(options)), key=lambda t: (utility(ev[t], penalty).mean(), -ev[t]['iterations'].mean()))
        i, j = options[best]
        return {'kind': 'fixed_pair', 'k1': CHECKPOINTS[i], 'k2': CHECKPOINTS[j]}
    rules = grid()[family]
    ev = [evaluate(rule, roots) for rule in rules]
    best = max(range(len(rules)), key=lambda t: (utility(ev[t], penalty).mean(), -ev[t]['iterations'].mean()))
    return {'kind': family, 'param': rules[best].param()}


def eval_control(spec, roots):
    if spec['kind'] == 'fixed_pair':
        return fixed_pair(CHECKPOINTS.index(spec['k1']), CHECKPOINTS.index(spec['k2']), roots)
    rule = next(r for r in grid()[spec['kind']] if r.param() == spec['param'])
    return evaluate(rule, roots)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use sbatch for fitting and analysis')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {}
    for task in ('cube', 'reacher'):
        directory = args.data / task
        # Open DEV outcomes only, even if test artifacts are present.
        roots = []
        for rid in range(100):
            path = directory / f'root_{rid:04d}.npz'
            if not path.exists() or not path.with_suffix('.json').exists():
                raise RuntimeError(f'incomplete dev root {path}')
            tr = load_root(path)
            if tr.checkpoints != CHECKPOINTS or not np.isfinite(tr.leaf_dist).all():
                raise RuntimeError(f'invalid tree {path}')
            roots.append(tr)
        manifest = json.loads((directory / 'manifest.json').read_text())
        groups = np.array([r['episode'] for r in manifest['roots'][:100]])
        folds = list(GroupKFold(n_splits=5).split(np.arange(100), groups=groups))
        res = {'n': 100, 'episodes': int(len(np.unique(groups))), 'variants': {},
               'dev_ids_sha256': hashlib.sha256(groups.tobytes()).hexdigest()}
        for penalty in (0., .002):
            families = ('fixed', 'fixed_pair', 'converge', 'gap', 'band', 'continuation')
            predictions = {f: {'success': np.zeros(100, bool), 'iterations': np.zeros(100)} for f in families}
            fold_specs = {f: [] for f in families}
            for fold, (train_idx, val_idx) in enumerate(folds):
                train = [roots[i] for i in train_idx]
                val = [roots[i] for i in val_idx]
                fold_dir = args.out / task / f'lambda_{penalty:g}' / f'fold_{fold}'
                fold_dir.mkdir(parents=True)
                for family in families:
                    spec = fit_policy(train, penalty) if family == 'continuation' else tune_control(family, train, penalty)
                    ev = outcomes(spec, val) if family == 'continuation' else eval_control(spec, val)
                    for key in ('success', 'iterations'):
                        predictions[family][key][val_idx] = ev[key]
                    fold_specs[family].append(spec if family != 'continuation' else {'kind': family})
                    (fold_dir / f'{family}.json').write_text(json.dumps(spec))
                (fold_dir / 'validation_roots.json').write_text(json.dumps(val_idx.tolist()))
            variant = {}
            full_dir = args.out / task / f'lambda_{penalty:g}' / 'all_dev'
            full_dir.mkdir(parents=True)
            for family in families:
                ev = predictions[family]
                variant[family] = {'success': float(ev['success'].mean()), 'iterations': float(ev['iterations'].mean()),
                                   'utility': float(utility(ev, penalty).mean()), 'fold_specs': fold_specs[family]}
                for ref in ('fixed', 'fixed_pair'):
                    variant[family][f'vs_{ref}'] = bootstrap(ev['success'], predictions[ref]['success'], groups)
                    variant[family][f'iterations_vs_{ref}'] = bootstrap(ev['iterations'], predictions[ref]['iterations'], groups)
                spec = fit_policy(roots, penalty) if family == 'continuation' else tune_control(family, roots, penalty)
                (full_dir / f'{family}.json').write_text(json.dumps(spec))
            np.savez_compressed(full_dir / 'oof.npz', groups=groups,
                                **{f'{f}_{k}': v for f, d in predictions.items() for k, v in d.items()})
            res['variants'][str(penalty)] = variant
        report[task] = res
        print(task, json.dumps({l: {f: {'success': v['success'], 'iterations': v['iterations']} for f, v in d.items()} for l, d in res['variants'].items()}), flush=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
