#!/usr/bin/env python3
"""One bounded simplification, same episode-grouped folds as v2."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cemstop.budget_stump import fit_budget, outcomes, initial_features
from cemstop.continuation import outcomes as continuation_outcomes
from cemstop.evaluate import load_root
from refine_v2 import bootstrap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use sbatch')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {}
    for task in ('cube', 'reacher'):
        roots = [load_root(args.data / task / f'root_{i:04d}.npz') for i in range(100)]
        if any(not (args.data / task / f'root_{i:04d}.json').exists() for i in range(100)):
            raise RuntimeError('incomplete roots')
        manifest = json.loads((args.data / task / 'manifest.json').read_text())
        groups = np.array([r['episode'] for r in manifest['roots'][:100]])
        report[task] = {}
        for lam in (0., .002):
            base = args.previous / task / f'lambda_{lam:g}'
            cv_s, cv_i = np.zeros(100, bool), np.zeros(100)
            train_scores, specs = [], []
            for f in range(5):
                val = json.loads((base / f'fold_{f}' / 'validation_roots.json').read_text())
                train = [r for i, r in enumerate(roots) if i not in val]
                spec = fit_budget(train, lam)
                ev = outcomes(spec, [roots[i] for i in val])
                cv_s[val], cv_i[val] = ev['success'], ev['iterations']
                tr_ev = outcomes(spec, train)
                train_scores.append(float(tr_ev['success'].mean()))
                specs.append(spec)
                out = args.out / task / f'lambda_{lam:g}' / f'fold_{f}'
                out.mkdir(parents=True)
                (out / 'budget_stump.json').write_text(json.dumps(spec))
                (out / 'validation_roots.json').write_text(json.dumps(val))
            # Structural causality: first-iteration features cannot read future entries.
            for r in roots:
                x = initial_features(r.costs1, r.extra['means1'], r.extra['std1'])
                c, m, s = r.costs1.copy(), r.extra['means1'].copy(), r.extra['std1'].copy()
                c[1:], m[1:], s[1:] = np.nan, np.nan, np.nan
                assert np.array_equal(x, initial_features(c, m, s))
            comparisons = {}
            with np.load(base / 'all_dev/oof.npz') as ref:
                for label in ('fixed', 'fixed_pair', 'gap', 'continuation'):
                    comparisons[label] = {'success': bootstrap(cv_s, ref[label + '_success'], groups),
                                          'iterations': bootstrap(cv_i, ref[label + '_iterations'], groups)}
            old = json.loads((base / 'all_dev/continuation.json').read_text())
            old_training = continuation_outcomes(old, roots)
            spec = fit_budget(roots, lam)
            tr = outcomes(spec, roots)
            out = args.out / task / f'lambda_{lam:g}' / 'all_dev'
            out.mkdir(parents=True)
            (out / 'budget_stump.json').write_text(json.dumps(spec))
            np.savez_compressed(out / 'oof.npz', success=cv_s, iterations=cv_i)
            report[task][str(lam)] = {'success': float(cv_s.mean()), 'iterations': float(cv_i.mean()),
                                     'vs': comparisons, 'fold_specs': specs, 'train_fold_success': train_scores,
                                     'all_dev_train_success': float(tr['success'].mean()),
                                     'v2_continuation_all_dev_train_success': float(old_training['success'].mean())}
        print(task, json.dumps(report[task]), flush=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
