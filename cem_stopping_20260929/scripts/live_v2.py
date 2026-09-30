#!/usr/bin/env python3
"""Deploy fold-specific rules on dev, checking exact tree equivalence and timing."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policies', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--roots', type=int, default=100)
    p.add_argument('--budget-policies', type=Path)
    p.add_argument('--plain-fixed', action='store_true')
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use sbatch')
    import torch
    import stable_worldmodel as swm
    import cemstop.tasks as tasks
    from cemstop.continuation import path
    from cemstop.evaluate import load_root, path_of
    from cemstop.online import run_online
    from cemstop.rules import CHECKPOINTS, Fixed, Gap, Converge, Band

    # Snapshot code still imports the already-validated source restore helpers.
    tasks.DIAG_SCRIPTS = Path(os.environ['CEMSTOP_REPO']) / 'diagnosis/scripts'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.out.mkdir(parents=True, exist_ok=False)
    arms = {'continuation': (.002, 'continuation'), 'fixed_pair': (.002, 'fixed_pair'),
            'fixed_quality': (0., 'fixed'), 'gap_v1': (0., 'gap')}
    if args.budget_policies:
        arms = {'budget_quality': (0., 'budget_stump'), 'budget_compute': (.002, 'budget_stump'),
                'fixed_pair': (.002, 'fixed_pair'), 'fixed_quality': (0., 'fixed')}
    all_rows = []
    for name in ('cube', 'reacher'):
        task = tasks.TASKS[name](swm)
        manifest = json.loads((args.data / name / 'manifest.json').read_text())
        root_specs = {r['root']: tasks.Root(**r) for r in manifest['roots'] if r['root'] < 100}
        try:
            # Warm-up all relevant GPU paths before timed comparison.
            warm_spec = {'kind': 'fixed', 'param': 30}
            run_online(task, root_specs[0], manifest['base_seed'], warm_spec)
            for rid in range(args.roots):
                tr = load_root(args.data / name / f'root_{rid:04d}.npz')
                fold = None
                for f in range(5):
                    ids = json.loads((args.policies / name / 'lambda_0.002' / f'fold_{f}' / 'validation_roots.json').read_text())
                    if rid in ids:
                        fold = f
                        break
                if fold is None:
                    raise RuntimeError('missing OOF policy')
                # Rotate order to reduce systematic timing bias.
                labels = list(arms)
                labels = labels[rid % len(labels):] + labels[:rid % len(labels)]
                for label in labels:
                    lam, family = arms[label]
                    source = args.budget_policies if family == 'budget_stump' else args.policies
                    spec = json.loads((source / name / f'lambda_{lam:g}' / f'fold_{fold}' / f'{family}.json').read_text())
                    if family == 'budget_stump':
                        from cemstop.budget_stump import path as budget_path
                        i, j = budget_path(spec, tr)
                    elif family == 'continuation':
                        i, j = path(spec, tr)
                    elif family == 'fixed_pair':
                        i = CHECKPOINTS.index(spec['k1'])
                        j = None if tr.p1_term[i] else CHECKPOINTS.index(spec['k2'])
                    else:
                        cls = {'fixed': Fixed, 'gap': Gap, 'converge': Converge, 'band': Band}[family]
                        i, j = path_of(cls(spec['param']), tr)
                    live = run_online(task, root_specs[rid], manifest['base_seed'], spec, plain_fixed=args.plain_fixed)
                    ks = [CHECKPOINTS[i]] + ([] if j is None else [CHECKPOINTS[j]])
                    expected_success = bool(tr.leaf_success[i, 0 if j is None else j])
                    expected_dist = float(tr.leaf_dist[i, 0 if j is None else j])
                    if live['ks'] != ks or live['success'] != expected_success or live['distance'] != expected_dist:
                        raise RuntimeError(f'live/tree outcome mismatch {name} {rid} {label}: {live["ks"]} vs {ks}')
                    for stage, k in enumerate(ks):
                        expected_costs = tr.costs1[:k] if stage == 0 else tr.costs2[i, :k]
                        expected_mean = tr.extra['means1'][i] if stage == 0 else tr.extra['means2'][i, j]
                        same_cost = live['costs'][stage] is None or np.array_equal(live['costs'][stage], expected_costs)
                        if not same_cost or not np.array_equal(live['means'][stage], expected_mean):
                            raise RuntimeError(f'live/tree prefix mismatch {name} {rid} {label}')
                    row = {k: v for k, v in live.items() if k not in ('costs', 'means')}
                    row.update(task=name, root=rid, episode=root_specs[rid].episode, arm=label, matched=True)
                    all_rows.append(row)
                    with (args.out / 'episodes.jsonl').open('a') as fp:
                        fp.write(json.dumps(row) + '\n')
                print(name, rid, 'LIVE_MATCH_OK', flush=True)
        finally:
            task.close()
    summary = {}
    for name in ('cube', 'reacher'):
        summary[name] = {}
        for label in arms:
            rows = [r for r in all_rows if r['task'] == name and r['arm'] == label]
            if len(rows) != args.roots:
                raise RuntimeError('incomplete live evaluation')
            summary[name][label] = {'n': len(rows), **{
                k: float(np.mean([r[k] for r in rows])) for k in
                ('success', 'iterations', 'planning_seconds', 'episode_seconds')}}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('LIVE_V2_OK', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
