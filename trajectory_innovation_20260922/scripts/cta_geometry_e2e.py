"""One bounded PushT iteration: relabel -> codec/WM -> closed loop -> report.

Development only. No metric gates between stages, no new candidate collection,
no privileged input to learned deployment, no code changes to previous artifacts.
"""
import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import cta_aggregate as aggregate
import cta_closed_loop as closed
import cta_train as ct
from cta_round2 import Recorder
from ti_wm.contract import require_compute
from ti_wm.cta_geometry import native_geometry_check, registration_score


def ranking_summary(scores, labels, roots, ci=False):
    """Keep per-candidate choices in the score archive, not the JSON summary."""
    result = ct.ranking_metrics(scores, labels, roots, ci=ci)
    result.pop('chosen')
    return result


def recover_training(previous, run):
    """Recover completed training artifacts after a reporting failure; never retrain."""
    source = previous / 'train'
    for name in ('cta.pt', 'dev_scores.npz', 'metrics.jsonl'):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    final = [json.loads(line) for line in (source / 'metrics.jsonl').read_text().splitlines()
             if line.strip()]
    if not final or final[-1].get('stage') != 'final' or final[-1].get('step') != 9000:
        raise ValueError('Recovery requires the completed 3000+6000-update run')
    run.mkdir()
    report = {'status': 'RECOVERED', 'recovered_from': str(previous.resolve()),
              'checkpoint_sha256': closed.sha256(source / 'cta.pt'),
              'final_logged_metrics': final[-1], 'ladder': {}, 'native_coverage_diagnostic': {}}
    with np.load(source / 'dev_scores.npz') as saved, np.load(previous / 'features/dev/meta.npz') as meta:
        for key in ('root', 'decision', 'cov8', 'native_cov8'):
            if not np.array_equal(saved[key], meta[key]):
                raise ValueError(f'Recovery score/feature mismatch: {key}')
        for tier in ct.TIERS:
            report['ladder'][tier] = ranking_summary(saved[tier], saved['cov8'], saved['root'], ci=True)
            report['native_coverage_diagnostic'][tier] = ranking_summary(
                saved[tier], saved['native_cov8'], saved['root'])
    for name in ('cta.pt', 'dev_scores.npz'):
        (run / name).symlink_to((source / name).resolve())
    (run / 'train_report.json').write_text(json.dumps(report, indent=2, default=float))
    return {'checkpoint': str(run / 'cta.pt'), 'report': str(run / 'train_report.json'),
            'recovered_from': str(source), 'checkpoint_sha256': report['checkpoint_sha256'],
            'training_updates_this_job': 0}


def relabel(features, collection, out):
    out.mkdir()
    report = {'target': 'negative_mean_vertex_distance_divided_by_512',
              'legacy_cov8_field_contains': 'registration_target', 'splits': {}}
    for split, lo, hi in [('train', 30250, 31049), ('dev', 2000, 2099)]:
        dest = out / split
        dest.mkdir()
        with np.load(features / split / 'meta.npz') as archive:
            meta = {key: archive[key] for key in archive.files}
        rows = []
        for path in sorted(collection.glob('shard_*.npz')):
            if not lo <= int(path.stem.split('_')[1]) <= hi:
                continue
            with np.load(path) as raw:
                rows.append({key: raw[key] for key in ('root', 'decision', 't', 'chunk', 'cov8', 'phys8')})
        if not rows:
            raise ValueError(f'No raw {split} shards')
        raw = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
        for key in ('root', 'decision', 't', 'chunk', 'cov8'):
            if not np.array_equal(raw[key], meta[key]):
                raise ValueError(f'Raw/feature identity mismatch: {split}/{key}')
        if raw['phys8'].shape != (len(meta['root']), 8, 10):
            raise ValueError('Unexpected physical-state schema')
        labels = registration_score(raw['phys8'][..., 4:7]).astype(np.float32)
        if not np.isfinite(labels).all() or (labels > 0).any():
            raise ValueError('Invalid registration target')
        native = meta['cov8'].copy()
        meta.update(native_cov8=native, cov8=labels)
        np.savez(dest / 'meta.npz', **meta)
        for key in ('cur', 'prev', 'end', 'seg'):
            (dest / f'{key}.npy').symlink_to((features / split / f'{key}.npy').resolve())
        cov_flat = np.ptp(native, axis=1) <= ct.RANK_MARGIN
        geo_spread = np.ptp(labels, axis=1) > ct.RANK_MARGIN
        report['splits'][split] = {
            'decisions': len(labels), 'roots': len(np.unique(meta['root'])),
            'native_coverage_flat_banks': int(cov_flat.sum()),
            'registration_informative_banks': int(geo_spread.sum()),
            'coverage_flat_registration_informative': int((cov_flat & geo_spread).sum()),
            'raw_files': [str(p) for p in sorted(collection.glob('shard_*.npz'))
                          if lo <= int(p.stem.split('_')[1]) <= hi],
        }
    for name in ('pca.pt', 'goals.npy'):
        (out / name).symlink_to((features / name).resolve())
    (out / 'target_report.json').write_text(json.dumps(report, indent=2))
    return report


def train(a, features, run):
    run.mkdir()
    parent = torch.load(a.parent / 'cta.pt', map_location='cpu')
    cfg = {**parent['config'], 'features': str(features), 'parent': str(a.parent),
           'parent_sha256': closed.sha256(a.parent / 'cta.pt'), 'seed': 0,
           'steps1': a.source_steps, 'steps2': a.wm_steps, 'steps3': 0, 'lam': 0.,
           'adapt_steps': 0, 'samples': 0, 'lr': 1e-4, 'warmup': 200,
           'eval_every': 2000, 'eval_n': 600, 'slurm_job': os.environ['SLURM_JOB_ID'],
           'target': 'negative_mean_vertex_distance_divided_by_512',
           'sampling': 'all_decisions_uniform', 'device': 'cuda',
           'scope': 'warmstarted development iteration; not a target-only causal ablation'}
    report = {'status': 'RUNNING', 'config': cfg}
    log = Recorder(run, cfg)
    t0 = time.perf_counter()
    try:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        rng = np.random.default_rng(0)
        device = torch.device('cuda')
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
        train_split, dev = ct.Split(features / 'train'), ct.Split(features / 'dev')
        if set(train_split.root) != set(range(30250, 31050)) or set(dev.root) != set(range(2000, 2100)):
            raise ValueError('Unexpected root split; sealed roots must stay unopened')
        # Future reconstruction sees every phase; tied labels generate no forced preference.
        train_split.rank_pool = np.arange(train_split.n)
        goals = torch.from_numpy(np.load(features / 'goals.npy')).to(device)
        models = ct.build(cfg, device)
        for name, model in models.items():
            model.load_state_dict(parent['state'][name], strict=True)
        norms = parent['norms']
        report['before'], _ = ct.ladder(models, dev, goals, device, amp,
                                       n=cfg['eval_n'], ci=False, norms=norms)
        step = ct.stage1(models, train_split, dev, goals, cfg, device, log, rng, norms, amp)
        pca = {key: parent[key] for key in ('pca_mean', 'pca_basis')}

        def save(name):
            torch.save({'config': cfg, 'state': {k: m.state_dict() for k, m in models.items()},
                        'norms': norms, **pca}, run / name)

        save('codec_stage.pt')
        codes = ct.source_indices(models, train_split, device, amp, batch=cfg['wm_decisions'])
        step = ct.stage2(models, train_split, dev, goals, codes, cfg, device, log, rng, amp, step)
        save('cta.pt')
        report['ladder'], scores = ct.ladder(models, dev, goals, device, amp, samples=0, norms=norms)
        with np.load(features / 'dev' / 'meta.npz') as meta:
            native = meta['native_cov8']
            decisions = meta['decision']
        report['native_coverage_diagnostic'] = {
            tier: ranking_summary(value, native, dev.root) for tier, value in scores.items()}
        np.savez(run / 'dev_scores.npz', root=dev.root, decision=decisions,
                 cov8=dev.cov8, native_cov8=native, **scores)
        log.log({'stage': 'final', **ct.brief(report['ladder'])}, step=step)
        report.update(status='DONE', seconds=time.perf_counter() - t0)
    except Exception:
        report.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        try:
            (run / 'train_report.json').write_text(json.dumps(report, indent=2, default=float))
        finally:
            log.logger.finish()
    return {'checkpoint': str(run / 'cta.pt'), 'report': str(run / 'train_report.json'),
            'seconds': report['seconds']}


def main(a):
    require_compute()
    if a.first != 2100 or a.count != 10:
        raise ValueError('This development iteration is locked to roots 2100-2109')
    a.run.mkdir(parents=True, exist_ok=True)
    path = a.run / 'e2e_report.json'
    if path.exists():
        raise FileExistsError(path)
    report = {'status': 'RUNNING', 'job': os.environ['SLURM_JOB_ID'],
              'scope': 'development only, ten roots; no statistically confirmed improvement claim'}

    def record(stage):
        report['stage'] = stage
        path.write_text(json.dumps(report, indent=2, default=float))
        print(f'E2E stage: {stage}', flush=True)

    try:
        train_run = a.run / 'train'
        if a.resume_from is not None:
            record('recover_completed_training')
            previous = json.loads((a.resume_from / 'e2e_report.json').read_text())
            report['geometry'] = native_geometry_check()
            report['labels'] = previous['labels']
            report['training'] = recover_training(a.resume_from, train_run)
        else:
            record('native_geometry_and_labels')
            report['geometry'] = native_geometry_check()
            features = a.run / 'features'
            report['labels'] = relabel(a.features, a.collection, features)
            record('codec_and_world_model')
            report['training'] = train(a, features, train_run)
        gc.collect()
        torch.cuda.empty_cache()
        record('closed_loop')
        cl_run = a.run / 'closed'
        cl_run.mkdir()
        # Diagnostics are part of the same run, not preconditions for deploying the learned arm.
        closed.main(SimpleNamespace(
            run=cl_run, prep=a.prep, smoke=a.smoke, train_run=train_run,
            dev_shard=a.collection / 'shard_2000_2049.npz', first=a.first, count=a.count,
            arms='P0,PHYS8,GEOM8,FULL8,CODE8,CTA8,CTA8E,DIRECT8', device='cuda',
            max_decisions=None, no_log_candidates=False, skip_done=[]))
        record('aggregate')
        aggregate.main([cl_run], a.run / 'aggregate', a.first, a.count)
        # Report the native continuous score as well as binary success.
        records = [json.loads(line) for line in (cl_run / 'roots_2100_2109.jsonl').read_text().splitlines()]
        report['mean_max_coverage'] = {
            arm: float(np.mean([r[arm]['max_coverage'] for r in records])) for arm in records[0]['arms']}
        report.update(status='DONE', aggregate=str(a.run / 'aggregate' / 'summary.json'))
        record('done')
    except Exception:
        report.update(status='FAILED', error=traceback.format_exc())
        path.write_text(json.dumps(report, indent=2, default=float))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run', 'parent', 'features', 'collection', 'prep', 'smoke'):
        p.add_argument(f'--{name}', type=Path, required=True)
    p.add_argument('--source-steps', type=int, default=3000)
    p.add_argument('--resume-from', type=Path, help='Recover a completed training run and continue evaluation')
    p.add_argument('--wm-steps', type=int, default=6000)
    p.add_argument('--first', type=int, default=2100)
    p.add_argument('--count', type=int, default=10)
    main(p.parse_args())
