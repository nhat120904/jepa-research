"""Matched reader adaptation followed by learned-only closed-loop evaluation.

Source/WM and target stay fixed. No metric gates or oracle selectors. See protocol.
"""
import argparse
import copy
import gc
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import cta_closed_loop as closed
import cta_train as ct
from ti_wm.contract import require_compute
from ti_wm.cta_runtime import Planner, run_episode
from ti_wm.gates import mcnemar_exact, paired_diff
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer
from ti_wm.sibling import rank_loss
from ti_wm.wb import Logger

ARMS = ('P0', 'DIRECT8', 'BASE_G', 'BASE_E', 'CTRL_G', 'CTRL_E', 'ADAPT_G', 'ADAPT_E')
ROUTES = {'BASE_G': ('base', 'CTA8'), 'BASE_E': ('base', 'CTA8E'),
          'CTRL_G': ('control', 'CTA8'), 'CTRL_E': ('control', 'CTA8E'),
          'ADAPT_G': ('method', 'CTA8'), 'ADAPT_E': ('method', 'CTA8E')}


def write_json(path, value):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=default))
    tmp.replace(path)


def mixed_codes(source, greedy, soft, mode, k=8):
    """One evidence type per whole sibling bank, never different types across siblings."""
    if source.shape != greedy.shape or source.shape != soft.shape or source.shape[0] != len(mode) * k:
        raise ValueError('code shapes do not match whole banks')
    if not ((mode >= 0) & (mode <= 2)).all():
        raise ValueError('unknown code mode')
    choice = mode.repeat_interleave(k)[:, None, None]
    return torch.where(choice == 0, source, torch.where(choice == 1, greedy, soft))


@torch.inference_mode()
def cache_codes(models, split, banks, device, amp, out):
    """Predict once with frozen models; only train-root informative banks are cached."""
    enc, wm = models['enc'], models['wm']
    codebook = enc.fsq.codebook
    result = {key: [] for key in ('source', 'greedy', 'soft')}
    for start in range(0, len(banks), 16):
        i = torch.as_tensor(banks[start:start + 16])
        ctx = split.context(i, device)
        with amp():
            src = enc(ctx, split.future(i, device))
            memory = wm.encode(ctx, split.actions(i, device))
            pred = wm.decode(memory)
            soft = wm.logits(memory, pred).float().softmax(-1) @ codebook
        for name, code in (('source', src), ('greedy', codebook[pred]), ('soft', soft)):
            result[name].append(code.float().view(len(i), ct.K, *code.shape[1:]).cpu())
        if start % 1600 == 0:
            print(f'cache {start}/{len(banks)} banks', flush=True)
    result = {key: torch.cat(value) for key, value in result.items()}
    np.savez(out, banks=banks, root=split.root[banks], **{k: v.numpy() for k, v in result.items()})
    return result


def train(a):
    report = {'status': 'RUNNING', 'job': os.environ['SLURM_JOB_ID'], 'stage': 'load'}
    path = a.run / 'train_report.json'
    log = None
    started = time.perf_counter()
    try:
        parent = torch.load(a.parent / 'cta.pt', map_location='cpu')
        if parent['config'].get('target') != 'negative_mean_vertex_distance_divided_by_512':
            raise ValueError('Requires the geometry checkpoint')
        cfg = {**parent['config'], 'features': str(a.features.resolve()),
               'parent': str(a.parent.resolve()), 'parent_sha256': closed.sha256(a.parent / 'cta.pt'),
               'slurm_job': os.environ['SLURM_JOB_ID'], 'adapt_steps': 3000, 'lr_adapt': 3e-5,
               'adapt_mix_source_greedy_soft': [0.5, 0.25, 0.25], 'seed': 0,
               'sampling': 'geometry_informative_train_banks', 'lam': 0., 'steps3': 0,
               'frozen': [k for k in parent['state'] if k != 'reader']}
        report['config'] = cfg
        write_json(path, report)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda')
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
        train_split, dev = ct.Split(a.features / 'train'), ct.Split(a.features / 'dev')
        if set(train_split.root) != set(range(30250, 31050)) or set(dev.root) != set(range(2000, 2100)):
            raise ValueError('Unexpected train/dev split')
        banks = train_split.spread
        if not len(banks):
            raise ValueError('No informative training banks')
        goals = torch.from_numpy(np.load(a.features / 'goals.npy')).to(device)
        models = ct.build(cfg, device)
        for name, model in models.items():
            model.load_state_dict(parent['state'][name], strict=True)
            model.eval().requires_grad_(False)
        readers = {name: copy.deepcopy(models['reader']).train().requires_grad_(True)
                   for name in ('control', 'method')}
        optimizers = {name: torch.optim.AdamW(reader.parameters(), lr=cfg['lr_adapt'], weight_decay=cfg['wd'])
                      for name, reader in readers.items()}
        log = Logger(a.run, 'reader_refine', cfg, name=f"reader_{cfg['slurm_job']}")
        report.update(stage='cache_frozen_predictions', train_banks=len(banks))
        write_json(path, report)
        cache = cache_codes(models, train_split, banks, device, amp, a.run / 'train_codes.npz')
        report.update(stage='matched_reader_training')
        write_json(path, report)
        rng, mode_rng = np.random.default_rng(0), np.random.default_rng(917)
        for step in range(cfg['adapt_steps']):
            rows = rng.integers(0, len(banks), cfg['decisions'])
            i = torch.as_tensor(banks[rows])
            ctx = train_split.context(i, device)
            goal = ct.sample_goals(goals, rng, len(i), device)
            labels = train_split.cov[i].to(device)
            codes = {key: value[rows].flatten(0, 1).to(device) for key, value in cache.items()}
            mode = torch.as_tensor(mode_rng.choice(3, len(i), p=(.5, .25, .25)), device=device)
            mixed = mixed_codes(codes['source'], codes['greedy'], codes['soft'], mode)
            values = {'step': step + 1}
            for name, code in (('control', codes['source']), ('method', mixed)):
                with amp():
                    scores = readers[name](ctx, code, goal).float().view(-1, ct.K)
                loss = rank_loss(scores, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{name} reader loss')
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(readers[name].parameters(), 1., error_if_nonfinite=True)
                optimizers[name].step()
                values[f'{name}/loss'] = float(loss)
                values[f'{name}/grad'] = float(grad)
            if step % 100 == 0 or step + 1 == cfg['adapt_steps']:
                with (a.run / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(values) + '\n')
                log.log(values, step=step + 1)
                print(values, flush=True)
            if (step + 1) % 1000 == 0 or step + 1 == cfg['adapt_steps']:
                for name, reader in readers.items():
                    dest = a.run / name
                    dest.mkdir(exist_ok=True)
                    state = {**parent['state'], 'reader': {k: v.detach().cpu() for k, v in reader.state_dict().items()}}
                    torch.save({**parent, 'config': {**cfg, 'reader_variant': name,
                               'reader_updates': step + 1}, 'state': state}, dest / 'cta.pt')
        # Actual invariance check: only the two reader copies were optimized.
        report['non_reader_unchanged'] = all(
            torch.equal(value.detach().cpu(), parent['state'][name][key])
            for name, model in models.items() if name != 'reader' for key, value in model.state_dict().items())
        if not report['non_reader_unchanged']:
            raise RuntimeError('Frozen module changed')
        del cache, optimizers, train_split
        gc.collect()
        torch.cuda.empty_cache()
        report.update(stage='offline_and_runtime_consistency', variants={})
        write_json(path, report)
        for name, reader in readers.items():
            models['reader'] = reader.eval()
            ladder, scores = ct.ladder(models, dev, goals, device, amp, samples=0, norms=parent['norms'])
            with np.load(a.features / 'dev/meta.npz') as meta:
                np.savez(a.run / name / 'dev_scores.npz', root=dev.root, decision=meta['decision'],
                         cov8=dev.cov8, native_cov8=meta['native_cov8'], **scores)
            report['variants'][name] = {'ladder': ladder, 'checkpoint_sha256': closed.sha256(a.run / name / 'cta.pt')}
            write_json(a.run / name / 'train_report.json', report['variants'][name])
            write_json(path, report)
        del readers, models, dev, goals, parent
        gc.collect()
        torch.cuda.empty_cache()
        # Same inference contract, now including the deployed expected-code branch.
        visual = VisualScorer('cuda')
        goal_frames = np.load(a.smoke / 'goal_frames.npz')['frames']
        report['preflight'] = {}
        for name, folder in (('base', a.parent), ('control', a.run / 'control'), ('method', a.run / 'method')):
            planner = Planner(folder / 'cta.pt', visual, goal_frames)
            report['preflight'][name] = closed.preflight(
                planner, a.dev_shard, folder, tiers={**closed.PREFLIGHT_TIERS, 'CTA8E': 'pred_soft'})
            del planner
            torch.cuda.empty_cache()
            write_json(path, report)
        report.update(status='DONE', stage='done', seconds=time.perf_counter() - started)
        write_json(path, report)
    except Exception:
        report.update(status='FAILED', error=traceback.format_exc())
        write_json(path, report)
        raise
    finally:
        if log is not None:
            log.finish()


def evaluate(a):
    trained = json.loads((a.train_run / 'train_report.json').read_text())
    if trained['status'] != 'DONE' or not trained['non_reader_unchanged']:
        raise ValueError('Training did not complete correctly')
    if a.count != 10 or a.first not in range(2100, 2200, 10):
        raise ValueError('Expected one of ten fixed dev shards')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    base = Path(trained['config']['parent'])
    smoke = json.loads((a.smoke / 'smoke.json').read_text())
    if smoke['status'] != 'SMOKE_PASS':
        raise ValueError('Invalid environment smoke')
    cloner = CLONERS[smoke['clone_method']]
    runner = PolicyRunner(a.prep / 'checkpoint', 'cuda')
    planner = Planner(base / 'cta.pt', VisualScorer('cuda'), np.load(a.smoke / 'goal_frames.npz')['frames'])
    readers = {'base': planner.models['reader']}
    hashes = {'base': closed.sha256(base / 'cta.pt')}
    if hashes['base'] != trained['config']['parent_sha256']:
        raise ValueError('Parent checkpoint changed')
    for name in ('control', 'method'):
        path = a.train_run / name / 'cta.pt'
        hashes[name] = closed.sha256(path)
        if hashes[name] != trained['variants'][name]['checkpoint_sha256']:
            raise ValueError('Adapted checkpoint changed')
        blob = torch.load(path, map_location='cpu')
        readers[name] = copy.deepcopy(readers['base'])
        readers[name].load_state_dict(blob['state']['reader'], strict=True)
        readers[name].eval()
        del blob
    path = a.run / f'roots_{a.first}_{a.first + a.count - 1}.jsonl'
    wins = {arm: 0 for arm in ARMS}
    with path.open('x') as stream:
        for root in range(a.first, a.first + a.count):
            start = time.perf_counter()
            rec = {'root': root, 'arms': list(ARMS), 'checkpoint_hashes': hashes,
                   'train_job': trained['job'], 'selected_branch_only': True}
            for arm in ARMS:
                name, runtime_arm = ROUTES.get(arm, ('base', arm))
                planner.models['reader'] = readers[name]
                rec[arm] = run_episode(root, runtime_arm, runner, cloner, planner, log_candidates=False)
                wins[arm] += rec[arm]['success']
            rec['seconds'] = time.perf_counter() - start
            stream.write(json.dumps(rec) + '\n')
            stream.flush()
            print(root, {arm: rec[arm]['success'] for arm in ARMS}, f"{rec['seconds']:.0f}s", flush=True)
    write_json(a.run / f'summary_{a.first}.json', {'status': 'DONE', 'wins': wins, 'count': a.count, 'hashes': hashes})


def validate_records(records, roots):
    if len(records) != len(roots) or {r['root'] for r in records} != set(roots):
        raise ValueError('Missing, duplicate or unexpected roots')
    if any(r['arms'] != list(ARMS) for r in records):
        raise ValueError('Unexpected arms')
    if len({json.dumps(r['checkpoint_hashes'], sort_keys=True) for r in records}) != 1:
        raise ValueError('Mixed checkpoints')
    return sorted(records, key=lambda r: r['root'])


def aggregate(a):
    recs = [json.loads(line) for path in sorted(a.closed_run.glob('shard_*/roots_*.jsonl'))
            for line in path.read_text().splitlines() if line.strip()]
    recs = validate_records(recs, list(range(2100, 2200)))
    outcomes = {arm: np.array([r[arm]['success'] for r in recs], float) for arm in ARMS}
    coverage = {arm: np.array([r[arm]['max_coverage'] for r in recs]) for arm in ARMS}
    contrasts = [(arm, 'P0') for arm in ARMS if arm != 'P0']
    contrasts += [(f'ADAPT_{mode}', f'{ref}_{mode}') for mode in ('G', 'E') for ref in ('BASE', 'CTRL')]
    contrasts += [('ADAPT_G', 'DIRECT8'), ('ADAPT_E', 'DIRECT8')]
    result = {'status': 'DONE', 'scope': '100 reused dev roots, seed 0; not sealed confirmation',
              'primary': 'ADAPT_G-P0', 'secondary': 'ADAPT_E-P0',
              'success': {arm: {'count': int(v.sum()), 'n': len(v), 'rate': float(v.mean())} for arm, v in outcomes.items()},
              'mean_max_coverage': {arm: float(v.mean()) for arm, v in coverage.items()},
              'contrasts': {}, 'checkpoint_hashes': recs[0]['checkpoint_hashes'], 'decisions': {}}
    for left, right in contrasts:
        result['contrasts'][f'{left}-{right}'] = {
            'success': paired_diff(outcomes[left], outcomes[right]),
            'mcnemar': mcnemar_exact(outcomes[left], outcomes[right]),
            'max_coverage': paired_diff(coverage[left], coverage[right])}
    for arm in ARMS:
        decisions = [d for r in recs for d in r[arm]['decisions']]
        scores = [d['score'] for d in decisions if 'score' in d]
        result['decisions'][arm] = {'count': len(decisions),
            'override_fraction': float(np.mean([d['chosen'] != 0 for d in decisions])),
            'exact_score_tie_fraction': float(np.mean([np.ptp(s) == 0 for s in scores])) if scores else None}
    write_json(a.run / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main(a):
    require_compute()
    a.run.mkdir(parents=True, exist_ok=True)
    {'train': train, 'closed': evaluate, 'aggregate': aggregate}[a.mode](a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train', 'closed', 'aggregate'), required=True)
    for name in ('run', 'parent', 'features', 'smoke', 'prep', 'dev-shard', 'train-run', 'closed-run'):
        parser.add_argument(f'--{name}', type=Path, required=name == 'run')
    parser.add_argument('--first', type=int)
    parser.add_argument('--count', type=int)
    main(parser.parse_args())
