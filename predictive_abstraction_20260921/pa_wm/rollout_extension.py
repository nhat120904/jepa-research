"""Frozen-reference replay plus controlled 4-vs-16-step continuation.

Image-feature-only training. Real-state refresh is a non-deployable diagnostic.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .runtime import require_slurm
from .branch_data import BranchStore
from .codec_rescue import FrameMetric, aggregates, groups_for
from .metric_bridge import prepare
from .local_transition import PatchTransition, unroll, window_batch
from .readout_diagnostic import selection, event_diagnostic


def refreshed(model, states, actions, interval):
    """Reads TRUE intermediate states every interval. Never a planning result."""
    if interval < 1 or states.shape[1] != actions.shape[1]+1:
        raise ValueError('Bad refresh cadence or transition alignment')
    predicted = []
    state = states[:, 0]
    for t, action in enumerate(actions.unbind(1)):
        if t % interval == 0:
            state = states[:, t]
        if model is not None:
            state = model(state, action)
        predicted.append(state)
    return torch.stack(predicted, 1)


def continuation_schedule():
    # First two match optimizer updates; last matches unrolled transition count.
    return [('more4_updates', 4, 600), ('more16', 16, 600), ('more4_transitions', 4, 2400)]


def paired_change(before, after, field):
    if [r['prefix'] for r in before] != [r['prefix'] for r in after]:
        raise ValueError('Bootstrap pairs must share prefix order')
    def value(row):
        return row['query_mse'][6] if field == 'ordered_mse' else row[field]
    delta = np.asarray([value(a)-value(b) for a, b in zip(before, after)])
    rng = np.random.default_rng(20260928)
    ci = np.quantile(rng.choice(delta, (2000, len(delta)), replace=True).mean(1), [.025, .975])
    return {'reduction': float(delta.mean()), 'prefix_ci': ci.tolist()}


def main():
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU sbatch required')
    old = json.loads((args.checkpoint.parent/'result.json').read_text())
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    store = BranchStore(args.features)
    torch.testing.assert_close(store.projection, ckpt['projection'])
    weights = ckpt['metric']
    metric = FrameMetric(len(weights['mean']), weights['mean'], weights['std'])
    metric.load_state_dict(weights)
    metric = metric.cuda().eval().requires_grad_(False)
    prepare(store, args.features, metric)
    patches, dim = store.patches, store.projection.shape[1]
    trajectories = {}
    with torch.no_grad():
        for i, row in enumerate(store.rows):
            raw = torch.cat([row['history'][-1:], row['future']]).float().cuda()
            projected = (raw @ store.projection.cuda()).flatten(1)
            trajectories[i] = ((projected-metric.mean)/metric.std).reshape(-1, patches, dim)
    ids = store.train
    lengths = torch.tensor([store.rows[i]['horizon'] for i in ids], device='cuda')
    states = torch.zeros(len(ids), 49, patches, dim, device='cuda')
    actions = torch.zeros(len(ids), 48, 2, device='cuda')
    for j, i in enumerate(ids):
        h = store.rows[i]['horizon']
        states[j, :h+1] = trajectories[i]
        actions[j, :h] = store.rows[i]['actions'].cuda()
    delta = torch.cat([trajectories[i][1:]-trajectories[i][:-1] for i in ids])
    scale = delta.square().mean(0).sqrt().clamp_min(.01)
    torch.testing.assert_close(scale.cpu(), ckpt['models']['step4']['delta_scale'])

    def model_from(name):
        model = PatchTransition(patches, dim, delta_scale=scale).cuda()
        model.load_state_dict(ckpt['models'][name])
        return model

    base = model_from('step4').eval().requires_grad_(False)
    blind = model_from('step4_no_action').eval().requires_grad_(False)
    result = {'status': 'CONTROLLED_ROLLOUT_EXTENSION_DEVELOPMENT', 'test_read': False,
              'control_evaluated': False, 'seed': 20260928, 'arms': {}, 'fit': {}}

    def record(name, split, prefix, h, predicted, real, truth, anchors):
        if not torch.isfinite(predicted).all():
            raise FloatingPointError(name)
        emb = metric.embed(predicted.flatten(-2))
        trace = torch.exp(-(emb[:, :, None]-anchors[:, None]).square().mean(-1))
        p = aggregates(trace).cpu().numpy()
        y = aggregates(truth).cpu().numpy()
        row = {'prefix': prefix, 'horizon': h,
               'state_mse': float((predicted-real[:, 1:]).square().mean()),
               'query_mse': ((p-y)**2).mean(0).tolist(),
               'events': event_diagnostic(trace.cpu().numpy(), truth.cpu().numpy()),
               'default_regret': float(y[:, 6].max()-y[0, 6]), **selection(p[:, 6], y[:, 6])}
        result['arms'].setdefault(name, {}).setdefault(split, []).append(row)

    def evaluate(models, reference_only=False):
        with torch.inference_mode():
            for split in ('train', 'val'):
                for (prefix, h), indices in groups_for(store, split).items():
                    real = torch.stack([trajectories[i] for i in indices])
                    act = torch.stack([store.rows[i]['actions'] for i in indices]).cuda()
                    truth = torch.stack([store.rows[i]['trace'] for i in indices]).cuda()
                    anchors = torch.stack([store.rows[i]['metric_anchors'] for i in indices]).cuda()
                    if reference_only:
                        variants = {'base4': unroll(base, real[:, 0], act),
                                    'old_no_action': unroll(blind, real[:, 0], torch.zeros_like(act))}
                        for k in (4, 8, 16):
                            variants[f'base4_refresh{k}'] = refreshed(base, real, act, k)
                            variants[f'persistence_refresh{k}'] = refreshed(None, real, act, k)
                    else:
                        variants = {name: unroll(model, real[:, 0], act) for name, model in models.items()}
                        variants['more16_shuffled'] = unroll(models['more16'], real[:, 0], act.roll(1, 0))
                    for name, pred in variants.items():
                        record(name, split, prefix, h, pred, real, truth, anchors)
                print('evaluation:', 'reference' if reference_only else 'continuation', split, flush=True)

    evaluate({}, reference_only=True)
    # Fail closed BEFORE continuation if frozen reference is not reproducible.
    for split in ('train', 'val'):
        for h in sorted({r['horizon'] for r in result['arms']['base4'][split]}):
            rows = [r for r in result['arms']['base4'][split] if r['horizon'] == h]
            ref = old['arms']['step4_open'][split]['summary'][f'bank{h}_h{h}']
            if (abs(np.mean([r['state_mse'] for r in rows])-ref['state_mse']) > 1e-4
                    or np.max(np.abs(np.mean([r['query_mse'] for r in rows], 0)-ref['query_mse'])) > 1e-4
                    or abs(np.mean([r['regret_first'] for r in rows])-ref['regret_first']) > 1e-4):
                raise AssertionError('53741 frozen reference mismatch')
    result['reference_reproduced'] = True
    print('53741 reproduced before training', flush=True)
    models = {}
    for name, h, steps in continuation_schedule():
        torch.manual_seed(20260928)
        model = model_from('step4').train()
        generator = torch.Generator(device='cuda').manual_seed(20260928)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        started = time.monotonic()
        for step in range(steps):
            x, a, target = window_batch(states, actions, lengths, generator, h)
            pred = unroll(model, x, a)
            loss = ((pred-target)/scale).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(name)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 0 and norm.item() == 0:
                raise AssertionError('No continuation gradient')
            opt.step()
            if (step+1) % 300 == 0:
                print(f'{name} {step+1}/{steps} loss={loss.item():.6f}', flush=True)
        models[name] = model.eval()
        result['fit'][name] = {'steps': steps, 'train_horizon': h, 'transition_examples': steps*h*32,
                              'seconds': time.monotonic()-started, 'last_loss': float(loss.detach()),
                              'parameters': sum(p.numel() for p in model.parameters())}
    evaluate(models)
    result['summary'] = {}
    for name, splits in result['arms'].items():
        result['summary'][name] = {}
        for split, all_rows in splits.items():
            summary = {}
            for h in sorted({r['horizon'] for r in all_rows}):
                rows = [r for r in all_rows if r['horizon'] == h]
                summary[h] = {k: float(np.mean([r[k] for r in rows])) for k in
                              ('state_mse', 'regret_first', 'regret_uniform_ties', 'default_regret')}
                summary[h]['query_mse'] = np.mean([r['query_mse'] for r in rows], 0).tolist()
                summary[h]['events'] = {k: sum(r['events'][k] for r in rows) for k in rows[0]['events']}
            result['summary'][name][split] = summary
    result['paired_val'] = {}
    progress = {}
    for h in (48, 64):
        after = [r for r in result['arms']['more16']['val'] if r['horizon'] == h]
        result['paired_val'][h] = {}
        for name in ('base4', 'more4_updates', 'more4_transitions'):
            before = [r for r in result['arms'][name]['val'] if r['horizon'] == h]
            result['paired_val'][h][name] = {k: paired_change(before, after, k) for k in
                                           ('state_mse', 'ordered_mse', 'regret_uniform_ties')}
        m = result['summary']['more16']['val'][h]
        b = result['summary']['base4']['val'][h]
        controls = [result['summary'][n]['val'][h] for n in ('more4_updates', 'more4_transitions')]
        progress[h] = bool(m['query_mse'][6] <= .8*b['query_mse'][6]
                           and m['query_mse'][6] <= .9*min(c['query_mse'][6] for c in controls)
                           and m['regret_uniform_ties'] <= b['regret_uniform_ties']+1e-8)
    result['query_progress_gate'] = {'by_horizon': progress, 'pass': all(progress.values())}
    result['limitations'] = ['one seed; reused development validation; not a method/control win',
        'refresh reads realized future states; persistence refresh is a matched non-deployable control',
        'transition-count matching is approximate compute matching, not identical wall time or optimizer updates',
        'no-action checkpoint is frozen old reference, not a continuation-budget-matched arm',
        'feature loss only; visual query evaluation still task-designed; no new query/arena test']
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    torch.save({'models': {k: v.cpu().state_dict() for k, v in models.items()},
                'metric': ckpt['metric'], 'projection': store.projection}, args.output/'checkpoints.pt')
    lines = ['# Controlled rollout extension', '',
             'Refresh variants see real intermediate futures, NOT deployable planning.', '',
             '| Arm | H | Train / val feature MSE | Val ordered MSE | Regret first / uniform | Missed / positive |',
             '|---|---:|---:|---:|---:|---:|']
    for name, splits in result['summary'].items():
        for h, m in splits['val'].items():
            train = splits['train'].get(h)
            train_mse = '—' if train is None else f"{train['state_mse']:.6f}"
            e = m['events']
            lines.append(f"| {name} | {h} | {train_mse} / {m['state_mse']:.6f} | {m['query_mse'][6]:.6f} | "
                         f"{m['regret_first']:.6f} / {m['regret_uniform_ties']:.6f} | "
                         f"{e['missed_below_0_1']}/{e['positive_events']} |")
    lines += ['', 'Development query-progress gate:', json.dumps(result['query_progress_gate'], indent=2),
              '', 'Paired prefix bootstrap (positive means improvement):', json.dumps(result['paired_val'], indent=2)]
    report = '\n'.join(lines)+'\n'
    (args.output/'REPORT.md').write_text(report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
