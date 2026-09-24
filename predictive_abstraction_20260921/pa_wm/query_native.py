"""Final bounded Wall pilot: query-space learning, no latent-coordinate loss.

All four arms receive identical RGB-derived targets. This is not generic JEPA or a
published frame-WM reproduction. Cross-query labels are evaluation-only.
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
from .models import SpatialActionSummary
from .fixed_summary_forecast import time_queries
from .codec_rescue import FrameMetric, aggregates, groups_for, candidate_difference_loss
from .metric_bridge import prepare
from .readout_diagnostic import selection, event_diagnostic
from .queries import chroma_features, similarity
from .rollout_extension import paired_change


class QueryNative(nn.Module):
    def __init__(self, patches, dim, kind='summary', width=128):
        super().__init__()
        if kind not in ('summary', 'frame', 'direct'):
            raise ValueError(kind)
        self.kind = kind
        self.context = SpatialActionSummary(patches, dim, width=width)
        if kind != 'direct':
            self.future_time = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
            self.future_attn = nn.MultiheadAttention(width, 4, batch_first=True)
            self.future_norm = nn.LayerNorm(width)
        self.query = nn.Sequential(nn.Linear(68, width), nn.GELU(), nn.Linear(width, width))
        self.read_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(),
                                     nn.Linear(64, 1), nn.Sigmoid())

    def encode(self, history, actions):
        context = self.context.context(history, actions)
        if self.kind == 'direct':
            return context
        h = actions.shape[1]
        count = 16 if self.kind == 'summary' else h
        q = self.future_time(time_queries(count, h, context.device, context.dtype))[None].expand(len(history), -1, -1)
        x, _ = self.future_attn(q, context, context, need_weights=False)
        return self.future_norm(q+x)

    def decode(self, representation, anchors, horizon):
        b, n = anchors.shape[:2]
        times = time_queries(horizon, horizon, anchors.device, anchors.dtype)
        q = self.query(torch.cat([anchors[:, None].expand(-1, horizon, -1, -1),
                                 times[None, :, None].expand(b, -1, n, -1)], -1)).flatten(1, 2)
        x, _ = self.read_attn(q, representation, representation, need_weights=False)
        return self.readout(q+x).reshape(b, horizon, n)


def query_loss(trace, truth, groups):
    error = (trace-truth).square()
    positive = truth > .05
    terms = [error[mask].mean() for mask in (positive, ~positive) if mask.any()]
    pred, target = aggregates(trace), aggregates(truth)
    return (torch.stack(terms).mean() + (pred-target).square().mean()
            + .1*candidate_difference_loss(pred, target, groups))


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
    store = BranchStore(args.features)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    torch.testing.assert_close(store.projection, checkpoint['projection'])
    state = checkpoint['metric']
    metric = FrameMetric(len(state['mean']), state['mean'], state['std'])
    metric.load_state_dict(state)
    metric = metric.cuda().eval().requires_grad_(False)
    mean, std = prepare(store, args.features, metric)
    groups = {split: groups_for(store, split) for split in ('train', 'val')}
    # Query swap is fixed by prefix ordering within SAME split/layout, not outcome.
    first = {}
    for i, row in enumerate(store.rows):
        first.setdefault(row['prefix_id'], i)
    layout_groups = {}
    for prefix, i in first.items():
        r = store.rows[i]
        layout_groups.setdefault((r['split'], tuple(r['layout'])), []).append(prefix)
    donor = {}
    for prefixes in layout_groups.values():
        prefixes = sorted(prefixes)
        assert len(prefixes) > 1
        donor.update({p: prefixes[(j+1) % len(prefixes)] for j, p in enumerate(prefixes)})
    manifest = json.loads((args.features/'manifest.json').read_text())
    rgb_anchors = {}
    for prefix, i in first.items():
        with np.load(Path(manifest['dataset_root'])/store.rows[i]['source']) as raw:
            rgb_anchors[prefix] = chroma_features(raw['anchors'])
    for row in store.rows:
        other = donor[row['prefix_id']]
        row['cross_anchors'] = store.rows[first[other]]['metric_anchors']
        if row['split'] == 'val':
            with np.load(Path(manifest['dataset_root'])/row['source']) as raw:
                f = chroma_features(raw['rgb'][1:])
                a, b = rgb_anchors[other]
                row['cross_trace'] = torch.from_numpy(np.stack([similarity(f, a), similarity(f, b)], -1))

    def batch(ids, cross=False):
        b = store.batch(ids, 'cuda')
        a = torch.stack([store.rows[i]['cross_anchors' if cross else 'metric_anchors'] for i in ids]).cuda()
        truth = torch.stack([store.rows[i]['cross_trace' if cross else 'trace'] for i in ids]).cuda()
        return b, (a-mean)/std, truth

    result = {'status': 'FINAL_QUERY_NATIVE_WALL_DEVELOPMENT', 'seed': 20260929,
              'test_read': False, 'control_evaluated': False, 'cross_query_donors': donor,
              'arms': {}, 'training': {}, 'auto_follow_on': False}
    weights = {}
    for name, kind, blind in [('summary', 'summary', False), ('frame', 'frame', False),
                               ('direct', 'direct', False), ('no_action', 'summary', True)]:
        torch.manual_seed(20260929)
        model = QueryNative(store.patches, store.dim, kind).cuda()
        rng = np.random.default_rng(20260929)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        started = time.monotonic()
        for step in range(2000):
            h = int(rng.choice([32, 48]))
            keys = [k for k in groups['train'] if k[1] == h]
            selected = rng.choice(len(keys), 2, replace=False)
            ids = [i for j in selected for i in groups['train'][keys[j]]]
            b, anchors, truth = batch(ids)
            actions = torch.zeros_like(b['actions']) if blind else b['actions']
            z = model.encode(b['history'], actions)
            loss = query_loss(model.decode(z, anchors, h), truth, 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(name)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 0 and norm.item() == 0:
                raise AssertionError('No query gradient')
            opt.step()
            if (step+1) % 500 == 0:
                print(f'{name} {step+1}/2000 loss={loss.item():.6f}', flush=True)
        model.eval()
        result['training'][name] = {'seconds': time.monotonic()-started, 'last_loss': float(loss.detach()),
                                    'parameters': sum(p.numel() for p in model.parameters())}
        result['arms'][name] = {}
        with torch.inference_mode():
            for split in ('train', 'val'):
                for cross in ([False, True] if split == 'val' else [False]):
                    label = split + ('_cross_query' if cross else '')
                    rows = []
                    for (prefix, h), ids in groups[split].items():
                        b, anchors, truth = batch(ids, cross)
                        actions = torch.zeros_like(b['actions']) if blind else b['actions']
                        z = model.encode(b['history'], actions)
                        trace = model.decode(z, anchors, h)
                        predicted, target = aggregates(trace).cpu().numpy(), aggregates(truth).cpu().numpy()
                        shuffled = model.decode(model.encode(b['history'], actions.roll(1, 0)), anchors, h)
                        rows.append({'prefix': prefix, 'horizon': h,
                            'mse': ((predicted-target)**2).mean(0).tolist(),
                            'shuffled_ordered_mse': float((aggregates(shuffled).cpu().numpy()[:, 6]-target[:, 6]).dot(
                                aggregates(shuffled).cpu().numpy()[:, 6]-target[:, 6])/len(ids)),
                            'zero_ordered_mse': float((target[:, 6]**2).mean()),
                            'positive_ordered': int((target[:, 6] >= .5).sum()),
                            'events': event_diagnostic(trace.cpu().numpy(), truth.cpu().numpy()),
                            'default_regret': float(target[:, 6].max()-target[0, 6]),
                            **selection(predicted[:, 6], target[:, 6])})
                    summary = {}
                    for h in sorted({r['horizon'] for r in rows}):
                        subset = [r for r in rows if r['horizon'] == h]
                        summary[h] = {key: float(np.mean([r[key] for r in subset])) for key in
                                      ('regret_first', 'regret_uniform_ties', 'default_regret',
                                       'zero_ordered_mse', 'shuffled_ordered_mse')}
                        summary[h]['mse'] = np.mean([r['mse'] for r in subset], 0).tolist()
                        summary[h]['positive_ordered'] = sum(r['positive_ordered'] for r in subset)
                        summary[h]['events'] = {k: sum(r['events'][k] for r in subset) for k in subset[0]['events']}
                    result['arms'][name][label] = {'summary': summary, 'groups': rows}
        weights[name] = model.cpu().state_dict()
        del model, opt

    # Operational stop rule. No theorem of impossibility and no acceptance claim.
    usable = {}
    for name in ('summary', 'frame', 'direct'):
        checks = []
        for h in (48, 64):
            m = result['arms'][name]['val']['summary'][h]
            n = result['arms']['no_action']['val']['summary'][h]
            checks.append(m['mse'][6] <= .8*min(n['mse'][6], m['zero_ordered_mse'])
                          and m['regret_first'] <= m['default_regret']+.01
                          and m['mse'][6] <= .9*m['shuffled_ordered_mse'])
        usable[name] = bool(all(checks))
    if not any(usable.values()):
        verdict = 'STOP_CURRENT_WALL_METHOD_IMPLEMENTATION'
    elif not usable['summary']:
        verdict = 'STOP_COMPACT_VARIANT_CONTROLS_HAVE_SIGNAL'
    else:
        beats = all(result['arms']['summary']['val']['summary'][h]['mse'][6]
                    <= .9*min(result['arms'][n]['val']['summary'][h]['mse'][6] for n in ('frame', 'direct'))
                    for h in (48, 64))
        verdict = 'CANDIDATE_FOR_FRESH_CONFIRMATION' if beats else 'NO_COMPACT_ACCURACY_ADVANTAGE_STOP_TUNING'
    result['decision'] = {'usable_by_arm': usable, 'verdict': verdict,
                          'cross_query_is_required_for_future_claim_but_not_this_native_feasibility_gate': True}
    result['paired_val'] = {}
    for h in (48, 64):
        result['paired_val'][h] = {}
        def paired_rows(name):
            return [{**r, 'query_mse': r['mse']} for r in result['arms'][name]['val']['groups']
                    if r['horizon'] == h]
        for name in ('frame', 'direct', 'no_action'):
            result['paired_val'][h][name] = {field: paired_change(paired_rows(name), paired_rows('summary'), field)
                                            for field in ('ordered_mse', 'regret_uniform_ties')}
    result['limitations'] = ['one seed, reused validation, 24 train prefixes',
        'same loss and updates, but direct has different parameter/compute count',
        'RGB-derived task-designed supervision, not simulator labels or generic latent JEPA',
        'frame control is ordered future tokens, NOT reproduced published frame world model',
        'cross-query pairs may lack positive ordered events: report counts, never resample to get positive',
        'no accuracy advantage does not exclude unmeasured compute advantage; no further tuning in this cycle']
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    torch.save({'models': weights, 'metric': state, 'projection': store.projection}, args.output/'checkpoints.pt')
    lines = ['# Final query-native Wall pilot', '', f'Operational verdict: {verdict}', '',
             '| Arm | Split | H | Ordered MSE | Regret first / uniform | Default regret | Positive ordered |',
             '|---|---|---:|---:|---:|---:|---:|']
    for name, arm in result['arms'].items():
        for split, data in arm.items():
            for h, m in data['summary'].items():
                lines.append(f"| {name} | {split} | {h} | {m['mse'][6]:.6f} | "
                             f"{m['regret_first']:.6f} / {m['regret_uniform_ties']:.6f} | "
                             f"{m['default_regret']:.6f} | {m['positive_ordered']} |")
    report = '\n'.join(lines)+'\n'
    (args.output/'REPORT.md').write_text(report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
