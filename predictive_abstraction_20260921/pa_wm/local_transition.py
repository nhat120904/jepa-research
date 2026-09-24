"""Short spatial transitions before another trajectory-abstraction experiment.

Training uses only cached image features/actions. Frozen visual metric is evaluation
only. Teacher-forced evaluation sees real future history: diagnostic, NOT planning.
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
from .models import temporal_encoder
from .readout_diagnostic import selection, event_diagnostic


class PatchTransition(nn.Module):
    def __init__(self, patches=16, dim=32, width=64, delta_scale=None):
        super().__init__()
        self.input = nn.Linear(dim, width)
        self.action = nn.Linear(2, width)
        self.spatial = nn.Parameter(torch.randn(patches, width) * .02)
        self.blocks = temporal_encoder(width, layers=2)
        self.output = nn.Linear(width, dim)
        self.register_buffer('delta_scale', torch.ones(patches, dim)
                             if delta_scale is None else delta_scale)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state, action):
        tokens = self.input(state) + self.spatial + self.action(action)[:, None]
        return state + self.output(self.blocks(tokens)) * self.delta_scale


def unroll(model, initial, actions):
    state = initial
    predictions = []
    for action in actions.unbind(1):
        state = model(state, action)
        predictions.append(state)
    return torch.stack(predictions, 1)


def teacher_forced(model, states, actions):
    b, h = actions.shape[:2]
    if states.shape[1] != h + 1:
        raise ValueError('states[t] -- actions[t] --> states[t+1]')
    return model(states[:, :-1].flatten(0, 1), actions.flatten(0, 1)).reshape(
        b, h, *states.shape[2:])


def window_batch(states, actions, lengths, generator, horizon, batch=32):
    row = torch.randint(len(states), (batch,), device=states.device, generator=generator)
    start = (torch.rand(batch, device=states.device, generator=generator)
             * (lengths[row] - horizon + 1)).long()
    t = start[:, None] + torch.arange(horizon, device=states.device)
    return states[row, start], actions[row[:, None], t], states[row[:, None], t + 1]


def main():
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--readout-result', type=Path, required=True)
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
    prepare(store, args.features, metric)
    patches, dim = store.patches, store.projection.shape[1]
    # Preserve spatial locations, rather than merging the frame to one token.
    trajectories = {}
    with torch.no_grad():
        for i, row in enumerate(store.rows):
            raw = torch.cat([row['history'][-1:], row['future']]).float().cuda()
            projected = (raw @ store.projection.cuda()).flatten(1)
            trajectories[i] = ((projected - metric.mean) / metric.std).reshape(
                -1, patches, dim)
    ids = store.train
    # Branches at different bank horizons use DIFFERENT action sequences; retain both.
    lengths = torch.tensor([store.rows[i]['horizon'] for i in ids], device='cuda')
    states = torch.zeros(len(ids), 49, patches, dim, device='cuda')
    actions = torch.zeros(len(ids), 48, 2, device='cuda')
    for j, i in enumerate(ids):
        h = store.rows[i]['horizon']
        states[j, :h+1] = trajectories[i]
        actions[j, :h] = store.rows[i]['actions'].cuda()
    delta = torch.cat([trajectories[i][1:] - trajectories[i][:-1] for i in ids])
    scale = delta.square().mean(0).sqrt().clamp_min(.01)
    result = {'status': 'LOCAL_TRANSITION_DEVELOPMENT', 'test_read': False,
              'training_targets': 'projected frozen image features only; no query/reward/state targets',
              'seed': 20260927, 'arms': {}, 'fit': {}, 'tiny': {}}

    # Tiny debug fit: eight highest-motion one-step transitions from TRAIN only.
    previous = torch.cat([trajectories[i][:-1] for i in ids])
    following = torch.cat([trajectories[i][1:] for i in ids])
    flat_actions = torch.cat([store.rows[i]['actions'].cuda() for i in ids])
    motion = ((following-previous)/scale).square().mean((1, 2))
    indices = motion.topk(8).indices
    torch.manual_seed(20260927)
    tiny = PatchTransition(patches, dim, delta_scale=scale).cuda()
    opt = torch.optim.AdamW(tiny.parameters(), lr=3e-4, weight_decay=.01)
    baseline = float(motion[indices].mean())
    for step in range(400):
        pred = tiny(previous[indices], flat_actions[indices])
        loss = ((pred-following[indices])/scale).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(tiny.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    tiny.eval()
    with torch.no_grad():
        final = float(((tiny(previous[indices], flat_actions[indices])-following[indices])
                       / scale).square().mean())
    result['tiny'] = {'persistence_loss': baseline, 'final_loss': final,
                      'pass': bool(np.isfinite(final) and final < .5 * baseline)}
    print('tiny fit:', result['tiny'], flush=True)
    del tiny, previous, following, flat_actions, opt
    if not result['tiny']['pass']:
        result['status'] = 'STOP_TINY_TRANSITION_FIT'
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        (args.output/'REPORT.md').write_text('# STOP: tiny transition fit failed\n')
        return

    models = {}
    for name, horizon, blind in [('step1', 1, False), ('step4', 4, False), ('step4_no_action', 4, True)]:
        torch.manual_seed(20260927)
        model = PatchTransition(patches, dim, delta_scale=scale).cuda()
        generator = torch.Generator(device='cuda').manual_seed(20260927)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        started = time.monotonic()
        first = None
        for step in range(1200):
            x, a, target = window_batch(states, actions, lengths, generator, horizon)
            if blind:
                a = torch.zeros_like(a)
            pred = unroll(model, x, a)
            loss = ((pred-target)/scale).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(name)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 0:
                first = float(loss.detach())
                if norm.item() == 0:
                    raise AssertionError('No predictor gradient')
            opt.step()
            if (step+1) % 300 == 0:
                print(f'{name} {step+1}/1200 loss={loss.item():.6f}', flush=True)
        result['fit'][name] = {'first_loss': first, 'last_loss': float(loss.detach()),
                              'seconds': time.monotonic()-started, 'unroll_train': horizon,
                              'parameters': sum(p.numel() for p in model.parameters())}
        models[name] = model.eval()

    def record(name, split, prefix, bank_h, pred, actual, trace_truth, anchors):
        if not torch.isfinite(pred).all():
            raise FloatingPointError('Nonfinite evaluation: '+name)
        with torch.no_grad():
            embedding = metric.embed(pred.flatten(-2))
            trace = torch.exp(-(embedding[:, :, None] - anchors[:, None]).square().mean(-1))
        # Report native banks separately; short-horizon diagnosis uses bank H48 only.
        for h in ([4, 8, 16, 32, 48] if bank_h == 48 else [bank_h]):
            p = aggregates(trace[:, :h]).cpu().numpy()
            y = aggregates(trace_truth[:, :h]).cpu().numpy()
            key = f'bank{bank_h}_h{h}'
            row = {'prefix': prefix, 'key': key,
                   'state_mse': float((pred[:, :h]-actual[:, :h]).square().mean()),
                   'scaled_state_mse': float(((pred[:, :h]-actual[:, :h])/scale).square().mean()),
                   'query_mse': ((p-y)**2).mean(0).tolist(),
                   'events': event_diagnostic(trace[:, :h].cpu().numpy(), trace_truth[:, :h].cpu().numpy()),
                   'default_regret': float(y[:, 6].max()-y[0, 6]), **selection(p[:, 6], y[:, 6])}
            result['arms'].setdefault(name, {}).setdefault(split, {'groups': []})['groups'].append(row)

    with torch.inference_mode():
        for split in ('train', 'val'):
            for (prefix, bank_h), indices in groups_for(store, split).items():
                real = torch.stack([trajectories[i] for i in indices])
                act = torch.stack([store.rows[i]['actions'] for i in indices]).cuda()
                truth = torch.stack([store.rows[i]['trace'] for i in indices]).cuda()
                anchors = torch.stack([store.rows[i]['metric_anchors'] for i in indices]).cuda()
                variants = {'observed': real[:, 1:],
                            'persistence_open': real[:, :1].expand(-1, bank_h, -1, -1),
                            'persistence_tf': real[:, :-1]}
                for name, model in models.items():
                    a = torch.zeros_like(act) if name.endswith('no_action') else act
                    variants[name+'_open'] = unroll(model, real[:, 0], a)
                    variants[name+'_tf'] = teacher_forced(model, real, a)
                # Matched frozen model with actions reassigned WITHIN prefix, not time reversed.
                variants['step4_shuffled_open'] = unroll(models['step4'], real[:, 0], act.roll(1, 0))
                for name, pred in variants.items():
                    record(name, split, prefix, bank_h, pred, real[:, 1:], truth, anchors)
            print('evaluation complete:', split, flush=True)
    for arm in result['arms'].values():
        for split in arm.values():
            summary = {}
            for key in sorted({r['key'] for r in split['groups']}):
                rows = [r for r in split['groups'] if r['key'] == key]
                summary[key] = {k: float(np.mean([r[k] for r in rows])) for k in
                                ('state_mse', 'scaled_state_mse', 'regret_first', 'regret_uniform_ties', 'default_regret')}
                summary[key]['query_mse'] = np.mean([r['query_mse'] for r in rows], axis=0).tolist()
                summary[key]['events'] = {k: sum(r['events'][k] for r in rows) for k in rows[0]['events']}
                summary[key]['prefixes'] = len(rows)
            split['summary'] = summary
    def error(arm, key):
        return result['arms'][arm]['val']['summary'][key]['state_mse']
    reference = json.loads(args.readout_result.read_text())['arms']['observed_kernel']['val']['summary']
    for h in (32, 48, 64):
        current = result['arms']['observed']['val']['summary'][f'bank{h}_h{h}']
        if (np.max(np.abs(np.asarray(current['query_mse'])-reference[str(h)]['mse'])) > 1e-4
                or abs(current['regret_first']-reference[str(h)]['regret_first']) > 1e-4):
            raise AssertionError('Observed-future reference did not reproduce 53700')
    result['observed_reference_reproduced'] = True
    def improvement(model, reference, key):
        return 1-error(model, key)/max(error(reference, key), 1e-12)
    result['development_gate'] = {
        'one_step_vs_persistence_skill': improvement('step4_tf', 'persistence_tf', 'bank48_h48'),
        'open_h8_vs_persistence_skill': improvement('step4_open', 'persistence_open', 'bank48_h8'),
        'open_h8_vs_no_action_skill': improvement('step4_open', 'step4_no_action_open', 'bank48_h8')}
    result['development_gate']['pass'] = all(v >= .1 for v in result['development_gate'].values())
    result['limitations'] = ['one seed, 24 train/12 reused validation prefixes',
        'teacher forcing sees real intermediate futures; not a deployable oracle',
        'projected 4x4 DINOv3 grid, not full DINO-WM reproduction',
        'one-step/four-step training have different compute; diagnostic, not matched method comparison',
        'gate measures short dynamics only, not query ranking or physical success',
        'only 3 shared layouts, no memory/policy/MPC or new query generalization test']
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    torch.save({'models': {k: v.cpu().state_dict() for k,v in models.items()},
                'projection': store.projection, 'metric': checkpoint['metric']}, args.output/'checkpoints.pt')
    lines = ['# Local patch transition diagnostic', '',
             'TF uses real future history; OPEN uses only initial observation + proposed actions.',
             'Gate is developmental short-dynamics feasibility, NOT control or abstraction success.', '',
             '| Arm | Bank / H | Val state MSE | Query A-before-B MSE | Regret first / uniform | Missed events |',
             '|---|---|---:|---:|---:|---:|']
    for name, arm in result['arms'].items():
        for key, m in arm['val']['summary'].items():
            e = m['events']
            lines.append(f"| {name} | {key} | {m['state_mse']:.6f} | {m['query_mse'][6]:.6f} | "
                         f"{m['regret_first']:.6f} / {m['regret_uniform_ties']:.6f} | "
                         f"{e['missed_below_0_1']}/{e['positive_events']} |")
    lines += ['', 'Development gate:', json.dumps(result['development_gate'], indent=2)]
    report = '\n'.join(lines)+'\n'
    (args.output/'REPORT.md').write_text(report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
