"""Round 3: parallel FSQ WM, task losses through a frozen reader, and endpoint WM.

All compute runs in Slurm. No task-score gates between training and control.
"""
import argparse
import gc
import json
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import cta_closed_loop as closed
import cta_train as ct
from cta_reader_refine import write_json
from ti_wm.contract import candidate_seed, require_compute, select_candidate
from ti_wm.cta import Scorer, action_features, goal_scores
from ti_wm.cta_geometry import world_vertices
from ti_wm.cta_parallel import ParallelFSQWM, EndpointWM, weighted_rank, score_consistency, normalized_score
from ti_wm.cta_runtime import Planner
from ti_wm.gates import paired_diff, mcnemar_exact
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer, done, reset_branch, run_prefix
from ti_wm.wb import Logger

ARMS = ('P0', 'CTA3', 'NLL8', 'FRAME8', 'DIRECT3', 'CTA8', 'CTA8E', 'DIRECT8')
NEW_TIERS = {'CTA3': 'task', 'NLL8': 'nll', 'FRAME8': 'frame', 'DIRECT3': 'direct_new'}
STOP_REQUESTED = False


def request_checkpoint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def build(cfg, device):
    return {'nll': ParallelFSQWM(m=cfg['m']).to(device),
            'task': ParallelFSQWM(m=cfg['m']).to(device),
            'frame': EndpointWM().to(device),
            'direct': Scorer('action', layers=cfg['direct_layers']).to(device)}


def predict_score(name, nets, frozen, ctx, actions, goals):
    if name in ('nll', 'task'):
        expected, _ = nets[name](ctx, actions)
        return goal_scores(frozen['reader'], ctx, expected, goals)
    if name == 'frame':
        return goal_scores(frozen['full'], ctx, nets['frame'](ctx, actions), goals)
    return goal_scores(nets['direct'], ctx, actions, goals)


def group_masks(features, collection):
    """Translation strata plus corresponding-vertex strata that also include rotation."""
    arrays = []
    for name in ('shard_2000_2049.npz', 'shard_2050_2099.npz'):
        with np.load(collection / name) as raw:
            arrays.append({key: raw[key] for key in ('root', 'decision', 'phys8')})
    raw = {key: np.concatenate([row[key] for row in arrays]) for key in arrays[0]}
    with np.load(features / 'dev/meta.npz') as meta:
        for key in ('root', 'decision'):
            if not np.array_equal(meta[key], raw[key]):
                raise ValueError(f'Group metadata mismatch: {key}')
    pose = raw['phys8'][..., 4:7]
    centre = np.linalg.norm(pose[:, :, None, :2] - pose[:, None, :, :2], axis=-1)
    verts = world_vertices(pose)
    vertex = np.linalg.norm(verts[:, :, None] - verts[:, None, :], axis=-1).max(-1)
    cutoff = 512. / 96.
    masks = {'all': np.ones(len(pose), bool), 'center_ge_1px': centre.max((1, 2)) >= cutoff,
             'vertex_ge_1px': vertex.max((1, 2)) >= cutoff}
    masks['center_lt_1px'] = ~masks['center_ge_1px']
    masks['vertex_lt_1px'] = ~masks['vertex_ge_1px']
    return masks


@torch.inference_mode()
def evaluate_offline(nets, frozen, dev, goals, device, amp, parent, features, collection, run, normalizers):
    for net in nets.values():
        net.eval()
    with np.load(parent / 'dev_scores.npz') as saved:
        if not np.array_equal(saved['root'], dev.root) or not np.array_equal(saved['cov8'], dev.cov8):
            raise ValueError('Parent dev scores do not match geometry view')
        scores = {key: saved[key].copy() for key in ct.TIERS}
    scores.update({name: np.zeros((dev.n, ct.K), np.float32) for name in NEW_TIERS.values()})
    prediction = {'nll/nll_nats_per_token': 0., 'task/nll_nats_per_token': 0.,
                  'frame/image_normalized_mse': 0., 'frame/proprio_normalized_mse': 0.}
    for start in range(0, dev.n, 4):
        i = torch.arange(start, min(start + 4, dev.n))
        ctx, actions = dev.context(i, device), dev.actions(i, device)
        with amp():
            future = dev.future(i, device)
            source = frozen['enc'](ctx, future)
            for name in ('nll', 'task', 'frame', 'direct'):
                if name in ('nll', 'task'):
                    expected, logits = nets[name](ctx, actions)
                    prediction[f'{name}/nll_nats_per_token'] += float(nets[name].nll(logits, source)) * 3 * len(i)
                    value = goal_scores(frozen['reader'], ctx, expected, goals)
                elif name == 'frame':
                    predicted = nets[name](ctx, actions)
                    prediction['frame/image_normalized_mse'] += float(F.mse_loss(
                        predicted['end'], future['end'].float())) / normalizers['image'] * len(i)
                    prediction['frame/proprio_normalized_mse'] += float(F.mse_loss(
                        predicted['prop'], future['prop'].float())) / normalizers['prop'] * len(i)
                    value = goal_scores(frozen['full'], ctx, predicted, goals)
                else:
                    value = goal_scores(nets['direct'], ctx, actions, goals)
                key = 'direct_new' if name == 'direct' else name
                scores[key][start:start + len(i)] = value.view(len(i), ct.K).cpu().numpy()
    masks = group_masks(features, collection)
    report = {'prediction': {key: value / dev.n for key, value in prediction.items()},
              'groups': {}, 'rates': {'source_nominal_bits': 128, 'predicted_scalars': 48,
              'predicted_fp16_bits': 768, 'predicted_runtime_dtype': 'float32',
              'predicted_runtime_bits': 1536, 'endpoint_scalars_including_proprio': 256 * 128 + 4}, 'ladder': {}}
    denominator = dev.cov8.max(1) - dev.cov8[:, 0]
    for name, mask in masks.items():
        report['groups'][name] = {'banks': int(mask.sum()),
            'oracle_gain_share': float(denominator[mask].sum() / denominator.sum()) if denominator.sum() else None}
        report['ladder'][name] = {}
        if not mask.any():
            continue
        for tier, values in scores.items():
            metrics = ct.ranking_metrics(values[mask], dev.cov8[mask], dev.root[mask], ci=True)
            metrics.pop('chosen')
            report['ladder'][name][tier] = metrics
    with np.load(features / 'dev/meta.npz') as meta:
        np.savez(run / 'dev_scores.npz', root=dev.root, decision=meta['decision'], cov8=dev.cov8,
                 native_cov8=meta['native_cov8'], **scores)
    write_json(run / 'offline.json', report)
    return report


class R3Planner(Planner):
    def __init__(self, parent, visual, goal_frames, nets):
        super().__init__(parent / 'cta.pt', visual, goal_frames)
        self.nets = nets
        for net in nets.values():
            net.eval()

    @torch.inference_mode()
    def scores(self, arm, state, chunks=None, branches=None, segs=None, seed=0):
        if arm not in NEW_TIERS:
            return super().scores(arm, state, chunks, branches, segs, seed)
        ctx, agent = self.context(state, len(chunks))
        actions = action_features(torch.as_tensor(np.asarray(chunks), device=self.device), agent)
        name = {'CTA3': 'task', 'NLL8': 'nll', 'FRAME8': 'frame', 'DIRECT3': 'direct'}[arm]
        with self.amp():
            return predict_score(name, self.nets, self.models, ctx, actions, self.goals).float().cpu().tolist()


def train(a):
    report = {'status': 'RUNNING', 'job': os.environ['SLURM_JOB_ID'], 'stage': 'load'}
    path, log = a.run / 'train_report.json', None
    started = time.perf_counter()
    try:
        parent = torch.load(a.parent / 'cta.pt', map_location='cpu')
        if parent['config'].get('target') != 'negative_mean_vertex_distance_divided_by_512':
            raise ValueError('Expected geometry checkpoint')
        cfg = {**parent['config'], 'parent': str(a.parent.resolve()), 'features': str(a.features.resolve()),
               'parent_sha256': closed.sha256(a.parent / 'cta.pt'), 'r3_steps': 6000, 'r3_lr': 1e-4,
               'r3_banks': 32, 'r3_microbanks': 8, 'rank_scale': .01, 'rank_margin': .001,
               'dev_eval_every': 1000, 'checkpoint_every': 250,
               'consistency_weight': 1., 'rank_weight': 1., 'sampling': 'all_train_banks_uniform',
               'seed': 0, 'primary': 'CTA3-P0 normalized native episode score excluding reset'}
        report['config'] = cfg
        write_json(path, report)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        rng = np.random.default_rng(0)
        device = torch.device('cuda')
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
        frozen = ct.build(cfg, device)
        for name, model in frozen.items():
            model.load_state_dict(parent['state'][name], strict=True)
            model.eval().requires_grad_(False)
        nets = build(cfg, device)
        report['warm_started_keys'] = nets['nll'].warm_start(parent['state']['wm'])
        nets['task'].load_state_dict(nets['nll'].state_dict(), strict=True)
        nets['frame'].warm_start(parent['state']['wm'])
        nets['direct'].load_state_dict(parent['state']['direct'], strict=True)
        report['parameters'] = {name: sum(p.numel() for p in net.parameters()) for name, net in nets.items()}
        opts = {name: torch.optim.AdamW(net.parameters(), lr=cfg['r3_lr'], weight_decay=cfg['wd'])
                for name, net in nets.items()}
        split, dev = ct.Split(a.features / 'train'), ct.Split(a.features / 'dev')
        if set(split.root) != set(range(30250, 31050)) or set(dev.root) != set(range(2000, 2100)):
            raise ValueError('Unexpected data split')
        goals = torch.from_numpy(np.load(a.features / 'goals.npy')).to(device)
        log = Logger(a.run, 'parallel_wm', cfg, name=f"r3_{report['job']}")
        # Fit score/physical normalizers on TRAIN only, with independent RNG.
        norms = {'code_score': [], 'full_score': [], 'prop_delta': []}
        cal_rng = np.random.default_rng(381)
        for _ in range(32):
            i = torch.as_tensor(cal_rng.integers(0, split.n, cfg['r3_microbanks']))
            ctx, fut = split.context(i, device), split.future(i, device)
            goal = ct.sample_goals(goals, cal_rng, len(i), device)
            with torch.no_grad(), amp():
                src = frozen['enc'](ctx, fut)
                code_score = frozen['reader'](ctx, src, goal).float().view(-1, ct.K)
                full_score = frozen['full'](ctx, fut, goal).float().view(-1, ct.K)
            norms['code_score'].append(float((code_score - code_score.mean(1, keepdim=True)).square().mean()))
            norms['full_score'].append(float((full_score - full_score.mean(1, keepdim=True)).square().mean()))
            norms['prop_delta'].append(float((fut['prop'] - ctx['prop']).square().mean()))
        scales = {'code': max(float(np.sqrt(np.mean(norms['code_score']))), 1e-3),
                  'full': max(float(np.sqrt(np.mean(norms['full_score']))), 1e-3),
                  'prop': max(float(np.mean(norms['prop_delta'])), 1e-6)}
        report.update(stage='train', train_normalizers=scales, dev_history=[])
        first_update = 0
        resume_path = getattr(a, 'resume_from', None)
        if resume_path is not None:
            resumed = torch.load(resume_path, map_location='cpu')
            for key in ('r3_steps', 'r3_banks', 'r3_microbanks', 'r3_lr', 'parent_sha256',
                        'rank_scale', 'rank_margin', 'features'):
                if resumed['config'][key] != cfg[key]:
                    raise ValueError(f'Resume configuration mismatch: {key}')
            first_update = int(resumed['updates'])
            if not 0 <= first_update <= cfg['r3_steps']:
                raise ValueError('Invalid resume update')
            for name, net in nets.items():
                net.load_state_dict(resumed['networks'][name], strict=True)
                opts[name].load_state_dict(resumed['optimizers'][name])
            rng.bit_generator.state = resumed['numpy_rng']
            torch.set_rng_state(resumed['torch_rng'])
            torch.cuda.set_rng_state_all(resumed['cuda_rng'])
            scales = resumed['normalizers']
            report.update(resumed_from=str(resume_path), first_update=first_update,
                          train_normalizers=scales, dev_history=resumed.get('dev_history', []))
            del resumed
        write_json(path, report)
        train_seconds = 0.
        normalizers = {'image': parent['norms']['end'], 'prop': scales['prop']}

        def checkpoint(updates):
            tmp = a.run / 'round3.tmp.pt'
            torch.save({'config': cfg, 'updates': updates, 'normalizers': scales,
                        'networks': {name: net.state_dict() for name, net in nets.items()},
                        'optimizers': {name: opt.state_dict() for name, opt in opts.items()},
                        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all(),
                        'dev_history': report['dev_history']}, tmp)
            tmp.replace(a.run / 'round3.pt')

        for step in range(first_update, cfg['r3_steps']):
            tick_update = time.perf_counter()
            all_i = torch.as_tensor(rng.integers(0, split.n, cfg['r3_banks']))
            all_labels = split.cov[all_i].to(device)
            all_goals = ct.sample_goals(goals, rng, len(all_i), device)
            # A single global denominator makes accumulated ranking gradients
            # match the effective 32-bank batch even when microbatches have
            # different numbers of informative pairs.
            pair_count = ((all_labels[:, :, None] - all_labels[:, None, :]) > cfg['rank_margin']).sum()
            accumulation = cfg['r3_banks'] // cfg['r3_microbanks']
            values = {'step': step + 1, 'banks_seen': (step + 1) * cfg['r3_banks']}
            for opt in opts.values():
                opt.zero_grad(set_to_none=True)

            def add_metric(key, value):
                values[key] = values.get(key, 0.) + float(value)

            for offset in range(0, cfg['r3_banks'], cfg['r3_microbanks']):
                i = all_i[offset:offset + cfg['r3_microbanks']]
                ctx, fut, actions = split.context(i, device), split.future(i, device), split.actions(i, device)
                labels = all_labels[offset:offset + len(i)]
                goal = all_goals[offset * ct.K:(offset + len(i)) * ct.K]
                with torch.no_grad(), amp():
                    source = frozen['enc'](ctx, fut)
                    teacher = frozen['reader'](ctx, source, goal).float().view(-1, ct.K)
                    teacher_full = frozen['full'](ctx, fut, goal).float().view(-1, ct.K)
                for name in ('nll', 'task', 'frame', 'direct'):
                    tick_network = time.perf_counter()
                    with amp():
                        if name in ('nll', 'task'):
                            predicted, logits = nets[name](ctx, actions)
                            anchor = nets[name].nll(logits, source)
                            loss = anchor / accumulation
                            add_metric(f'{name}/nll_nats_per_token', anchor.detach() * 3 / accumulation)
                            if name == 'task':
                                scores = frozen['reader'](ctx, predicted, goal).float().view(-1, ct.K)
                                consistency = score_consistency(scores, teacher, scales['code']) / accumulation
                                ranking = weighted_rank(scores, labels, cfg['rank_scale'], cfg['rank_margin'],
                                                        pair_count=pair_count)
                                loss = loss + consistency + ranking
                                add_metric('task/consistency', consistency.detach())
                                add_metric('task/ranking', ranking.detach())
                        elif name == 'frame':
                            predicted = nets[name](ctx, actions)
                            image_loss = F.mse_loss(predicted['end'], fut['end'].float()) / normalizers['image']
                            prop_loss = F.mse_loss(predicted['prop'], fut['prop'].float()) / normalizers['prop']
                            scores = frozen['full'](ctx, predicted, goal).float().view(-1, ct.K)
                            loss = ((image_loss + prop_loss) / 2
                                    + score_consistency(scores, teacher_full, scales['full'])) / accumulation
                            loss = loss + weighted_rank(scores, labels, cfg['rank_scale'], cfg['rank_margin'],
                                                        pair_count=pair_count)
                            add_metric('frame/image', image_loss.detach() / accumulation)
                            add_metric('frame/proprio', prop_loss.detach() / accumulation)
                        else:
                            scores = nets['direct'](ctx, actions, goal).float().view(-1, ct.K)
                            loss = weighted_rank(scores, labels, cfg['rank_scale'], cfg['rank_margin'],
                                                 pair_count=pair_count)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(name)
                    loss.backward()
                    add_metric(f'{name}/loss', loss.detach())
                    torch.cuda.synchronize()
                    add_metric(f'{name}/seconds_per_update', time.perf_counter() - tick_network)
            # Clip and update once per effective batch, not once per microbatch.
            for name in nets:
                grad = torch.nn.utils.clip_grad_norm_(nets[name].parameters(), 1., error_if_nonfinite=True)
                opts[name].step()
                values[f'{name}/grad'] = float(grad)
            torch.cuda.synchronize()
            elapsed_update = time.perf_counter() - tick_update
            train_seconds += elapsed_update
            values.update(seconds_per_update=elapsed_update,
                          mean_seconds_per_update=train_seconds / (step - first_update + 1),
                          estimated_remaining_train_seconds=train_seconds / (step - first_update + 1)
                                                            * (cfg['r3_steps'] - step - 1))
            if step == first_update or (step + 1) % 100 == 0 or step + 1 == cfg['r3_steps']:
                with (a.run / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(values) + '\n')
                log.log(values, step=step + 1)
                print(values, flush=True)
                report.update(last_update=step + 1, timing=values, wall_seconds=time.perf_counter() - started)
                write_json(path, report)
            if (step + 1) % cfg['checkpoint_every'] == 0 or STOP_REQUESTED:
                checkpoint(step + 1)
            if STOP_REQUESTED:
                report.update(status='INTERRUPTED_TIME_LIMIT', stage='checkpointed', last_update=step + 1)
                write_json(path, report)
                raise SystemExit(75)
            if (step + 1) % cfg['dev_eval_every'] == 0 and step + 1 < cfg['r3_steps']:
                eval_started = time.perf_counter()
                report.update(stage=f'dev_eval_{step + 1}')
                write_json(path, report)
                eval_dir = a.run / f'dev_step_{step + 1}'
                eval_dir.mkdir()
                dev_report = evaluate_offline(nets, frozen, dev, goals, device, amp,
                                             a.parent, a.features, a.collection, eval_dir, normalizers)
                entry = {'step': step + 1, 'seconds': time.perf_counter() - eval_started,
                         'prediction': dev_report['prediction'],
                         'retained_gap': {key: val['retained_gap']['ratio']
                                          for key, val in dev_report['ladder']['all'].items()}}
                report['dev_history'].append(entry)
                write_json(a.run / 'dev_curve.json', report['dev_history'])
                log.log({'dev': entry}, step=step + 1)
                print({'dev': entry}, flush=True)
                for net in nets.values():
                    net.train()
                report.update(stage='train')
                write_json(path, report)
        # Also materialize the complete checkpoint after resuming at step 6000.
        checkpoint(cfg['r3_steps'])
        report['frozen_unchanged'] = all(torch.equal(value.detach().cpu(), parent['state'][name][key])
            for name, model in frozen.items() for key, value in model.state_dict().items())
        if not report['frozen_unchanged']:
            raise RuntimeError('Source/reader/reference checkpoint changed')
        report['checkpoint_sha256'] = closed.sha256(a.run / 'round3.pt')
        report.update(stage='offline')
        write_json(path, report)
        del opts, split
        gc.collect()
        torch.cuda.empty_cache()
        eval_started = time.perf_counter()
        dev_report = evaluate_offline(nets, frozen, dev, goals, device, amp,
                                      a.parent, a.features, a.collection, a.run, normalizers)
        entry = {'step': cfg['r3_steps'], 'seconds': time.perf_counter() - eval_started,
                 'prediction': dev_report['prediction'],
                 'retained_gap': {key: val['retained_gap']['ratio']
                                  for key, val in dev_report['ladder']['all'].items()}}
        report['dev_history'].append(entry)
        write_json(a.run / 'dev_curve.json', report['dev_history'])
        log.log({'dev': entry}, step=cfg['r3_steps'])
        print({'dev': entry}, flush=True)
        del frozen, dev, parent
        gc.collect()
        torch.cuda.empty_cache()
        report.update(stage='runtime_consistency')
        write_json(path, report)
        planner = R3Planner(a.parent, VisualScorer('cuda'), np.load(a.smoke / 'goal_frames.npz')['frames'], nets)
        report['preflight'] = closed.preflight(planner, a.collection / 'shard_2000_2049.npz', a.run,
            tiers={**NEW_TIERS, 'CTA8': 'pred', 'CTA8E': 'pred_soft', 'DIRECT8': 'direct'},
            allow_matching_flat=True)
        report.update(status='DONE', stage='done', seconds=time.perf_counter() - started)
        write_json(path, report)
    except Exception:
        report.update(status='FAILED', error=traceback.format_exc())
        write_json(path, report)
        raise
    finally:
        if log is not None:
            log.finish()


def episode(root, arm, runner, cloner, planner):
    state = reset_branch(root)
    state.max_coverage = 0.  # Explicit Round-3 metric amendment: exclude reset.
    decisions, decision = [], 0
    start = time.perf_counter()
    while not done(state):
        bank = runner.bank(state.hist, [candidate_seed(root, decision, k) for k in range(1 if arm == 'P0' else 8)])
        rec = {'d': decision, 't': state.t}
        if arm == 'P0':
            chosen = 0
        else:
            tick = time.perf_counter()
            rec['score'] = planner.scores(arm, state, chunks=bank)
            torch.cuda.synchronize()
            rec['score_seconds'] = time.perf_counter() - tick
            chosen = select_candidate(rec['score'])
        rec['chosen'] = chosen
        decisions.append(rec)
        state = run_prefix(state, bank[chosen], cloner=cloner)
        decision += 1
    result = {'success': state.success, 'score': normalized_score(state.max_coverage, state.env.success_threshold),
              'max_coverage': state.max_coverage, 'steps': state.t, 'decisions': decisions,
              'seconds': time.perf_counter() - start, 'reset_excluded': True}
    state.env.close()
    return result


def evaluate(a):
    report = json.loads((a.train_run / 'train_report.json').read_text())
    if report['status'] != 'DONE' or not report['frozen_unchanged']:
        raise ValueError('Training incomplete')
    if a.first not in range(2100, 2200, 10) or a.count != 10:
        raise ValueError('Unexpected dev roots')
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint = a.train_run / 'round3.pt'
    if closed.sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Checkpoint drift')
    blob = torch.load(checkpoint, map_location='cpu')
    parent = Path(blob['config']['parent'])
    if (closed.sha256(parent / 'cta.pt') != blob['config']['parent_sha256']
            or blob['updates'] != 6000 or blob['config']['r3_banks'] != 32):
        raise ValueError('Parent/updates drift')
    nets = build(blob['config'], 'cuda')
    for name, net in nets.items():
        net.load_state_dict(blob['networks'][name], strict=True)
        net.eval()
    del blob
    smoke = json.loads((a.smoke / 'smoke.json').read_text())
    if smoke['status'] != 'SMOKE_PASS':
        raise ValueError('Invalid clone contract')
    cloner = CLONERS[smoke['clone_method']]
    runner = PolicyRunner(a.prep / 'checkpoint', 'cuda')
    planner = R3Planner(parent, VisualScorer('cuda'), np.load(a.smoke / 'goal_frames.npz')['frames'], nets)
    with (a.run / f'roots_{a.first}_{a.first + a.count - 1}.jsonl').open('x') as stream:
        for root in range(a.first, a.first + a.count):
            rec = {'root': root, 'arms': list(ARMS), 'checkpoint_sha256': report['checkpoint_sha256']}
            for arm in ARMS:
                rec[arm] = episode(root, arm, runner, cloner, planner)
            stream.write(json.dumps(rec) + '\n')
            stream.flush()
            print(root, {arm: (rec[arm]['success'], round(rec[arm]['score'], 3)) for arm in ARMS}, flush=True)


def aggregate(a):
    rows = [json.loads(line) for path in sorted(a.closed_run.glob('shard_*/roots_*.jsonl'))
            for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 100 or {r['root'] for r in rows} != set(range(2100, 2200)):
        raise ValueError('Missing/duplicate/unexpected roots')
    rows.sort(key=lambda row: row['root'])
    if len({r['checkpoint_sha256'] for r in rows}) != 1 or any(r['arms'] != list(ARMS) for r in rows):
        raise ValueError('Inconsistent experiment identity')
    if not all(r[arm]['reset_excluded'] for r in rows for arm in ARMS):
        raise ValueError('Inconsistent score definition')
    score = {arm: np.array([r[arm]['score'] for r in rows]) for arm in ARMS}
    success = {arm: np.array([r[arm]['success'] for r in rows], float) for arm in ARMS}
    out = {'status': 'DONE', 'primary': 'CTA3-P0 normalized score', 'n': len(rows),
           'scope': 'reused development roots, single training seed, not sealed confirmation',
           'arms': {}, 'contrasts': {}}
    for arm in ARMS:
        decisions = [d for r in rows for d in r[arm]['decisions']]
        times = [d['score_seconds'] for d in decisions if 'score_seconds' in d]
        out['arms'][arm] = {'mean_score': float(score[arm].mean()), 'successes': int(success[arm].sum()),
            'mean_max_coverage': float(np.mean([r[arm]['max_coverage'] for r in rows])),
            'seconds_per_episode': float(np.mean([r[arm]['seconds'] for r in rows])),
            'scorer_seconds_per_decision': float(np.mean(times)) if times else 0.,
            'seconds_per_decision_including_policy_and_environment': sum(r[arm]['seconds'] for r in rows) / len(decisions),
            'override_fraction': float(np.mean([d['chosen'] != 0 for d in decisions]))}
    pairs = [(arm, 'P0') for arm in ARMS if arm != 'P0'] + [('CTA3', arm) for arm in ('NLL8', 'FRAME8', 'DIRECT3', 'CTA8', 'CTA8E')]
    for left, right in pairs:
        out['contrasts'][f'{left}-{right}'] = {'score': paired_diff(score[left], score[right]),
            'success': paired_diff(success[left], success[right]), 'mcnemar': mcnemar_exact(success[left], success[right])}
    write_json(a.run / 'comparison.json', out)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train', 'closed', 'aggregate'), required=True)
    for key in ('run', 'parent', 'features', 'collection', 'smoke', 'prep', 'train-run', 'closed-run'):
        parser.add_argument(f'--{key}', type=Path, required=key == 'run')
    parser.add_argument('--first', type=int)
    parser.add_argument('--count', type=int)
    parser.add_argument('--resume-from', type=Path, help='Resume a complete optimizer/RNG checkpoint into a fresh run')
    args = parser.parse_args()
    require_compute()
    signal.signal(signal.SIGUSR1, request_checkpoint)
    args.run.mkdir(parents=True, exist_ok=True)
    {'train': train, 'closed': evaluate, 'aggregate': aggregate}[args.mode](args)
