"""Audit event labels against public observations and separate WM/support/quantization errors."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--parent', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert os.environ.get('SLURM_JOB_ID'), 'Use sbatch for data/model work'
sys.path.insert(0, str(a.parent / 'source'))
import numpy as np
import torch
from s_entities import state_entities
from u_wm import Model, event_input

torch.set_num_threads(4)
a.out.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, a.out / 'source.py')
registry = json.loads((a.parent / 'job_registry.json').read_text())
report = {'job_id': os.environ['SLURM_JOB_ID'], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'families': {}}
for family in ('puzzle', 'cube', 'scene'):
    prep = a.parent / 'prep' / family
    complete = json.loads((a.parent / 'runs' / family / ('train_' + registry['training'][family]) / 'training_complete.json').read_text())
    ck = torch.load(complete['model'], map_location='cpu', weights_only=False)
    M = Model(ck, 'cpu')
    layout = json.loads((prep / 'front' / 'layout.json').read_text())
    dataset = {'puzzle': 'puzzle-4x5-play-v0', 'cube': 'cube-triple-play-v0', 'scene': 'scene-play-v0'}[family]
    fr = {'labels': {}, 'predictions': {}}
    for split in ('train', 'val'):
        ev = dict(np.load(prep / 'events' / f'events_{split}.npz'))
        obs = np.load(prep / 'cache' / dataset / f'{split}_observations.npy', mmap_mode='r')
        terminals = np.load(prep / 'cache' / dataset / f'{split}_terminals.npy', mmap_mode='r')
        starts = np.r_[0, np.flatnonzero(terminals) + 1]
        ends = np.r_[np.flatnonzero(terminals), len(obs)-1]
        before_idx = np.maximum(ev['t_start'] - 1, starts[ev['episode']])
        after_idx = np.minimum(ev['t'] + 1, ends[ev['episode']])
        real_before = state_entities(obs[before_idx], layout)[0]
        real_after = state_entities(obs[after_idx], layout)[0]
        if family == 'puzzle':
            mismatch_b = (real_before[..., 2:5] != ev['before'][..., 2:5]).any((1, 2))
            mismatch_a = (real_after[..., 2:5] != ev['after'][..., 2:5]).any((1, 2))
            masks = (real_before[..., 2] != real_after[..., 2])
            group = {}
            for e in range(M.K):
                ids = np.flatnonzero(ev['e'] == e)
                counts = Counter(tuple(np.flatnonzero(masks[i])) for i in ids)
                modal, count = counts.most_common(1)[0]
                group[str(e)] = {'events': len(ids), 'modal_changed_objects': modal, 'modal_fraction': count/len(ids)}
            fr['labels'][split] = {'events': len(ev['e']), 'before_bit_mismatch_events': int(mismatch_b.sum()), 'after_bit_mismatch_events': int(mismatch_a.sum()), 'by_actor': group}
        else:
            fr['labels'][split] = {'events': len(ev['e']), 'before_max_abs_diff': float(np.abs(real_before-ev['before']).max()), 'after_max_abs_diff': float(np.abs(real_after-ev['after']).max())}
        # All VAL; a deterministic balanced-by-actor TRAIN sample to bound CPU cost.
        ids = np.arange(len(ev['e'])) if split == 'val' else np.concatenate([np.flatnonzero(ev['e'] == e)[::max(1, int((ev['e'] == e).sum())//256)][:256] for e in range(M.K)])
        pred_raw, pred_final = [], []
        for start in range(0, len(ids), 256):
            ii = ids[start:start+256]
            S, e, x = ev['before'][ii], ev['e'][ii], ev['target'][ii]
            xin = event_input(S, e, x, M.sc.tol_pos) if M.pos_only else x
            with torch.no_grad():
                raw = M.sc.denorm(M.wm(torch.as_tensor(M.sc.norm(S)).float(), torch.as_tensor(e).long(), torch.as_tensor(M.sc.norm(xin)).float()).numpy())
            pred_raw.append(raw)
            pred_final.append(M.step_batch(S, list(zip(e, x))))
        raw, final = np.concatenate(pred_raw), np.concatenate(pred_final)
        y, b = ev['after'][ids], ev['before'][ids]
        changed = (np.linalg.norm(y[..., :2]-b[..., :2], axis=-1)>M.sc.tol_pos) | (np.abs(y[..., 2:5]-b[..., 2:5]).max(-1)>M.app_tol)
        stat = {'evaluated_events': len(ids), 'changed_fraction': float(changed.mean())}
        for name, pred in [('raw', raw), ('canonical_only', M.canonical(raw)), ('support_and_canonical', final)]:
            bad = (np.linalg.norm(pred[..., :2]-y[..., :2], axis=-1)>M.sc.tol_pos) | (np.abs(pred[..., 2:5]-y[..., 2:5]).max(-1)>M.app_tol)
            stat[name] = {'exact_event_fraction': float((~bad.any(-1)).mean()), 'changed_error_fraction': float(bad[changed].mean()), 'unchanged_error_fraction': float(bad[~changed].mean())}
            if family == 'puzzle':
                bitbad = ((pred[..., 2:5]>.5)!=(y[..., 2:5]>.5)).any(-1)
                stat[name]['binary_exact_event_fraction'] = float((~bitbad.any(-1)).mean())
        fr['predictions'][split] = stat
    if family == 'puzzle':
        samples = []
        for seed, job in registry['evaluations'][family].items():
            path = a.parent/'runs'/family/f'eval_seed{seed}_{job}'/'loop/u_closed_loop.json'
            episodes = json.loads(path.read_text())['episodes']
            for task in range(1, 6):
                episode = next(v for v in episodes if v['task'] == task)
                for event in episode['events'][:6]:
                    S = np.asarray(event['read_start'], np.float32)
                    e = event['e']; x = np.asarray(event['x'], np.float32)
                    xin = event_input(S[None], [e], x[None], M.sc.tol_pos) if M.pos_only else x[None]
                    with torch.no_grad():
                        raw = M.sc.denorm(M.wm(torch.as_tensor(M.sc.norm(S[None])).float(), torch.tensor([e]), torch.as_tensor(M.sc.norm(xin)).float()).numpy())[0]
                    final = M.step_batch(S[None], [(e,x)])[0]
                    y = np.asarray(event['read_end'], np.float32)
                    wrong = np.flatnonzero((raw[:,2]>.5)!=(y[:,2]>.5))
                    if len(wrong):
                        samples.append({'seed': seed, 'task': task, 'actor': e, 'time': event['t'], 'raw_wrong_objects': wrong.tolist(), 'raw_values_wrong': raw[wrong,2:5].tolist(), 'true_values_wrong': y[wrong,2:5].tolist(), 'full_wrong_objects': np.flatnonzero((final[:,2]>.5)!=(y[:,2]>.5)).tolist(), 'current_bits': S[:,2].tolist(), 'event_target': x.tolist()})
        fr['closed_loop_examples'] = samples
    report['families'][family] = fr
    print('FAMILY', family, json.dumps(fr, default=lambda x: int(x))[:16000], flush=True)
(a.out/'audit.json').write_text(json.dumps(report, indent=2, default=lambda x: int(x))+'\n')
print('OUTPUT', a.out/'audit.json', flush=True)
