"""Recover a timed-out unified h stage without retraining its WM or BC skill.

The old checkpoints do not contain Adam/RNG state. This explicitly records the
optimizer reset and regenerated imagined pool; it does not claim bitwise resume.
All model/data work runs only in a batch allocation.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--family')
ap.add_argument('--parent-job')
ap.add_argument('--aggregate', action='store_true')
a = ap.parse_args()
if 'SLURM_JOB_ID' not in os.environ:
    raise RuntimeError('Must run through sbatch')
sys.path[:0] = [str(a.root / 'source'), str(a.root)]
from stage_utils import load_protocol, record_run, run_script, cli_options, sha_file
spec = load_protocol(a.root)
if a.aggregate:
    import subprocess
    result = subprocess.run([sys.executable, str(a.root/'aggregate.py'), '--root', str(a.root)])
    registry = json.loads((a.root/'job_registry.json').read_text())
    note = ('Puzzle h training resumed after scheduler timeout. Adam/RNG reset and imagined pool '
            'regenerated because original checkpoints lack optimizer/RNG state. Same WM/BC weights '
            'and 150k effective h steps; additional recovery compute and lost trailing updates disclosed. '
            'This is not a bitwise-uninterrupted matched training run.')
    path = a.root/'results.json'
    if path.exists():
        report = json.loads(path.read_text())
        report['recovery'] = registry.get('recovery')
        report['limits'] += ' ' + note
        path.write_text(json.dumps(report, indent=2) + '\n')
    path = a.root/'RESULTS.md'
    if path.exists():
        with path.open('a') as stream:
            stream.write('\n' + note + '\n')
    raise SystemExit(result.returncode)
if not a.family or not a.parent_job:
    ap.error('--family and --parent-job required for training recovery')
parent = a.root / 'runs' / a.family / ('train_' + a.parent_job)
out = a.root / 'runs' / a.family / ('train_' + os.environ['SLURM_JOB_ID'])
record_run(a.root, out, dict(stage='h_recovery', family=a.family, parent_job=a.parent_job))
if (parent / 'training_complete.json').exists():
    shutil.copy2(parent / 'training_complete.json', out / 'training_complete.json')
    print('PARENT_ALREADY_COMPLETE_NO_TRAINING', flush=True)
    raise SystemExit(0)

import numpy as np
import torch
from u_wm import Model, Scale, make_h, h_input
torch.set_num_threads(2)
torch.manual_seed(0)
rng = np.random.default_rng(0)
prep = a.root / 'prep' / a.family
events = {k: dict(np.load(prep / 'events' / f'events_{k}.npz')) for k in ('train', 'val')}
partials = sorted((parent / 'wm').glob('h_step_*.pt'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
if not partials:
    raise RuntimeError('No h checkpoint: refusing to restart training silently')
checkpoint = partials[-1]
ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
start = int(ck['h_step']); target = int(spec['wm']['h_steps'])
assert ck['training_stage'] == 'h_partial' and 0 < start < target and start % 1000 == 0
dev = 'cuda'; M = Model(ck, dev); sc = M.sc
h = make_h(ck['K'], ck['h_width'], ck['h_absdiff']).to(dev)
h.load_state_dict(ck['h']); h_tgt = M.h
states = M.canonical(np.concatenate([events['train']['before'], events['train']['after']]).astype(np.float32))
recovery = dict(parent_job=a.parent_job, checkpoint=str(checkpoint), checkpoint_sha256=sha_file(checkpoint),
    checkpoint_step=start, final_step=target, optimizer_reset=True, rng_reset=True,
    imagined_pool_regenerated=True, source_sha256=sha_file(Path(__file__)))
(out / 'recovery.json').write_text(json.dumps(recovery, indent=2) + '\n')
print('RECOVERY', json.dumps(recovery), flush=True)
t0 = time.time()
pS = states[rng.integers(0, len(states), spec['wm']['h_walk_pool'])].copy()
pS[:, :, 5] = pS[:, :, 5] > .5
pG = pS.copy(); kk = rng.integers(0, spec['wm']['h_walk_max'] + 1, len(pS))
for w in range(spec['wm']['h_walk_max']):
    rows, evs = [], []
    for b in np.nonzero(kk > w)[0]:
        c = M.candidates(pG[b], pG[b])
        if c:
            rows.append(b); evs.append(c[rng.integers(len(c))])
    for c0 in range(0, len(rows), 8192):
        pG[rows[c0:c0+8192]] = M.step_batch(pG[rows[c0:c0+8192]], evs[c0:c0+8192])
ho = torch.optim.AdamW(h.parameters(), lr=1e-4, weight_decay=1e-5)
(out / 'wm').mkdir()
for step in range(start, target):
    i = rng.integers(0, len(states), 128); j = rng.integers(0, len(states), 128)
    S, G = states[i], states[j]
    w = rng.integers(0, len(pS), 64)
    S = np.concatenate([S[:64], pS[w]]); G = np.concatenate([G[:64], pG[w]])
    evs_all, owner = [], []
    for b in range(len(S)):
        evs = M.candidates(S[b], G[b]); evs_all += evs; owner += [b] * len(evs)
    y = np.zeros(len(S), np.float32)
    if evs_all:
        owner = np.asarray(owner)
        succ = M.step_batch(S[owner], evs_all); done = M.at_goal(succ, G[owner])
        with torch.no_grad():
            sn = torch.as_tensor(sc.norm(succ), device=dev).float().reshape(len(succ), -1)
            gn = torch.as_tensor(sc.norm(G[owner]), device=dev).float().reshape(len(succ), -1)
            hv = h_tgt(h_input(torch, sn, gn, ck['h_absdiff'])).squeeze(-1).cpu().numpy()
        y = np.full(len(S), 30., np.float32)
        np.minimum.at(y, owner, 1 + np.where(done, 0., hv))
    y[M.at_goal(S, G)] = 0.
    sn = torch.as_tensor(sc.norm(S), device=dev).float().reshape(len(S), -1)
    gn = torch.as_tensor(sc.norm(G), device=dev).float().reshape(len(S), -1)
    loss = ((h(h_input(torch, sn, gn, ck['h_absdiff'])).squeeze(-1) - torch.as_tensor(y, device=dev)) ** 2).mean()
    assert torch.isfinite(loss), 'Non-finite recovery loss'
    ho.zero_grad(set_to_none=True); loss.backward(); ho.step()
    if step % 1000 == 999:
        h_tgt.load_state_dict(h.state_dict())
    if (step + 1) % 5000 == 0:
        print(dict(h_step=step+1, loss=float(loss), minutes=(time.time()-t0)/60), flush=True)
        torch.save({**ck, 'h':h.state_dict(), 'h_step':step+1, 'training_stage':'h_partial',
                    'recovery':recovery}, out / 'wm' / f'h_step_{step+1}.pt')
ck.update(h=h.state_dict(), h_step=target, training_stage='complete', recovery=recovery)
torch.save(ck, out / 'wm' / 'u_model.pt')
for name in ('wm_eval.json',):
    shutil.copy2(parent / 'wm' / name, out / 'wm' / name)
run_script(a.root, 'train_event_support.py', cli_options(dict(model=out/'wm/u_model.pt',
    events=prep/'events', out=out/'support', **spec['support'])), out/'support.log')
shutil.copytree(parent / 'skill_final', out / 'skill_final')
wm_ck = torch.load(out/'support/u_model.pt', map_location='cpu', weights_only=False)
skill_ck = torch.load(out/'skill_final/u_skill.pt', map_location='cpu', weights_only=False)
assert wm_ck['h_step'] == target and wm_ck['event_support']['steps'] == spec['support']['steps']
assert skill_ck['include_side_effects'] and skill_ck['release_clipped_at_episode_boundary']
complete = dict(family=a.family, K=ck['K'], h_steps=target, support_steps=spec['support']['steps'],
    skill_initial_steps=spec['skill']['steps'], skill_continued_steps=skill_ck['training_steps'],
    model=str(out/'support/u_model.pt'), skill=str(out/'skill_final/u_skill.pt'), recovery=recovery)
(out/'training_complete.json').write_text(json.dumps(complete, indent=2) + '\n')
print('TRAINING_COMPLETE', json.dumps(complete), flush=True)
