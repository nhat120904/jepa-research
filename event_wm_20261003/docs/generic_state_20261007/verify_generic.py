"""Integrated checks after generic preparation (compute node): data alignment, model interfaces, env reset/goal reading.
Mirrors unified_rerun verify_prepared.py; the proprio-independence check uses the DISCOVERED agent dimensions (no
layout knowledge), and entities never acted on are reported instead of asserted (the front end may find parts the
agent never targets)."""
import argparse
import json
from pathlib import Path

import numpy as np

from stage_utils import load_protocol

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--family', required=True)
a = ap.parse_args()
spec = load_protocol(a.root)
data = spec['families'][a.family]
out = a.root / 'prep' / a.family
cache = out / 'cache' / data['dataset']
from g_entities import GenericMap
from skill_segments import segment_frames

L = json.loads((out / 'front/layout.json').read_text())
K = L['K']
G = GenericMap(L)
checks = {}
for split, expected in [('train', spec['data']['train_episodes']), ('val', spec['data']['val_episodes'])]:
    obs = np.load(cache / f'{split}_observations.npy', mmap_mode='r')
    terminals = np.load(cache / f'{split}_terminals.npy')
    ends = np.flatnonzero(terminals)
    ev = np.load(out / 'events' / f'events_{split}.npz')
    assert len(ends) == expected
    assert ev['before'].shape[1:] == (K, 6)
    assert np.isfinite(ev['before']).all() and np.isfinite(ev['after']).all()
    first = np.r_[0, ends[:-1] + 1][ev['episode']]
    last = ends[ev['episode']]
    assert np.all((ev['seg_start'] >= first) & (ev['seg_start'] <= ev['t_start']) & (ev['t_start'] <= ev['t']) & (ev['t'] <= last))
    for i in np.flatnonzero(ev['t'] + 10 > last):
        frames = list(segment_frames(ev['seg_start'][i], ev['t'][i], 10, last[i]))
        assert frames[0] >= first[i] and frames[-1] == last[i]
    counts = np.bincount(ev['e'], minlength=K)
    checks[split] = dict(frames=len(obs), episodes=len(ends), obs_dim=obs.shape[1], events=len(ev['e']),
                         multi_object_events=int(ev['knock'].sum()), acted_counts=counts.tolist(),
                         never_acted=np.flatnonzero(counts == 0).tolist())
import torch

torch.set_num_threads(2)
from event_support import features, make_support
from u_wm import h_input, make_h, make_wm

S = torch.zeros((2, K, 6)); E = torch.arange(2) % K; X = torch.zeros((2, 6))
with torch.no_grad():
    assert make_wm(K)(S, E, X).shape == (2, K, 6)
    flat = S.flatten(1)
    assert make_h(K, 2048, True)(h_input(torch, flat, flat, True)).shape == (2, 1)
    assert make_support(K)(features(torch, S, E, X)).shape == (2, 1)
import gymnasium
import ogbench  # noqa: F401

env = gymnasium.make(data['env'])
assert env.spec.max_episode_steps == data['horizon']
task_checks = []
for task in range(1, 6):
    ob, info = env.reset(seed=task * 100, options=dict(task_id=task, render_goal=False))
    S_, eff = G(ob[None]); G_, _ = G(info['goal'][None])
    assert S_.shape == G_.shape == (1, K, 6) and np.isfinite(S_).all() and np.isfinite(G_).all()
    modified = ob.copy(); modified[L['agent_dims']] += 10
    np.testing.assert_array_equal(G(modified[None])[0], S_)
    task_checks.append(dict(task=task, start=np.round(S_[0], 2).tolist(), goal=np.round(G_[0], 2).tolist()))
env.close()
checks.update(K=K, model_interfaces='passed', entity_state_independent_of_discovered_agent_dims='passed',
              native_horizon=data['horizon'], task_checks=task_checks)
(out / 'preparation_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
print('PREPARED', a.family, json.dumps({k: v for k, v in checks.items() if k != 'task_checks'}), flush=True)
