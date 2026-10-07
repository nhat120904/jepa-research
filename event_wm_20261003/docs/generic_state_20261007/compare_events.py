"""PRIVILEGED diagnostics (report only, never read by the method): event quality of the generic front end vs the layout
adapter, both extracted with the same current u_events.py.

True acted object of an event = the object whose documented-layout block is most active over [t_start - 1, t + 1]:
moved objects (block of 9: xyz) by displacement / 0.4 (one cube width in observation units); buttons (block of 4:
one-hot, joint pos, joint vel) by joint excursion / its 99th-percentile excursion; drawer / window by joint displacement
/ its range. Entities are mapped to objects through their dimensions (generic) or by construction (adapter).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from g_entities import privileged_groups
from stage_utils import load_protocol

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--family', required=True)
a = ap.parse_args()
spec = load_protocol(a.root)
data = spec['families'][a.family]
out = a.root / 'prep' / a.family
obs = np.load(out / 'cache' / data['dataset'] / 'val_observations.npy').astype(np.float64)
term = np.load(out / 'cache' / data['dataset'] / 'val_terminals.npy')
_, objs = privileged_groups(data['dataset'], obs.shape[1])
names = list(objs)
L = json.loads((out / 'front/layout.json').read_text())
name_of = {j: nm for nm, ds in objs.items() for j in ds}
gen_map = []
for e in L['entities']:
    o = sorted({name_of.get(j, 'agent') for j in e['dims']})
    gen_map.append(o[0] if len(o) == 1 else '+'.join(o))
adapter_map = {'cube': [f'cube{k}' for k in range(len(names))], 'puzzle': [f'button{k}' for k in range(len(names))],
               'scene': ['cube', 'button0', 'button1', 'drawer', 'window']}[a.family]


def activity(t0, t1):
    t0, t1 = max(t0 - 1, 0), min(t1 + 1, len(obs) - 1)
    act = {}
    for nm, ds in objs.items():
        blk = obs[t0:t1 + 1, ds]
        if len(ds) == 9:
            act[nm] = np.linalg.norm(blk[-1, :3] - blk[0, :3]) / 0.4
        elif len(ds) == 4:
            act[nm] = np.abs(blk[:, 2] - blk[0, 2]).max() / scale[nm]
        else:
            act[nm] = abs(blk[-1, 0] - blk[0, 0]) / scale[nm]
    return act


scale = {}
for nm, ds in objs.items():
    if len(ds) == 4:
        scale[nm] = float(np.percentile(np.abs(obs[:, ds[2]] - np.median(obs[:, ds[2]])), 99.9)) + 1e-9
    elif len(ds) == 2:
        scale[nm] = float(obs[:, ds[0]].max() - obs[:, ds[0]].min()) + 1e-9
res = {'entity_objects_generic': gen_map, 'entity_objects_adapter': adapter_map}
for arm, d, emap in (('generic', 'events', gen_map), ('adapter', 'adapter_events', adapter_map)):
    ev = np.load(out / d / 'events_val.npz')
    rep = json.loads((out / d / 'report.json').read_text())
    truth, hit, none = [], [], 0
    for t0, t1, e in zip(ev['t_start'], ev['t'], ev['e']):
        act = activity(int(t0), int(t1))
        best = max(act, key=act.get)
        if act[best] < 0.5:
            none += 1
            continue
        truth.append(best)
        hit.append(emap[e] == best)
    res[arm] = {'events_val': int(len(ev['e'])), 'per_episode': rep['val']['per_episode'],
                'privileged_one_to_one': rep['val'].get('privileged_one_to_one'),
                'privileged_one_to_one_moves_2cm': rep['val'].get('privileged_one_to_one_moves_2cm'),
                'acted_correct': float(np.mean(hit)) if hit else None, 'events_with_true_actor': len(hit),
                'events_without_clear_actor': none, 'thresholds': {k: rep.get(k) for k in ('thr_pos', 'tol_pos', 'r_pos', 'r_app', 'thr_app')}}
(out / 'compare_events.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
print('COMPARE', a.family, json.dumps(res, default=float), flush=True)
