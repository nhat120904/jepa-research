"""Integrated data alignment, public-state and model-interface checks on a compute node."""
import argparse
import json
from pathlib import Path
import numpy as np
from stage_utils import load_protocol

ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--family',required=True)
a=ap.parse_args();spec=load_protocol(a.root);data=spec['families'][a.family]
out=a.root/'prep'/a.family;cache=out/'cache'/data['dataset']
from s_entities import state_entities
from skill_segments import segment_frames
L=json.loads((out/'front/layout.json').read_text());K=L['K'];checks={}
for split,expected in [('train',1000),('val',100)]:
    obs=np.load(cache/f'{split}_observations.npy',mmap_mode='r')
    terminals=np.load(cache/f'{split}_terminals.npy');ends=np.flatnonzero(terminals)
    ev=np.load(out/'events'/f'events_{split}.npz')
    assert len(ends)==expected and obs.shape[0]==expected*1001
    assert ev['before'].shape[1:]==(K,6)
    assert np.isfinite(ev['before']).all() and np.isfinite(ev['after']).all()
    assert (np.bincount(ev['e'],minlength=K)>0).all()
    first=np.r_[0,ends[:-1]+1][ev['episode']];last=ends[ev['episode']]
    assert np.all((ev['seg_start']>=first)&(ev['seg_start']<=ev['t_start'])&(ev['t_start']<=ev['t'])&(ev['t']<=last))
    for i in np.flatnonzero(ev['t']+10>last):
        frames=list(segment_frames(ev['seg_start'][i],ev['t'][i],10,last[i]))
        assert frames[0]>=first[i] and frames[-1]==last[i]
    checks[split]=dict(frames=len(obs),episodes=len(ends),obs_dim=obs.shape[1],events=len(ev['e']),
        multi_object_events=int(ev['knock'].sum()),acted_counts=np.bincount(ev['e'],minlength=K).tolist(),
        release_segments_clipped_at_reset=int(np.sum(ev['t']+10>last)))
import torch
torch.set_num_threads(2)
from u_wm import make_wm,make_h,h_input,finite_rest_support
from event_support import make_support,features
S=torch.zeros((2,K,6));E=torch.arange(2)%K;X=torch.zeros((2,6))
with torch.no_grad():
    assert make_wm(K)(S,E,X).shape==(2,K,6)
    flat=S.flatten(1)
    assert make_h(K,2048,True)(h_input(torch,flat,flat,True)).shape==(2,1)
    assert make_support(K)(features(torch,S,E,X)).shape==(2,1)
import gymnasium
import ogbench
env=gymnasium.make(data['env'])
assert env.spec.max_episode_steps==data['horizon']
task_checks=[]
for task in range(1,6):
    obs,info=env.reset(seed=task*100,options=dict(task_id=task,render_goal=False))
    S,eff=state_entities(obs[None],L);G,_=state_entities(info['goal'][None],L)
    assert S.shape==G.shape==(1,K,6) and np.isfinite(S).all() and np.isfinite(G).all()
    modified=obs.copy();modified[:19]+=10
    np.testing.assert_array_equal(state_entities(modified[None],L)[0],S)
    task_checks.append(dict(task=task,state_shape=list(S.shape),goal_shape=list(G.shape)))
env.close()
checks.update(K=K,model_interfaces='passed',public_state_proprio_independence='passed',native_horizon=data['horizon'],task_checks=task_checks)
(out/'preparation_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('PREPARED',a.family,json.dumps(checks),flush=True)
