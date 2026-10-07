"""Diagnostic controls for harder roots, plus search state-alias attribution."""
import json, os, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from u_wm import Model
from lightsout import solver

if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('Use sbatch')
torch.set_num_threads(2)
root=Path(os.environ['DIAG_DIR']);out=root/('job_'+os.environ['SLURM_JOB_ID']);out.mkdir()
M=Model(torch.load('/mnt/data/nhatnc129/jepa/event_wm/state_puzzle-4x5-play-v0_57614/wm_57636/u_model.pt',map_location='cpu',weights_only=False),'cuda')
orig_step,orig_h,orig_key=M.step_batch,M.heuristic,M.key
eps=json.loads((root/'dev.json').read_text())['episodes']
A,R,piv,span=solver(4,5)
def bits(s):return (np.asarray(s)[...,2:5].mean(-1)>.5).astype(np.uint8)
def exact_h(S,G):
    t=((bits(S)^bits(G)[None])@R[:,20:].T)%2
    x=np.zeros((len(S),20),np.uint8);x[:,piv]=t[:,:len(piv)]
    h=(x[:,None]^span[None]).sum(-1).min(-1).astype(np.float32);h[t[:,len(piv):].any(-1)]=30
    return h
def exact_step(S,es):
    z=np.array(S,copy=True);z[...,2:5]=(bits(S)^A[:,[e for e,x in es]].T)[...,None];z[...,5]=0;return z
def project(S):
    z=np.array(S,copy=True)
    for k,p in enumerate(M.proto):
        d=(((z[:,k,None,:2]-p[None,:,:2])/M.sc.thr_pos)**2).sum(-1)+(((z[:,k,None,2:5]-p[None,:,2:5])/M.app_tol[k])**2).sum(-1)
        z[:,k]=p[d.argmin(-1)]
    return z
report=dict(job_id=os.environ['SLURM_JOB_ID'],controls=[])
for name,wm,h,ts in [('raw_WM_oracle_h',orig_step,exact_h,(3,4,5)),('oracle_WM_learned_h',exact_step,orig_h,(3,4,5))]:
    M.step_batch,M.heuristic=wm,h
    for task in ts:
        ep=next(ep for ep in eps if ep['task']==task and ep['episode']==0)
        S=np.array(ep['read_start'],np.float32);G=np.array(ep['read_goal'],np.float32)
        t0=time.time();plan,info=M.plan(S,G,max_expansions=20000)
        real=S.copy()
        if plan is not None:
            for ev in plan:real=exact_step(real[None],[ev])[0]
        row=dict(mode=name,task=task,found=plan is not None,length=len(plan) if plan is not None else None,
                 reference_goal=bool(M.at_goal(real,G)) if plan is not None else None,expanded=info['expanded'],seconds=time.time()-t0,best_h=info.get('best_h'))
        report['controls'].append(row);print('CONTROL',row,flush=True)
ep=next(ep for ep in eps if ep['task']==2 and ep['episode']==0);S=np.array(ep['read_start'],np.float32);G=np.array(ep['read_goal'],np.float32)
M.step_batch,M.heuristic=orig_step,orig_h
for mode in ('raw_key','prototype_key_only'):
    pairs=set();raws=set();logical=set()
    def tracked_key(state):
        rk=orig_key(state);bk=bits(state).tobytes();pairs.add((bk,rk));raws.add(rk);logical.add(bk)
        return rk if mode=='raw_key' else orig_key(project(state[None])[0])
    M.key=tracked_key
    t0=time.time();plan,info=M.plan(S,G,max_expansions=20000)
    hist=Counter(b for b,r in pairs)
    row=dict(mode=mode,task=2,found=plan is not None,length=len(plan) if plan is not None else None,
             expanded=info['expanded'],seconds=time.time()-t0,unique_raw_keys=len(raws),unique_logical_boards=len(logical),
             max_raw_keys_per_logical_board=max(hist.values()),logical_boards_with_multiple_keys=sum(v>1 for v in hist.values()))
    report['controls'].append(row);print('KEY',row,flush=True)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
