"""Matched WM continuations on original vs corrected labels, with fixed h/support/skill."""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--parent', type=Path, required=True)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--family', required=True)
p.add_argument('--steps', type=int, default=10000)
p.add_argument('--lr', type=float, default=1e-4)
a = p.parse_args()
assert os.environ.get('SLURM_JOB_ID'), 'Use sbatch'
source_hashes = json.loads((a.run/'SOURCE_SHA256SUMS.json').read_text())
for name, expected in source_hashes.items():
    assert hashlib.sha256((a.run/name).read_bytes()).hexdigest() == expected, name
sys.path.insert(0, str(a.run / 'source'))
import numpy as np
import torch
import torch.nn.functional as F
from u_wm import Model, Scale, event_input, make_wm

torch.set_num_threads(2)
root = a.run / ('repair_' + os.environ['SLURM_JOB_ID'])
root.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, root/'source.py')
spec = json.loads((a.parent/'protocol.json').read_text())
registry = json.loads((a.parent/'job_registry.json').read_text())
complete = json.loads((a.parent/'runs'/a.family/('train_'+registry['training'][a.family])/'training_complete.json').read_text())
prep = a.parent/'prep'/a.family
fixed = a.run/('events_'+a.family)
old = prep/'events'
val = dict(np.load(fixed/'events_val.npz'))
parent = torch.load(complete['model'], map_location='cpu', weights_only=False)
sc = Scale(parent['thr_pos'], parent['thr_app'], parent['tol_pos'])
app_unit = parent['thr_app'] if np.isfinite(parent['thr_app']) else .1
unit = torch.tensor([parent['tol_pos']/32]*2+[2*app_unit]*3, device='cuda')
manifest = {'job_id':os.environ['SLURM_JOB_ID'], 'family':a.family, 'steps_per_arm':a.steps,
            'source_hashes':source_hashes,
            'lr':a.lr, 'parent_model':complete['model'], 'parent_sha256':hashlib.sha256(Path(complete['model']).read_bytes()).hexdigest(),
            'fixed_skill':complete['skill'], 'h_support_skill_unchanged':True,
            'development_seed':6, 'fresh_evaluation_seed':8, 'arms':{}}

def evaluate(ck):
    M = Model(ck, 'cuda')
    y, b = val['after'], val['before']
    preds=[]
    for start in range(0,len(y),512):
        sl=slice(start,start+512)
        preds.append(M.step_batch(b[sl],list(zip(val['e'][sl],val['target'][sl]))))
    pred=np.concatenate(preds)
    bad=(np.linalg.norm(pred[...,:2]-y[...,:2],axis=-1)>M.sc.tol_pos)|(np.abs(pred[...,2:5]-y[...,2:5]).max(-1)>M.app_tol)|((pred[...,5]>.5)!=(y[...,5]>.5))
    changed=(np.linalg.norm(y[...,:2]-b[...,:2],axis=-1)>M.sc.tol_pos)|(np.abs(y[...,2:5]-b[...,2:5]).max(-1)>M.app_tol)
    return {'events':len(y),'exact_event_fraction':float((~bad.any(-1)).mean()),'changed_error_fraction':float(bad[changed].mean()),'unchanged_error_fraction':float(bad[~changed].mean())}

manifest['parent_on_corrected_val']=evaluate(parent)
(root/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PARENT_ON_CORRECTED_VAL',manifest['parent_on_corrected_val'],flush=True)
for arm, path in [('old_labels',old),('fixed_labels',fixed)]:
    torch.manual_seed(0)
    tr=dict(np.load(path/'events_train.npz'))
    ref=dict(np.load(old/'events_train.npz'))
    assert np.array_equal(tr['t'],ref['t']) and np.array_equal(tr['t_start'],ref['t_start'])
    assert len(tr['e'])==len(ref['e']), 'No extra data or dropped segments in this comparison'
    x=event_input(tr['before'],tr['e'],tr['target'],sc.tol_pos) if parent.get('event_pos_only') else tr['target']
    T={k:torch.as_tensor(v,device='cuda').float() for k,v in {'s':sc.norm(tr['before']),'x':sc.norm(x),'y':sc.norm(tr['after'])}.items()}
    E=torch.as_tensor(tr['e'],device='cuda').long()
    wm=make_wm(parent['K']).to('cuda');wm.load_state_dict(parent['wm'])
    opt=torch.optim.AdamW(wm.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1,(s+1)/100)*.5*(1+math.cos(math.pi*min(1,s/a.steps))))
    t0=time.time();armroot=root/arm;armroot.mkdir()
    for step in range(a.steps):
        idx=torch.randint(0,len(E),(512,),device='cuda')
        cont,logit=wm(T['s'][idx],E[idx],T['x'][idx],logits=True)
        loss=(((cont-T['y'][idx,...,:5])/unit)**2).mean()+F.binary_cross_entropy_with_logits(logit,T['y'][idx,...,5])
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(wm.parameters(),1.);opt.step();sched.step()
        if (step+1)%1000==0:print(arm,{'step':step+1,'loss':float(loss),'minutes':(time.time()-t0)/60},flush=True)
    wm.eval();ck={**parent,'wm':wm.state_dict(),'wm_label_repair':{'arm':arm,'extra_steps':a.steps,'lr':a.lr,'h_trained_with_parent_wm':True}}
    torch.save(ck,armroot/'u_model.pt')
    meta={'train_events_sha256':hashlib.sha256((path/'events_train.npz').read_bytes()).hexdigest(),'corrected_val':evaluate(ck),'evaluations':{}}
    manifest['arms'][arm]=meta
    (root/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('VAL',arm,meta['corrected_val'],flush=True)
    del wm,T,E,opt,sched
    torch.cuda.empty_cache()

for arm in manifest['arms']:
    for seed in (6,8):
        out=root/arm/('loop_seed'+str(seed))
        cmd=[sys.executable,str(a.run/'source/u_closed_loop.py'),'--env',spec['families'][a.family]['env'],
             '--model',str(root/arm/'u_model.pt'),'--skill',complete['skill'],'--state-layout',str(prep/'front/layout.json'),
             '--events',str(fixed),'--cache',str(prep/'cache'/spec['families'][a.family]['dataset']),
             '--seed',str(seed),'--out',str(out)]
        for key,value in spec['evaluation'].items():
            if key=='seeds':continue
            cmd+=['--'+key.replace('_','-')]+([str(v) for v in value] if isinstance(value,list) else [str(value)])
        print('EVAL',arm,seed,flush=True)
        with (root/arm/('eval_seed'+str(seed)+'.log')).open('w') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((out/'u_closed_loop.json').read_text())
        grid={(v['task'],v['episode']) for v in result['episodes']}
        assert len(result['episodes'])==100 and len(grid)==100
        assert grid=={(t,e) for t in range(1,6) for e in range(20)}
        manifest['arms'][arm]['evaluations'][str(seed)]=result['summary']
        print('RESULT',arm,seed,result['summary']['success'],result['summary']['by_task'],flush=True)
        (root/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('REPAIR_COMPLETE',root/'results.json',flush=True)
