"""Bounded repair: train-derived finite rest support + continued Bellman training.
No puzzle transition rules or exact-distance labels enter training/search.
Cached successors use the existing candidate generator; validate goal candidates are covered.
"""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from u_wm import Model, h_input

if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('Use sbatch')
torch.set_num_threads(2)
torch.manual_seed(0)
rng=np.random.default_rng(61006)
root=Path(os.environ['REPAIR_DIR']);out=root/('job_'+os.environ['SLURM_JOB_ID']);out.mkdir()
base=Path('/mnt/data/nhatnc129/jepa/event_wm/state_puzzle-4x5-play-v0_57614')
cp=base/'wm_57636/u_model.pt';ck=torch.load(cp,map_location='cpu',weights_only=False)
tr=np.load(base/'events/events_train.npz')
all_states=np.concatenate([tr['before'],tr['after']]).astype(np.float32)
# Conservative support discovery: near-exact finite attributes (<=8 values at 1e-5 precision).
# Does not cluster or snap general continuous positions/heights to a handful of prototypes.
support=[]
for k in range(ck['K']):
 fields=[]
 for c in range(6):
  v=np.unique(np.round(all_states[:,k,c],5))
  fields.append(v.tolist() if len(v)<=8 else None)
 support.append(fields)
ck['rest_support']=support
M=Model(ck,'cuda'); states=np.unique(M.canonical(all_states).reshape(len(all_states),-1),axis=0).reshape(-1,M.K,6)
if len(states)>32768:states=states[rng.choice(len(states),32768,replace=False)]
# State-independent event lists are allowed for caching only after checking every sampled state.
# This experiment is limited to finite rest support; the general cube implementation still has goal candidates.
for fields in support:
 if any(v is None for v in fields):raise RuntimeError('Finite-support cached experiment inapplicable to continuous attributes')
N=len(states);t0=time.time();counts=[];succ=[]
for begin in range(0,N,256):
 ss=states[begin:begin+256];owner=[];ev=[]
 for i,s in enumerate(ss):
  cs=M.candidates(s,s);counts.append(len(cs));ev.extend(cs);owner.extend([i]*len(cs))
 if not ev:raise RuntimeError('No candidates')
 part=[]
 for q in range(0,len(ev),2048):part.append(M.step_batch(ss[np.asarray(owner[q:q+2048])],ev[q:q+2048]))
 succ.append(np.concatenate(part))
print('CACHE',dict(states=N,candidate_counts=sorted(set(counts)),seconds=round(time.time()-t0,1)),flush=True)
if len(set(counts))!=1:raise RuntimeError('Variable candidate count: use general online training')
E=counts[0];succ=np.concatenate(succ).reshape(N,E,M.K,6)
# Validate cached action targets cover the actual goal-dependent generator for random train goals.
for i,j in zip(rng.integers(N,size=1024),rng.integers(N,size=1024)):
 c0=M.candidates(states[i],states[i]);c1=M.candidates(states[i],states[j])
 if len(c0)!=len(c1) or any(not any(e==f and np.allclose(x,y,atol=1e-5) for f,y in c0) for e,x in c1):
  raise RuntimeError('Goal candidates differ: cache would change Bellman operator')
# New imagined goals, canonicalized after every step, using only learned WM and prototypes.
P=60000;pindex=rng.integers(N,size=P);pG=states[pindex].copy();walk=rng.integers(0,31,size=P)
for w in range(30):
 rows=np.flatnonzero(walk>w)
 for begin in range(0,len(rows),1024):
  ix=rows[begin:begin+1024];ev=[]
  for s in pG[ix]:
   cs=M.candidates(s,s);ev.append(cs[rng.integers(len(cs))])
  pG[ix]=M.step_batch(pG[ix],ev)
print('GOALS',dict(pairs=P,seconds=round(time.time()-t0,1)),flush=True)
# Same h architecture, continued from frozen checkpoint; change only support/coverage/training throughput.
h=M.h;h.train();target=Model(ck,'cuda').h;opt=torch.optim.AdamW(h.parameters(),lr=1e-4,weight_decay=1e-5)
S=torch.as_tensor(M.sc.norm(states).reshape(N,-1),device='cuda')
C=torch.as_tensor(M.sc.norm(succ).reshape(N,E,-1),device='cuda')
PG=torch.as_tensor(M.sc.norm(pG).reshape(P,-1),device='cuda')
log=[];B=512;steps=15000
for step in range(steps):
 # Half real goals, half imagined walk goals; S and successor representations now use identical support.
 ix=rng.integers(N,size=B);gx=rng.integers(N,size=B);pw=rng.integers(P,size=B//2);ix[B//2:]=pindex[pw]
 st=S[ix];gt=torch.cat([S[gx[:B//2]],PG[pw]])
 children=C[ix];gc=gt[:,None].expand(-1,E,-1)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  hv=target(h_input(torch,children.reshape(B*E,-1),gc.reshape(B*E,-1),M.absdiff)).reshape(B,E).float()
  # Canonical state equality is the original tolerance goal test for these finite supports.
  done=(children-gc).abs().amax(-1)<1e-5
  y=(1+hv.masked_fill(done,0).min(1).values).clamp(max=30)
  y.masked_fill_((st-gt).abs().amax(-1)<1e-5,0)
 with torch.autocast('cuda',dtype=torch.bfloat16):pred=h(h_input(torch,st,gt,M.absdiff)).squeeze(-1).float()
 loss=(pred-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 if (step+1)%1000==0:target.load_state_dict(h.state_dict())
 if step%1000==0 or step==steps-1:
  row=dict(step=step,loss=float(loss),target_mean=float(y.mean()),minutes=round((time.time()-t0)/60,2));log.append(row);print('TRAIN',row,flush=True)
ck['h']=h.cpu().state_dict();ck['repair']=dict(parent_checkpoint=str(cp),parent_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
    train_steps=steps,batch=B,seed=61006,finite_support_precision=1e-5,pool_states=N,walk_pairs=P)
torch.save(ck,out/'u_model.pt')
(out/'training.json').write_text(json.dumps(dict(config=ck['repair'],support=support,log=log),indent=2)+'\n')
# Evaluation uses saved dev roots only; never used for sampling, labels or early stopping.
M=Model(torch.load(out/'u_model.pt',map_location='cpu',weights_only=False),'cuda')
eps=json.loads((root/'dev.json').read_text())['episodes'];results=[]
for ep in eps:
 if ep['episode']!=0:continue
 start=np.asarray(ep['read_start'],np.float32);goal=np.asarray(ep['read_goal'],np.float32)
 t=time.time();plan,info=M.plan(start,goal,max_expansions=20000)
 row=dict(task=ep['task'],found=plan is not None,length=len(plan) if plan is not None else None,
    h0=float(M.heuristic(start[None],goal)[0]),seconds=time.time()-t,info={k:v for k,v in info.items() if k!='best_plan'})
 results.append(row);print('SEARCH',row,flush=True)
(out/'search.json').write_text(json.dumps(results,indent=2)+'\n')
