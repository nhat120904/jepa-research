import json,os,time
from pathlib import Path
import numpy as np
import torch
from u_wm import Model
if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Use sbatch')
torch.set_num_threads(2)
root=Path(os.environ['REPAIR_DIR']);out=root/('job_'+os.environ['SLURM_JOB_ID']);out.mkdir()
cp=root/'job_57684/u_model.pt';M=Model(torch.load(cp,map_location='cpu',weights_only=False),'cuda')
def slow(S):
 z=np.array(S,np.float32,copy=True)
 for k,fields in enumerate(M.rest_support):
  for c,values in enumerate(fields):
   if values is not None:
    v=np.asarray(values,np.float32);z[...,k,c]=v[np.abs(z[...,k,c,None]-v).argmin(-1)]
 return z
rng=np.random.default_rng(0)
for shape in [(M.K,6),(512,M.K,6)]:
 z=rng.normal(size=shape).astype(np.float32)
 assert np.array_equal(slow(z),M.canonical(z))
 for k in range(M.K):assert np.array_equal(M.canonical_entity(z.reshape(-1,M.K,6)[0,k],k),slow(z.reshape(-1,M.K,6)[0])[k])
print('CANONICAL_EQUIVALENCE_PASSED',flush=True)
eps=json.loads((root/'dev.json').read_text())['episodes'];results=[]
for ep in eps:
 if ep['episode']!=0:continue
 S=np.asarray(ep['read_start'],np.float32);G=np.asarray(ep['read_goal'],np.float32);t=time.time()
 plan,info=M.plan(S,G,max_expansions=20000)
 row=dict(task=ep['task'],found=plan is not None,length=len(plan) if plan is not None else None,h0=float(M.heuristic(S[None],G)[0]),seconds=time.time()-t,info={k:v for k,v in info.items() if k!='best_plan'})
 results.append(row);print('SEARCH',row,flush=True)
 (out/'search.json').write_text(json.dumps(results,indent=2)+'\n')
