"""Attribute heuristic sensitivity and test generic projection on frozen puzzle roots."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from u_wm import Model

if "SLURM_JOB_ID" not in os.environ:
    raise RuntimeError("Use sbatch")
torch.set_num_threads(2)
root=Path(os.environ["DIAG_DIR"]); out=root/("job_"+os.environ["SLURM_JOB_ID"]); out.mkdir()
M=Model(torch.load("/mnt/data/nhatnc129/jepa/event_wm/state_puzzle-4x5-play-v0_57614/wm_57636/u_model.pt",map_location="cpu",weights_only=False),"cuda")
orig_step,orig_h=M.step_batch,M.heuristic
old=json.loads((root/"job_57669/report.json").read_text())
eps=json.loads((root/"dev.json").read_text())["episodes"]

def project(states):
    result=np.array(states,copy=True)
    for k,p in enumerate(M.proto):
        d=(((result[:,k,None,:2]-p[None,:,:2])/M.sc.thr_pos)**2).sum(-1)
        d+=(((result[:,k,None,2:5]-p[None,:,2:5])/M.app_tol[k])**2).sum(-1)
        result[:,k]=p[d.argmin(-1)]
    return result

ep=next(ep for ep in eps if ep["task"]==2 and ep["episode"]==0)
start=np.array(ep["read_start"],np.float32); goal=np.array(ep["read_goal"],np.float32)
best=next(r for r in old["searches"] if r["mode"]=="raw_WM_learned_h")
s=start.copy()
for ev in best["commands"]:s=orig_step(s[None],[(ev["e"],np.array(ev["x"],np.float32))])[0]
clean=project(s[None])[0]
variants={"raw":s,"prototype_projected_all":clean}
z=s.copy();z[:,:2]=clean[:,:2];variants["positions_only"]=z
z=s.copy();z[:,2:5]=clean[:,2:5];variants["appearance_only"]=z
sensitivity={name:float(orig_h(z[None],goal)[0]) for name,z in variants.items()}
print("SENSITIVITY",sensitivity,flush=True)
report=dict(job_id=os.environ["SLURM_JOB_ID"],sensitivity=sensitivity,raw_best_state=s.tolist(),projected_best_state=clean.tolist(),searches=[])
for mode in ("raw_WM_projected_h_input","projected_WM_learned_h"):
    if mode=="raw_WM_projected_h_input":
        M.step_batch=orig_step
        M.heuristic=lambda Ss,G:orig_h(project(Ss),G)
    else:
        M.step_batch=lambda Ss,es:project(orig_step(Ss,es))
        M.heuristic=orig_h
    for task in (2,3,4,5):
        ep=next(ep for ep in eps if ep["task"]==task and ep["episode"]==0)
        S=np.array(ep["read_start"],np.float32);G=np.array(ep["read_goal"],np.float32)
        t0=time.time();plan,info=M.plan(S,G,max_expansions=20000)
        row=dict(mode=mode,task=task,found=plan is not None,length=len(plan) if plan is not None else None,
                 expanded=info["expanded"],seconds=time.time()-t0,best_h=info.get("best_h"))
        if plan is not None:
            row["commands"]=[dict(e=int(e),x=x.tolist()) for e,x in plan]
        report["searches"].append(row)
        print("SEARCH",{k:v for k,v in row.items() if k!="commands"},flush=True)
(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
