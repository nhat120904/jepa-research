"""One shared training recipe for every family; no family-dependent optimization settings."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from stage_utils import load_protocol,cli_options,record_run,run_script

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--family',required=True)
a=ap.parse_args();spec=load_protocol(a.root);data=spec['families'][a.family]
prep=a.root/'prep'/a.family;out=a.root/'runs'/a.family/('train_'+os.environ['SLURM_JOB_ID'])
record_run(a.root,out,dict(stage='training',family=a.family,env=data['env'],initialization='scratch',training_seed=0))
wm_cmd=[sys.executable,str(a.root/'source/u_wm.py'),*cli_options(dict(events=prep/'events',out=out/'wm',**spec['wm']))]
skill_common=dict(cache=prep/'cache'/data['dataset'],events=prep/'events',train_frames=0,val_frames=0,
                  hist_gap=spec['skill']['hist_gap'],chunk=spec['skill']['chunk'],release=spec['skill']['release'],cond=spec['skill']['cond'])
with (out/'wm.log').open('w') as log:
    print('WM_COMMAND',wm_cmd,flush=True)
    wm=subprocess.Popen(wm_cmd,stdout=log,stderr=subprocess.STDOUT)
    try:
        run_script(a.root,'u_skill.py',cli_options(dict(**skill_common,steps=spec['skill']['steps'],lr=spec['skill']['lr'],out=out/'skill_initial')),out/'skill_initial.log')
        run_script(a.root,'u_skill.py',cli_options(dict(**skill_common,**spec['skill_continue'],init_model=out/'skill_initial/u_skill.pt',out=out/'skill_final')),out/'skill_final.log')
        if wm.wait()!=0:
            raise RuntimeError('WM/h training failed; inspect wm.log')
    finally:
        if wm.poll() is None:
            wm.terminate()
            wm.wait()
run_script(a.root,'train_event_support.py',cli_options(dict(model=out/'wm/u_model.pt',events=prep/'events',out=out/'support',**spec['support'])),out/'support.log')
import torch
torch.set_num_threads(2)
wm_ck=torch.load(out/'support/u_model.pt',map_location='cpu',weights_only=False)
skill_ck=torch.load(out/'skill_final/u_skill.pt',map_location='cpu',weights_only=False)
assert wm_ck['training_stage']=='complete' and wm_ck['h_step']==spec['wm']['h_steps']
assert wm_ck['event_support']['steps']==spec['support']['steps']
assert skill_ck['include_side_effects'] and skill_ck['release_clipped_at_episode_boundary']
assert skill_ck['training_steps']==spec['skill_continue']['steps']
complete=dict(family=a.family,K=wm_ck['K'],h_steps=wm_ck['h_step'],support_steps=wm_ck['event_support']['steps'],
    skill_initial_steps=spec['skill']['steps'],skill_continued_steps=skill_ck['training_steps'],
    model=str(out/'support/u_model.pt'),skill=str(out/'skill_final/u_skill.pt'))
(out/'training_complete.json').write_text(json.dumps(complete,indent=2)+'\n')
print('TRAINING_COMPLETE',json.dumps(complete),flush=True)
