import argparse
import json
import os
from pathlib import Path
from stage_utils import load_protocol,cli_options,record_run,run_script
from result_checks import validate_episode_grid

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--family',required=True)
ap.add_argument('--training-job',required=True);ap.add_argument('--seed',type=int,required=True)
a=ap.parse_args();spec=load_protocol(a.root);data=spec['families'][a.family]
assert a.seed in spec['evaluation']['seeds']
train=a.root/'runs'/a.family/('train_'+a.training_job)
assert json.loads((train/'protocol.json').read_text())==spec
complete=json.loads((train/'training_complete.json').read_text())
assert complete['h_steps']==spec['wm']['h_steps'] and complete['support_steps']==spec['support']['steps']
out=a.root/'runs'/a.family/(f'eval_seed{a.seed}_'+os.environ['SLURM_JOB_ID'])
record_run(a.root,out,dict(stage='evaluation',family=a.family,env=data['env'],training_job=a.training_job,seed=a.seed,native_horizon=data['horizon']))
prep=a.root/'prep'/a.family
options={k:v for k,v in spec['evaluation'].items() if k!='seeds'}
run_script(a.root,'u_closed_loop.py',cli_options(dict(env=data['env'],model=complete['model'],skill=complete['skill'],
    state_layout=prep/'front/layout.json',events=prep/'events',cache=prep/'cache'/data['dataset'],seed=a.seed,out=out/'loop',**options)))
result=json.loads((out/'loop/u_closed_loop.json').read_text())
validate_episode_grid(result['episodes'],spec['evaluation']['tasks'],spec['evaluation']['episodes'])
print('EVALUATION_COMPLETE',a.family,a.seed,result['summary']['success'],flush=True)
