"""Login-safe scheduler submission, quota audit and durable job registration. No model/data loading."""
import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

def command(args):
    return subprocess.check_output(args,text=True).strip()

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
if (a.root/'job_registry.json').exists():
    raise RuntimeError('Registry already exists: inspect jobs before resubmitting; duplicates forbidden')
spec=json.loads((a.root/'protocol.json').read_text())
jobs={'training':{},'evaluations':{},'quota_audits':[]}
reserved={'gres/gpu':24.,'cpu':217.83333333333334,'mem':600405.3333333334}
known=set()

def save():
    tmp=a.root/'job_registry.tmp'
    tmp.write_text(json.dumps(jobs,indent=2)+'\n')
    tmp.replace(a.root/'job_registry.json')

def guard(stage):
    queued=command(['squeue','-h','-u','nhatnc129','-o','%i|%j|%T'])
    for row in queued.splitlines():
        if row.split('|')[0] not in known:
            raise RuntimeError('New peer job requires resource/duplicate inspection: '+row)
    if known:
        command(['sacct','-j',','.join(sorted(known)),'--format=JobID,State,ExitCode','-n','-P'])
    month=datetime.date.today().replace(day=1).isoformat()
    raw=command(['sreport','-t','hours','-T','gres/gpu,cpu,mem','cluster','UserUtilizationByAccount',f'start={month}','end=now','-P','-n'])
    totals={r:{} for r in reserved}
    for row in raw.splitlines():
        cols=row.split('|')
        if len(cols)>=6 and cols[4] in totals:
            users=totals[cols[4]];users[cols[1]]=users.get(cols[1],0)+float(cols[5])
    audit={'stage':stage,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'resources':{},'squeue':queued}
    for resource,budget in reserved.items():
        users=totals[resource];ranking=sorted(users.values(),reverse=True)
        if len(ranking)<5 or 'nhatnc129' not in users:
            raise RuntimeError('Monthly ranking unavailable for '+resource)
        own=users['nhatnc129'];ceiling=.9*ranking[4]
        audit['resources'][resource]={'used':own,'planned_full_protocol':budget,'ceiling':ceiling}
        if own+budget>ceiling:
            raise RuntimeError(f'Quota exceeded: {resource} {own}+{budget}>{ceiling}; reduce/wait before submission')
    jobs['quota_audits'].append(audit)
    (a.root/f'quota_{stage}.txt').write_text(raw+'\n')
    print('QUOTA_OK',stage,audit['resources'],flush=True)
    save()

def submit(script,arguments=(),dependency=None,name=None):
    cmd=['sbatch','--parsable',f'--export=ALL,RUN_DIR={a.root}']
    if dependency:cmd.append('--dependency='+dependency)
    if name:cmd.append('--job-name='+name)
    cmd+=[str(a.root/script),*map(str,arguments)]
    job=command(cmd).split(';')[0]
    known.add(job)
    return job

guard('prepare')
jobs['preparation']=submit('prepare.sbatch');save()
print('SUBMITTED','preparation',jobs['preparation'],flush=True)
for family in spec['families']:
    guard('train_'+family)
    job=submit('train.sbatch',[family],'afterok:'+jobs['preparation'],'ew_un_'+family)
    jobs['training'][family]=job;save();print('SUBMITTED','train',family,job,flush=True)
for family in spec['families']:
    jobs['evaluations'][family]={}
    for seed in spec['evaluation']['seeds']:
        guard(f'eval_{family}_{seed}')
        job=submit('evaluate.sbatch',[family,jobs['training'][family],seed],
                   'afterok:'+jobs['training'][family],f'ew_ue_{family}_{seed}')
        jobs['evaluations'][family][str(seed)]=job;save();print('SUBMITTED','eval',family,seed,job,flush=True)
all_eval=[job for runs in jobs['evaluations'].values() for job in runs.values()]
jobs['aggregation']=submit('aggregate.sbatch',dependency='afterany:'+':'.join(all_eval));save()
print('SUBMITTED','aggregate',jobs['aggregation'],flush=True)
