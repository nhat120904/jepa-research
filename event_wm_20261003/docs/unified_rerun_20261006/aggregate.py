"""Complete/partial evaluation aggregation, on a small dependent CPU job."""
import argparse
import json
import math
from pathlib import Path
from result_checks import validate_episode_grid
from stage_utils import load_protocol,sha_file

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
spec=load_protocol(a.root);registry=json.loads((a.root/'job_registry.json').read_text())
report={'protocol':spec,'families':{},'complete':True,'limits':'One training seed. State adapters and per-family checkpoint dimensions differ. Native benchmark horizons retained; no published-baseline or zero-shot claim.'}
for family in spec['families']:
    combined=[];arms={}
    for seed in spec['evaluation']['seeds']:
        job=registry['evaluations'][family][str(seed)]
        path=a.root/'runs'/family/f'eval_seed{seed}_{job}'/'loop/u_closed_loop.json'
        if not path.exists():
            report['complete']=False
            partial=list((path.parent/'episodes').glob('*.json'))
            arms[str(seed)]={'complete':False,'completed_episode_files':len(partial),'job_id':job}
            continue
        d=json.loads(path.read_text())
        try:
            validate_episode_grid(d['episodes'])
        except ValueError as error:
            report['complete']=False
            arms[str(seed)]={'complete':False,'reason':str(error),'job_id':job}
            continue
        args=d['summary']['args']
        for key,value in spec['evaluation'].items():
            if key not in ('seeds','tasks'):
                assert str(value)==args[key],(family,seed,key)
        assert str(seed)==args['seed'] and str(spec['evaluation']['tasks'])==args['tasks']
        combined+=d['episodes']
        arms[str(seed)]={'complete':True,'successes':sum(e['success'] for e in d['episodes']),
            'episodes':100,'by_task':d['summary']['by_task'],'sha256':sha_file(path),'job_id':job}
    family_report={'by_seed':arms,'complete':len(combined)==200}
    if len(combined)==200:
        k=sum(e['success'] for e in combined);n=len(combined);z=1.959963984540054;den=1+z*z/n
        c=(k/n+z*z/(2*n))/den;h=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
        family_report.update(successes=k,episodes=n,success_rate=k/n,
            episode_wilson95_percent=[100*(c-h),100*(c+h)],
            by_task={str(t):{'successes':sum(e['success'] for e in combined if e['task']==t),'episodes':40} for t in range(1,6)})
    report['families'][family]=family_report
(a.root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# One-version STATE rerun results','',f"Complete protocol: {report['complete']}",'',
       '| Family | Seed6 | Seed7 | Total |','|---|---:|---:|---:|']
for family,r in report['families'].items():
    seed_cells=[f"{r['by_seed'][str(s)]['successes']}/100" if r['by_seed'][str(s)]['complete'] else 'partial/missing' for s in (6,7)]
    total=f"{r['successes']}/200" if r['complete'] else 'incomplete'
    lines.append(f"| {family} | {seed_cells[0]} | {seed_cells[1]} | {total} |")
lines+=['',report['limits'],'','WM30k,h150k,BC40k+10k,support6k;1000TRAIN/100play-validation episodes per family. Final checkpoints, frozen source/config, max20000expansions, no finite-only successor cache/LHBL. No historical scores substituted.']
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('AGGREGATED',report['complete'],{k:v.get('successes') for k,v in report['families'].items()},flush=True)
if not report['complete']:
    raise SystemExit(1)
