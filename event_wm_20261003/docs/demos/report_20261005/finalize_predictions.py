"""Match Model.plan's initial covered-bit quantization; export every selected step.

Reuses all previous images, videos and commands. No search or simulator replay.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from export_report import BG, INK, font, save_json, state_panel
from u_wm import Model


def main():
    if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('Use sbatch')
    ap=argparse.ArgumentParser(); ap.add_argument('--src',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); torch.set_num_threads(1)
    for p in a.src.iterdir():
        if p.is_file() and p.suffix in ('.png','.mp4','.json','.js'):
            shutil.copy2(p,a.out/p.name)
    d=json.loads((a.src/'report_data.json').read_text())
    ref=json.loads((a.src/'inputs/reference.json').read_text())
    model_path=Path(ref['summary']['args']['model'])
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==d['checkpoint_sha256']['model']
    M=Model(torch.load(model_path,map_location='cpu',weights_only=False),'cpu')
    if os.environ.get('REPORT_RERENDER')=='1':
        import gymnasium
        import ogbench
        from u_reader import make_reader
        from export_report import tile
        args=ref['summary']['args']
        rk=torch.load(args['reader'],map_location='cpu',weights_only=False)
        reader=make_reader(rk['K'],agent=rk.get('agent',False)).eval()
        reader.load_state_dict(rk['reader'])
        def observe(raw):
            with torch.no_grad():
                s,_=reader(torch.as_tensor(raw).permute(2,0,1)[None].float()/255)
            s=s[0].numpy(); s[:,5]=1/(1+np.exp(-s[:,5])); return s
        # Gym Env.close does not release this OGBench renderer. Creating the
        # next env before the old Python object is destroyed can invalidate
        # OSMesa's current context. Reuse one env / renderer across task resets.
        env=gymnasium.make(args['env'])
        for p in d['plans']:
            if p['task'] not in (3,4): continue
            try:
                raw,info=env.reset(seed=int(args['seed'])*10000+p['task']*100,
                                   options=dict(task_id=p['task'],render_goal=False))
                raw=np.array(raw,copy=True); goal=np.array(info['goal'],copy=True)
                stats=dict(observation_mean=float(raw.mean()),goal_mean=float(goal.mean()),
                           observation_black_fraction=float((raw==0).all(-1).mean()),
                           goal_black_fraction=float((goal==0).all(-1).mean()),
                           dtype=str(raw.dtype),shape=list(raw.shape),LP_NUM_THREADS=os.environ.get('LP_NUM_THREADS'))
                print('RENDER_CHECK',p['task'],stats,flush=True)
                if (raw.shape!=(64,64,3) or raw.dtype!=np.uint8 or raw.mean()<20 or goal.mean()<20
                        or stats['observation_black_fraction']>.25 or stats['goal_black_fraction']>.25):
                    raise RuntimeError('Broken reset rendering; do not export or trust this planning case')
                S,G=observe(raw),observe(goal)
                plan,pi=M.plan(S,G,max_expansions=int(args['max_expansions']))
                p['full_plan_found']=plan is not None; chosen=plan if plan is not None else pi.get('best_plan',[])
                p['commands']=[dict(e=int(e),target=np.asarray(x).tolist()) for e,x in chosen]
                p['states']=[S.tolist()]; p['goal']=G.tolist(); p['expanded']=int(pi.get('expanded',0))
                p['mode']='Corrected offline CPU planning on validated reset images; no execution'
                p['render_validation']=stats
                tile(raw,320).save(a.out/p['observation']); tile(goal,320).save(a.out/p['goal_image'])
            finally:
                pass
        env.close()
        if env.unwrapped._renderer is not None:
            env.unwrapped._renderer.close()
    corrections=[]
    for p in d['plans']:
        initial=np.array(p['states'][0],np.float32); state=initial.copy(); state[:,5]=state[:,5]>.5
        pred=[initial.tolist()]
        for cmd in p['commands']:
            state=M.step(state,[(cmd['e'],np.array(cmd['target'],np.float32))])[0]
            pred.append(state.tolist())
        delta=float(np.abs(np.array(pred)-np.array(p['states'])).max())
        p['states']=pred; p['predicted_goal_test']=bool(M.at_goal(state,np.array(p['goal'])))
        panels=[state_panel(initial,'Observed entity state')]
        for j,cmd in enumerate(p['commands']):
            x=cmd['target']; panels.append(state_panel(np.array(pred[j+1]),f'WM prediction {j+1}',f"E{cmd['e']} -> ({x[0]:.1f}, {x[1]:.1f})"))
        panels.append(state_panel(np.array(p['goal']),'Goal entity state'))
        im=Image.new('RGB',(len(panels)*332+20,540),BG); dr=ImageDraw.Draw(im)
        label='Full plan found' if p['full_plan_found'] else 'Partial fallback; no full plan'
        dr.text((16,8),f"Task {p['task']}: {label} | {p['expanded']} expansions",font=font(23),fill=INK)
        for j,panel in enumerate(panels): im.paste(panel,(10+j*332,57))
        dr.text((16,487),'All selected events shown. Dots are reader / WM image coordinates, not rendered future images.',font=font(14),fill='#536c80')
        dr.text((16,515),'Initial covered bits quantized exactly as Model.plan; no new search or execution.',font=font(14),fill='#536c80')
        im.save(a.out/p['storyboard']); save_json(a.out/f"plan_task{p['task']}.json",p)
        corrections.append(dict(task=p['task'],max_change=delta,predicted_goal_test=p['predicted_goal_test']))
    d['original_export_job_id']=d['job_id']; d['job_id']=os.environ['SLURM_JOB_ID']
    d['finalization']={'description':'Match covered-bit quantization; include all steps; validate task 3/4 reset images with one OSMesa context.',
                       'corrections':corrections,'new_search_tasks':[3,4] if os.environ.get('REPORT_RERENDER')=='1' else [],
                       'no_new_execution':True}
    save_json(a.out/'report_data.json',d)
    (a.out/'report_data.js').write_text('window.REPORT_DATA = '+json.dumps(d,ensure_ascii=False,allow_nan=False)+';\n')
    print('FINALIZED',json.dumps(corrections),flush=True)


if __name__=='__main__': main()
