"""Export a report pack from cached data and fixed checkpoints on a CPU worker.

No training; no privileged state is provided to reader, WM, planner or skill.
Existing closed-loop videos are annotated, not rerun. New tasks 3/4 use reset
images only, with the same seed formula and search budget as the reference.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = '#f7fafc'
INK = '#142c40'
PALETTE = ['#4a93d2', '#c5a21d', '#d35773']
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def tile(raw, size=384):
    return Image.fromarray(np.asarray(raw, np.uint8)).resize((size, size), Image.Resampling.NEAREST)


def state_panel(state, title, subtitle='', size=320):
    im = Image.new('RGB', (size, size + 90), BG)
    d = ImageDraw.Draw(im)
    d.text((12, 8), title, font=font(18), fill=INK)
    d.text((12, 36), subtitle, font=font(13), fill='#536c80')
    margin, top, extent = 24, 70, size - 48
    for q in (0, 16, 32, 48, 64):
        x = margin + extent * q / 64
        y = top + extent * q / 64
        d.line((x, top, x, top+extent), fill='#dbe4ec')
        d.line((margin, y, margin+extent, y), fill='#dbe4ec')
    for k, s in enumerate(state):
        x, y = margin + extent*s[0]/64, top + extent*s[1]/64
        rgb = tuple(int(np.clip(c, 0, 1)*255) for c in s[2:5])
        r = 10
        d.ellipse((x-r, y-r, x+r, y+r), fill=rgb if s[5] <= .5 else BG,
                  outline=INK, width=2)
        d.text((x+12, y-12), f'E{k}' + (' covered' if s[5] > .5 else ''), font=font(12), fill=INK)
    d.text((12, size+55), 'Entity coordinates in the 64 x 64 image', font=font(12), fill='#536c80')
    return im


def export_events(root, out):
    import imageio.v2 as imageio
    configs = [
        ('cube', 'cube_reader_clean.json', 'Cube: move red E2 across the table'),
        ('puzzle', 'puzzle_reader_clean.json', 'Puzzle: appearance changes around inferred E5'),
        ('excluded', 'cube_reader_excluded.json', 'Cube: segment flagged by the contamination filter'),
    ]
    results = []
    for slug, meta_file, title in configs:
        meta = json.loads((out/'inputs'/meta_file).read_text())
        obs = np.load(root/'cache'/meta['env']/'val_observations.npy', mmap_mode='r')
        ts, te, coarse = meta['event_start'], meta['refined_end'], meta['coarse_end']
        ep = meta['episode']; lo = max(ep*1001, ts-10)
        hi = min((ep+1)*1001-1, max(te, coarse)+10)
        fs = np.unique(np.r_[np.arange(lo, hi+1, 2), ts, te, hi]).astype(int)
        before, after = np.asarray(meta['before']), np.asarray(meta['after'])
        acted = meta['acted_entity']; ids = meta['changed_entities']
        with np.load(Path(meta['source_front'])/'entities_val.npz') as z:
            positions = z['pos'][fs]
        def annotated_raw(t, pos=None, size=384):
            im = tile(obs[t], size)
            d = ImageDraw.Draw(im)
            if pos is not None:
                for k in ids:
                    x, y = pos[k, :2]*size/64
                    col = '#ffffff'
                    d.ellipse((x-9,y-9,x+9,y+9), outline=col, width=2)
                    d.text((x+10,y-14), f'E{k}', font=font(14), fill='white', stroke_width=2, stroke_fill='black')
            return im
        sheet = Image.new('RGB', (1200, 550), BG); d = ImageDraw.Draw(sheet)
        d.text((16,12), title, font=font(24), fill=INK)
        picks = [meta['frame_range'][0], (ts+te)//2, min((ep+1)*1001-1, max(te,coarse)+6)]
        for j, (t, label) in enumerate(zip(picks, ['Before', 'During', 'After'])):
            x = 16+j*396
            sheet.paste(annotated_raw(t, before if j==0 else (after if j==2 else None)), (x,76))
            d.text((x,48), f'{label} | source frame {t}', font=font(18), fill=INK)
        d.text((16,475), f'Event label: E{acted}; changed identities: {ids}; interval {ts}-{te}', font=font(18), fill=INK)
        if slug == 'cube':
            note = f'E2 position: ({before[2,0]:.1f}, {before[2,1]:.1f}) -> ({after[2,0]:.1f}, {after[2,1]:.1f})'
        elif slug == 'puzzle':
            note = 'Entity E5 is inferred as the acted identity; IDs 3, 5, 7, 12 change appearance.'
        else:
            note = 'The filter flag is an inferred discrepancy, not proof of a physical collision.'
        d.text((16,507), note, font=font(15), fill='#536c80')
        sheet.save(out/f'event_{slug}_before_after.png')
        with imageio.get_writer(str(out/f'event_{slug}.mp4'), fps=6, codec='libx264', macro_block_size=1) as w:
            for j, t in enumerate(fs):
                canvas=Image.new('RGB',(1040,550),BG); dr=ImageDraw.Draw(canvas)
                phase='Before' if t<ts else ('Event interval' if t<=te else 'After')
                dr.text((18,10), title, font=font(23), fill=INK)
                canvas.paste(annotated_raw(int(t),positions[j],448),(18,55))
                dr.text((494,66), phase, font=font(30), fill=INK)
                dr.text((494,116), f'Frame {t} | event {ts}-{te}', font=font(19), fill=INK)
                dr.text((494,164), f'Acted identity: E{acted}', font=font(21), fill=INK)
                dr.text((494,204), f'Changed identities: {ids}', font=font(18), fill=INK)
                dr.text((494,256),'Saved event label, inferred from images',font=font(16),fill='#536c80')
                dr.text((494,289),'Kept for skill: '+str(not meta['excluded_from_skill']),font=font(18),fill=INK)
                dr.text((494,341),'Offline play data; not a policy rollout',font=font(16),fill='#536c80')
                dr.text((494,374),'Playback is resampled, not real-time',font=font(16),fill='#536c80')
                for x, st, txt in ((494,before[acted],'Start'),(750,after[acted],'Target')):
                    dr.text((x,422),txt,font=font(18),fill=INK)
                    dr.text((x,456),f'pos ({st[0]:.1f}, {st[1]:.1f})',font=font(18),fill=INK)
                dr.text((18,520),'SAM2 / reader + rules create the label; boundaries and IDs are not ground truth.',font=font(15),fill='#536c80')
                w.append_data(np.asarray(canvas))
        results.append(dict(slug=slug, title=title, metadata=meta, video=f'event_{slug}.mp4',
                            storyboard=f'event_{slug}_before_after.png', frames=fs.tolist(), fps=6))
        print('EVENT',slug,flush=True)
    return results


def export_planning(root, out, reference, manifest):
    import torch
    import gymnasium
    import ogbench  # register environments
    from u_reader import make_reader
    from u_wm import Model
    torch.set_num_threads(1)
    args=reference['summary']['args']
    wm_path=Path(args['model'])
    assert hashlib.sha256(wm_path.read_bytes()).hexdigest()==manifest['checkpoint_sha256']['model']
    model=Model(torch.load(wm_path,map_location='cpu',weights_only=False),'cpu')
    rk=torch.load(args['reader'],map_location='cpu',weights_only=False)
    reader=make_reader(rk['K'],agent=rk.get('agent',False)).eval()
    reader.load_state_dict(rk['reader'])
    def observe(raw):
        with torch.no_grad():
            s,_=reader(torch.as_tensor(raw).permute(2,0,1)[None].float()/255)
        s=s[0].numpy(); s[:,5]=1/(1+np.exp(-s[:,5])); return s
    results=[]
    reset_env=None
    for task in range(1,6):
        tracefile=out/'inputs'/f'task{task}_ep0.json'
        if tracefile.exists():
            data=json.loads(tracefile.read_text()); p=data['planner_trace'][0]
            S=np.asarray(p['reader_state'],np.float32); G=np.asarray(p['goal_reader_state'],np.float32)
            plan=[(int(e['e']),np.asarray(e['target'],np.float32)) for e in p['events']]
            found=p['found']; expanded=p['expanded']; mode='Recorded first plan from CPU closed-loop replay 57497'
            initial=Image.open(root/'demo_unified_cube_57497'/f'task{task}_ep0_initial.png')
            initial.crop((16,72,336,392)).save(out/f'task{task}_initial.png')
            initial.crop((368,72,688,392)).save(out/f'task{task}_goal.png')
        else:
            if reset_env is None: reset_env=gymnasium.make(args['env'])
            env=reset_env
            try:
                raw,info=env.reset(seed=int(args['seed'])*10000+task*100,
                                   options=dict(task_id=task,render_goal=False))
                raw=np.array(raw,copy=True); goal=np.array(info['goal'],copy=True)
                if (raw.mean()<20 or goal.mean()<20 or (raw==0).all(-1).mean()>.25
                        or (goal==0).all(-1).mean()>.25):
                    raise RuntimeError('Broken reset rendering; do not use this planning demo')
                S,G=observe(raw),observe(goal)
                tile(raw,320).save(out/f'task{task}_initial.png'); tile(goal,320).save(out/f'task{task}_goal.png')
                pl,pi=model.plan(S,G,max_expansions=int(args['max_expansions']))
                found=pl is not None; expanded=pi.get('expanded',0)
                plan=pl if pl is not None else pi.get('best_plan',[])
                mode='New offline CPU planning on reset images only; no execution'
            finally:
                pass
        preds=[S.tolist()]; state=S.copy(); state[:,5]=state[:,5]>.5; cmds=[]
        for e,x in plan:
            state=model.step(state,[(e,x)])[0]
            preds.append(state.tolist()); cmds.append(dict(e=e,target=x.tolist()))
        rec=dict(task=task,episode=0,mode=mode,full_plan_found=bool(found),expanded=int(expanded),
                 commands=cmds,states=preds,goal=G.tolist(),predicted_goal_test=bool(model.at_goal(state,G)),
                 observation=f'task{task}_initial.png',goal_image=f'task{task}_goal.png',
                 note='Predicted states are entity diagrams, not rendered future observations. Planning-only is not closed-loop success.')
        shown=min(len(plan),6)
        panels=[state_panel(S,'Observed entity state')]
        for j in range(shown):
            e,x=plan[j]
            panels.append(state_panel(np.asarray(preds[j+1]),f'WM prediction {j+1}',f'E{e} -> ({x[0]:.1f}, {x[1]:.1f})'))
        panels.append(state_panel(G,'Goal entity state'))
        im=Image.new('RGB',(len(panels)*332+20,540),BG); d=ImageDraw.Draw(im)
        label='Full plan found' if found else 'Partial fallback; no full plan'
        d.text((16,8),f'Task {task}: {label} | {expanded} expansions',font=font(23),fill=INK)
        for j,panel in enumerate(panels): im.paste(panel,(10+j*332,57))
        d.text((16,487),'Observed / predicted / goal states are explicitly separated; dots are reader / WM coordinates.',font=font(14),fill='#536c80')
        d.text((16,515),'Same checkpoint and search budget as the reference. These figures are not a new success-rate estimate.',font=font(14),fill='#536c80')
        rec['storyboard']=f'plan_task{task}.png'; im.save(out/rec['storyboard'])
        save_json(out/f'plan_task{task}.json',rec); results.append(rec)
        print('PLAN',task,found,len(plan),expanded,flush=True)
    if reset_env is not None:
        reset_env.close()
        if reset_env.unwrapped._renderer is not None: reset_env.unwrapped._renderer.close()
    return results


def export_rollouts(root,out):
    import imageio.v2 as imageio
    results=[]
    for task in (1,2,5):
        data=json.loads((out/'inputs'/f'task{task}_ep0.json').read_text())
        plans=data['planner_trace']; ep=data['cpu_demo_episode']; steps=ep['steps']
        src=root/'demo_unified_cube_57497'/f'task{task}_ep0.mp4'
        frames=0
        with imageio.get_reader(str(src)) as video, imageio.get_writer(str(out/f'rollout_task{task}.mp4'),
                fps=5,codec='libx264',macro_block_size=1) as writer:
            for j,raw in enumerate(video):
                step=min(4*j,steps)
                pidx=max([i for i,p in enumerate(plans) if p['step']<=step],default=0)
                p=plans[pidx]
                im=Image.new('RGB',(704,532),BG); im.paste(Image.fromarray(raw),(0,0)); d=ImageDraw.Draw(im)
                if p['events']:
                    cmd=p['events'][0]; x=cmd['target']
                    text=f"Selected event: E{cmd['e']} -> ({x[0]:.1f}, {x[1]:.1f})"
                else: text='No selected event in the current plan'
                d.text((16,457),text,font=font(19),fill=INK)
                d.text((16,490),f"Planner call {pidx+1} | full plan: {p['found']} | CPU replay, not benchmark",font=font(14),fill='#536c80')
                writer.append_data(np.asarray(im)); frames+=1
                if j==0: im.save(out/f'rollout_task{task}_poster.png')
            im.save(out/f'rollout_task{task}_final.png')
        ended=[]
        for ev in ep['events']:
            error=float(np.linalg.norm(np.asarray(ev['final'])[:2]-np.asarray(ev['x'])[:2]))
            ended.append(dict(e=ev['e'],target=ev['x'],reader_final=ev['final'],error_px=round(error,2),
                              steps=ev['steps'],others_changed=ev['others_changed']))
        rec=dict(task=task,episode=0,success=ep['success'],steps=steps,replans=ep['replans'],timeouts=ep['timeouts'],
                 reference_success=data['reference_episode']['success'],reference_steps=data['reference_episode']['steps'],
                 plans=[{k:p[k] for k in ('step','found','expanded','events')} for p in plans],event_end_diagnostics=ended,
                 video=f'rollout_task{task}.mp4',poster=f'rollout_task{task}_poster.png',frames=frames,fps=5,
                 note='Recorded CPU replay; privileged trace is excluded from the report payload. Event end is not necessarily target arrival.')
        save_json(out/f'rollout_task{task}.json',rec); results.append(rec)
        print('ROLLOUT',task,frames,flush=True)
    return results


def main():
    if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('Use sbatch on a CPU worker')
    os.environ.setdefault('LP_NUM_THREADS','1')
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    src=a.root/'demo_unified_cube_57497'
    ref=json.loads((a.out/'inputs'/'reference.json').read_text()); man=json.loads((src/'manifest.json').read_text())
    events=export_events(a.root,a.out)
    plans=export_planning(a.root,a.out,ref,man)
    rollouts=export_rollouts(a.root,a.out)
    totals=[]
    for task in range(1,6):
        eps=[e for e in ref['episodes'] if e['task']==task]
        totals.append(dict(task=task,success=sum(e['success'] for e in eps),episodes=len(eps)))
    summary=dict(job_id=os.environ['SLURM_JOB_ID'],checkpoint_sha256=man['checkpoint_sha256'],reference=man['reference'],
                 reference_results=totals,events=events,plans=plans,rollouts=rollouts,
                 limitations=['Selected development examples, not representative success-rate estimates.',
                 'CPU replay differs from the GPU reference on task 2.',
                 'New tasks 3/4 contain planning only; no new closed-loop rollout.',
                 'Pseudo-label boundaries, entity IDs and reader-space errors are not physical ground truth.',
                 'WM prediction is not supplied as whole-scene input to the skill.'])
    save_json(a.out/'report_data.json',summary)
    (a.out/'report_data.js').write_text('window.REPORT_DATA = '+json.dumps(summary,ensure_ascii=False,allow_nan=False)+';\n')
    print('REPORT_DONE',a.out,flush=True)


if __name__=='__main__': main()
