"""Export saved SAM2 proposals and actual event labels, on a Slurm CPU node.

No model inference, simulator, privileged state, or changes to the method.
"""
import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = [(87, 182, 255), (255, 208, 83), (255, 103, 133), (147, 231, 140),
          (199, 146, 255), (255, 163, 84), (105, 229, 218), (234, 153, 215)]


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Submit through sbatch')
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except OSError:
        font = small = ImageFont.load_default()
    cases = []
    configs = [
        ('cube', 'visual-cube-triple-play-v0', 'uident_57366_cf', 'segfull_cube'),
        ('puzzle', 'visual-puzzle-4x5-play-v0', 'uident_57376_full2', 'segfull_puzzle'),
    ]
    for family, env, run, seg_run in configs:
        base = a.root / run / env
        obs = np.load(a.root / 'cache' / env / 'val_observations.npy', mmap_mode='r')
        seg_path = a.root / seg_run / 'segments_val.npz'
        # Load only cached segmentation members needed for selected frames.
        with np.load(seg_path) as z:
            seg_t, seg_mask, sampled = z['t'], z['mask'], z['frames']
        order = np.argsort(seg_t, kind='stable')
        sorted_t = seg_t[order]
        variants = [('sam2_teacher', base / 'front', base / 'events')]
        variants += [('reader_clean', base / 'self1000/front', base / 'self1000/events_ref')]
        if family == 'cube':
            variants += [('reader_excluded', base / 'self1000/front', base / 'self1000/events_ref')]
        else:
            variants = variants[1:]  # One puzzle example of a multi-entity appearance event.
        for mode, front, event_dir in variants:
            disc = json.loads((front / 'discover.json').read_text())
            report = json.loads((event_dir / 'report.json').read_text())
            with np.load(event_dir / 'events_val.npz') as z:
                ev = {k: z[k] for k in z.files}
            duration = ev['t'] - ev['t_start']
            pool = (ev['t'] < 30 * 1001) & (duration >= 20) & (duration <= 350)
            pool &= ev['knock'] if mode == 'reader_excluded' else ~ev['knock']
            if family == 'puzzle':
                changes = np.abs(ev['after'][..., 2:5] - ev['before'][..., 2:5]).max(-1) > report['thr_app']
                pool &= changes.sum(1) >= 3
            idx = np.flatnonzero(pool)
            if not len(idx):
                raise RuntimeError(f'No requested example: {family} {mode}')
            i = int(idx[0])
            before, after = ev['before'][i], ev['after'][i]
            acted, ts, te = int(ev['e'][i]), int(ev['t_start'][i]), int(ev['t'][i])
            coarse = int(ev.get('t_coarse', ev['t'])[i])
            # Current data episodes contain 1001 observations; do not cross an episode boundary.
            episode = int(ev['episode'][i])
            lo, hi = max(episode * 1001, ts - 35), min((episode + 1) * 1001 - 1, max(te, coarse) + 35)
            frames = sampled[(sampled >= lo) & (sampled <= hi)]
            if not len(frames):
                raise RuntimeError('Selected interval has no stored SAM2 frames')
            with np.load(front / 'entities_val.npz') as z:
                positions, appearances, area = z['pos'][frames], z['app'][frames], z['area'][frames]
            dp = np.linalg.norm(after[:, :2] - before[:, :2], axis=-1)
            da = np.abs(after[:, 2:5] - before[:, 2:5]).max(-1)
            changed = np.flatnonzero((dp > report['tol_pos']) | (da > report['thr_app']))
            name = f'{family}_{mode}'
            metadata = dict(name=name, env=env, split='val', source_front=str(front),
                            source_events=str(event_dir), source_sam2=str(seg_path), event_index=i,
                            episode=episode, acted_entity=acted, event_start=ts, refined_end=te,
                            coarse_end=coarse, frame_range=[int(frames[0]), int(frames[-1])],
                            changed_entities=changed.tolist(), excluded_from_skill=bool(ev['knock'][i]),
                            contaminated=bool(ev.get('contaminated', np.zeros(len(ev['t']), bool))[i]),
                            before=before.tolist(), after=after.tolist(),
                            selection='First validation event satisfying the displayed category and duration bounds',
                            note='Recorded offline play, not a policy rollout. Mask colours identify proposals within a frame, not persistent objects. Entity colours identify inferred IDs. Playback 2 sampled frames per second, sample stride 5; event endpoints are saved labels, not ground truth.')

            def panel(t, j):
                raw = np.asarray(obs[t])
                rgb = raw.astype(np.float32).copy()
                sl = order[np.searchsorted(sorted_t, t, 'left'):np.searchsorted(sorted_t, t, 'right')]
                masks = np.unpackbits(seg_mask[sl], axis=-1)[..., :64].astype(bool)
                # Draw saved, filtered SAM2 proposals; proposals may overlap and split an object into faces.
                for m, mask in enumerate(masks):
                    col = np.asarray(COLORS[m % len(COLORS)], np.float32)
                    rgb[mask] = .45 * rgb[mask] + .55 * col
                    interior = mask.copy()
                    interior[1:, :] &= mask[:-1, :]; interior[:-1, :] &= mask[1:, :]
                    interior[:, 1:] &= mask[:, :-1]; interior[:, :-1] &= mask[:, 1:]
                    rgb[mask & ~interior] = col
                canvas = Image.new('RGB', (1008, 624), '#141821')
                d = ImageDraw.Draw(canvas)
                d.text((16, 10), f'{family.upper()} | {mode} | validation episode {episode}, event {i}', font=font, fill='white')
                titles = ['Original observation', f'SAM2 proposals: {len(masks)} masks',
                          'SAM2 + rules: entity state' if mode == 'sam2_teacher' else 'Learned reader: cached entity state']
                for x, im, title in zip((16, 344, 672), (raw, np.clip(rgb, 0, 255).astype(np.uint8), raw), titles):
                    canvas.paste(Image.fromarray(im).resize((320, 320), Image.Resampling.NEAREST), (x, 64))
                    d.text((x, 42), title, font=small, fill='#ccd7e8')
                for k, p in enumerate(positions[j]):
                    if area[j, k] <= 0:
                        continue
                    x, y = 672 + float(p[0]) * 5, 64 + float(p[1]) * 5
                    col = COLORS[k % len(COLORS)]
                    r = 8 if k == acted else 4
                    d.ellipse((x-r, y-r, x+r, y+r), outline=col, width=3 if k == acted else 2)
                    d.text((x+5, y-13), f'E{k}', font=small, fill=col, stroke_width=1, stroke_fill='black')
                phase = 'BEFORE' if t < ts else ('EVENT INTERVAL' if t <= te else 'AFTER')
                d.text((16, 395), f'frame {t} | {phase} | acted E{acted} | changed IDs {changed.tolist()}', font=font, fill='#ffcf89')
                d.text((16, 420), f'Saved start {ts}; refined end {te}; coarse end {coarse}; excluded from skill: {bool(ev["knock"][i])}', font=small, fill='#c5d2e5')
                # The temporal endpoints are labelled, not predictions inferred by this viewer.
                x0, x1, yy = 16, 992, 455
                d.line((x0, yy, x1, yy), fill='#506077', width=3)
                scale = lambda f: x0 + (int(f)-lo) / max(1, hi-lo) * (x1-x0)
                d.line((scale(ts), yy, scale(te), yy), fill='#91c7ff', width=8)
                d.line((scale(t), yy-9, scale(t), yy+9), fill='white', width=2)
                d.text((16, 472), 'Before -> after: saved event labels. Position in pixels; RGB in [0,1].', font=small, fill='#c5d2e5')
                ids = list(dict.fromkeys([acted] + changed.tolist()))[:5]
                for row, k in enumerate(ids):
                    b, f = before[k], after[k]
                    rgb_before = ','.join(f'{float(v):.2f}' for v in b[2:5])
                    rgb_after = ','.join(f'{float(v):.2f}' for v in f[2:5])
                    txt = f'E{k}: pos ({b[0]:.1f},{b[1]:.1f}) -> ({f[0]:.1f},{f[1]:.1f}); RGB [{rgb_before}] -> [{rgb_after}]'
                    d.text((16, 493 + row*19), txt, font=small, fill=COLORS[k % len(COLORS)])
                d.text((16, 600), 'Offline data demo | 2 sampled frames/sec, stride 5 | inferred masks/IDs; no simulator state', font=small, fill='#9facbf')
                return canvas

            with imageio.get_writer(str(a.out / f'{name}.mp4'), fps=2, codec='libx264', macro_block_size=1) as writer:
                for j, t in enumerate(frames):
                    im = panel(int(t), j)
                    writer.append_data(np.asarray(im))
            picks = [0, int(np.argmin(np.abs(frames-ts))), int(np.argmin(np.abs(frames-(ts+te)/2))), int(np.argmin(np.abs(frames-(coarse+5)))), len(frames)-1]
            sheet = Image.new('RGB', (1008, 624*len(picks)), '#141821')
            for row, j in enumerate(picks):
                sheet.paste(panel(int(frames[j]), j), (0, row*624))
            sheet.save(a.out / f'{name}_storyboard.png')
            panel(int(frames[picks[3]]), picks[3]).save(a.out / f'{name}_after.png')
            (a.out / f'{name}.json').write_text(json.dumps(metadata, indent=2))
            cases.append({k: metadata[k] for k in ('name','env','event_index','episode','acted_entity','event_start','refined_end','coarse_end','changed_entities','excluded_from_skill')})
            print('EXPORTED', json.dumps(cases[-1]), flush=True)
    (a.out / 'manifest.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'], cases=cases,
        purpose='Explain frozen SAM2 proposals, rule-derived identities, learned reader states, and actual event labels; no new benchmark or training.'), indent=2))


if __name__ == '__main__':
    main()
