"""Bounded, matched Round-2 co-design continuation of a Round-1 checkpoint.

See docs/CTA_ROUND2_PROTOCOL.md. All numerical work requires a compute node.
The lambda=0 arm receives exactly the same source/WM/baseline update budgets.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import cta_train as ct
from ti_wm.contract import require_compute, select_candidate
from ti_wm.cta import fsq_predictability_nll, goal_scores
from ti_wm.gates import cluster_ratio
from ti_wm.sibling import rank_loss
from ti_wm.wb import Logger


class Recorder:
    def __init__(self, run, cfg):
        self.path = run / 'metrics.jsonl'
        self.logger = Logger(run, 'codesign', cfg, name=f"r2_{cfg['slurm_job']}_lam{cfg['lam']}")

    def log(self, values, step=None):
        with self.path.open('a') as f:
            f.write(json.dumps({'step': step, **values}, default=float) + '\n')
        self.logger.log(values, step=step)


def paired_gap(scores_a, scores_b, labels, roots):
    """Root-cluster CI of the difference in retained gaps, with shared denominator."""
    pick = lambda scores: np.array([select_candidate(row.tolist()) for row in scores])
    rows = np.arange(len(roots))
    delta = labels[rows, pick(scores_a)] - labels[rows, pick(scores_b)]
    den = labels.max(1) - labels[:, 0]
    uniq = np.unique(roots)
    return cluster_ratio(np.array([delta[roots == r].sum() for r in uniq]),
                         np.array([den[roots == r].sum() for r in uniq]))


@torch.inference_mode()
def diagnostics(models, split, goals, device, amp, n):
    """True-prefix decoding is privileged and diagnostic only, never a deployed arm.

    Measures exposure mismatch and label differences hidden by exact code collisions.
    It cannot establish a closed-loop ceiling or prove a unique cause.
    """
    for model in models.values():
        model.eval()
    n = min(n, split.n)
    scores = np.zeros((n, ct.K), np.float32)
    conflicts = pairs = 0
    wm, enc = models['wm'], models['enc']
    for start in range(0, n, 4):
        i = torch.arange(start, min(start + 4, n))
        ctx = split.context(i, device)
        with amp():
            source = enc.fsq.codes_to_indices(enc(ctx, split.future(i, device)))
            mem = wm.encode(ctx, split.actions(i, device))
            predicted = wm.decode(mem)
            teacher = wm.logits(mem, source).argmax(-1)
            s = goal_scores(models['reader'], ctx, enc.fsq.codebook[teacher], goals)
        scores[start:start + len(i)] = s.view(len(i), ct.K).cpu().numpy()
        pred = predicted.view(len(i), ct.K, -1).cpu().numpy()
        labels = split.cov8[start:start + len(i)]
        relevant = np.abs(labels[:, :, None] - labels[:, None, :]) > ct.RANK_MARGIN
        relevant &= np.triu(np.ones((ct.K, ct.K), bool), 1)[None]
        conflicts += int(((pred[:, :, None] == pred[:, None, :]).all(-1) & relevant).sum())
        pairs += int(relevant.sum())
    return {'n_decisions': n, 'privileged_true_prefix': ct.ranking_metrics(
                scores, split.cov8[:n], split.root[:n], ci=False)['retained_gap'],
            'label_distinct_pairs': pairs, 'label_distinct_pairs_with_same_predicted_code': conflicts,
            'collision_fraction_on_label_distinct_pairs': conflicts / pairs if pairs else None}


def codesign(models, train, dev, goals, cfg, device, log, rng, norms, amp):
    ema = copy.deepcopy(models['enc']).eval().requires_grad_(False)
    source_groups = ('codec', 'full', 'direct')
    opt_c, sch_c = ct.optimizer(models, source_groups, cfg['lr3'], cfg['wd'], cfg['steps3'], cfg['warmup_r2'])
    opt_w, sch_w = ct.optimizer(models, ('wm',), cfg['lr3'], cfg['wd'], cfg['wm_steps3'], cfg['warmup_r2'])
    source_steps = wm_steps = total = 0
    while source_steps < cfg['steps3']:
        frozen = copy.deepcopy(models['wm']).eval().requires_grad_(False)
        for name in ('enc', 'reader', 'dec', 'full', 'direct'):
            models[name].train()
        for _ in range(cfg['src_block']):
            i = torch.as_tensor(rng.choice(train.rank_pool, cfg['decisions']))
            ctx, fut, act = train.context(i, device), train.future(i, device), train.actions(i, device)
            cov, goal = train.cov[i].to(device), ct.sample_goals(goals, rng, len(i), device)
            with amp():
                code, scores, l_rank, l_rec, l_sat = ct.codec_losses(models, ctx, fut, goal, cov, norms)
                with torch.no_grad():
                    indices = models['enc'].fsq.codes_to_indices(code.detach())
                    logits = frozen.logits(frozen.encode(ctx, act), indices)
                l_pred = fsq_predictability_nll(code, logits, models['enc'].fsq)
                l_full = rank_loss(models['full'](ctx, fut, goal).float().view(-1, ct.K), cov)
                l_direct = rank_loss(models['direct'](ctx, act, goal).float().view(-1, ct.K), cov)
            loss = l_rank + cfg['rec'] * l_rec + cfg['sat'] * l_sat + cfg['lam'] * l_pred + l_full + l_direct
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite source objective')
            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            ct.clip(models, source_groups)
            opt_c.step()
            sch_c.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), models['enc'].parameters()):
                    target.lerp_(source, 1 - cfg['ema'])
            source_steps += 1
            total += 1
            if source_steps % cfg['log_every'] == 0:
                log.log({'stage': 'source', 'source_steps': source_steps, 'rank': float(l_rank),
                         'rec': float(l_rec), 'sat': float(l_sat), 'categorical_nll_surrogate': float(l_pred),
                         'rank_direct': float(l_direct), 'rank_full': float(l_full)}, step=total)
        del frozen
        models['wm'].train(), models['prior'].train()
        for _ in range(cfg['wm_block']):
            # Uniform over all decisions, independent of task labels.
            i = torch.as_tensor(rng.integers(0, train.n, cfg['wm_decisions']))
            ctx, act = train.context(i, device), train.actions(i, device)
            with torch.no_grad(), amp():
                target = ema.fsq.codes_to_indices(ema(ctx, train.future(i, device)))
            with amp():
                logits, ce_w, ce_p, (contrast, accuracy) = ct.wm_losses(
                    models, ctx, act, target, cfg['contrast_banks'] if cfg['wm_contrast'] else 0)
                loss = ce_w + ce_p + cfg['wm_contrast'] * contrast
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite WM objective')
            opt_w.zero_grad(set_to_none=True)
            loss.backward()
            ct.clip(models, ('wm',))
            opt_w.step()
            sch_w.step()
            wm_steps += 1
            total += 1
            if wm_steps % cfg['log_every'] == 0:
                log.log({'stage': 'wm', 'wm_steps': wm_steps,
                         **ct.wm_log('wm', logits, ce_w, ce_p, target, cfg['m']),
                         'contrast': float(contrast), 'contrast_accuracy': float(accuracy)}, step=total)
        if source_steps % cfg['eval_every'] == 0 or source_steps == cfg['steps3']:
            rep, _ = ct.ladder(models, dev, goals, device, amp, ('code', 'pred', 'pred_soft', 'direct'),
                               n=cfg['eval_n'], ci=False, norms=norms)
            log.log({'stage': 'dev', **ct.brief(rep)}, step=total)
        print(f"source={source_steps}/{cfg['steps3']} wm={wm_steps}/{cfg['wm_steps3']}", flush=True)
    assert wm_steps == cfg['wm_steps3']
    # Export the ONLINE encoder, which is the encoder used to train the reader.
    # Do not silently replace it with EMA. Retarget WM to this fixed encoder next.
    codes = ct.source_indices(models, train, device, amp, batch=cfg['wm_decisions'])
    align_cfg = {**cfg, 'steps2': cfg['align_steps'], 'lr': cfg['lr3'], 'warmup': cfg['warmup_r2']}
    return ct.stage2(models, train, dev, goals, codes, align_cfg, device, log, rng, amp, total)


def main(a):
    require_compute()
    if a.src_block <= 0 or a.source_steps <= 0 or a.source_steps % a.src_block or a.align_steps <= 0:
        raise ValueError('positive source steps must be divisible by source block; alignment must be positive')
    if a.lam < 0 or a.wm_block <= 0 or a.warmup <= 0 or a.eval_every <= 0:
        raise ValueError('lambda must be nonnegative and WM block positive')
    a.run.mkdir(parents=True, exist_ok=True)
    if (a.run / 'train_report.json').exists() or (a.run / 'cta.pt').exists():
        raise FileExistsError('refuse to overwrite a run')
    parent = torch.load(a.parent / 'cta.pt', map_location='cpu')
    cfg = {**parent['config'], 'lam': a.lam, 'steps3': a.source_steps,
           'wm_steps3': a.source_steps // a.src_block * a.wm_block,
           'src_block': a.src_block, 'wm_block': a.wm_block, 'align_steps': a.align_steps,
           'lr3': a.lr, 'warmup_r2': a.warmup, 'eval_every': a.eval_every,
           'eval_n': a.eval_n, 'device': a.device, 'parent': str(a.parent),
           'slurm_job': os.environ['SLURM_JOB_ID'], 'predictability_loss': 'fsq_multilinear_nll',
           'export_encoder': 'online_then_frozen_wm_alignment', 'adapt_steps': 0,
           'n_train': a.n_train, 'n_dev': a.n_dev, 'seed': a.seed, 'samples': a.samples}
    if a.smoke:
        cfg.update(decisions=1, wm_decisions=1, contrast_banks=1, log_every=1)
    report = {'status': 'RUNNING', 'config': cfg}
    log = Recorder(a.run, cfg)
    t0 = time.perf_counter()
    try:
        features = Path(cfg['features'])
        train, dev = ct.Split(features / 'train', a.n_train), ct.Split(features / 'dev', a.n_dev)
        if not set(train.root.tolist()) <= set(range(30250, 31050)):
            raise ValueError('unexpected training roots')
        if not set(dev.root.tolist()) <= set(range(2000, 2100)):
            raise ValueError('unexpected offline dev roots')
        if not a.smoke and (a.n_train is not None or a.n_dev is not None):
            raise ValueError('data caps are smoke-only')
        device = torch.device(a.device)
        amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if device.type == 'cuda' else contextlib.nullcontext
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)
        goals = torch.from_numpy(np.load(features / 'goals.npy')).to(device)
        if a.smoke:
            goals = goals[:1]
        models = ct.build(cfg, device)
        for name, model in models.items():
            model.load_state_dict(parent['state'][name], strict=True)
        norms = parent['norms']
        report['before'] = diagnostics(models, dev, goals, device, amp, a.eval_n)
        # The two arms use exactly the same RNG stream, including goal sampling.
        torch.manual_seed(a.seed)
        step = codesign(models, train, dev, goals, cfg, device, log, rng, norms, amp)
        # Persist before the expensive full ladder; a failed eval remains FAILED.
        pca = {k: parent[k] for k in ('pca_mean', 'pca_basis')}
        checkpoint = {'config': cfg, 'state': {k: v.state_dict() for k, v in models.items()}, 'norms': norms, **pca}
        torch.save(checkpoint, a.run / 'cta.pt')
        # Ensure compatibility with the existing inference state layouts.
        restored = torch.load(a.run / 'cta.pt', map_location='cpu')
        for name in models:
            models[name].load_state_dict(restored['state'][name], strict=True)
        del restored, checkpoint
        torch.manual_seed(a.seed + 10000)
        ladder, scores = ct.ladder(models, dev, goals, device, amp, ct.TIERS, samples=a.samples, norms=norms)
        report['ladder'] = ladder
        report['after'] = diagnostics(models, dev, goals, device, amp, a.eval_n)
        np.savez(a.run / 'dev_scores.npz', root=dev.root, decision=np.arange(dev.n), cov8=dev.cov8, **scores)
        if not a.smoke:
            old = np.load(a.parent / 'dev_scores.npz')
            for key, expected in (('root', dev.root), ('decision', np.arange(dev.n)), ('cov8', dev.cov8)):
                if not np.array_equal(old[key], expected):
                    raise ValueError(f'parent dev alignment mismatch: {key}')
            report['paired_gap_vs_parent'] = {tier: paired_gap(scores[tier], old[tier], dev.cov8, dev.root)
                                              for tier in ct.TIERS}
            report['paired_gap_vs_matched_direct'] = {tier: paired_gap(scores[tier], scores['direct'], dev.cov8, dev.root)
                                                       for tier in ('pred', 'pred_soft')}
        with (a.parent / 'cta.pt').open('rb') as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
            report['parent_sha256'] = digest.hexdigest()
        report['updates'] = {'source_reader_full_direct': a.source_steps,
                             'wm_prior': cfg['wm_steps3'] + a.align_steps}
        log.log({'stage': 'final', **ct.brief(ladder)}, step=step)
        report.update(status='SMOKE_OK' if a.smoke else 'DONE', seconds=time.perf_counter() - t0)
        print(json.dumps(report, indent=2, default=float), flush=True)
    except Exception:
        report.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        (a.run / 'train_report.json').write_text(json.dumps(report, indent=2, default=float))
        log.logger.finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--lam', type=float, required=True)
    p.add_argument('--source-steps', type=int, default=3000)
    p.add_argument('--src-block', type=int, default=200)
    p.add_argument('--wm-block', type=int, default=400)
    p.add_argument('--align-steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--eval-n', type=int, default=600)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--n-train', type=int)
    p.add_argument('--n-dev', type=int)
    p.add_argument('--smoke', action='store_true')
    main(p.parse_args())
