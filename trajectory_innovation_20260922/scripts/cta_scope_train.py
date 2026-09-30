"""Actual-future endpoint versus trajectory coding; fixed offline experiment."""
import argparse
import contextlib
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from cta_train import Split
from ti_wm.contract import require_compute
from ti_wm.cta import goal_scores, saturation_penalty, vocab_size
from ti_wm.cta_eval import bank_distinct, perplexity, ranking_metrics
from ti_wm.cta_scope import build, future_evidence, shared_initial_hash
from ti_wm.sibling import rank_loss
from ti_wm.wb import Logger


def endpoint_norm(split):
    rng = np.random.default_rng(42000)
    idx = np.sort(rng.choice(split.n, min(256, split.n), replace=False))
    cur, end = split.tok['cur'][idx].float(), split.tok['end'][idx].float()
    return max(float(((end - cur[:, None]) ** 2).mean()), 1e-8)


@torch.inference_mode()
def evaluate(models, split, goals, device, amp, path, norm):
    for model in models.values():
        model.eval()
    scores = np.zeros((split.n, 8), np.float32)
    indices, squared_errors = [], []
    start_time = time.perf_counter()
    for start in range(0, split.n, 4):
        i = torch.arange(start, min(start + 4, split.n))
        ctx = split.context(i, device)
        fut = future_evidence(split.future(i, device), path)
        with amp():
            code = models['enc'](ctx, fut)
            score = goal_scores(models['reader'], ctx, code, goals)
            reconstruction = models['dec'](ctx, code)
        scores[start:start + len(i)] = score.view(len(i), 8).cpu().numpy()
        indices.append(models['enc'].fsq.codes_to_indices(code).view(len(i), 8, -1).cpu().numpy())
        squared_errors.extend(((reconstruction - fut['end'].float()) ** 2).mean((1, 2)).cpu().tolist())
    idx = np.concatenate(indices)
    report = ranking_metrics(scores, split.cov8, split.root)
    report.pop('chosen')
    report.update(source_perplexity=perplexity(idx.reshape(-1, idx.shape[-1]), vocab_size()),
                  bank_distinct=bank_distinct(idx), endpoint_reconstruction_r2=1 - float(np.mean(squared_errors)) / norm,
                  evaluation_seconds_including_transfer_and_reconstruction=time.perf_counter() - start_time)
    flat = np.ptp(split.cov8, axis=1) <= 1e-3
    report['flat_bank_override_default_fraction'] = float((scores[flat].argmax(1) != 0).mean()) if flat.any() else None
    return report, scores, idx


def main(a):
    require_compute()
    if a.steps <= 0 or a.decisions <= 0:
        raise ValueError('positive steps and batch size required')
    if not a.smoke and (a.steps != 10000 or a.decisions != 16 or a.n_train is not None or a.n_dev is not None):
        raise ValueError('production settings are locked; caps are smoke-only')
    a.run.mkdir(parents=True, exist_ok=True)
    if (a.run / 'train_report.json').exists() or (a.run / 'codec.pt').exists():
        raise FileExistsError(a.run)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k != 'run'}
    cfg.update(m=16, nominal_bits=128, lr=3e-4, wd=.05, warmup=500,
               rec_weight=1., sat_weight=1., sampling='all_decisions_uniform',
               reconstruction_target='endpoint_only', goal_protocol='unchanged_16_goal_images')
    report = {'status': 'RUNNING', 'config': cfg, 'slurm_job': os.environ['SLURM_JOB_ID']}
    logger = Logger(a.run, 'codec_scope', cfg, name=f'scope_s{a.seed}_path{a.path}_{os.environ["SLURM_JOB_ID"]}')
    started = time.perf_counter()
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device(a.device)
        amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if device.type == 'cuda' else contextlib.nullcontext
        train, dev = Split(a.features / 'train', a.n_train), Split(a.features / 'dev', a.n_dev)
        if not ((train.root >= 30250) & (train.root <= 31049)).all() or not ((dev.root >= 2000) & (dev.root <= 2099)).all():
            raise ValueError('unexpected train/dev roots')
        models = build(a.seed, bool(a.path))
        report['shared_initial_sha256'] = shared_initial_hash(models)
        report['parameter_counts'] = {k: sum(p.numel() for p in model.parameters()) for k, model in models.items()}
        models = {k: model.to(device).train() for k, model in models.items()}
        goals = torch.from_numpy(np.load(a.features / 'goals.npy')).to(device)
        if a.smoke:
            goals = goals[:1]
        norm = endpoint_norm(train)
        report.update(endpoint_copy_mse=norm, train_decisions=train.n, dev_decisions=dev.n,
                      dev_roots=int(len(np.unique(dev.root))))
        params = [p for model in models.values() for p in model.parameters()]
        optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'])
        warmup = min(cfg['warmup'], a.steps)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: min(1., (step + 1) / warmup) * .5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps)))
        rng, batch_hash = np.random.default_rng(a.seed + 3_000_000), hashlib.sha256()
        selected_informative = 0
        train_start = time.perf_counter()
        for step in range(a.steps):
            batch = rng.integers(0, train.n, a.decisions, dtype=np.int64)
            goal_ids = rng.integers(0, len(goals), a.decisions, dtype=np.int64)
            batch_hash.update(batch.tobytes())
            batch_hash.update(goal_ids.tobytes())
            i = torch.from_numpy(batch)
            ctx = train.context(i, device)
            fut = future_evidence(train.future(i, device), bool(a.path))
            cov = train.cov[i].to(device)
            goal = goals[torch.as_tensor(goal_ids, device=device)].repeat_interleave(8, 0)
            with amp():
                code, pre = models['enc'](ctx, fut, return_pre=True)
                scores = models['reader'](ctx, code, goal).float().view(-1, 8)
                reconstruction = models['dec'](ctx, code)
                ranking = rank_loss(scores, cov)
                rec = ((reconstruction - fut['end'].float()) ** 2).mean() / norm
                sat = saturation_penalty(pre)
                loss = ranking + rec + sat
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite codec loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step()
            schedule.step()
            selected_informative += int((np.ptp(train.cov8[batch], axis=1) > 1e-3).sum())
            if (step + 1) % 100 == 0 or step + 1 == a.steps:
                values = {'step': step + 1, 'rank': float(ranking), 'endpoint_rec': float(rec),
                          'saturation': float(sat), 'seconds': time.perf_counter() - train_start}
                with (a.run / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(values) + '\n')
                logger.log(values, step=step + 1)
                print(json.dumps(values), flush=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        report.update(training_seconds=time.perf_counter() - train_start,
                      batch_goal_sha256=batch_hash.hexdigest(),
                      sampled_informative_fraction=selected_informative / (a.steps * a.decisions))
        torch.save({'config': cfg, 'state': {k: model.state_dict() for k, model in models.items()},
                    'endpoint_copy_mse': norm}, a.run / 'codec.pt')
        # Fixed last checkpoint; there is no dev-driven checkpoint selection.
        metrics, scores, indices = evaluate(models, dev, goals, device, amp, bool(a.path), norm)
        np.savez(a.run / 'dev_scores.npz', root=dev.root, decision=np.arange(dev.n), cov8=dev.cov8,
                 code=scores, source_indices=indices)
        report.update(status='SMOKE_OK' if a.smoke else 'DONE', metrics=metrics, total_seconds=time.perf_counter() - started)
        logger.summary(report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        report.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        (a.run / 'train_report.json').write_text(json.dumps(report, indent=2))
        logger.finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--path', type=int, choices=(0, 1), required=True)
    p.add_argument('--seed', type=int, choices=(0, 1), required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--decisions', type=int, default=16)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--n-train', type=int)
    p.add_argument('--n-dev', type=int)
    main(p.parse_args())
