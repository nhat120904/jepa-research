"""Frozen-model prefix/action audit. Privileged scores never enter deployment."""
import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import cta_train as ct
from cta_round2 import paired_gap
from ti_wm.contract import require_compute
from ti_wm.cta import goal_scores


@torch.inference_mode()
def main(a):
    require_compute()
    a.run.mkdir(parents=True, exist_ok=True)
    if (a.run / 'report.json').exists():
        raise FileExistsError(a.run / 'report.json')
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    blob = torch.load(a.train_run / 'cta.pt', map_location='cpu')
    cfg = blob['config']
    if cfg['lam'] != 0:
        raise ValueError('audit is locked to the noncollapsed lambda=0 control')
    device = torch.device(a.device)
    models = ct.build(cfg, device)
    for name, model in models.items():
        model.load_state_dict(blob['state'][name], strict=True)
        model.eval().requires_grad_(False)
    del blob
    split = ct.Split(Path(cfg['features']) / 'dev', a.n)
    if not ((split.root >= 2000) & (split.root <= 2099)).all():
        raise ValueError('unexpected diagnostic roots')
    goals = torch.from_numpy(np.load(Path(cfg['features']) / 'goals.npy')).to(device)
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if device.type == 'cuda' else contextlib.nullcontext
    enc, wm, prior, reader = (models[k] for k in ('enc', 'wm', 'prior', 'reader'))
    names = ('source', 'wm_free', 'prior_free', 'wm_true_prefix', 'prior_true_prefix', 'wm_wrong_action_true_prefix')
    scores = {k: np.zeros((split.n, 8), np.float32) for k in names}
    ce = {k: 0. for k in names[3:]}
    token_count = 0
    collisions = {'wm_free': 0, 'prior_free': 0}
    relevant_pairs = 0
    per_token_wrong = {k: np.zeros(cfg['m'], np.int64) for k in names if k != 'source'}
    for start in range(0, split.n, 4):
        i = torch.arange(start, min(start + 4, split.n))
        ctx, fut, act = split.context(i, device), split.future(i, device), split.actions(i, device)
        with amp():
            src = enc(ctx, fut)
            idx = enc.fsq.codes_to_indices(src)
            memory, context_memory = wm.encode(ctx, act), prior.encode(ctx)
            wrong_act = act.view(len(i), 8, *act.shape[1:]).roll(1, dims=1).flatten(0, 1)
            logits = {'wm_true_prefix': wm.logits(memory, idx),
                      'prior_true_prefix': prior.logits(context_memory, idx),
                      'wm_wrong_action_true_prefix': wm.logits(wm.encode(ctx, wrong_act), idx)}
            predicted = {'wm_free': wm.decode(memory), 'prior_free': prior.decode(context_memory),
                         **{k: v.argmax(-1) for k, v in logits.items()}}
            scores['source'][start:start + len(i)] = goal_scores(reader, ctx, src, goals).view(len(i), 8).cpu().numpy()
            for name, ids in predicted.items():
                scores[name][start:start + len(i)] = goal_scores(reader, ctx, enc.fsq.codebook[ids], goals).view(len(i), 8).cpu().numpy()
                per_token_wrong[name] += (ids != idx).sum(0).cpu().numpy()
            for name, logit in logits.items():
                ce[name] += float(F.cross_entropy(logit.float().flatten(0, 1), idx.flatten(), reduction='sum'))
        labels = split.cov8[start:start + len(i)]
        relevant = np.abs(labels[:, :, None] - labels[:, None, :]) > ct.RANK_MARGIN
        relevant &= np.triu(np.ones((8, 8), bool), 1)[None]
        relevant_pairs += int(relevant.sum())
        for name in collisions:
            ids = predicted[name].view(len(i), 8, -1).cpu().numpy()
            collisions[name] += int(((ids[:, :, None] == ids[:, None, :]).all(-1) & relevant).sum())
        token_count += idx.numel()
        if start % 100 == 0:
            print(f'audited {start + len(i)}/{split.n} decisions', flush=True)
    report = {'status': 'DONE', 'train_run': str(a.train_run), 'n': split.n,
        'n_roots': len(np.unique(split.root)), 'panel': 'fixed first 600 offline dev decisions, same panel as Round 2 diagnostic',
        'scope': 'no training; true-prefix variants use actual future information and are diagnostic only',
        'ladder': {}, 'teacher_forced_ce_nats': {k: v / token_count for k, v in ce.items()},
        'token_error_by_position': {k: (v / (split.n * 8)).tolist() for k, v in per_token_wrong.items()},
        'collision_on_label_distinct_pairs': {k: v / relevant_pairs if relevant_pairs else None for k, v in collisions.items()}}
    for name in names:
        metrics = ct.ranking_metrics(scores[name], split.cov8, split.root, ci=True)
        metrics.pop('chosen', None)
        report['ladder'][name] = metrics
    report['paired_gap'] = {f'{x}-minus-{y}': paired_gap(scores[x], scores[y], split.cov8, split.root)
        for x, y in (('wm_true_prefix', 'prior_true_prefix'),
                     ('wm_true_prefix', 'wm_wrong_action_true_prefix'), ('wm_free', 'prior_free'))}
    np.savez(a.run / 'scores.npz', root=split.root, cov8=split.cov8, **scores)
    (a.run / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train-run', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--n', type=int, default=600)
    p.add_argument('--device', default='cuda')
    main(p.parse_args())
