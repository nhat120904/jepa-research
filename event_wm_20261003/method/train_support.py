#!/usr/bin/env python3
"""Component 13 (method/README.md): train the event-support prior on TRAIN events and attach it to the model checkpoint.
Port of docs/generic_state_20261007/base_source/train_event_support.py (same NCE recipe: positives = play events, negatives =
the same event in a corrupted context; per-entity threshold = --quantile of the scores of genuine VAL events), with the
state length D from the checkpoint. Corruptions are density negatives, not physical failure labels. In the method it runs
between world_model.py --stage wm and --stage h, so the cost-to-go is trained with the candidates the search will use.
Entities never acted on in VAL get the lowest TRAIN quantile instead of an error (u_wm aborted: its K were all actable).
"""
import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--events', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--quantile', type=float, default=.02)
    ap.add_argument('--kind', choices=('joint', 'pair'), default='joint',
                    help='joint: one logit for (state, event) (NCE with corrupted contexts); pair: one logit per other entity '
                         '(event_support.make_pair_support), the event cost is the summed evidence against compatibility')
    ap.add_argument('--p-one', type=float, default=.5,
                    help='share of negatives with ONE other entity swapped from a donor event (the rest: every other entity from the '
                         'donor). 1.0 = pairwise compatibility only: a whole donor context also teaches context typicality, which '
                         'gave legal button presses in rarer contexts support ~.1 (scene task 2, 2026-10-11)')
    ap.add_argument('--seed', type=int, default=61006)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if a.device == 'cuda' and 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('runs under sbatch')
    import torch
    from event_support import make_support, features
    from world_model import Model
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ck = torch.load(a.model, map_location='cpu', weights_only=False)
    M = Model(ck, a.device)
    tr = dict(np.load(a.events / 'events_train.npz'))
    va = dict(np.load(a.events / 'events_val.npz'))
    for z in (tr, va):                                   # object events: only events whose acted entity is known before and after
        if 'target_known' in z:
            keep = z['target_known'].astype(bool)
            for k_ in [k_ for k_, v_ in z.items() if isinstance(v_, np.ndarray) and v_.ndim >= 1 and len(v_) == len(keep)]:
                z[k_] = z[k_][keep]
    S = torch.as_tensor(M.sc.norm(M.canonical(tr['before'])), dtype=torch.float32, device=a.device)
    E = torch.as_tensor(tr['e'], dtype=torch.long, device=a.device)
    X = torch.as_tensor(M.sc.norm(tr['target']), dtype=torch.float32, device=a.device)
    if a.kind == 'pair':
        return train_pair(a, torch, M, ck, tr, va, S, E, X, rng)
    support = make_support(M.K, M.D, a.width).to(a.device)
    opt = torch.optim.AdamW(support.parameters(), lr=a.lr, weight_decay=1e-4)
    row = torch.arange(512, device=a.device)
    log, t0 = [], time.time()
    for step in range(a.steps):
        idx = torch.as_tensor(rng.integers(len(S), size=512), device=a.device)
        st, ee, xx = S[idx], E[idx], X[idx]
        donor = torch.as_tensor(rng.integers(len(S), size=512), device=a.device)
        neg = S[donor].clone()
        neg[row, ee] = st[row, ee]
        one = torch.as_tensor(rng.random(512) < a.p_one, device=a.device)
        other = torch.as_tensor(rng.integers(M.K - 1, size=512), device=a.device)
        other += (other >= ee).long()
        ns = st.clone()
        ns[row, other] = neg[row, other]
        neg[one] = ns[one]
        changed = (neg - st).abs().amax((1, 2)) > 1e-5
        pos = support(features(torch, st, ee, xx)).squeeze(-1)
        loss = torch.nn.functional.softplus(-pos).mean()
        if changed.any():
            nlogit = support(features(torch, neg[changed], ee[changed], xx[changed])).squeeze(-1)
            loss = loss + torch.nn.functional.softplus(nlogit).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == a.steps - 1:
            item = dict(step=step, loss=float(loss.detach()), minutes=round((time.time() - t0) / 60, 2))
            log.append(item)
            print('SUPPORT', item, flush=True)
    support.eval()

    def scores(d):
        s = torch.as_tensor(M.sc.norm(M.canonical(d['before'])), dtype=torch.float32, device=a.device)
        e = torch.as_tensor(d['e'], dtype=torch.long, device=a.device)
        x = torch.as_tensor(M.sc.norm(d['target']), dtype=torch.float32, device=a.device)
        with torch.no_grad():
            return np.concatenate([support(features(torch, s[i:i + 4096], e[i:i + 4096], x[i:i + 4096])).sigmoid().squeeze(-1).cpu().numpy()
                                   for i in range(0, len(e), 4096)])

    sv, st_ = scores(va), scores(tr)
    floor = float(np.quantile(st_, a.quantile))
    thresholds = [float(np.quantile(sv[va['e'] == e], a.quantile)) if np.any(va['e'] == e) else floor for e in range(M.K)]
    retention = [float((sv[va['e'] == e] >= thresholds[e]).mean()) if np.any(va['e'] == e) else None for e in range(M.K)]
    parent_hash = hashlib.sha256(a.model.read_bytes()).hexdigest()
    ck['event_support'] = dict(weights=support.state_dict(), width=a.width, thresholds=thresholds,
                               training_definition='NCE context corruption; not labeled simulator failures',
                               parent_sha256=parent_hash, steps=a.steps, seed=a.seed, quantile=a.quantile, p_one=a.p_one)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out / 'model_support.pt')
    save_json(a.out / 'support_report.json', dict(training_events=len(E), validation_events=len(va['e']), steps=a.steps, seed=a.seed, p_one=a.p_one,
                                                  width=a.width, lr=a.lr, quantile=a.quantile, thresholds=thresholds,
                                                  entities_without_val_events=int(sum(r is None for r in retention)),
                                                  validation_event_retention=retention, parent_sha256=parent_hash, log=log))


def train_pair(a, torch, M, ck, tr, va, S, E, X, rng):
    """PAIRWISE support: per (event, other known entity k) NCE of k's genuine state against k's state in another event."""
    from event_support import known_entities, make_pair_support, pair_cost, pair_logits
    F = torch.nn.functional
    kn_tr = known_entities(M.canonical(tr['before'])) & (tr['before_known'] if 'before_known' in tr else True)
    KN = torch.as_tensor(kn_tr, device=a.device)
    support = make_pair_support(M.K, M.D, a.width).to(a.device)
    opt = torch.optim.AdamW(support.parameters(), lr=a.lr, weight_decay=1e-4)
    B = 512
    row = torch.arange(B, device=a.device)
    log, t0 = [], time.time()
    for step in range(a.steps):
        idx = torch.as_tensor(rng.integers(len(S), size=B), device=a.device)
        donor = torch.as_tensor(rng.integers(len(S), size=B), device=a.device)
        st, ee, xx, kn = S[idx], E[idx], X[idx], KN[idx]
        neg = S[donor].clone()
        neg[row, ee] = st[row, ee]
        not_e = ~F.one_hot(ee, M.K).bool()
        vp = kn & not_e
        vn = KN[donor] & not_e & ((neg - st).abs().amax(-1) > 1e-5)
        lp, ln = pair_logits(support, torch, st, ee, xx), pair_logits(support, torch, neg, ee, xx)
        loss = F.softplus(-lp[vp]).mean()
        if vn.any():
            loss = loss + F.softplus(ln[vn]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == a.steps - 1:
            item = dict(step=step, loss=float(loss.detach()), minutes=round((time.time() - t0) / 60, 2))
            log.append(item)
            print('PAIR SUPPORT', item, flush=True)
    support.eval()

    def costs(d, swap):
        Sd = M.canonical(d['before'])
        kn = known_entities(Sd) & (d['before_known'] if 'before_known' in d else True)
        s = torch.as_tensor(M.sc.norm(Sd), dtype=torch.float32, device=a.device)
        e = torch.as_tensor(d['e'], dtype=torch.long, device=a.device)
        x = torch.as_tensor(M.sc.norm(d['target']), dtype=torch.float32, device=a.device)
        k_ = torch.as_tensor(kn, device=a.device)
        if swap:                                                                 # one other entity from a random event
            g = np.random.default_rng(1)
            oth = g.integers(0, M.K - 1, len(d['e'])); oth += oth >= d['e']
            dn = g.integers(0, len(d['e']), len(d['e']))
            r_ = torch.arange(len(d['e']), device=a.device)
            o_, n_ = torch.as_tensor(oth, device=a.device), torch.as_tensor(dn, device=a.device)
            s = s.clone(); s[r_, o_] = s[n_, o_]
            k_ = k_.clone(); k_[r_, o_] = k_[n_, o_]
        with torch.no_grad():
            return np.concatenate([pair_cost(torch, pair_logits(support, torch, s[i:i + 2048], e[i:i + 2048], x[i:i + 2048]), k_[i:i + 2048]).cpu().numpy()
                                   for i in range(0, len(d['e']), 2048)])

    cg, cs = costs(va, False), costs(va, True)
    report = dict(kind='pair', training_events=len(E), validation_events=len(va['e']), steps=a.steps, seed=a.seed, width=a.width, lr=a.lr,
                  val_cost_genuine_q50_90_99=np.round(np.quantile(cg, [.5, .9, .99]), 3).tolist(),
                  val_cost_one_swapped_q50_90_99=np.round(np.quantile(cs, [.5, .9, .99]), 3).tolist(), log=log)
    print({k: v for k, v in report.items() if k != 'log'}, flush=True)
    parent_hash = hashlib.sha256(a.model.read_bytes()).hexdigest()
    ck['event_support'] = dict(kind='pair', weights=support.state_dict(), width=a.width, thresholds=[0.0] * M.K,
                               training_definition='pairwise NCE: genuine entity state vs the same entity in another event',
                               parent_sha256=parent_hash, steps=a.steps, seed=a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out / 'model_support.pt')
    save_json(a.out / 'support_report.json', dict(report, parent_sha256=parent_hash))


if __name__ == '__main__':
    main()
