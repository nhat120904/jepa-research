"""Shared learned event-support trainer. Corruptions are density negatives, not physical failure labels."""
import argparse
import hashlib
import os
import time
from pathlib import Path
import numpy as np
from common import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--events', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--quantile', type=float, default=.02)
    ap.add_argument('--seed', type=int, default=61006)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use sbatch')
    import torch
    from event_support import make_support, features
    from u_wm import Model
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ck = torch.load(a.model, map_location='cpu', weights_only=False)
    M = Model(ck, a.device)
    tr = np.load(a.events / 'events_train.npz')
    va = np.load(a.events / 'events_val.npz')
    S = torch.as_tensor(M.sc.norm(M.canonical(tr['before'])), dtype=torch.float32, device=a.device)
    E = torch.as_tensor(tr['e'], dtype=torch.long, device=a.device)
    X = torch.as_tensor(M.sc.norm(tr['target']), dtype=torch.float32, device=a.device)
    support = make_support(M.K, a.width).to(a.device)
    opt = torch.optim.AdamW(support.parameters(), lr=a.lr, weight_decay=1e-4)
    row = torch.arange(512, device=a.device)
    log, t0 = [], time.time()
    for step in range(a.steps):
        idx = torch.as_tensor(rng.integers(len(S), size=512), device=a.device)
        st, ee, xx = S[idx], E[idx], X[idx]
        donor = torch.as_tensor(rng.integers(len(S), size=512), device=a.device)
        neg = S[donor].clone()
        neg[row, ee] = st[row, ee]
        one = torch.as_tensor(rng.random(512) < .5, device=a.device)
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
            item = dict(step=step, loss=float(loss.detach()), minutes=round((time.time()-t0)/60, 2))
            log.append(item)
            print('SUPPORT', item, flush=True)
    support.eval()
    vs = torch.as_tensor(M.sc.norm(M.canonical(va['before'])), dtype=torch.float32, device=a.device)
    ve = torch.as_tensor(va['e'], dtype=torch.long, device=a.device)
    vx = torch.as_tensor(M.sc.norm(va['target']), dtype=torch.float32, device=a.device)
    with torch.no_grad():
        score = support(features(torch, vs, ve, vx)).sigmoid().squeeze(-1).cpu().numpy()
    if any(not np.any(va['e'] == e) for e in range(M.K)):
        raise RuntimeError('Cannot calibrate an actor missing from play validation')
    thresholds = [float(np.quantile(score[va['e'] == e], a.quantile)) for e in range(M.K)]
    retention = [float((score[va['e'] == e] >= thresholds[e]).mean()) for e in range(M.K)]
    parent_hash = hashlib.sha256(a.model.read_bytes()).hexdigest()
    ck['event_support'] = dict(weights=support.state_dict(), width=a.width, thresholds=thresholds,
        training_definition='NCE context corruption; not labeled simulator failures',
        parent_sha256=parent_hash, steps=a.steps, seed=a.seed, quantile=a.quantile)
    a.out.mkdir(parents=True, exist_ok=False)
    torch.save(ck, a.out / 'u_model.pt')
    save_json(a.out / 'support_report.json', dict(training_events=len(E), validation_events=len(ve),
        steps=a.steps, seed=a.seed, width=a.width, lr=a.lr, quantile=a.quantile,
        thresholds=thresholds, validation_event_retention=retention, parent_sha256=parent_hash, log=log))

if __name__ == '__main__':
    main()
