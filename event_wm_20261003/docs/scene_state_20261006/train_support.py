"""One bounded offline repair: learn conditional event support, keep WM/h/skill frozen, then closed loop."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from event_support import make_support, features
from u_wm import Model

if 'SLURM_JOB_ID' not in os.environ:
    raise RuntimeError('Use sbatch')
torch.set_num_threads(2)
torch.manual_seed(61006)
rng = np.random.default_rng(61006)
ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
a.out.mkdir()
base = a.root / 'job_57772/wm/u_model.pt'
ck = torch.load(base, map_location='cpu', weights_only=False)
M = Model(ck, 'cpu')
tr = np.load(a.root / 'prep_57771/events/events_train.npz')
va = np.load(a.root / 'prep_57771/events/events_val.npz')
S = torch.as_tensor(M.sc.norm(M.canonical(tr['before'])), dtype=torch.float32)
E = torch.as_tensor(tr['e'], dtype=torch.long)
X = torch.as_tensor(M.sc.norm(tr['target']), dtype=torch.float32)
groups = [np.flatnonzero(tr['e'] == k) for k in range(M.K)]
support = make_support(M.K)
opt = torch.optim.AdamW(support.parameters(), lr=3e-4, weight_decay=1e-4)
log, t0 = [], time.time()
for step in range(6000):
    idx = rng.integers(len(S), size=512)
    st, ee, xx = S[idx], E[idx], X[idx]
    neg = S[rng.integers(len(S), size=512)].clone()
    # Corrupt only context, preserving the acted object and request. Half replace one other object.
    neg[torch.arange(512), ee] = st[torch.arange(512), ee]
    one = torch.as_tensor(rng.random(512) < .5)
    other = torch.as_tensor(rng.integers(M.K - 1, size=512))
    other += (other >= ee).long()
    ns = st.clone()
    ns[torch.arange(512), other] = neg[torch.arange(512), other]
    neg[one] = ns[one]
    changed = (neg - st).abs().amax((1, 2)) > 1e-5
    pos_logits = support(features(torch, st, ee, xx)).squeeze(-1)
    neg_logits = support(features(torch, neg[changed], ee[changed], xx[changed])).squeeze(-1)
    loss = torch.nn.functional.softplus(-pos_logits).mean() + torch.nn.functional.softplus(neg_logits).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 1000 == 0 or step == 5999:
        row = dict(step=step, loss=float(loss.detach()), minutes=round((time.time() - t0) / 60, 2))
        log.append(row)
        print('SUPPORT_TRAIN', row, flush=True)
support.eval()
vs = torch.as_tensor(M.sc.norm(M.canonical(va['before'])), dtype=torch.float32)
ve = torch.as_tensor(va['e'], dtype=torch.long)
vx = torch.as_tensor(M.sc.norm(va['target']), dtype=torch.float32)
with torch.no_grad():
    vscore = support(features(torch, vs, ve, vx)).sigmoid().squeeze(-1).numpy()
threshold = [float(np.quantile(vscore[va['e'] == e], .02)) for e in range(M.K)]
retention = [float((vscore[va['e'] == e] >= threshold[e]).mean()) for e in range(M.K)]
# Diagnostic frequencies from TRAIN; this does not construct task-specific conditions.
discrete_context = []
for e in range(M.K):
    rows = tr['before'][tr['e'] == e]
    for j in range(M.K):
        for c in range(6):
            values = np.unique(np.round(tr['before'][:, j, c], 5))
            if 1 < len(values) <= 8:
                freq = [float(np.mean(np.abs(rows[:, j, c] - v) < 1e-4)) for v in values]
                discrete_context.append(dict(actor=e, entity=j, field=c, values=values.tolist(), frequencies=freq))
ck['event_support'] = dict(weights=support.state_dict(), width=256, thresholds=threshold,
                           training_definition='NCE context corruption; not labeled simulator failures',
                           parent_sha256=hashlib.sha256(base.read_bytes()).hexdigest(), steps=6000, seed=61006)
torch.save(ck, a.out / 'u_model.pt')
N = Model(ck, 'cpu')
dev = json.loads((a.root / 'job_57772/loop_dev/u_closed_loop.json').read_text())
first = []
for ep in dev['episodes']:
    s = np.asarray(ep['read_start'], np.float32)
    event = ep['first_events'][0]
    e, x = event['e'], np.asarray(event['x'], np.float32)
    with torch.no_grad():
        sc = support(features(torch, torch.as_tensor(M.sc.norm(s[None])), torch.tensor([e]),
                              torch.as_tensor(M.sc.norm(x[None])))).sigmoid().item()
    raw, gated = M.step(s, [(e, x)])[0], N.step(s, [(e, x)])[0]
    first.append(dict(task=ep['task'], actor=e, score=sc, threshold=threshold[e],
                      abstained=sc < threshold[e], raw_next=raw.tolist(), supported_next=gated.tolist()))
report = dict(steps=6000, training_events=len(E), thresholds=threshold, validation_event_retention=retention,
              source='TRAIN events only, quantile calibration on play validation', log=log,
              train_discrete_context=discrete_context, failed_run_first_request_diagnostics=first)
(a.out / 'support_report.json').write_text(json.dumps(report, indent=2) + '\n')
print('SUPPORT_REPORT', json.dumps({k: v for k, v in report.items() if k not in ('train_discrete_context', 'log')}), flush=True)
