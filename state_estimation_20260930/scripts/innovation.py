#!/usr/bin/env python3
"""How much of the history-3 rollout error is predictable from the last innovation?

Dataset windows with frames every 5 steps from t-20 to t+25. nu_t = z_t - f(z_t-15,
z_t-10, z_t-5; executed blocks) is the one-step innovation observed when replanning
at t (nu_prev the same one block earlier). Corrections of the 5-block history-3
rollout under the true actions (held-out windows, split by episode):
  post(g)     add g * nu_t to every predicted block
  dist(g)     disturbance model: add g * nu_t inside the autoregressive loop
  ema         post-hoc with the mean of nu_t and nu_prev
  diag        per-horizon, per-dimension gain fitted on training windows
  ridge(l)    per-horizon linear map D x D fitted on training windows
Also reports cos(nu_t, nu_prev) and cos(nu_t, e_1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
K_PAST = 4      # t-20, t-15, t-10, t-5
K_FUT = 5
STRIDE = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model work must run under sbatch")
    import torch

    import stable_worldmodel as swm
    from se import tasks as T
    from se.tasks import episode_column

    dev = "cuda"
    c = T.TASKS[a.task]
    dataset = T.load_dataset(swm, a.task)
    scaler = T.fit_processors(dataset, a.task)["action"]
    model = swm.wm.utils.load_pretrained(c["repo"]).to(dev).eval()
    model.requires_grad_(False)
    tf = T.image_transform()

    col = episode_column(dataset)
    ep_col = np.asarray(dataset.get_col_data(col))
    step = np.asarray(dataset.get_col_data("step_idx"))
    _, inv = np.unique(ep_col, return_inverse=True)
    last = np.zeros(inv.max() + 1, dtype=np.int64)
    np.maximum.at(last, inv, step)
    ok = np.nonzero((step >= K_PAST * STRIDE) & (step + K_FUT * STRIDE <= last[inv]))[0]
    g = np.random.default_rng(a.seed)
    pick = np.sort(ok[g.choice(len(ok), size=min(a.n, len(ok)), replace=False)])
    rows = dataset.get_row_data(pick)
    eps, ts = np.asarray(rows[col]), np.asarray(rows["step_idx"])
    nf = K_PAST + 1 + K_FUT

    def nxt(ctx, u):
        return model.predict(ctx, u)[:, -1]

    def rollout(ctx, u_ctx, u_fut, bias=None):
        emb = list(ctx.unbind(1))
        acts = list(u_ctx.unbind(1)) + list(u_fut.unbind(1))
        h, out = ctx.size(1), []
        for k in range(K_FUT):
            lo = max(0, h + k - 3)
            p = nxt(torch.stack(emb[lo:], 1), torch.stack(acts[lo:h + k], 1))
            if bias is not None:
                p = p + bias
            emb.append(p)
            out.append(p)
        return torch.stack(out, 1)

    Z, P, NU, NUP, EP = [], [], [], [], []
    DIST = {0.25: [], 0.5: [], 1.0: []}
    acts_all = np.asarray(dataset.get_col_data("action"), np.float32)
    offs = np.arange(-K_PAST * STRIDE, K_FUT * STRIDE + 1)
    for b0 in range(0, len(eps), a.batch):
        be, rows_t = eps[b0:b0 + a.batch], pick[b0:b0 + a.batch]
        idx = rows_t[:, None] + offs[None]                       # contiguous rows per episode
        assert (ep_col[idx] == be[:, None]).all() and (np.diff(step[idx], axis=1) == 1).all()
        fidx = idx[:, ::STRIDE]                                   # (B, nf) frame rows
        flat = np.unique(fidx)
        got = dataset.get_row_data(flat)["pixels"]
        got = got.numpy() if hasattr(got, "numpy") else np.asarray(got)
        if got.shape[-1] != 3:
            got = np.moveaxis(got, 1, -1)
        pos = {r: i for i, r in enumerate(flat)}
        pix = [got[[pos[r] for r in row]] for row in fidx]
        act = [acts_all[row[:-1]] for row in idx]
        pix = np.stack(pix)
        B = len(pix)
        x = torch.stack([tf(torch.from_numpy(im).permute(2, 0, 1)) for im in pix.reshape(-1, *pix.shape[2:])])
        x = x.reshape(B, nf, *x.shape[1:]).to(dev)
        raw = np.stack(act).astype(np.float32)
        norm = scaler.transform(raw.reshape(-1, raw.shape[-1])).reshape(B, nf - 1, -1)
        with torch.no_grad():
            z = model.encode({"pixels": x})["emb"]
            ae = model.action_encoder(torch.as_tensor(norm, device=dev, dtype=torch.float32))
            i0 = K_PAST
            fut, fa = z[:, i0 + 1:], ae[:, i0:i0 + K_FUT]
            p3 = rollout(z[:, i0 - 2:i0 + 1], ae[:, i0 - 2:i0], fa)
            nu = z[:, i0] - nxt(z[:, i0 - 3:i0], ae[:, i0 - 3:i0])
            nup = z[:, i0 - 1] - nxt(z[:, i0 - 4:i0 - 1], ae[:, i0 - 4:i0 - 1])
            for gam in DIST:
                DIST[gam].append((rollout(z[:, i0 - 2:i0 + 1], ae[:, i0 - 2:i0], fa, gam * nu) - fut).cpu().numpy())
        Z.append(fut.cpu().numpy())
        P.append(p3.cpu().numpy())
        NU.append(nu.cpu().numpy())
        NUP.append(nup.cpu().numpy())
        EP.append(be)
        print(f"batch {b0}", flush=True)
    Z, P, NU, NUP, EP = map(np.concatenate, (Z, P, NU, NUP, EP))
    DIST = {k: np.concatenate(v) for k, v in DIST.items()}
    E = Z - P                                  # (N, 5, D) h3 rollout errors
    ueps = np.unique(EP)
    test_eps = set(np.random.default_rng(0).choice(ueps, size=len(ueps) // 3, replace=False).tolist())
    te = np.array([e in test_eps for e in EP])
    tr = ~te

    def mse(err):
        return (err[te] ** 2).mean(-1).mean(0)

    rep = {"task": a.task, "n": int(len(EP)), "n_test": int(te.sum()),
           "h3": mse(E).round(6).tolist(),
           "cos_nu_nuprev": float(np.mean(np.sum(NU * NUP, -1) / (np.linalg.norm(NU, axis=-1) * np.linalg.norm(NUP, axis=-1) + 1e-9))),
           "cos_nu_e1": float(np.mean(np.sum(NU * E[:, 0], -1) / (np.linalg.norm(NU, axis=-1) * np.linalg.norm(E[:, 0], axis=-1) + 1e-9)))}
    for gam in (0.25, 0.5, 0.75, 1.0):
        rep[f"post({gam})"] = mse(E - gam * NU[:, None]).round(6).tolist()
    for gam, d in DIST.items():
        rep[f"dist({gam})"] = mse(d).round(6).tolist()
    rep["ema(0.5)"] = mse(E - 0.5 * (0.5 * (NU + NUP))[:, None]).round(6).tolist()
    # Diagonal gain per horizon and dimension: g = <e, nu> / <nu, nu> on training windows.
    gdiag = (E[tr] * NU[tr, None]).sum(0) / ((NU[tr, None] ** 2).sum(0) + 1e-9)   # (5, D)
    rep["diag"] = mse(E - gdiag[None] * NU[:, None]).round(6).tolist()
    for lam in (1.0, 10.0, 100.0):
        Xtr = np.concatenate([NU[tr], NUP[tr]], -1)
        Xte = np.concatenate([NU, NUP], -1)
        A = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ E[tr].reshape(tr.sum(), -1))
        pred = (Xte @ A).reshape(E.shape)
        rep[f"ridge({lam})"] = mse(E - pred).round(6).tolist()
    rel = {k: (np.array(v) / np.array(rep["h3"]) - 1).round(3).tolist()
           for k, v in rep.items() if isinstance(v, list) and k != "h3"}
    rep["relative_to_h3"] = rel
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
