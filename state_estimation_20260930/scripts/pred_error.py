#!/usr/bin/env python3
"""Offline decomposition of released-LeWM rollout error by planning start state.

Dataset windows (frames every 5 steps, t-15 .. t+25; true action blocks). For the
context ending at frame t we roll the frozen predictor 5 blocks ahead and compare
with the encodings of the true future frames:
  h1   context [z_t]                        (released planner)
  h2   context [z_t-5, z_t] + 1 action block
  h3   context [z_t-10, z_t-5, z_t] + 2 blocks (training context)
  h3+nu*g   h3 rollout plus g * innovation, nu = z_t - prediction of z_t from
            the context ending at t-5 (tests whether model errors persist)
  mhe(l)    context latents re-estimated by minimising ||x - z||^2 +
            l * one-step dynamics residuals inside the window (tests whether
            dynamics-consistent smoothing of clean encodings changes anything)
Also reports the cosine between nu and the next one-step error of h3.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

K_PAST = 3      # frames before t used (t-15, t-10, t-5)
K_FUT = 5
STRIDE = 5


def sample_windows(dataset, n, seed):
    from se.tasks import episode_column

    col = episode_column(dataset)
    ep_col = np.asarray(dataset.get_col_data(col))
    step = np.asarray(dataset.get_col_data("step_idx"))
    _, inv = np.unique(ep_col, return_inverse=True)
    last = np.zeros(inv.max() + 1, dtype=np.int64)
    np.maximum.at(last, inv, step)
    ok = (step >= K_PAST * STRIDE) & (step + K_FUT * STRIDE <= last[inv])
    rows = np.nonzero(ok)[0]
    g = np.random.default_rng(seed)
    pick = np.sort(rows[g.choice(len(rows), size=n, replace=False)])
    r = dataset.get_row_data(pick)
    return np.asarray(r[col]), np.asarray(r["step_idx"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model work must run under sbatch")
    import torch

    import stable_worldmodel as swm
    from se import tasks as T

    torch.manual_seed(0)
    dev = "cuda"
    c = T.TASKS[a.task]
    dataset = T.load_dataset(swm, a.task)
    scaler = T.fit_processors(dataset, a.task)["action"]
    model = swm.wm.utils.load_pretrained(c["repo"]).to(dev).eval()
    model.requires_grad_(False)
    tf = T.image_transform()
    eps, ts = sample_windows(dataset, a.n, a.seed)
    nf = K_PAST + 1 + K_FUT  # frames t-15 .. t+25

    def predict_next(ctx, acts):
        """ctx (B, h, D) latents, acts (B, h, A_emb) action embeddings -> (B, D)."""
        return model.predict(ctx, acts)[:, -1]

    def rollout(ctx, act_emb_ctx, act_emb_fut):
        """Autoregressive 5-block rollout; ctx (B, h, D), act_emb_ctx (B, h-1, E)
        executed blocks between context frames, act_emb_fut (B, 5, E)."""
        emb = list(ctx.unbind(1))
        acts = list(act_emb_ctx.unbind(1)) + list(act_emb_fut.unbind(1))
        h = ctx.size(1)
        out = []
        for k in range(K_FUT):
            lo = max(0, h + k - 3)
            e = torch.stack(emb[lo:], 1)
            u = torch.stack(acts[lo:h + k], 1)
            nxt = predict_next(e, u)
            emb.append(nxt)
            out.append(nxt)
        return torch.stack(out, 1)

    acc = {}

    def add(name, err):
        acc.setdefault(name, []).append(err.detach().float().cpu().numpy())

    for b0 in range(0, len(eps), a.batch):
        be, bt = eps[b0:b0 + a.batch], ts[b0:b0 + a.batch]
        lo = bt - K_PAST * STRIDE
        chunks = dataset.load_chunk(be, lo, bt + K_FUT * STRIDE + 1)
        pix, act = [], []
        for ch in chunks:
            p = ch["pixels"]
            p = p.numpy() if hasattr(p, "numpy") else np.asarray(p)
            if p.shape[-1] != 3:
                p = np.moveaxis(p, 1, -1)
            pix.append(p[::STRIDE][:nf])
            u = ch["action"]
            u = u.numpy() if hasattr(u, "numpy") else np.asarray(u)
            act.append(u[:(nf - 1) * STRIDE])
        pix = np.stack(pix)                       # (B, nf, H, W, C)
        B = len(pix)
        x = torch.stack([tf(torch.from_numpy(im).permute(2, 0, 1)) for im in pix.reshape(-1, *pix.shape[2:])])
        x = x.reshape(B, nf, *x.shape[1:]).to(dev)
        raw = np.stack(act).astype(np.float32)    # (B, (nf-1)*5, A)
        norm = scaler.transform(raw.reshape(-1, raw.shape[-1])).reshape(B, nf - 1, -1)
        with torch.no_grad():
            z = model.encode({"pixels": x})["emb"]                    # (B, nf, D)
            ae = model.action_encoder(torch.as_tensor(norm, device=dev))  # (B, nf-1, E)
        i0 = K_PAST                                   # index of frame t
        fut = z[:, i0 + 1:i0 + 1 + K_FUT]
        fut_act = ae[:, i0:i0 + K_FUT]
        base = ((fut - z[:, i0:i0 + 1]) ** 2).mean(-1)   # "no-motion" reference
        add("static_ref", base)
        with torch.no_grad():
            preds = {
                "h1": rollout(z[:, i0:i0 + 1], ae[:, i0:i0], fut_act),
                "h2": rollout(z[:, i0 - 1:i0 + 1], ae[:, i0 - 1:i0], fut_act),
                "h3": rollout(z[:, i0 - 2:i0 + 1], ae[:, i0 - 2:i0], fut_act),
            }
            # innovation of z_t given context ending at t-5 (3 frames t-15..t-5)
            zt_hat = predict_next(z[:, i0 - 3:i0], ae[:, i0 - 3:i0])
            nu = z[:, i0] - zt_hat
        for name, p in preds.items():
            add(name, ((p - fut) ** 2).mean(-1))
        for g in (0.25, 0.5, 1.0):
            add(f"h3+nu*{g}", ((preds["h3"] + g * nu[:, None] - fut) ** 2).mean(-1))
        e1 = fut[:, 0] - preds["h3"][:, 0]
        cos = torch.nn.functional.cosine_similarity(nu, e1, dim=-1)
        acc.setdefault("cos_nu_e1", []).append(cos.cpu().numpy())
        acc.setdefault("nu_norm", []).append(nu.pow(2).mean(-1).cpu().numpy())
        # MHE: re-estimate the three context latents.
        for lam in (0.3, 3.0):
            xv = z[:, i0 - 2:i0 + 1].clone().requires_grad_(True)
            opt = torch.optim.Adam([xv], lr=0.02)
            anchor = z[:, i0 - 3:i0 - 2]
            for _ in range(50):
                opt.zero_grad()
                seq = torch.cat([anchor, xv], 1)           # t-15, t-10, t-5, t
                pr = model.predict(seq[:, :3], ae[:, i0 - 3:i0])  # predicts t-10, t-5, t
                dyn = ((pr - xv) ** 2).mean(-1).sum(-1)
                obs = ((xv - z[:, i0 - 2:i0 + 1]) ** 2).mean(-1).sum(-1)
                (obs + lam * dyn).sum().backward()
                opt.step()
            with torch.no_grad():
                p = rollout(xv.detach(), ae[:, i0 - 2:i0], fut_act)
            add(f"mhe({lam})", ((p - fut) ** 2).mean(-1))
        print(f"batch {b0} done", flush=True)

    rep = {"task": a.task, "n": int(len(eps))}
    for k, v in acc.items():
        v = np.concatenate(v)
        if v.ndim == 2:
            rep[k] = {"mean_by_h": v.mean(0).round(5).tolist(), "median_by_h": np.median(v, 0).round(5).tolist()}
        else:
            rep[k] = {"mean": float(v.mean()), "median": float(np.median(v))}
    h1 = np.concatenate(acc["h1"])
    for k in acc:
        if k in ("h1", "static_ref") or np.concatenate(acc[k]).ndim != 2:
            continue
        d = np.concatenate(acc[k]) - h1
        rep[k]["paired_vs_h1_last"] = {"mean": float(d[:, -1].mean()),
                                       "frac_better": float((d[:, -1] < 0).mean())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
