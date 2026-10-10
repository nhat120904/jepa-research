#!/usr/bin/env python3
"""Event WM + cost-to-go + offline planning checks on the scene-memory-v2 reader events (sm2_reader.py output), the
train_planner.py recipe without the token code:
  1. event WM f(b, e) -> b' (planner.make_event_wm) on TRAIN events with a type in the vocabulary;
  2. cost-to-go h(b, g) by approximate value iteration inside the WM: start codes from TRAIN frames, goals = k random
     imagined events (k <= --kmax), y = 0 if b == g else 1 + min_e h_target(f(b, e), g); target copied every 1000 steps;
  3. checks: WM exactness on VAL events; PRIVILEGED (puzzle, scoring only): h against the true minimal number of presses
     d* between VAL frames, and batch weighted A* plans between VAL frames simulated with the true buttons that event
     types map to (majority over TRAIN events).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from planner import bwas, make_costtogo, make_event_wm

BINS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 99))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def pressed_buttons(cache, split, ev):
    """PRIVILEGED: majority pressed button (deepest button joint) around each event, -1 if none."""
    q = np.load(cache / f"{split}_qpos.npy", mmap_mode="r")
    out = []
    for ts, te in zip(ev["t_start"], ev["t"]):
        w = np.asarray(q[max(ts - 12, 0):te + 3, 14:])
        p = np.where(w.min(1) < -0.015, w.argmin(1), -1)
        u, c = np.unique(p[p >= 0], return_counts=True)
        out.append(int(u[c.argmax()]) if len(u) else -1)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True, help="sm2_reader.py output dir")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--family", choices=("cube", "puzzle", "scene"), required=True)
    ap.add_argument("--wm-steps", type=int, default=6000)
    ap.add_argument("--h-steps", type=int, default=100000)
    ap.add_argument("--h-batch", type=int, default=1024)
    ap.add_argument("--h-hidden", type=int, default=2048)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--target-every", type=int, default=1000)
    ap.add_argument("--pairs-per-bin", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    dev = a.device
    torch.manual_seed(0); rng = np.random.default_rng(0)
    t0 = time.time()
    tr, va = dict(np.load(a.events / "events_train.npz")), dict(np.load(a.events / "events_val.npz"))
    vocab = np.load(a.events / "vocab.npy")
    E, K = vocab.shape
    res = {"events": int(E), "bits": int(K)}
    a.out.mkdir(parents=True, exist_ok=True)

    wm = make_event_wm(K, E).to(dev)
    m = tr["e"] >= 0
    Xb, Xe, Y = (torch.as_tensor(tr[k][m], device=dev) for k in ("before", "e", "after"))
    opt = torch.optim.AdamW(wm.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(a.wm_steps):
        i = torch.randint(0, len(Xb), (1024,), device=dev)
        loss = F.binary_cross_entropy_with_logits(wm(Xb[i], Xe[i]), Y[i].float())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    wm.eval()
    mv = va["e"] >= 0
    with torch.no_grad():
        pv = (wm(torch.as_tensor(va["before"][mv], device=dev), torch.as_tensor(va["e"][mv], device=dev)) > 0).cpu().numpy()
    res.update(wm_val_exact=float((pv == va["after"][mv]).all(1).mean()), wm_val_bit=float((pv == va["after"][mv]).mean()),
               wm_val_events=int(mv.sum()), wm_train_events=int(m.sum()))
    print({"wm": res, "min": round((time.time() - t0) / 60, 1)}, flush=True)

    pool = torch.as_tensor(np.unique(tr["codes"][rng.integers(0, len(tr["codes"]), 400_000)], axis=0), device=dev)
    h = make_costtogo(K, a.h_hidden, True).to(dev)
    h_tgt = make_costtogo(K, a.h_hidden, True).to(dev)
    h_tgt.load_state_dict(h.state_dict())
    opt = torch.optim.AdamW(h.parameters(), lr=3e-4, weight_decay=1e-5)
    hlog, B = [], a.h_batch
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else dict(device_type="cpu", enabled=False)
    for step in range(a.h_steps):
        b = pool[torch.randint(0, len(pool), (B,), device=dev)]
        k = torch.randint(0, a.kmax + 1, (B,), device=dev)
        g = b.clone()
        with torch.no_grad():
            for j in range(a.kmax):
                act = k > j
                if not act.any():
                    break
                ev = torch.randint(0, E, (int(act.sum()),), device=dev)
                g[act] = (wm(g[act], ev) > 0).to(torch.uint8)
            ch = wm.successors(b)
            gg = g.unsqueeze(1).expand_as(ch)
            with torch.autocast(**amp):
                hc = h_tgt(ch.reshape(-1, K), gg.reshape(-1, K)).view(B, E).float()
            hc = hc.masked_fill((ch == gg).all(-1), 0.0)
            y = (1 + hc.min(1).values).clamp(max=a.kmax + 5)
            y = y.masked_fill((b == g).all(-1), 0.0)
        loss = F.mse_loss(h(b, g), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % a.target_every == 0:
            h_tgt.load_state_dict(h.state_dict())
        if step % 5000 == 0 or step == a.h_steps - 1:
            hlog.append({"step": step, "loss": loss.item(), "y_mean": y.mean().item(), "min": round((time.time() - t0) / 60, 1)})
            print(hlog[-1], flush=True)
    h.eval()
    res["h_log"] = hlog
    torch.save({"wm": wm.state_dict(), "h": h.state_dict(), "h_hidden": a.h_hidden, "h_xor": True, "events": E, "bits": K,
                "vocab": vocab, "events_dir": str(a.events)}, a.out / "planner.pt")

    if a.family == "puzzle":                                                     # PRIVILEGED checks (scoring only)
        from lightsout import min_presses, press
        rows, cols = 4, 5
        pb_tr = pressed_buttons(a.cache, "train", tr)
        t2b = {}
        for e in range(E):
            u, c = np.unique(pb_tr[(tr["e"] == e) & (pb_tr >= 0)], return_counts=True)
            t2b[e] = int(u[c.argmax()]) if len(u) else -1
            res.setdefault("type_to_button_purity", []).append(float(c.max() / c.sum()) if len(u) else None)
        res["type_to_button"] = t2b
        res["buttons_covered_by_types"] = int(len({b for b in t2b.values() if b >= 0}))
        S = np.load(a.cache / "val_button_states.npy", mmap_mode="r")
        C = va["codes"]; n = len(C)
        S = np.asarray(S[:n]).astype(np.uint8)
        i, j = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
        d = np.array([min_presses(S[x], S[y], rows, cols) for x, y in zip(i, j)])
        with torch.no_grad():
            hv = h(torch.as_tensor(C[i], device=dev), torch.as_tensor(C[j], device=dev)).cpu().numpy()
        res["h_spearman_dstar"] = spearman(hv, d)
        res["h_mean_by_dstar"] = {int(x): float(hv[d == x].mean()) for x in np.unique(d)}

        def simulate(state, events):
            s_ = state.astype(np.uint8)
            for e in events:
                if t2b.get(int(e), -1) < 0:
                    return None
                s_ = press(s_, t2b[int(e)], rows, cols)
            return s_

        plans = []
        for lo, hi in BINS:
            for c in np.nonzero((d >= lo) & (d <= hi))[0][:a.pairs_per_bin]:
                ts = time.time()
                path, info = bwas(C[i[c]], C[j[c]], wm, h, dev, batch=256, max_expansions=20000)
                sim = None if path is None else simulate(S[i[c]], path)
                plans.append({"dstar": int(d[c]), "found": info["found"], "solved": bool(sim is not None and (sim == S[j[c]]).all()),
                              "length": None if path is None else len(path), "expanded": info["expanded"], "sec": round(time.time() - ts, 2)})
        res["offline_plans"] = {f"{lo}-{hi}": {
            "n": len(p), "found": float(np.mean([x["found"] for x in p])), "solved": float(np.mean([x["solved"] for x in p])),
            "length_over_dstar": float(np.mean([x["length"] / x["dstar"] for x in p if x["length"]])) if any(x["length"] for x in p) else None,
            "expanded_median": float(np.median([x["expanded"] for x in p]))}
            for lo, hi in BINS for p in [[x for x in plans if lo <= x["dstar"] <= hi]] if p}
        print({k: res[k] for k in ("h_spearman_dstar", "buttons_covered_by_types", "offline_plans")}, flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / "planner_eval.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
