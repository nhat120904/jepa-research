#!/usr/bin/env python3
"""PRIVILEGED diagnostic (puzzle / scene button envs): can each button's state be read from one identity?

For every identity and button: best single-threshold accuracy of the button state from the identity's
appearance (projected on the direction between the two class means), on processed, visible val frames
(threshold fit on the first half of the frames, accuracy on the second half). Reports, per button, the
best identity and its accuracy, and the place of each identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads((a.front / "discover.json").read_text())
    E = np.load(a.front / "entities_val.npz")
    b = np.load(a.cache / "val_button_states.npy", mmap_mode="r")
    pf = np.nonzero(E["processed"])[0]
    B = np.asarray(b[pf]).astype(int)
    if B.ndim == 1:
        B = B[:, None]
    app, vis = E["app"][pf], E["area"][pf] >= 1
    K, nb = app.shape[1], B.shape[1]
    half = len(pf) // 2
    acc = np.zeros((K, nb))
    for k in range(K):
        for j in range(nb):
            y = B[:, j]
            tr = vis[:, k] & (np.arange(len(pf)) < half); te = vis[:, k] & (np.arange(len(pf)) >= half)
            if tr.sum() < 20 or te.sum() < 20 or len(np.unique(y[tr])) < 2:
                continue
            m1, m0 = app[tr, k][y[tr] == 1].mean(0), app[tr, k][y[tr] == 0].mean(0)
            w = m1 - m0
            s_tr, s_te = app[tr, k] @ w, app[te, k] @ w
            cands = np.unique(s_tr)
            best = max(cands[:: max(1, len(cands) // 200)], key=lambda c: ((s_tr > c) == y[tr]).mean())
            acc[k, j] = ((s_te > best) == y[te]).mean()
    per_button = [{"button": j, "best_identity": int(acc[:, j].argmax()), "acc": round(float(acc[:, j].max()), 3)} for j in range(nb)]
    out = {"K": K, "buttons": nb, "buttons_acc_ge_98": int(sum(p["acc"] >= 0.98 for p in per_button)),
           "buttons_acc_ge_90": int(sum(p["acc"] >= 0.90 for p in per_button)),
           "identities_used": len(set(p["best_identity"] for p in per_button if p["acc"] >= 0.9)),
           "per_button": per_button,
           "identity_places": [None if i["centre"] is None else [round(c, 1) for c in i["centre"]] for i in d["identities"]],
           "identity_visible_frac": np.round(vis.mean(0), 2).tolist()}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
