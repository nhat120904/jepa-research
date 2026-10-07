#!/usr/bin/env python3
"""Diagnostic: per identity, how often it changes in u_events events (acted or effect), and (PRIVILEGED, button
envs) its on/off appearance signal vs thr_app."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    ev = np.load(a.run / "events" / "events_train.npz")
    thr = float(ev["thr_app"])
    chg = np.abs(ev["after"][..., 2:5] - ev["before"][..., 2:5]).max(-1) > thr
    E = np.load(a.run / "front" / "entities_val.npz")
    pf = np.nonzero(E["processed"])[0]
    Bp = np.asarray(np.load(a.cache / "val_button_states.npy", mmap_mode="r")[pf]).astype(int)
    app, vis = E["app"][pf], E["area"][pf] >= 1
    out = []
    for k in range(chg.shape[1]):
        best = (None, 0.0)
        for j in range(Bp.shape[1]):
            v = vis[:, k]
            if v.sum() < 50 or len(np.unique(Bp[v, j])) < 2:
                continue
            sig = float(np.abs(app[v, k][Bp[v, j] == 1].mean(0) - app[v, k][Bp[v, j] == 0].mean(0)).max())
            if sig > best[1]:
                best = (j, sig)
        out.append({"id": k, "changed_in": int(chg[:, k].sum()), "acted": int((ev["e"] == k).sum()), "button": best[0],
                    "signal": round(best[1], 3), "visible": round(float(vis[:, k].mean()), 2)})
    print(json.dumps({"thr_app": thr, "ids": out}))


if __name__ == "__main__":
    main()
