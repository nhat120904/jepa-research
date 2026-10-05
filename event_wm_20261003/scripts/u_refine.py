#!/usr/bin/env python3
"""Unified backend, step 3b: refine event timing with the trained reader (self-training round 2).

u_events.py locates changes only at the sampled frames (every `stride` frames) and ends an event at the
frame before the acted identity's next rest run. The reader reads every frame, so an event's arrival time
is the first frame from which the reader keeps the acted identity at its after-state -- position within
r_pos and appearance within r_app (the rest-run noise radii) -- up to the old end + 1. Segments are rebuilt
from the refined ends (start = previous refined end + 1). A segment in which another identity changed
(reader state at the segment start vs at the arrival differs by > tol_pos / thr_app) and that change is not
one of the event's own changes contains a change the events missed and is flagged (excluded from skill
training like knocks).
Output: events_{split}.npz in the u_events format with refined t / seg_start / knock (+ t_coarse,
contaminated); labels symlinked; refine_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from u_reader import make_reader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True, help="u_events.py output")
    ap.add_argument("--reader", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    rep_in = json.loads((a.events / "report.json").read_text())
    r_pos, r_app = float(rep_in["r_pos"]), float(rep_in["r_app"])
    rk = torch.load(a.reader, map_location="cpu", weights_only=False)
    K = rk["K"]
    reader = make_reader(K, agent=rk.get("agent", False)).to(a.device).eval(); reader.load_state_dict(rk["reader"])
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {"r_pos": r_pos, "r_app": r_app}
    for split in ("train", "val"):
        obs = np.load(a.cache / f"{split}_observations.npy", mmap_mode="r")
        term = np.load(a.cache / f"{split}_terminals.npy")
        ev = dict(np.load(a.events / f"events_{split}.npz"))
        tol_pos, thr_app = float(ev["tol_pos"]), float(ev["thr_app"])
        n = len(term)
        ep_of = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)
        first = np.r_[0, np.nonzero(term)[0] + 1]
        last = np.r_[np.nonzero(term)[0], n - 1]
        eps = np.unique(ep_of[ev["t_start"]]) if len(ev["t_start"]) else np.zeros(0, int)
        pred = {}
        with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16):
            for e in eps:
                s0, s1 = first[e], last[e] + 1
                x = torch.as_tensor(np.asarray(obs[s0:s1]), device=a.device).permute(0, 3, 1, 2).float().div_(255.0)
                st, _ = reader(x)
                pred[e] = (s0, st.float().cpu().numpy()[..., :5])
        t_new = ev["t"].copy()
        for i, (t0, t1, k) in enumerate(zip(ev["t_start"], ev["t"], ev["e"])):
            s0, P = pred[ep_of[t0]]
            hi = min(t1 + 1, last[ep_of[t0]])
            seg = P[t0 - s0:hi - s0 + 1, k]
            dp = np.linalg.norm(seg[:, :2] - ev["after"][i, k, :2], axis=-1)
            da = np.abs(seg[:, 2:5] - ev["after"][i, k, 2:5]).max(-1)
            far = np.nonzero((dp > r_pos) | (da > (r_app if np.isfinite(r_app) else np.inf)))[0]
            t_new[i] = min(t0 + (far[-1] + 1 if len(far) else 0), t1)
        order = np.argsort(ev["t_start"], kind="stable")
        seg_start = np.zeros_like(t_new); last_end = {}
        for i in order:
            e = ep_of[ev["t_start"][i]]
            seg_start[i] = min(max(first[e], last_end.get(e, first[e] - 1) + 1), ev["t_start"][i])
            last_end[e] = max(last_end.get(e, -1), t_new[i])
        contaminated = np.zeros(len(t_new), bool)
        for i in range(len(t_new)):
            s0, P = pred[ep_of[ev["t_start"][i]]]
            A, Bn = P[seg_start[i] - s0], P[t_new[i] - s0]
            ch = (np.linalg.norm(Bn[:, :2] - A[:, :2], axis=-1) > tol_pos) | (np.abs(Bn[:, 2:5] - A[:, 2:5]).max(-1) > thr_app)
            # the event's own changes (acted identity and its effects, e.g. toggled neighbours) are not missed changes
            own = (np.linalg.norm(ev["after"][i, :, :2] - ev["before"][i, :, :2], axis=-1) > tol_pos) | \
                  (np.abs(ev["after"][i, :, 2:5] - ev["before"][i, :, 2:5]).max(-1) > thr_app)
            ch[ev["e"][i]] = False; ch[own] = False
            contaminated[i] = ch.any()
        out = dict(ev)
        out.update(t=t_new, seg_start=seg_start, knock=ev["knock"] | contaminated, t_coarse=ev["t"], contaminated=contaminated)
        np.savez_compressed(a.out / f"events_{split}.npz", **out)
        rep[split] = {"events": int(len(t_new)), "end_shift_pct": np.percentile(t_new - ev["t"], [10, 50, 90]).tolist() if len(t_new) else None,
                      "contaminated_frac": float(contaminated.mean()) if len(t_new) else None,
                      "excluded_frac": float(out["knock"].mean()) if len(t_new) else None}
        print(split, json.dumps(rep[split]), flush=True)
    (a.out / "refine_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    for f in ("labels_train.npz", "labels_val.npz", "report.json"):
        if (a.events / f).exists() and not (a.out / f).exists():
            os.symlink(a.events / f, a.out / f)


if __name__ == "__main__":
    main()
