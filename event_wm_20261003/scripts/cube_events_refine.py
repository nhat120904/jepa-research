#!/usr/bin/env python3
"""Cube direction A: refine label-free move timing with the CNN object reader (self-training round 2).

cube_events_px.py ends a move when the colour track sees the cube at rest again, i.e. only after the
gripper has left: ~14 frames after the cube actually stopped (PRIVILEGED check, job 57118 events). Skill
segments therefore started ~13 frames late (the frames right after a release were only ever labelled
with the previous move's target) and ended ~14 frames late. The reader localises the cube also under
the gripper, so a move's arrival time is the first frame from which the reader keeps the cube within
r px of its final rest position up to the old end (val check: -3 frames median w.r.t. qpos).
Segments are rebuilt from the refined ends; a segment in which another object changed place (reader
positions at the segment start vs the arrival, > the object-width threshold) contains a move the
events missed and is flagged (excluded from skill training like knocks).
Output: the cube_events_px.py format with refined t / seg_start / knock (+ t_colour, contaminated).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True, help="cube_events_px.py output")
    ap.add_argument("--reader", type=Path, required=True, help="cube_reader.pt")
    ap.add_argument("--r", type=float, default=1.0, help="px; arrival radius (= the rest-run radius)")
    ap.add_argument("--ref-events", type=Path, default=None, help="PRIVILEGED cube_events.py dir, diagnostic only")
    ap.add_argument("--discover", type=Path, default=None, help="for the object -> cube matching of the diagnostic")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    from train_cube_reader import make_cube_reader

    dev = "cuda"
    rk = torch.load(a.reader, map_location="cpu", weights_only=False)
    reader = make_cube_reader(rk["K"], coverage=rk.get("coverage", False)).to(dev).eval()
    reader.load_state_dict(rk["reader"])
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {"r": a.r}
    for split in ("train", "val"):
        obs = np.load(a.cache / f"{split}_observations.npy", mmap_mode="r")
        term = np.load(a.cache / f"{split}_terminals.npy")
        ev = dict(np.load(a.events / f"cube_events_{split}.npz"))
        n = len(obs)
        pred = np.zeros((n, rk["K"], 2), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, n, 4096):
                x = torch.as_tensor(np.array(obs[s:s + 4096]), device=dev).permute(0, 3, 1, 2).float().div_(255.0)
                pred[s:s + len(x)] = reader(x).float().cpu().numpy()
        ep_of = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)
        first = np.r_[0, np.nonzero(term)[0] + 1]
        thr = float(ev["thr_px"])
        t_new = ev["t"].copy()
        for i, (t0, t1, k) in enumerate(zip(ev["t_start"], ev["t"], ev["k"])):
            d = np.linalg.norm(pred[t0:t1 + 1, k] - ev["after"][i, k, :2], axis=-1)
            far = np.nonzero(d > a.r)[0]
            t_new[i] = t0 + (far[-1] + 1 if len(far) else 0)
        order = np.argsort(ev["t_start"], kind="stable")
        seg = np.zeros_like(t_new)
        last_end = {}
        for i in order:
            e = ep_of[ev["t_start"][i]]
            seg[i] = min(max(first[e], last_end.get(e, first[e] - 1) + 1), ev["t_start"][i])
            last_end[e] = max(last_end.get(e, -1), t_new[i])
        # contamination: another object changed place within [seg, t_new]
        K = pred.shape[1]
        others = np.ones((len(t_new), K), bool); others[np.arange(len(t_new)), ev["k"]] = False
        dpos = np.linalg.norm(pred[t_new] - pred[seg], axis=-1)
        contaminated = ((dpos > thr) & others).any(1)
        out = dict(ev)
        out.update(t=t_new, seg_start=seg, knock=ev["knock"] | contaminated, t_colour=ev["t"], contaminated=contaminated)
        np.savez_compressed(a.out / f"cube_events_{split}.npz", **out)
        r = {"moves": int(len(t_new)), "end_shift_pct": np.percentile(t_new - ev["t"], [10, 50, 90]).tolist(),
             "contaminated_frac": float(contaminated.mean()), "excluded_frac": float(out["knock"].mean()),
             "segment_len_pct": np.percentile(t_new + 10 - seg, [10, 50, 90, 99]).tolist()}
        if a.ref_events is not None and a.discover is not None:                   # PRIVILEGED diagnostic
            disc = json.loads((a.discover / "discover.json").read_text())
            o2c = {d["object"]: d["cube"] for d in disc["privileged_diagnostic"] if d}
            Q = np.load(a.ref_events / f"cube_events_{split}.npz")
            used = np.zeros(len(Q["t"]), bool); lag_e, lag_s = [], []
            for i, (t0, t1, k) in enumerate(zip(ev["t_start"], ev["t"], ev["k"])):
                c = np.nonzero((Q["k"] == o2c[int(k)]) & ~used & (Q["t_start"] <= t1 + 30) & (Q["t"] >= t0 - 30))[0]
                if len(c):
                    j = c[np.abs(Q["t"][c] - t1).argmin()]; used[j] = True
                    lag_e.append(t_new[i] - Q["t"][j]); lag_s.append(seg[i] - Q["seg_start"][j])
            r["privileged_end_lag_pct"] = np.percentile(lag_e, [10, 25, 50, 75, 90]).tolist()
            r["privileged_seg_start_lag_pct"] = np.percentile(lag_s, [10, 25, 50, 75, 90]).tolist()
        rep[split] = r
        print(split, json.dumps(r), flush=True)
    (a.out / "refine_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    for f in ("labels_train.npz", "labels_val.npz"):                         # labels unchanged (closed loop reads them)
        if (a.events / f).exists() and not (a.out / f).exists():
            os.symlink(a.events / f, a.out / f)


if __name__ == "__main__":
    main()
