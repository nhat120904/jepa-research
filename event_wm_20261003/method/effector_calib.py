#!/usr/bin/env python3
"""Component 5d: calibration of the learned effector point (effector3.py) from a first events pass, frames and actions only.

Why. effector3's point is where the commanded translation is most visible, which is not always where the gripper touches
things: PRIVILEGED checks put it 1.4-1.7 px from the pressed light on puzzles, but 6.8 px above the pressed button on
scene and 16 px above the grasped cube on cube-triple (a wrist, not the fingers). The offset is systematic: subtracting
the median vector leaves 0.6 px (scene buttons, p50) and 4.5 px (cube grasps).

Rule (one for every environment). From a first pass of events_objects.py --effector (no offset, no lag):
  OFFSET = the median, over TRAIN events with at least one known change, of the centroid of the changed entities'
           positions before the event minus the effector point at the event's contact moment (t_core0, the latest
           departure). Every interaction happens where the gripper touches, so the changed entities surround it.
  LAG    = the reading lag under the effector: for every change of an entity that the calibrated effector came within one
           object width (thr_pos) of during the change's window, the frames from the effector's last such frame to the
           change's arrival (the new state read); L = ceil(p90). It replaces the p90 of within-event arrival spreads,
           which also counts how long an interaction lasts (cube-triple: 88 frames, the carry).
Outputs (--out): effector_calib.json {offset: [du, dv], lag: L, ...}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goal_maps import delta_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True, help="first pass: events_objects.py --effector (no offset, no lag)")
    ap.add_argument("--entities", type=Path, required=True, help="objects.py output the events were built from")
    ap.add_argument("--effector", type=Path, required=True, help="effector3.py output (track_train.npy)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(a.events / "events_train.npz"))
    pos = np.load(a.entities / "entities_train.npz")["pos"].astype(np.float64)
    n = len(pos)
    eff = np.asarray(np.load(a.effector / "track_train.npy", mmap_mode="r")[:n, :2], np.float64)
    term = np.load(a.cache / "train_terminals.npy")[:n]
    ep = np.concatenate([[0], np.cumsum(term[:-1])])
    thr_pos = float(z["thr_pos"])
    # OFFSET: changed entities' centroid before the event - effector at the contact moment
    ch = delta_mask(z["before"], z["after"], z["before_known"].astype(bool), z["after_known"].astype(bool), float(z["tol_pos"]),
                    np.asarray(z["thr_app_id"]))
    use = np.flatnonzero(ch.any(1))
    c0 = np.clip(z["t_core"][use, 0], 0, n - 1)
    cen = np.array([z["before"][i][ch[i], :2].mean(0) for i in use])
    vec = cen - eff[c0]
    ok = np.isfinite(vec).all(1)
    vec = vec[ok]
    offset = np.median(vec, 0)
    raw, res = np.linalg.norm(vec, axis=1), np.linalg.norm(vec - offset, axis=1)
    # LAG: last frame of the calibrated effector within one object width of the entity's new position -> its arrival
    effc = eff + offset
    chg = np.load(a.events / "changes_train.npz")["changes"]                    # (t0, t1, k, episode, before-rest end)
    lab = np.load(a.events / "labels_train.npz")                                # rest labels (the new state from the arrival on)
    labp, valid = lab["pos"], lab["valid"]
    lags = []
    for t0, t1, k, _, _ in chg:
        t0, t1 = max(int(t0) - 1, 0), min(int(t1), n - 1)
        if t1 <= t0 or ep[t0] != ep[t1]:
            continue
        if not valid[t1, k]:
            continue
        p_new = labp[t1, k].astype(np.float64)
        near = np.flatnonzero(np.linalg.norm(effc[t0:t1 + 1] - p_new, axis=1) <= thr_pos)
        if len(near):
            lags.append(t1 - (t0 + int(near[-1])))
    lags = np.asarray(lags)
    lag = int(np.ceil(np.percentile(lags, 90))) if len(lags) else 0
    rep = {"events": str(a.events), "offset": [round(float(offset[0]), 3), round(float(offset[1]), 3)],
           "lag": lag, "thr_pos": thr_pos, "events_used": int(len(vec)),
           "distance_raw_p50_p90": np.percentile(raw, [50, 90]).round(2).tolist(),
           "distance_calibrated_p50_p90": np.percentile(res, [50, 90]).round(2).tolist(),
           "changes_with_effector_near": int(len(lags)), "changes": int(len(chg)),
           "lag_p50_p90": (np.percentile(lags, [50, 90]).round(1).tolist() if len(lags) else None)}
    (a.out / "effector_calib.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
