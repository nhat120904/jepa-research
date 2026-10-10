#!/usr/bin/env python3
"""Component 8b (method/README.md): acted entity by explanation, with the world model of the first round.

An event's acted entity is the identity whose action best explains what was observed: among the known changed identities
of the event (and its current acted entity), the candidate c whose world-model prediction f(before, c, after[c]) is closest
to the observed after-state over the entities known after the event (squared errors in the change units: position /
tol_pos, appearance / the entity's appearance unit). The contact rule of events_objects.py picks a touched identity near the
centre of the changes; on Lights-Out-like effects a neighbour of the pressed light is touched too, and the world model
trained on those labels learns a blurred toggle structure (puzzle-4x5 VAL, PRIVILEGED: side effects within tolerance .948
on single presses with the right acted light, .524 with a wrong one). Events that no single action explains (the best
loss above the 2-means split of log best losses, when bimodal: Ashman D > 2), such as two interactions in one event, are
marked target_known = False, unless the best candidate is also the contact rule's choice (two independent signals agree
on the cause): they stay in the event list (skill segments, states) but do not train the model. Puzzle-4x5 (2026-10-09,
PRIVILEGED scoring), training labels of target-known events: contact rule .858 (24.0k events); explained only (small-loss
selection) .993 but 14.5k events, 23 of them acting on light 2, whose presses the second model then mispredicted in 157 of
177 closed-loop presses (the split separates exact explanations from ANY error of the first model, so events showing an
effect the first model missed were dropped); explained or agreed .931 with 23.2k events (160 on light 2); without
trimming (--no-trim) .809.
Input: an events dir (events_objects.py) and its first-round world model (world_model.py --stage wm). Output: an events dir
with e / target / target_known replaced, other files copied, relabel_report.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from utils import save_json, two_means_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="world_model.py --stage wm output (wm_stage.pt)")
    ap.add_argument("--no-trim", dest="trim", action="store_false", help="keep events no single action explains as targets")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch

    from world_model import Scale, event_input, make_wm

    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    K, D = ck["K"], ck["D"]
    wm = make_wm(K, D).to(a.device).eval(); wm.load_state_dict(ck["wm"])
    sc = Scale(ck["thr_pos"], ck["thr_app"], ck["tol_pos"])
    tol = float(ck["tol_pos"])
    a.out.mkdir(parents=True, exist_ok=True)
    for f in a.events.iterdir():
        if f.name not in ("events_train.npz", "events_val.npz"):
            shutil.copy(f, a.out / f.name)
    rep = {"events": str(a.events), "model": str(a.model)}
    split_thr = None
    for split in ("train", "val"):
        z = dict(np.load(a.events / f"events_{split}.npz"))
        n = len(z["e"])
        bk, ak = z["before_known"].astype(bool), z["after_known"].astype(bool)
        unit = np.asarray(z["app_unit_id"], np.float64)
        unit = np.where(np.isfinite(unit), unit, float(ck["thr_app"]) if np.isfinite(ck["thr_app"]) else 0.1)
        b4, af = z["before"], z["after"]
        chg = bk & ak & ((np.linalg.norm(af[..., :2] - b4[..., :2], axis=-1) > tol) | (np.abs(af[..., 2:5] - b4[..., 2:5]).max(-1) > unit))
        pairs = []                                                               # (event, candidate)
        for i in range(n):
            cands = set(np.flatnonzero(chg[i]).tolist()) | {int(z["e"][i])}
            pairs += [(i, c) for c in sorted(cands) if bk[i, c] and ak[i, c]]
        if not pairs:
            continue
        P = np.array(pairs)
        loss = np.zeros(len(P))
        for s0 in range(0, len(P), 8192):
            ii, cc = P[s0:s0 + 8192, 0], P[s0:s0 + 8192, 1]
            S = b4[ii]; X = af[ii, cc]
            Xin = event_input(S, cc, X, tol) if ck.get("event_pos_only") else X
            with torch.no_grad():
                pred = sc.denorm(wm(sc.norm(torch.as_tensor(S, device=a.device).float()), torch.as_tensor(cc, device=a.device).long(),
                                    sc.norm(torch.as_tensor(Xin, device=a.device).float())).cpu().numpy())
            err = np.concatenate([(pred[..., :2] - af[ii][..., :2]) / tol, (pred[..., 2:5] - af[ii][..., 2:5]) / unit[None, :, None]], -1)
            m = ak[ii]
            loss[s0:s0 + 8192] = ((err ** 2).mean(-1) * m).sum(1) / np.maximum(m.sum(1), 1)
        best = np.full(n, np.inf); arg = z["e"].astype(np.int64).copy()
        for (i, c), l_ in zip(P, loss):
            if l_ < best[i]:
                best[i], arg[i] = l_, c
        agree = arg == z["e"]                                                    # the contact rule's choice explains best
        changed_e = (arg != z["e"]) & np.isfinite(best)
        fin = np.isfinite(best)
        if split_thr is None:                                                    # the TRAIN split defines "unexplained"
            th, Dsep, _ = two_means_threshold(np.log(best[fin] + 1e-9))
            split_thr = (float(np.exp(th)), float(Dsep))
        unexplained = fin & (best > split_thr[0]) & ~agree if (a.trim and split_thr[1] > 2) else np.zeros(n, bool)
        z["e"] = arg
        z["target"] = af[np.arange(n), arg]
        z["target_known"] = bk[np.arange(n), arg] & ak[np.arange(n), arg] & ~unexplained
        np.savez_compressed(a.out / f"events_{split}.npz", **z)
        rep[split] = {"events": int(n), "acted_changed": int(changed_e.sum()), "acted_changed_share": float(changed_e.mean()),
                      "unexplained": int(unexplained.sum()), "target_known_share": float(z["target_known"].mean())}
        print(split, json.dumps(rep[split]), flush=True)
    rep["unexplained_loss_split"] = split_thr
    save_json(a.out / "relabel_report.json", rep)


if __name__ == "__main__":
    main()
