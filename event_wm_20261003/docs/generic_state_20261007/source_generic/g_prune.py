#!/usr/bin/env python3
"""Generic front end, final step: drop entities that carry no planning information.

An entity is kept if, in the TRAIN output of u_events.py, it is the acted entity of some event or its contact-free
rest state ever changes (consecutive rest labels differing by more than tol_pos / thr_app). Otherwise it never changes
across any interaction (e.g. a part that springs back to the same rest once the agent has left) and is removed from
the entity tables and layout; u_events.py is then rerun by the caller. Event endpoint readings are not used for this:
they are taken right at the end of a change, while such a part may still be held. No family-specific rule.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    a = ap.parse_args()
    ev = np.load(a.events / "events_train.npz")
    rep = json.loads((a.events / "report.json").read_text())
    tol_pos, thr_app = float(rep["tol_pos"]), float(rep["thr_app"])
    K = ev["before"].shape[1]
    acted = np.bincount(ev["e"], minlength=K)
    lab = np.load(a.events / "labels_train.npz")
    changed = np.zeros(K, np.int64)
    for k in range(K):
        v = np.nonzero(lab["valid"][:, k])[0]
        if len(v) > 1:
            dp = np.linalg.norm(np.diff(lab["pos"][v, k], axis=0), axis=-1) > tol_pos
            da = np.abs(np.diff(lab["app"][v, k], axis=0)).max(-1) > thr_app
            changed[k] = int((dp | da).sum())
    keep = np.nonzero((acted > 0) | (changed > 0))[0]
    out = {"K_before": int(K), "kept": keep.tolist(), "acted": acted.tolist(), "rest_changes": changed.tolist()}
    if len(keep) < K:
        bak = a.front.parent / (a.front.name + "_unpruned")
        shutil.copytree(a.front, bak)
        L = json.loads((a.front / "layout.json").read_text())
        L["pruned_entities"] = [L["entities"][k] for k in range(K) if k not in set(keep.tolist())]
        L["entities"] = [L["entities"][k] for k in keep]
        L["K"] = int(len(keep))
        (a.front / "layout.json").write_text(json.dumps(L) + "\n")
        d = json.loads((a.front / "discover.json").read_text())
        d["objects"] = int(len(keep)); d["groups"] = [[k] for k in range(len(keep))]
        d["table"] = [d["table"][k] for k in keep]; d["identities"] = [d["identities"][k] for k in keep]
        (a.front / "discover.json").write_text(json.dumps(d, indent=1) + "\n")
        for split in ("train", "val"):
            z = dict(np.load(a.front / f"entities_{split}.npz"))
            for k_ in ("pos", "app", "area"):
                z[k_] = z[k_][:, keep]
            np.savez(a.front / f"entities_{split}.npz", **z)
    (a.front / "prune.json").write_text(json.dumps(out, indent=1) + "\n")
    print("PRUNE", json.dumps({"K_before": K, "K_after": int(len(keep))}), flush=True)


if __name__ == "__main__":
    main()
