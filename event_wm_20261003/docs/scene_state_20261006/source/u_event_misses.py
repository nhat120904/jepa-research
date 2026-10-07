#!/usr/bin/env python3
"""PRIVILEGED diagnostic for u_events.py on cube envs: why are reference moves missed?

Reference move = contiguous qpos motion interval of one cube with net xy displacement >= 2 cm. For each
reference move: detected (an event overlaps within 15 frames) or missed; for misses, which identity
(PRIVILEGED identity -> cube map from discover.json), whether that identity had a rest label within
`win` frames before the start and after the end, and the rest-run gap around the move.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="dir with front/ and events/")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--win", type=int, default=50)
    a = ap.parse_args()
    disc = json.loads((a.run / "front" / "discover.json").read_text())
    ev = np.load(a.run / "events" / f"events_{a.split}.npz")
    lab = np.load(a.run / "events" / f"labels_{a.split}.npz")
    ent = np.load(a.run / "front" / f"entities_{a.split}.npz")
    q = np.load(a.cache / f"{a.split}_qpos.npy", mmap_mode="r")
    term = np.load(a.cache / f"{a.split}_terminals.npy")
    n = len(term)
    starts = np.r_[0, np.nonzero(term)[0] + 1]; ends = np.r_[np.nonzero(term)[0] + 1, n]
    starts, ends = starts[starts < ends], ends[starts < ends]
    proc = ent["processed"]
    eps = [e for e in range(len(starts)) if proc[starts[e]]]
    cube_of = {d["object"]: d["cube"] for d in disc["privileged_diagnostic"] if d and d["median_err_cm"] < 5}
    ident_of = {c: k for k, c in cube_of.items()}
    sl = [c for c in (14, 21, 28, 35) if c + 3 <= q.shape[1]]
    ts, te = ev["t_start"], ev["t"]
    rows = []
    for e in eps:
        s0, s1 = starts[e], ends[e]
        Q = np.asarray(q[s0:s1])
        for ci, c in enumerate(sl):
            mv = np.r_[False, np.abs(np.diff(Q[:, c:c + 3], axis=0)).max(1) > 2e-3]
            d_ = np.diff(np.r_[0, mv.astype(np.int8), 0]); xs, ys = np.nonzero(d_ == 1)[0], np.nonzero(d_ == -1)[0] - 1
            for x, y in zip(xs, ys):
                disp = float(np.linalg.norm(Q[min(y + 1, len(Q) - 1), c:c + 2] - Q[max(x - 1, 0), c:c + 2]))
                if disp < 0.02:
                    continue
                X, Y = s0 + x, s0 + y
                hit = bool(np.any((ts <= Y + 15) & (te >= X - 15)))
                k = ident_of.get(ci)
                r = {"cube": ci, "identity": k, "t0": int(X), "t1": int(Y), "dur": int(Y - X + 1), "disp_cm": round(disp * 100, 1), "detected": hit}
                if k is not None:
                    v = lab["valid"][:, k]
                    r["rest_before"] = bool(v[max(X - a.win, s0):X].any())
                    r["rest_after"] = bool(v[Y + 1:min(Y + 1 + a.win, s1)].any())
                    vis = ent["area"][:, k] >= 1
                    r["visible_after_frac"] = round(float(vis[Y + 1:min(Y + 1 + a.win, s1)][proc[Y + 1:min(Y + 1 + a.win, s1)]].mean()), 2) \
                        if proc[Y + 1:min(Y + 1 + a.win, s1)].any() else None
                    nxt = [x2 for x2, c2 in zip(xs, [ci] * len(xs)) if x2 > y]
                    r["frames_to_next_move_same_cube"] = int(nxt[0] - y) if nxt else None
                rows.append(r)
    det = [r for r in rows if r["detected"]]; miss = [r for r in rows if not r["detected"]]
    out = {"moves": len(rows), "recall": len(det) / max(1, len(rows)),
           "recall_by_cube": {ci: float(np.mean([r["detected"] for r in rows if r["cube"] == ci])) for ci in range(len(sl)) if any(r["cube"] == ci for r in rows)},
           "cube_to_identity": {int(c): int(k) for c, k in ident_of.items()},
           "misses_with_identity": len([r for r in miss if r["identity"] is not None]),
           "misses_no_rest_after": len([r for r in miss if r.get("rest_after") is False]),
           "misses_no_rest_before": len([r for r in miss if r.get("rest_before") is False]),
           "miss_next_move_within_50": len([r for r in miss if (r.get("frames_to_next_move_same_cube") or 999) <= 50]),
           "miss_examples": miss[:12]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
