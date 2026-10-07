#!/usr/bin/env python3
"""Build the state-track event demo page (local; 2026-10-05).

Inputs: rendered val frames (render_frames.py, job 57626), the official OGBench state val datasets, and the val events
of the state-track runs 57613 (cube-triple) and 57614 (puzzle-4x5) written by u_events.py --per-frame.
Ground truth (cube motion from qpos, button toggles) is PRIVILEGED and shown for checking only.
Output: SITE/index.html (data inline) and SITE/sprite_<fam>_ep<k>.webp (all frames of an episode, 32 x 32 tiles).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FR, VAL, EV, SITE = (Path(p) for p in sys.argv[1:5])
SITE.mkdir(parents=True, exist_ok=True)
TILE, GRID, RW = 128, 32, 160                       # sprite tile px, tiles per row, render px
FAM = {
    "cube": {"ds": "cube-triple-play-v0", "run": "cube-triple-play-v0_57613",
             "cam": ((1.053, -0.014, 0.639), (0.000, 1.000, 0.000, -0.628, 0.001, 0.778))},
    "puzzle": {"ds": "puzzle-4x5-play-v0", "run": "puzzle-4x5-play-v0_57614",
               "cam": ((0.905, 0.000, 0.762), (0.000, 1.000, 0.000, -0.771, 0.000, 0.637))},
}


def projector(cam, fovy=45.0, size=RW):
    pos, ax = np.array(cam[0]), np.array(cam[1])
    xa, ya = ax[:3] / np.linalg.norm(ax[:3]), ax[3:] / np.linalg.norm(ax[3:])
    za = np.cross(xa, ya)
    f = (size / 2) / np.tan(np.radians(fovy) / 2)

    def proj(p):                                     # world (..., 3) -> image (..., 2) in render px
        d = np.asarray(p) - pos
        xc, yc, zc = d @ xa, d @ ya, d @ za
        return np.stack([size / 2 + f * xc / -zc, size / 2 - f * yc / -zc], -1)
    return proj


def canvas_to_m(uv):
    uv = np.asarray(uv, np.float64)
    return np.stack([(uv[..., 0] - 32) / 80 + 0.425, (uv[..., 1] - 32) / 80], -1)


def cube_truth(xyz):
    """PRIVILEGED: motion intervals per cube (speed > 2 mm/step, gaps <= 3 merged, net xy displacement >= 2 cm)."""
    out = []
    v = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    for c in range(xyz.shape[1]):
        mv = v[:, c] > 2e-3
        d = np.diff(np.r_[0, mv.astype(np.int8), 0]); st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        ms = []
        for x, y in zip(st, en):
            if ms and x - ms[-1][1] <= 3:
                ms[-1][1] = y
            else:
                ms.append([x, y])
        for x, y in ms:
            if np.linalg.norm(xyz[min(y, len(xyz) - 1), c, :2] - xyz[x, c, :2]) >= 0.02:
                out.append({"a": int(x), "b": int(y), "k": c})
    return sorted(out, key=lambda r: r["a"])


CROSS = []
for i in range(20):
    r, c = divmod(i, 5); m = {i}
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 <= r + dr < 4 and 0 <= c + dc < 5:
            m.add((r + dr) * 5 + c + dc)
    CROSS.append(m)


def puzzle_truth(bs):
    """PRIVILEGED: press frames (state t -> t+1 toggles) and the pressed button (centre of the toggled cross)."""
    out = []
    for t in np.nonzero((bs[1:] != bs[:-1]).any(1))[0]:
        tog = set(np.nonzero(bs[t + 1] != bs[t])[0].tolist())
        pressed = next((i for i, m in enumerate(CROSS) if m == tog), -1)
        out.append({"a": int(t), "b": int(t + 1), "k": pressed, "tog": sorted(tog)})
    return out


data = {"families": {}}
for fam, cfg in FAM.items():
    z = np.load(VAL / f"{cfg['ds']}-val.npz")
    term, qpos, obs = z["terminals"], z["qpos"], z["observations"]
    ends = np.nonzero(term)[0]; starts = np.r_[0, ends[:-1] + 1]
    ev = dict(np.load(EV / cfg["run"] / "events_val.npz"))
    L = json.loads((EV / cfg["run"] / "layout.json").read_text())
    rep = json.loads((EV / cfg["run"] / "report.json").read_text())
    proj = projector(cfg["cam"])
    tol_pos, thr_app = float(ev["tol_pos"]), float(ev["thr_app"])
    eps = []
    for k in (0, 1):
        a, b = int(starts[k]), int(ends[k]) + 1
        fz = np.load(FR / f"{cfg['ds']}_val_ep{k}.npz")["frames"]
        T = len(fz)
        sheet = Image.new("RGB", (GRID * TILE, int(np.ceil(T / GRID)) * TILE))
        for t in range(T):
            sheet.paste(Image.fromarray(fz[t]).resize((TILE, TILE), Image.LANCZOS), ((t % GRID) * TILE, (t // GRID) * TILE))
        name = f"sprite_{fam}_ep{k}.webp"
        sheet.save(SITE / name, "WEBP", quality=78, method=6)
        sc = TILE / RW
        if fam == "cube":
            xyz = np.stack([qpos[a:b, 14 + 7 * i:17 + 7 * i] for i in range(3)], 1).astype(np.float64)
            img = proj(xyz) * sc
            p0 = proj(xyz[0]); col = []
            for (u, v) in p0:                                             # entity colour = its pixels in frame 0
                u, v = int(np.clip(u, 2, RW - 3)), int(np.clip(v, 2, RW - 3))
                col.append("#%02x%02x%02x" % tuple(fz[0][v - 2:v + 3, u - 2:u + 3].reshape(-1, 3).mean(0).astype(int)))
            series = {"z": np.round(xyz[..., 2], 4).T.tolist(), "img": np.round(img, 1).transpose(1, 0, 2).tolist()}
            truth = cube_truth(xyz)
            names = [f"cube {i}" for i in range(3)]
        else:
            bs = z["button_states"][a:b].astype(np.int8)
            loc = canvas_to_m(np.array(L["loc_uv"]))
            eff = obs[:, 12:15] / 10 + np.array([0.425, 0.0, 0.0])
            depth = obs[:, 19 + 2::4]
            zpress = float(np.median(eff[(depth < 0).any(1), 2]))        # effector height while a button is down
            img = proj(np.c_[loc, np.full(len(loc), zpress)]) * sc
            col = ["#7a8691"] * 20
            series = {"bits": ["".join(map(str, r)) for r in bs.tolist()], "img": np.round(img, 1).tolist()}
            truth = puzzle_truth(bs)
            names = [f"btn {i}" for i in range(20)]
        sel = (ev["t_start"] >= a) & (ev["t_start"] < b)
        events = []
        for i in np.nonzero(sel)[0]:
            bef, aft, e = ev["before"][i], ev["after"][i], int(ev["e"][i])
            ch = ((np.linalg.norm(aft[:, :2] - bef[:, :2], axis=-1) > tol_pos) | (np.abs(aft[:, 2:5] - bef[:, 2:5]).max(-1) > thr_app))
            ch[e] = True
            s0, s1 = int(ev["t_start"][i]) - a, int(ev["t"][i]) - a
            if fam == "cube":
                tm = canvas_to_m(aft[e, :2]); tz = float(aft[e, 2]) / 0.25 / 10
                tgt = {"x": round(float(tm[0]), 3), "y": round(float(tm[1]), 3), "z": round(tz, 3),
                       "img": np.round(proj(np.r_[tm, tz]) * sc, 1).tolist()}
                hit = [r for r in truth if r["a"] <= s1 + 5 and r["b"] >= s0 - 5]
                match = {"n": len(hit), "same_object": bool(hit) and all(r["k"] == e for r in hit)}
            else:
                tgt = {"state": int(round(float(aft[e, 2])))}
                hit = [r for r in truth if s0 - 3 <= r["a"] <= s1 + 3]
                match = {"n": len(hit), "same_object": len(hit) == 1 and hit[0]["k"] == e}
            events.append({"a": s0, "b": s1, "e": e, "changed": np.nonzero(ch)[0].tolist(), "tgt": tgt, "match": match})
        eps.append({"k": k, "T": T, "sprite": name, "tile": TILE, "grid": GRID, "series": series, "truth": truth,
                    "events": events})
        print(fam, k, "frames", T, "events", len(events), "truth", len(truth), flush=True)
    data["families"][fam] = {"names": names, "colors": col, "episodes": eps, "report": {s: rep[s] for s in ("thr_pos", "tol_pos", "thr_app") if s in rep}}

html = (Path(__file__).parent / "template.html").read_text()
(SITE / "index.html").write_text(html.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print("written", SITE / "index.html", round((SITE / "index.html").stat().st_size / 1e6, 2), "MB", flush=True)
