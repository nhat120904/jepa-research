#!/usr/bin/env python3
"""PRIVILEGED diagnostic references (2026-10-05): compact tables for offline failure analysis of the unified
pipeline. Read-only on caches and run dirs; never used for training or by the method.

Outputs (OUT/):
  cube_triple_{split}_xyz.npy      float16 (n, 3, 3) cube xyz per frame (train) / float32 (val)
  cube_triple_val_qpos.npy         float32 full qpos (val; arm + gripper + cubes)
  cube_triple_moves_{split}.npz    per motion interval: start, end, cube, xyz0, xyz1 (speed > 2e-3 m/step, gaps <= 3 merged)
  puzzle45_{split}_buttons.npy     packed button states (n, ceil(20/8))
  puzzle45_presses_{split}.npz     per toggle frame t (state t -> t+1): toggled mask, pressed button (centre of the cross), n_toggled
  scene_val_qpos.npy / scene_val_buttons.npy / scene_train_buttons_packed.npy
  ent_<tag>_val.npz                compact entity tables on processed frames: frame idx, pos, app, area,
                                   min agent-pixel distance per identity, agent area
  obs_<family>_val_ep{0,1}.npy     raw val frames of the first episodes (uint8)
"""
import json, os, sys
from pathlib import Path
import numpy as np

R = Path("/mnt/data/nhatnc129/jepa/event_wm"); C = R / "cache"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
rep = {}

def eps(term):
    n = len(term); s = np.r_[0, np.nonzero(term)[0] + 1]; e = np.r_[np.nonzero(term)[0] + 1, n]
    k = s < e
    return s[k], e[k]

# ---------------- cube-triple ----------------
for split in ("train", "val"):
    q = np.load(C / "visual-cube-triple-play-v0" / f"{split}_qpos.npy", mmap_mode="r")
    term = np.load(C / "visual-cube-triple-play-v0" / f"{split}_terminals.npy")
    xyz = np.stack([np.asarray(q[:, 14 + 7 * i:17 + 7 * i], np.float32) for i in range(3)], 1)
    np.save(OUT / f"cube_triple_{split}_xyz.npy", xyz.astype(np.float16 if split == "train" else np.float32))
    if split == "val":
        np.save(OUT / "cube_triple_val_qpos.npy", np.asarray(q, np.float32))
    S, E = eps(term)
    rows = []
    for a, b in zip(S, E):
        v = np.linalg.norm(np.diff(xyz[a:b], axis=0), axis=-1)            # (T-1, 3)
        for c in range(3):
            mv = v[:, c] > 2e-3
            d = np.diff(np.r_[0, mv.astype(np.int8), 0]); st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
            # merge gaps <= 3 frames
            ms = []
            for x, y in zip(st, en):
                if ms and x - ms[-1][1] <= 3:
                    ms[-1][1] = y
                else:
                    ms.append([x, y])
            for x, y in ms:
                rows.append((a + x, a + y, c, *xyz[a + x, c], *xyz[min(a + y, b - 1), c]))
    rows = np.array(rows, np.float64)
    o = np.argsort(rows[:, 0], kind="stable"); rows = rows[o]
    np.savez_compressed(OUT / f"cube_triple_moves_{split}.npz", start=rows[:, 0].astype(np.int64), end=rows[:, 1].astype(np.int64),
                        cube=rows[:, 2].astype(np.int64), xyz0=rows[:, 3:6].astype(np.float32), xyz1=rows[:, 6:9].astype(np.float32))
    disp = np.linalg.norm(rows[:, 6:8] - rows[:, 3:5], axis=-1)
    rep[f"cube_{split}"] = {"frames": int(len(term)), "episodes": int(len(S)), "motion_intervals": int(len(rows)),
                            "moves_ge_2cm": int((disp >= 0.02).sum()), "per_episode_ge_2cm": float((disp >= 0.02).sum() / len(S))}
    print(rep[f"cube_{split}"], flush=True)

# ---------------- puzzle-4x5 ----------------
ROWS, COLS = 4, 5
cross = []
for i in range(ROWS * COLS):
    r, c = divmod(i, COLS); m = np.zeros(ROWS * COLS, bool); m[i] = True
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            m[rr * COLS + cc] = True
    cross.append(m)
cross = np.array(cross)
for split in ("train", "val"):
    b = np.asarray(np.load(C / "visual-puzzle-4x5-play-v0" / f"{split}_button_states.npy", mmap_mode="r")).astype(bool)
    term = np.load(C / "visual-puzzle-4x5-play-v0" / f"{split}_terminals.npy")
    np.save(OUT / f"puzzle45_{split}_buttons.npy", np.packbits(b, axis=-1))
    S, E = eps(term)
    same = np.ones(len(b) - 1, bool); same[E[:-1] - 1] = False
    tog = (b[1:] != b[:-1]) & same[:, None]
    t = np.nonzero(tog.any(1))[0]
    tm = tog[t]
    match = (tm[:, None, :] == cross[None]).all(-1)                      # exact cross
    pressed = np.where(match.any(1), match.argmax(1), -1)
    np.savez_compressed(OUT / f"puzzle45_presses_{split}.npz", t=t, toggled=np.packbits(tm, axis=-1), pressed=pressed, n_toggled=tm.sum(1))
    rep[f"puzzle_{split}"] = {"frames": int(len(term)), "episodes": int(len(S)), "toggle_frames": int(len(t)),
                              "exact_cross_frac": float((pressed >= 0).mean()), "per_episode": float(len(t) / len(S)),
                              "press_counts": np.bincount(pressed[pressed >= 0], minlength=20).tolist()}
    print(rep[f"puzzle_{split}"], flush=True)

# ---------------- scene ----------------
sc = C / "visual-scene-play-v0"
np.save(OUT / "scene_val_qpos.npy", np.asarray(np.load(sc / "val_qpos.npy", mmap_mode="r"), np.float32))
np.save(OUT / "scene_val_buttons.npy", np.asarray(np.load(sc / "val_button_states.npy", mmap_mode="r")))
rep["scene_cache_info"] = json.loads((sc / "cache_info.json").read_text()) if (sc / "cache_info.json").exists() else None

# ---------------- compact entity tables (processed val frames) ----------------
ENT = {
    "cube_front": R / "uident_57366_cf/visual-cube-triple-play-v0/front",
    "cube_self": R / "uident_57366_cf/visual-cube-triple-play-v0/self1000/front",
    "puzzle_front": R / "uident_57376_full2/visual-puzzle-4x5-play-v0/front",
    "puzzle_self": R / "uident_57376_full2/visual-puzzle-4x5-play-v0/self1000/front",
    "scene_front": R / "uident_57379_tapp/visual-scene-play-v0/front",
    "cubedouble_front": R / "family_visual-cube-double-play-v0_57420/front",
}
vg, ug = np.mgrid[0:64, 0:64]
for tag, d in ENT.items():
    f = d / "entities_val.npz"
    if not f.exists():
        print("missing", f); continue
    z = np.load(f)
    pos, app, area, agent, proc = z["pos"], z["app"], z["area"], z["agent"], z["processed"]
    idx = np.nonzero(proc)[0]
    P = pos[idx].astype(np.float32); K = P.shape[1]
    dist = np.full((len(idx), K), np.inf, np.float32); aarea = np.zeros(len(idx), np.int32)
    for j, t in enumerate(idx):
        am = np.unpackbits(agent[t], axis=-1)[:, :64].astype(bool)
        aarea[j] = am.sum()
        if am.any():
            dist[j] = np.hypot(ug[am][:, None] - P[j, :, 0][None], vg[am][:, None] - P[j, :, 1][None]).min(0)
    np.savez_compressed(OUT / f"ent_{tag}_val.npz", frame=idx, pos=P, app=app[idx].astype(np.float32), area=area[idx],
                        agent_dist=dist, agent_area=aarea, agent_packed=agent[idx])
    rep[f"ent_{tag}"] = {"processed": int(len(idx)), "K": int(K)}
    print(tag, rep[f"ent_{tag}"], flush=True)

# ---------------- a few raw val episodes ----------------
for fam, env, n_ep in (("cube", "visual-cube-triple-play-v0", 2), ("puzzle", "visual-puzzle-4x5-play-v0", 2), ("scene", "visual-scene-play-v0", 1)):
    o = np.load(C / env / "val_observations.npy", mmap_mode="r"); term = np.load(C / env / "val_terminals.npy")
    S, E = eps(term)
    for e in range(n_ep):
        np.save(OUT / f"obs_{fam}_val_ep{e}.npy", np.asarray(o[S[e]:E[e]]))
(OUT / "refs_report.json").write_text(json.dumps(rep, indent=1) + "\n")
print("done", flush=True)
