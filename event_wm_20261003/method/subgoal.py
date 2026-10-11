#!/usr/bin/env python3
"""Component 16b (low level L1, with gcivl.py): SUBGOAL IMAGES for one planned event. The image-goal policy (gcivl.py) is
given the event-start frame with the entities the event is predicted to change drawn in their predicted next state, so
the policy only has to reach a goal one event away. No labels: frames, the objects table (objects.py), its typical colours.
One rule per entity KIND, the kinds being those objects.py already tells apart from data:

DISCRETE places (snapped states: puzzle lights, scene buttons). Per place k and state s, the per-pixel median of TRAIN
  frames in which k reads s and the agent mask leaves its own disc clear. REGION = the place's own pixels (objects.py
  disc) dilated by 1 px whose two state medians differ beyond the 2-means split of those differences (a window of 2 object
  widths took neighbour pixels: see main). Drawn by pasting the target state.
MOVERS (cubes; G4 2026-10-10). EXEMPLAR = the per-pixel median of crops (side 2 * ceil(1.5 w) + 1) centred on the mover's
  rest readings with no agent pixel within 2 object widths; MASK = crop pixels whose median differs from the median of the
  same crops of the typical-colour image (objects.py front.npz trgb: the empty scene) beyond the 2-means split. Drawn by
  erasing the mover at its current position (typical colours) and pasting the exemplar at its next position (not when
  its next state is covered).
CONTINUOUS places (not snapped: a sliding drawer, window; G4). A BANK of TRAIN frames with the place read and the agent
  clear of its sliding window (its readings' 95th-percentile spread around the median position + 2 object widths);
  REGION = window pixels whose temporal std over the bank exceeds the 2-means split. Drawn by copying the region from the
  bank frame whose reading (position, appearance; standardised over the bank) is nearest the next state.
Outputs (--out): subgoals.npz, subgoal_report.json, subgoal_panels.png (VAL frames: frame, a rendered next state).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from utils import save_json


def two_means(x):
    """1-D 2-means split (Lloyd from the min / max)."""
    x = np.sort(np.asarray(x, np.float64).ravel())
    if len(x) < 2 or x[0] == x[-1]:
        return np.inf
    lo, hi = x[0], x[-1]
    for _ in range(50):
        thr = (lo + hi) / 2
        a_, b_ = x[x <= thr], x[x > thr]
        if not len(a_) or not len(b_):
            break
        lo2, hi2 = a_.mean(), b_.mean()
        if lo2 == lo and hi2 == hi:
            break
        lo, hi = lo2, hi2
    return (lo + hi) / 2


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def state_index(app, states):
    """app (..., A), states (2, A) -> nearest state index (...)."""
    return np.argmin(np.linalg.norm(np.asarray(app)[..., None, :] - states, axis=-1), axis=-1)


def render(frame, targets, Z):
    """discrete places only: frame (64, 64, 3) uint8, targets {place id: state index} -> subgoal image."""
    out = np.array(frame, copy=True)
    pos = {int(k): j for j, k in enumerate(Z["ids"])}
    for k, s in targets.items():
        j = pos.get(int(k))
        if j is None:
            continue
        m = Z["regions"][j]
        out[m] = Z["images"][j, int(s)][m]
    return out


def _stamp(out, patch, mask, centre, src=None):
    """paste patch[mask] (P, P, ...) centred at centre (u, v) px into out; src: an image to take the pixels from instead."""
    P = mask.shape[0]; h = P // 2
    cu, cv = int(round(float(centre[0]))), int(round(float(centre[1])))
    ys, xs = np.nonzero(mask)
    yy, xx = ys - h + cv, xs - h + cu
    ok = (yy >= 0) & (yy < 64) & (xx >= 0) & (xx < 64)
    if src is None:
        out[yy[ok], xx[ok]] = patch[ys[ok], xs[ok]]
    else:
        out[yy[ok], xx[ok]] = src[yy[ok], xx[ok]]


def render_state(frame, S, S_next, changed, Z):
    """every kind: frame (64, 64, 3) uint8, current / next entity states (K, D) (pos 2, app A, covered), changed (K,) bool
    -> subgoal image (64, 64, 3) uint8 with the changed entities drawn in their next state."""
    out = np.array(frame, copy=True)
    S, S_next = np.asarray(S), np.asarray(S_next)
    disc = {int(k): j for j, k in enumerate(Z.get("ids", []))}
    mov = {int(k): j for j, k in enumerate(Z.get("mover_ids", []))}
    cont = {int(k): j for j, k in enumerate(Z.get("cont_ids", []))}
    bg = Z["background"] if "background" in Z else None
    ks = [int(k) for k in np.flatnonzero(changed)]
    for k in ks:                                                                # movers leave first (their old place shows the empty scene)
        if k in mov and bg is not None:
            _stamp(out, None, ndimage.binary_dilation(Z["mover_mask"][mov[k]]), S[k, :2], src=bg)   # 1 px more: no leftover rim
    for k in ks:
        if k in disc:
            j = disc[k]; m = Z["regions"][j]
            out[m] = Z["images"][j, int(state_index(S_next[k, 2:-1], Z["states"][j]))][m]
        elif k in mov:
            if S_next[k, -1] < 0.5:                                             # covered next: only erased
                j = mov[k]; _stamp(out, Z["mover_patch"][j], Z["mover_mask"][j], S_next[k, :2])
        elif k in cont:
            j = cont[k]
            r = np.r_[S_next[k, :2], S_next[k, 2:-1]]
            z = (Z["cont_read"][j] - Z["cont_mu"][j]) / Z["cont_sd"][j]
            i = int(np.argmin(np.linalg.norm(z - (r - Z["cont_mu"][j]) / Z["cont_sd"][j], axis=-1)))
            m = Z["cont_region"][j]
            out[m] = Z["cont_frames"][j][i][m]
    return out


def unpack_agent(agent, t):
    return np.unpackbits(agent[t], axis=-1)[:, :64].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--objects", type=Path, required=True, help="objects.py output (entities_train.npz, discover.json, objects_report.json, front.npz)")
    ap.add_argument("--per-state", type=int, default=1500, help="agent-free frames per place and state / per mover for the median")
    ap.add_argument("--bank", type=int, default=1500, help="frames per continuous place")
    ap.add_argument("--region-window", action="store_true",
                    help="ablation: discrete place regions from a window of 2 object widths (before 2026-10-11)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    E = np.load(a.objects / "entities_train.npz")
    app, area, posE = E["app"], E["area"], E["pos"]
    agent = E["agent"]
    disc = json.loads((a.objects / "discover.json").read_text())
    rep_o = json.loads((a.objects / "objects_report.json").read_text())
    w = float(disc["object_width"])
    snapped = {int(k) for k, v in rep_o.get("place_states", {}).items() if v.get("snapped")}
    obs = np.load(a.cache / "train_observations.npy", mmap_mode="r")
    bg = np.clip(np.load(a.objects / "front.npz")["trgb"], 0, 255).astype(np.uint8)   # the empty scene (typical colours)
    import pickle
    idents = pickle.loads((a.objects / "idents.pkl").read_bytes())["idents"]
    vv, uu = np.mgrid[0:64, 0:64]
    out = {"background": bg}
    ids, S, Rg, IM, rep = [], [], [], [], {"object_width": w, "places": {}, "movers": {}, "continuous": {}}
    m_ids, m_patch, m_mask = [], [], []
    c_ids, c_region, c_read, c_mu, c_sd, c_frames = [], [], [], [], [], []
    P = 2 * int(np.ceil(1.5 * w)) + 1; h = P // 2
    for k, ident in enumerate(disc["identities"]):
        seen = np.flatnonzero(area[:, k] >= 1)
        if len(seen) < 50:
            continue
        if ident.get("anchor") == "location" and k in snapped and ident.get("centre") is not None:
            cu, cv = ident["centre"]
            # within the place's own pixels + 1 px: a per-pixel median turns any correlation of a neighbour's state (or of the
            # arm's whereabouts) with this place's state into a full contrast, so a window of 2 object widths took neighbour
            # pixels (2026-10-11: puzzle-4x5 G1 9 of 20 lights, 4x4 7 of 15; drawing one light redrew a neighbour in a state
            # the event does not reach; scene buttons had none outside)
            win = (np.hypot(uu - cu, vv - cv) <= 2 * w) if a.region_window else ndimage.binary_dilation(idents[k]["disc"], iterations=1)
            own = np.hypot(uu - cu, vv - cv) <= max(float(ident.get("radius") or 1.0), 1.0)
            u_, c_ = np.unique(np.round(app[seen, k], 4), axis=0, return_counts=True)
            if len(u_) < 2:
                continue
            states = u_[np.argsort(-c_)[:2]].astype(np.float32)
            st = state_index(app[seen, k], states)
            meds = []
            for s in (0, 1):
                cand = seen[st == s]
                cand = rng.choice(cand, min(len(cand), 8 * a.per_state), replace=False) if len(cand) else cand
                free = [t for t in np.sort(cand) if not unpack_agent(agent, t)[own].any()][:a.per_state]
                if len(free) < 20:
                    meds = None
                    break
                meds.append(np.median(np.asarray(obs[np.array(free)]), axis=0).astype(np.uint8))
            if meds is None:
                rep["places"][k] = "too few agent-free frames"
                continue
            d = np.abs(meds[1].astype(np.int32) - meds[0].astype(np.int32)).sum(-1)
            region = win & (d > two_means(d[win]))
            ids.append(k); S.append(states); Rg.append(region); IM.append(np.stack(meds))
            rep["places"][k] = {"region_px": int(region.sum()), "states": states.round(4).tolist()}
        elif ident.get("anchor") != "location":                                 # MOVER
            cand = np.sort(rng.choice(seen, min(len(seen), 12 * a.per_state), replace=False))
            free = []
            for t in cand:
                am = unpack_agent(agent, t)
                p = posE[t, k]
                if not (np.hypot(uu[am] - p[0], vv[am] - p[1]) <= 2 * w).any():
                    free.append(t)
                if len(free) >= a.per_state:
                    break
            if len(free) < 20:
                rep["movers"][k] = "too few agent-free rest frames"
                continue
            X = np.pad(np.asarray(obs[np.array(free)]), ((0, 0), (h, h), (h, h), (0, 0)), mode="edge")
            B = np.pad(bg, ((h, h), (h, h), (0, 0)), mode="edge")
            crops, bcrops = [], []
            for i, t in enumerate(free):
                cu, cv = (int(round(float(x))) for x in posE[t, k])
                crops.append(X[i, cv:cv + P, cu:cu + P]); bcrops.append(B[cv:cv + P, cu:cu + P])
            med = np.median(np.stack(crops), 0).astype(np.uint8); bmed = np.median(np.stack(bcrops), 0)
            d = np.abs(med.astype(np.float64) - bmed).sum(-1)
            # the mover with its darker side faces and shadow: half the 2-means split of the median-vs-empty differences,
            # the connected part around the centre (the top-face split alone left dark rims where cubes were erased)
            lab, _ = ndimage.label(d > 0.5 * two_means(d))
            mask = lab == lab[h, h] if lab[h, h] > 0 else d > two_means(d)
            m_ids.append(k); m_patch.append(med); m_mask.append(mask)
            rep["movers"][k] = {"patch": P, "mask_px": int(mask.sum()), "frames": len(free)}
        else:                                                                   # CONTINUOUS place
            c0 = np.median(posE[seen, k], 0)
            spread = float(np.percentile(np.linalg.norm(posE[seen, k] - c0, axis=-1), 95))
            win = np.hypot(uu - c0[0], vv - c0[1]) <= spread + 2 * w
            cand = np.sort(rng.choice(seen, min(len(seen), 12 * a.bank), replace=False))
            # pass 1: the agent clear of the place's own pixels (objects.py disc; the whole sliding window is rarely clear,
            # and the segmenter keeps a static robot part next to the scene window); the region from the temporal std of
            # those frames, without the pixels the agent covers in more than half of them (part of the robot, not the place)
            own = idents[k]["disc"]
            frees = [(t, unpack_agent(agent, t)) for t in cand]
            frees = [(t, am) for t, am in frees if not am[own].any()][:4 * a.bank]
            if len(frees) < 50:
                rep["continuous"][k] = "too few agent-free frames"
                continue
            free = [t for t, _ in frees]
            agf = np.mean([am for _, am in frees], 0)
            X = np.asarray(obs[np.array(free)])
            sd = X.astype(np.float32).std(0).sum(-1)
            region = win & (sd > two_means(sd[win])) & (agf <= 0.5)
            # pass 2: the bank keeps the frames whose agent leaves the region itself clear (nothing of the arm is copied)
            movers_ = [j for j, d_ in enumerate(disc["identities"]) if d_.get("anchor") != "location"]
            reg_d = ndimage.binary_dilation(region, iterations=max(1, int(round(w))))
            def mover_in(t):                                                    # a mover read inside the region would be copied too
                return any(area[t, j] >= 1 and reg_d[int(np.clip(round(float(posE[t, j, 1])), 0, 63)), int(np.clip(round(float(posE[t, j, 0])), 0, 63))]
                           for j in movers_)
            protos = [np.asarray(d_["proto"], np.float64) for d_ in idents if d_.get("anchor") == "mover"]
            bgr = bg[region].astype(np.float64); bgc = (bgr / (bgr.sum(-1, keepdims=True) + 1.0))[:, :2]
            def mover_colour(x):                                                # a hidden (unread) mover in the region: its colour
                if not protos:
                    return False
                px = x[region].astype(np.float64); c = (px / (px.sum(-1, keepdims=True) + 1.0))[:, :2]
                dp = np.min([np.linalg.norm(c - pr, axis=-1) for pr in protos], 0)
                moved = np.abs(px - bgr).sum(-1) > 30
                return int((moved & (dp < np.linalg.norm(c - bgc, axis=-1))).sum()) >= 3
            keep = np.array([not unpack_agent(agent, t)[region].any() and not mover_in(t) and not mover_colour(X[i])
                             for i, t in enumerate(free)])
            if keep.sum() < 20:
                rep["continuous"][k] = "too few frames with the region clear"
                continue
            free = list(np.array(free)[keep][:a.bank]); X = X[keep][:a.bank]
            rd = np.c_[posE[free, k], app[free, k]].astype(np.float32)
            c_ids.append(k); c_region.append(region); c_read.append(rd); c_mu.append(rd.mean(0)); c_sd.append(rd.std(0) + 1e-6); c_frames.append(X)
            rep["continuous"][k] = {"region_px": int(region.sum()), "bank": len(free), "window_radius": round(spread + 2 * w, 1)}
    out.update(ids=np.array(ids, np.int64), states=np.array(S, np.float32).reshape(-1, 2, app.shape[-1]),
               regions=np.array(Rg, bool).reshape(-1, 64, 64), images=np.array(IM, np.uint8).reshape(-1, 2, 64, 64, 3))
    if m_ids:
        out.update(mover_ids=np.array(m_ids, np.int64), mover_patch=np.stack(m_patch), mover_mask=np.stack(m_mask))
    if c_ids:
        nb = max(len(r) for r in c_read)                                        # equal banks (stacked): shorter ones repeat entries
        rep_idx = [np.resize(np.arange(len(r)), nb) for r in c_read]
        out.update(cont_ids=np.array(c_ids, np.int64), cont_region=np.stack(c_region), cont_read=np.stack([r[i] for r, i in zip(c_read, rep_idx)]),
                   cont_mu=np.stack(c_mu), cont_sd=np.stack(c_sd), cont_frames=np.stack([f[i] for f, i in zip(c_frames, rep_idx)]))
    np.savez_compressed(a.out / "subgoals.npz", **out)
    rep.update(n_places=len(ids), n_movers=len(m_ids), n_continuous=len(c_ids))
    save_json(a.out / "subgoal_report.json", rep)
    print(json.dumps({"discrete": len(ids), "movers": len(m_ids), "continuous": len(c_ids),
                      "movers_mask_px": [rep["movers"][k]["mask_px"] for k in m_ids],
                      "continuous_region_px": [rep["continuous"][k]["region_px"] for k in c_ids]}), flush=True)
    # panels (diagnostic): VAL frames, each changed kind drawn: discrete toggled, movers shifted by 2 object widths,
    # continuous places to the bank state farthest from the current reading
    try:
        from PIL import Image
        Z = load(a.out / "subgoals.npz")
        vo = np.load(a.cache / "val_observations.npy", mmap_mode="r")
        Ev = np.load(a.objects / "entities_val.npz")
        K = Ev["pos"].shape[1]
        rows = []
        for t in np.sort(rng.choice(min(len(vo), len(Ev["app"])), 6, replace=False)):
            Sc = np.zeros((K, 2 + app.shape[-1] + 1), np.float32); Sc[:, :2] = Ev["pos"][t]; Sc[:, 2:-1] = Ev["app"][t]
            Sn = Sc.copy(); ch = Ev["area"][t] >= 1
            for j, k in enumerate(Z["ids"]):
                Sn[k, 2:-1] = Z["states"][j][1 - int(state_index(Sc[k, 2:-1], Z["states"][j]))]
            for k in Z.get("mover_ids", []):
                Sn[k, 0] = np.clip(Sc[k, 0] + 2 * w, 4, 60)
            for j, k in enumerate(Z.get("cont_ids", [])):
                d = np.linalg.norm((Z["cont_read"][j] - np.r_[Sc[k, :2], Sc[k, 2:-1]]) / Z["cont_sd"][j], axis=-1)
                Sn[k, :2] = Z["cont_read"][j][int(np.argmax(d)), :2]; Sn[k, 2:-1] = Z["cont_read"][j][int(np.argmax(d)), 2:]
            sg = render_state(vo[t], Sc, Sn, ch, Z)
            rows.append(np.concatenate([vo[t], np.full((64, 2, 3), 255, np.uint8), sg], 1))
        img = np.concatenate(rows, 0)
        Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(a.out / "subgoal_panels.png")
    except Exception as e:                                                      # panels are a diagnostic only
        print("panels failed:", e, flush=True)


if __name__ == "__main__":
    main()
