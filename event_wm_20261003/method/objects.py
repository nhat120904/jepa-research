#!/usr/bin/env python3
"""Component 7 (method/README.md, object entities): objects from the agent-free view -> entity table in the u_events format
(entities_{split}.npz: pos px, app RGB/255, area, agent, processed; discover.json). One rule set for every environment:
thresholds are 2-means splits of data statistics or the majority (1/2), no per-environment setting.

CHANGED PIXELS. Per pixel the TYPICAL colour is the mode of its agent-free colours quantized to --levels per channel over
TRAIN discovery frames (the empty table, a drawer in its most frequent state). A pixel is CHANGED when it leaves its typical
bin with a chromaticity (rgb / sum) change above tau_c, the 2-means split of log chromaticity distances of pixels that leave
their bin: shading and shadows keep the chromaticity and are not changes (a brightening test made the arm's shadow, where it
is the typical state, look like an object when the shadow left). Agent pixels (segmenter p > .5, dilated by one token)
are not observed.
AGENT COLOURS. The mask is coarse (tokens, dilated, with the arm's shadow): inside it a changed pixel is the agent's when its
colour bin is more frequent among changed pixels inside the mask than among changed pixels at least w from it (objects,
places); the others are objects seen through the margin or the shadow. Movers and the stored agent mask use this refined
mask (cube-triple VAL: cubes visible 75-88% of frames instead of 32%; scene cube 57% instead of 8%); places keep the mask.
INSTANCES per frame: connected components of changed pixels (8-connected, 1-pixel gaps bridged); position = centroid,
appearance = mean chromaticity. Object width w = diameter of the median discovery instance. An instance is FREE when it
is at least w from the agent, UNTOUCHED when no 8-neighbour is agent, at REST when the same exact centroid is found one
discovery frame later (--discover-episodes TRAIN episodes, every --stride frames).

PLACES (things that stay where they are). Candidate pixels: (i) masks of free rest instances whose exact centroid recurs in
at least two episodes and in more than half of the episodes showing any free rest instance within w (deterministic
rendering repeats a fixed thing's states exactly; a free object rests at continuous positions); (ii) pixels changed in
more than half of their free observations (no dominant appearance: a drawer, a window). A place is a connected
component of candidates that is PERSISTENT: its state (changed or not) flips between consecutive discovery frames at less
than half the rate of independent draws (flip rate / 2p(1-p) < 1/2); the arm's unsegmented parts flicker and fail.
State: position = centroid of its unoccluded changed pixels (its centre when none: a sliding part moves it), appearance =
mean colour of its unoccluded pixels weighted by their temporal variance; unobserved while the agent covers more than
half of that weight. DISCRETE places (a light, a button): full-view appearances bimodal along their principal axis
(Ashman D > 2) with each side dominated by one exact reading (more than half of the side: deterministic rendering repeats
a state exactly) snap to their state, in full view and through SeeThrough codes alike: appearance = the mean of its side,
position = the place centre (one state = one exact value). Dev VAL dominant shares per side: puzzle lights .72-1.0, scene
buttons .99-1.0, scene window .20 / .25 and drawer .20 / .13 (D 4.4 / 3.8: they slide, the reading is continuous and the
changed-pixel centroid tracks the slide). History: snapping the appearance only left light 3 of puzzle-4x5 at (39.0, 25.0)
in full view and (37.5, 24.5) through the lookup in one state, a 1.6 px "move" above tol_pos that made spurious events;
centre positions for every place lost the slide (scene privileged |corr| window .98 -> .72, drawer .97 -> .62).
MOVERS (things that move). Untouched rest instances that touch no place. M = the median over episodes of the most of
them seen at once; M = 0: no movers. Identities = colour clusters (area-weighted k-means of chromaticities); K grows from M
while one more cluster halves the frames in which one cluster is seen twice (one object is never seen twice at once),
by more than twice the Poisson noise of that count; a cluster seen in fewer than half of the discovery episodes is not an object of the environment. Per frame the
changed pixels of instances that touch no place go to the nearest prototype if nearer than their typical colour; an
identity is observed at the centroid of its largest component when that has more than half of its median rest area
(an edge pixel of another object is not the object). State = position, appearance = prototype colour.
Assumptions, stated: a fixed camera and deterministic rendering (exact recurrence; a noisy camera needs a tolerance);
objects of one environment differ in chromaticity from each other and from what they rest on, or stay at their places.
TABLE: every frame of --episodes TRAIN / --val-episodes VAL episodes, in parallel over episode chunks (--workers).
PRIVILEGED (scoring only, discover.json / objects_report.json): cube envs identity -> cube fit (cm); otherwise the best
|correlation| of each identity's state with a button state, the drawer / window joint or the cube position.
History (local, 2026-10-08): SAM 2 identity rules gave a colour-merge radius of 0 on exact single-colour segments (13,783
clusters); touching links chained all classes; recursive 2-means of chromaticities gave 70 types; change-frequency
places took the cube table (2-means always splits, also with one class); free (>= w) rest movers rarely showed all three
cubes at once (count 2; colour co-occurrence fixes K); an undilated mask raised visibility but let the purple gripper
pass as the red scene cube.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from utils import episode_bounds, save_json, two_means_threshold

EIGHT = np.ones((3, 3), bool)
VV, UU = np.mgrid[0:64, 0:64]


def split2(x, fallback):
    x = np.asarray(x, np.float64)
    if len(x) < 4:
        return fallback, 0.0
    th, D, _ = two_means_threshold(x)
    return (th if D > 2 else fallback), D


def agent_mask(seg, idx, dilate=1):
    from frontend import dilate_tokens
    m = np.asarray(seg[idx]) > 127
    m = dilate_tokens(m.reshape(len(m), 256), dilate).reshape(-1, 16, 16)
    return np.repeat(np.repeat(m, 4, 1), 4, 2)


def quant_key(x, L):
    q = (x.astype(np.int32) * L) // 256
    return (q[..., 0] * L + q[..., 1]) * L + q[..., 2]


def typical_colours(src, frames, L, chunk=2000):
    nb = L ** 3
    cnt = np.zeros((4096, nb), np.int64); sm = np.zeros((4096, nb, 3), np.float64)
    pix = np.arange(4096)
    for s in range(0, len(frames), chunk):
        f = frames[s:s + chunk]
        x, ag = src.get(f)
        x = x.reshape(len(f), 4096, 3)
        k = quant_key(x, L)
        free = ~ag.reshape(len(f), 4096)
        for j in range(len(f)):
            m = free[j]
            np.add.at(cnt, (pix[m], k[j, m]), 1)
            np.add.at(sm, (pix[m], k[j, m]), x[j, m])
    mode = cnt.argmax(1)
    rgb = sm[pix, mode] / np.maximum(cnt[pix, mode], 1)[:, None]
    return mode.reshape(64, 64), rgb.reshape(64, 64, 3).astype(np.float32), float((cnt.max(1) > 0).mean())


def material(x, key, tkey, trgb):
    """-> (pixel left its typical bin, chromaticity distance to the typical colour)."""
    diff = key != tkey
    xs = x.astype(np.float32)
    s_x, s_t = xs.sum(-1, keepdims=True) + 1.0, trgb.sum(-1, keepdims=True) + 1.0
    return diff, np.linalg.norm(xs / s_x - trgb / s_t, axis=-1)


def changed(x, agent, P):
    """materially changed pixels of frames x (..., 64, 64, 3): left the typical bin with a chromaticity change > tau_c."""
    diff, chroma = material(x, quant_key(x, P["L"]), P["tkey"], P["trgb"])
    return diff & (chroma > P["tau_c"]) & ~agent


def refine_agent(x, ag, P):
    """agent mask -> (agent pixels, changed object pixels): inside the mask, changed pixels whose colour bin is not an agent
    colour (discover: AGENT COLOURS) are objects, not agent."""
    key = quant_key(x, P["L"])
    diff, chroma = material(x, key, P["tkey"], P["trgb"])
    call = diff & (chroma > P["tau_c"])
    ag_r = ag & ~(call & ~P["agent_bin"][key])
    return ag_r, call & ~ag_r


def chroma_of(px):
    """mean chromaticity (r, g of rgb / sum) of pixels (n, 3)."""
    px = px.astype(np.float64)
    c = px / (px.sum(-1, keepdims=True) + 1.0)
    return c[:, :2].mean(0)


def frame_instances(x, agent, P):
    """one frame -> list of (pos (2,), area, radius, chroma (2,), mask): connected components of changed pixels."""
    ch = changed(x, agent, P)
    if not ch.any():
        return []
    lab, n = ndimage.label(ndimage.binary_dilation(ch, EIGHT), structure=EIGHT)
    lab = np.where(ch, lab, 0)
    out = []
    for sl_i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        m = np.zeros((64, 64), bool); m[sl] = lab[sl] == sl_i
        a = int(m.sum())
        if not a:
            continue
        u, v = UU[m].mean(), VV[m].mean()
        r = float(np.sqrt(((UU[m] - u) ** 2 + (VV[m] - v) ** 2).mean()))
        out.append((np.array([u, v], np.float32), a, r, chroma_of(x[m]), m))
    return out


def kmeans_w(X, wt, K, seed=0, restarts=5, iters=50):
    """area-weighted k-means -> (sse, centres, labels)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        c = X[rng.choice(len(X), K, replace=False, p=wt / wt.sum())]
        for _ in range(iters):
            lb = np.linalg.norm(X[:, None] - c[None], axis=-1).argmin(1)
            c = np.stack([(X[lb == k] * wt[lb == k, None]).sum(0) / wt[lb == k].sum() if (lb == k).any() else c[k] for k in range(K)])
        sse = float((wt * np.linalg.norm(X - c[lb], axis=-1) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, c, lb)
    return best


def violations(F, lb):
    """number of frames with >= 2 instances in which two of them carry the same label (one object cannot be seen twice)."""
    by = {}
    for f_, l_ in zip(F, lb):
        by.setdefault(int(f_), []).append(int(l_))
    return sum(len(v) != len(set(v)) for v in by.values() if len(v) >= 2)


def unpack(b):
    return np.unpackbits(b, axis=-1)[..., :64].astype(bool)


def recurring(ep, pos, w):
    """instances (episode, exact centroid) -> index of one instance (the first) per exact centroid that recurs in at least
    two episodes and in more than half of the episodes showing any instance within w of it (deterministic rendering
    repeats a fixed thing's states exactly; a free object rests at continuous positions)."""
    pos = np.asarray(pos, np.float64)
    key = np.round(pos[:, 0] * 1000).astype(np.int64) * 100000 + np.round(pos[:, 1] * 1000).astype(np.int64)
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    pairs = np.unique(np.c_[inv, np.asarray(ep).astype(np.int64)], axis=0)
    n_exact = np.bincount(pairs[:, 0], minlength=len(uk))
    upos = np.zeros((len(uk), 2)); np.add.at(upos, inv, pos); upos /= np.maximum(np.bincount(inv, minlength=len(uk)), 1)[:, None]
    eps_u = [[] for _ in range(len(uk))]
    for u_, e_ in pairs:
        eps_u[u_].append(e_)
    tree = cKDTree(upos) if len(uk) else None
    out = [first[u_] for u_ in np.flatnonzero(n_exact >= 2)
           if n_exact[u_] > 0.5 * len({e for v_ in tree.query_ball_point(upos[u_], w) for e in eps_u[v_]})]
    return np.array(out, np.int64)


def change_instances(x0, x1, unobs, P):
    """two frames of one episode -> CHANGE instances [(pos (2,), area, mask)]: connected components (1-pixel gaps bridged)
    of the pixels observed in both that leave their colour bin with a chromaticity change above tau_c (shading is not a
    change), not touching an unobserved pixel (a cut instance has no fixed centroid)."""
    a, b = x0.astype(np.float32), x1.astype(np.float32)
    cd = np.linalg.norm(a / (a.sum(-1, keepdims=True) + 1.0) - b / (b.sum(-1, keepdims=True) + 1.0), axis=-1)
    d = (quant_key(x0, P["L"]) != quant_key(x1, P["L"])) & (cd > P["tau_c"]) & ~unobs
    if not d.any():
        return []
    lab, n = ndimage.label(ndimage.binary_dilation(d, EIGHT), structure=EIGHT)
    lab = np.where(d, lab, 0)
    near = ndimage.binary_dilation(unobs, EIGHT)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        m = np.zeros((64, 64), bool); m[sl] = lab[sl] == i
        if m.any() and not (m & near).any():
            out.append((np.array([UU[m].mean(), VV[m].mean()]), int(m.sum()), m))
    return out


def discover(a, src, frames, P, rep):
    """object identities from TRAIN discovery frames (module doc: PLACES and MOVERS)."""
    nF = len(frames)
    term = np.load(a.cache / "train_terminals.npy"); ep_of = np.concatenate([[0], np.cumsum(term[:-1])])
    eps = ep_of[frames]
    rows, masks, dts, chg = [], [], [], []                                      # rows: frame idx, episode, u, v, area, cr, cg, agent dist
    drows, dmasks, prev = [], [], None                                          # change instances (iii): episode, u, v
    for c0 in range(0, nF, 1000):
        fr = frames[c0:c0 + 1000]
        X, AG = src.get(fr)
        chg.append(np.packbits(changed(X, AG, P), axis=-1))
        for j in range(len(fr)):
            dt = ndimage.distance_transform_edt(~AG[j]) if AG[j].any() else np.full((64, 64), 1e3)
            dts.append(dt.astype(np.float16))
            for p_, ar, r_, cr, mi in frame_instances(X[j], AG[j], P):
                rows.append((c0 + j, ep_of[fr[j]], p_[0], p_[1], ar, cr[0], cr[1], float(dt[mi].min())))
                masks.append(np.packbits(mi, axis=-1))
            e_ = ep_of[fr[j]]
            if prev is not None and prev[2] == e_:
                for p_, ar, mi in change_instances(prev[0], X[j], prev[1] | AG[j], P):
                    drows.append((e_, p_[0], p_[1])); dmasks.append(np.packbits(mi, axis=-1))
            prev = (X[j], AG[j], e_)
    R = np.array(rows, np.float64).reshape(-1, 8)
    chg = np.concatenate(chg)
    rep["discovery_instances_per_frame"] = float(len(R) / nF)
    if not len(R):
        return [], np.zeros((64, 64), bool), 1.0
    w = float(max(2.0 * np.sqrt(np.median(R[:, 4]) / np.pi), 1.0))            # object width: diameter of the median instance
    F = R[:, 0].astype(np.int64)
    free = R[:, 7] >= w                                                         # FREE: at least one object width from the agent
    # AGENT COLOURS: a changed pixel inside the agent mask is the agent's when its colour bin is more frequent among changed
    # pixels inside the mask than among changed pixels at least w from it (objects, places); otherwise it is an object
    # seen through the mask's margin or the arm's shadow (refine_agent)
    nb = P["L"] ** 3
    c_in = np.zeros(nb); c_out = np.zeros(nb)
    for c0 in range(0, nF, max(1, nF // 5000)):
        xx, aa = src.get(frames[c0:c0 + 1]); x, ag = xx[0], aa[0]
        key = quant_key(x, P["L"]); diff, chroma = material(x, key, P["tkey"], P["trgb"])
        call = diff & (chroma > P["tau_c"])
        np.add.at(c_in, key[call & ag], 1); np.add.at(c_out, key[call & (dts[c0] >= w)], 1)
    P["agent_bin"] = (c_in / max(c_in.sum(), 1)) > (c_out / max(c_out.sum(), 1))
    rep["agent_colours"] = {"bins_agent": int(P["agent_bin"].sum()), "changed_px_inside": int(c_in.sum()), "changed_px_free": int(c_out.sum()),
                            "inside_share_agent": float(c_in[P["agent_bin"]].sum() / max(c_in.sum(), 1))}
    untouched = R[:, 7] >= 2                                                    # no 8-neighbour of the instance is agent
    key = np.round(R[:, 2] * 1000).astype(np.int64) * 100000 + np.round(R[:, 3] * 1000).astype(np.int64)
    nxt = set(zip((F - 1).tolist(), key.tolist()))
    rest = np.array([(f_, k_) in nxt for f_, k_ in zip(F.tolist(), key.tolist())], bool)   # same exact centroid one discovery frame later
    # PLACES (i): free rest instances whose exact centroid recurs in other episodes more than any centroid within w
    fi = np.flatnonzero(free & rest)
    cand = np.zeros((64, 64), bool)
    for i in fi[recurring(R[fi, 1], R[fi, 2:4], w)]:
        cand |= unpack(masks[i])
    n_recur_px = int(cand.sum())
    # PLACES (iii): change instances (two consecutive discovery frames of an episode) whose exact centroid recurs as in (i).
    # A place switching between states of similar frequency has no typical colour (the per-pixel mode mixes its states,
    # so (i) sees fragments of it), but its switch between two states changes the same exact pixels every time.
    DR = np.array(drows, np.float64).reshape(-1, 3)
    ch_idx = recurring(DR[:, 0], DR[:, 1:3], w) if len(DR) else np.zeros(0, np.int64)
    cand_ch = np.zeros((64, 64), bool)
    for i in ch_idx:
        cand_ch |= unpack(dmasks[i])
    rep["change_instances"] = {"instances": int(len(DR)), "recurring_centroids": int(len(ch_idx)), "pixels": int(cand_ch.sum()),
                               "pixels_new": int((cand_ch & ~cand).sum()), "used": bool(a.change_places)}
    # off by default (2026-10-10): (iii) added 0-7 place pixels on puzzle-3x3 / 4x4 / 4x5 (no measurable gain), but 62 on
    # scene, where they joined the window place (44 -> 105 px; privileged |corr| .98 -> .92, window acted .94 -> .86)
    if a.change_places:
        cand |= cand_ch
    # PLACES (ii): pixels changed in more than half of their free observations (no dominant appearance: a drawer, a window)
    nchg = np.zeros((64, 64)); nfree = np.zeros((64, 64))
    for i in range(nF):
        fm = dts[i] >= w
        nchg += unpack(chg[i]) & fm; nfree += fm
    share = nchg / np.maximum(nfree, 1)
    cand |= (share > 0.5) & (nfree > 0)
    lab, n_pl = ndimage.label(ndimage.binary_dilation(cand, EIGHT), structure=EIGHT)
    lab = np.where(cand, lab, 0)
    comps = [lab == i for i in range(1, n_pl + 1) if (lab == i).any()]
    # persistence: a place keeps its state between consecutive discovery frames (both observed: more than half of its
    # pixels free) far more often than independent draws would; rho = flip rate / 2p(1-p) < 1/2, p = share of observed
    # frames in which more than half of its observed pixels are changed
    C = len(comps)
    st_obs = np.zeros((nF, C), bool); st_chg = np.zeros((nF, C), bool)
    for i in range(nF):
        fm = dts[i] >= w; ch = unpack(chg[i])
        for c_, m in enumerate(comps):
            nfm = (fm & m).sum()
            st_obs[i, c_] = 2 * nfm > m.sum()
            st_chg[i, c_] = 2 * (ch & fm & m).sum() > nfm
    same_ep = eps[1:] == eps[:-1]
    places, kept_info = [], []
    for c_, m in enumerate(comps):
        o = st_obs[:, c_]; z = st_chg[:, c_]
        pr = same_ep & o[1:] & o[:-1]
        p_ = float(z[o].mean()) if o.any() else 0.0
        flip = float((z[1:] != z[:-1])[pr].mean()) if pr.any() else 1.0
        rho = flip / (2 * p_ * (1 - p_)) if 0 < p_ < 1 else np.inf
        kept_info.append({"centre": [round(float(UU[m].mean()), 1), round(float(VV[m].mean()), 1)], "area": int(m.sum()),
                          "p_changed": round(p_, 3), "rho": round(float(rho), 3)})
        if rho < 0.5:
            places.append({"anchor": "location", "disc": m, "centre": np.array([UU[m].mean(), VV[m].mean()], np.float64)})
    # a place whose changed pixels carry agent colours at least as often as the agent mask's own changed pixels is a part of
    # the agent the segmenter misses (a robot pixel that flips with the arm's pose), not an object
    a_in = rep["agent_colours"]["inside_share_agent"]
    cnt = np.zeros((len(places), 2))
    for c0 in range(0, nF, max(1, nF // 5000)):
        x = src.get(frames[c0:c0 + 1])[0][0]; key = quant_key(x, P["L"])
        diff, chroma = material(x, key, P["tkey"], P["trgb"])
        call = diff & (chroma > P["tau_c"]) & (dts[c0] >= w)
        for c_, d in enumerate(places):
            mm = d["disc"] & call
            cnt[c_] += (P["agent_bin"][key[mm]].sum(), mm.sum())
    share_ag = cnt[:, 0] / np.maximum(cnt[:, 1], 1)
    for c_, d in enumerate(places):
        kept_info_d = next(q for q in kept_info if q["centre"] == [round(float(d["centre"][0]), 1), round(float(d["centre"][1]), 1)])
        kept_info_d["agent_colour_share"] = round(float(share_ag[c_]), 3)
    places = [d for c_, d in enumerate(places) if share_ag[c_] < a_in]
    place_px = np.zeros((64, 64), bool)
    for d in places:
        place_px |= d["disc"]
    on_place = np.array([(unpack(mk) & place_px).any() for mk in masks]) if place_px.any() else np.zeros(len(R), bool)
    rep["places"] = {"object_width_px": w, "free_rest_instances": int(len(fi)), "recurring_px": n_recur_px,
                     "changed_majority_px": int(((share > 0.5) & (nfree > 0)).sum()), "candidates": kept_info,
                     "places": len(places), "place_pixels": int(place_px.sum()),
                     "areas": [int(d["disc"].sum()) for d in places], "centres": [np.round(d["centre"], 1).tolist() for d in places]}
    # MOVERS: rest instances that touch neither the agent (which may cut them) nor a place. M = median over episodes of the
    # most of them seen at once (0: no movers). Identities = colour clusters (area-weighted k-means of chromaticity) with K from M up while one more
    # cluster halves the number of frames in which one cluster is seen twice (beyond Poisson noise); a cluster seen in fewer than half of the
    # discovery episodes is not an object of the environment.
    mvr = untouched & rest & ~on_place
    # as for places: an instance whose pixels carry agent colours at least as often as the agent mask's own changed pixels
    # is the agent's (or shares its colours; a grey window frame sliding in scene), not a mover
    cand_i = np.flatnonzero(mvr)
    for f_ in np.unique(F[cand_i]):
        x = src.get(frames[f_:f_ + 1])[0][0]; key = quant_key(x, P["L"])
        for i_ in cand_i[F[cand_i] == f_]:
            if P["agent_bin"][key[unpack(masks[i_])]].mean() >= a_in:
                mvr[i_] = False
    rep["movers_agent_coloured_dropped"] = int(len(cand_i) - mvr.sum())
    Rm, Fm = R[mvr], F[mvr]
    cnt = np.bincount(Fm, minlength=nF)
    per_ep = np.array([cnt[eps == e].max() for e in np.unique(eps)])
    M = int(round(np.median(per_ep))) if len(per_ep) else 0
    rep["movers"] = {"rest_instances": int(len(Rm)), "max_at_once_per_episode_median": M,
                     "max_at_once_histogram": np.bincount(per_ep.astype(np.int64)).tolist() if len(per_ep) else []}
    if getattr(a, "discover_only", False):                                      # diagnostics of the discovery instances
        np.savez(a.out / "discovery_rows.npz", R=R, free=free, rest=rest, on_place=on_place, share=share, DR=DR,
                 key=np.round(R[:, 2] * 1000).astype(np.int64) * 100000 + np.round(R[:, 3] * 1000).astype(np.int64))
    movers = []
    if M > 0 and len(Rm) > M:
        X, wt = Rm[:, 5:7], Rm[:, 4]
        fits, viol = {}, {}
        for K in range(M, min(int(per_ep.max()) + 1, len(Rm)) + 1):
            fits[K] = kmeans_w(X, wt, K); viol[K] = violations(Fm, fits[K][2])
        K = M
        # one more cluster is an object when it halves the violating frames and the drop exceeds twice its Poisson noise
        while K + 1 in viol and viol[K + 1] <= viol[K] / 2 and viol[K] - viol[K + 1] > 2 * np.sqrt(viol[K]):
            K += 1
        _, cent, lb = fits[K]
        n_ep = len(np.unique(eps))
        presence = np.array([len(np.unique(Rm[lb == k, 1])) / n_ep for k in range(K)])
        rep["movers"].update({"violations": {int(k): round(v, 4) for k, v in viol.items()}, "K": K,
                              "prototypes_chroma": np.round(cent, 4).tolist(),
                              "prototype_mass": np.round([wt[lb == k].sum() / wt.sum() for k in range(K)], 3).tolist(),
                              "presence": np.round(presence, 3).tolist()})
        for k in np.flatnonzero(presence >= 0.5):
            movers.append({"anchor": "mover", "proto": cent[k].astype(np.float32), "colour": np.r_[cent[k], 1 - cent[k].sum()].astype(np.float32),
                           "area": float(np.median(Rm[lb == k, 4]))})
    # place pixel weights: temporal variance over agent-free frames
    if places:
        fs = frames[:: max(1, nF // 3000)]
        X, AG = src.get(fs); X = X.astype(np.float64) / 255.0
        for L_ in places:
            D_ = L_["disc"]; px = X[:, D_]; ok = ~AG[:, D_]
            c = np.maximum(ok.sum(0), 1)[:, None]
            s1 = (px * ok[..., None]).sum(0); s2 = (px ** 2 * ok[..., None]).sum(0)
            wv = np.clip(s2 / c - (s1 / c) ** 2, 0, None).sum(-1)
            L_["w"] = wv if wv.sum() > 0 else np.ones_like(wv)
    return movers + places, place_px, w


def grow_fragments(src, frames, idents, w, place_px, rep, P):
    """FRAGMENTS (F1b, 2026-10-11): a location place below half the median place area -- a light under the robot's resting arm
    that raw frames and views both show only by an edge (puzzle-3x3 top-middle light: 2 px, PRIVILEGED |corr| .10; the light's
    other pixels are agent-free in 2-50% of frames) -- grows by the pixels within one object width whose colour, in frames
    where both are agent-free, follows its own (|corr| of their chromaticity along the fragment's main axis of colour change
    with the fragment's >= the 2-means split of the window's correlations, at least .5; intensity barely changes between a
    light's states: .12-.42). Its per-pixel weights are recomputed as in discover(). -> place_px."""
    from utils import two_means_threshold
    locs = [j for j, d in enumerate(idents) if d["anchor"] == "location"]
    if len(locs) < 3:
        return place_px
    areas = np.array([int(idents[j]["disc"].sum()) for j in locs]); med = float(np.median(areas))
    small = [j for j, ar in zip(locs, areas) if ar < 0.5 * med]
    rep["fragments_grown"] = []
    if not small:
        return place_px
    fs = frames[:: max(1, len(frames) // 8000)]
    X, AG = src.get(fs)
    # the pixel-level agent (refine_agent, as the tables read it): the dilated token mask left the 3x3 fragment agent-free in
    # 28 of 8040 frames, the refined one in ~5%
    AG = np.stack([refine_agent(X[i], AG[i], P)[0] for i in range(len(X))])
    Xf = X.astype(np.float32)
    CH = (Xf / (Xf.sum(-1, keepdims=True) + 1.0))[..., :2]                      # chromaticity (r, g): states differ in colour
    for j in small:
        d = idents[j]; D_ = d["disc"]; cu, cv = d["centre"]
        win = (np.hypot(UU - cu, VV - cv) <= w) & ~D_ & ~place_px
        ok_d = ~AG[:, D_].any(1)
        if ok_d.sum() < 50:
            rep["fragments_grown"].append({"identity": int(j), "skipped": "agent-free frames", "n": int(ok_d.sum()), "frames": int(len(fs))})
            continue
        cc = CH[:, D_].mean(1)                                                   # (n, 2) the fragment's colour
        u_ = np.linalg.svd(cc[ok_d] - cc[ok_d].mean(0), full_matrices=False)[2][0]   # its main axis of change
        core = cc @ u_
        ys, xs = np.nonzero(win)
        cs = np.zeros(len(ys))
        for n_, (y, x) in enumerate(zip(ys, xs)):
            okp = ok_d & ~AG[:, y, x]
            sig = CH[okp, y, x] @ u_
            if okp.sum() >= 50 and core[okp].std() > 0 and sig.std() > 0:
                cs[n_] = abs(np.corrcoef(core[okp], sig)[0, 1])
        if not len(cs) or cs.max() < 0.5:
            rep["fragments_grown"].append({"identity": int(j), "skipped": "no pixel follows it", "max_corr": round(float(cs.max()) if len(cs) else 0.0, 3)})
            continue
        thr = max(0.5, two_means_threshold(cs)[0]) if len(cs) >= 2 else 0.5
        add = np.zeros((64, 64), bool); add[ys[cs >= thr], xs[cs >= thr]] = True
        new = D_ | add
        px = X[:, new].astype(np.float64) / 255.0; ok = ~AG[:, new]
        c = np.maximum(ok.sum(0), 1)[:, None]
        s1 = (px * ok[..., None]).sum(0); s2 = (px ** 2 * ok[..., None]).sum(0)
        wv = np.clip(s2 / c - (s1 / c) ** 2, 0, None).sum(-1)
        rep["fragments_grown"].append({"identity": int(j), "area": [int(D_.sum()), int(new.sum())], "threshold": round(float(thr), 3),
                                       "centre": [np.round(d["centre"], 1).tolist(), [round(float(UU[new].mean()), 1), round(float(VV[new].mean()), 1)]]})
        d["disc"], d["centre"], d["w"] = new, np.array([UU[new].mean(), VV[new].mean()], np.float64), (wv if wv.sum() > 0 else np.ones_like(wv))
        place_px = place_px | new
    return place_px


def place_tokens(idents, tokens):
    """indices of the token entities (memory_entities.py, 16 x 16 grid of 4 px tokens) that overlap each place."""
    out = {}
    for j, d in enumerate(idents):
        if d["anchor"] == "location":
            out[j] = [i for i, t in enumerate(tokens) if d["disc"][(t // 16) * 4:(t // 16) * 4 + 4, (t % 16) * 4:(t % 16) * 4 + 4].any()]
    return out


def place_reading(x, ch, d):
    """a place seen in full -> (chromaticity of its pixels weighted by their temporal variance, centroid of its changed
    pixels or its centre). Exact for an unchanged state under deterministic rendering; shading does not change it."""
    D_, wv = d["disc"], d["w"]
    px = x[D_].astype(np.float32)
    cpx = px / (px.sum(-1, keepdims=True) + 1.0)
    cm = D_ & ch
    return (wv[:, None] * cpx).sum(0) / wv.sum(), ((UU[cm].mean(), VV[cm].mean()) if cm.any() else tuple(d["centre"]))


def see_lookup(a, src, frames, P, idents):
    """places read through the agent: per place, the SeeThrough codes of its tokens (memory_entities.py tables) -> the
    reading most often seen in full with those codes (TRAIN discovery frames). Codes are deterministic per state."""
    z = np.load(a.tokens / "entities_train.npz")
    rd, dg = z["area"], np.load(a.tokens / "app_train.npy", mmap_mode="r")
    ptk = place_tokens(idents, z["tokens"])
    cnt = {j: {} for j in ptk}
    for c0 in range(0, len(frames), 1000):
        fr = frames[c0:c0 + 1000]
        X, AG = src.get(fr); R_ = rd[fr]; G_ = np.asarray(dg[fr])
        for i in range(len(fr)):
            ch = None
            for j, tk in ptk.items():
                d = idents[j]
                if not tk or AG[i][d["disc"]].any() or not (R_[i, tk] >= 1).all():
                    continue
                if ch is None:
                    ch = refine_agent(X[i], AG[i], P)[1]
                ap_, pp = place_reading(X[i], ch, d)
                val = (tuple(np.round(ap_, 5).tolist()), tuple(np.round(pp, 3).tolist()))
                c_ = cnt[j].setdefault(G_[i, tk].tobytes(), {}); c_[val] = c_.get(val, 0) + 1
    for j, tk in ptk.items():
        idents[j]["tokens"] = tk
        idents[j]["see"] = {k: max(c.items(), key=lambda kv: kv[1])[0] for k, c in cnt[j].items()}
    return {int(j): len(idents[j]["see"]) for j in ptk}


def reader_context(idents, P):
    """constants of the per-frame reading: mover indices and prototypes, typical chromaticity per pixel."""
    mov = [j for j, d in enumerate(idents) if d["anchor"] == "mover"]
    return {"mov": mov, "protos": np.stack([idents[j]["proto"] for j in mov]) if mov else np.zeros((0, 2)),
            "tch": (P["trgb"] / (P["trgb"].sum(-1, keepdims=True) + 1.0))[..., :2]}


def read_partial_movers(x, ag, P, idents, place_px, ctx, lo=0.25):
    """PARTIAL readings of movers in one frame: the largest component of a mover's colour holds more than lo but at most half
    of its median rest area (read_frame drops those: less than half visible). Used on the goal image only, where an object
    under another in a stack shows a side face (cube-triple 3-stack goals: 11-16 px of 29). -> {identity: (u, v)}"""
    out = {}
    mov, protos, tch = ctx["mov"], ctx["protos"], ctx["tch"]
    _, ch = refine_agent(x, ag, P)
    if not mov or not ch.any():
        return out
    lab, n = ndimage.label(ndimage.binary_dilation(ch, EIGHT), structure=EIGHT)
    lab = np.where(ch, lab, 0)
    mv = ch & ~np.isin(lab, np.unique(lab[place_px & (lab > 0)]))
    if not mv.any():
        return out
    px = x[mv].astype(np.float32)
    c = (px / (px.sum(-1, keepdims=True) + 1.0))[:, :2]
    d = np.linalg.norm(c[:, None] - protos[None], axis=-1)
    own, ok = d.argmin(1), d.min(1) < np.linalg.norm(c - tch[mv], axis=-1)
    for a_i, j in enumerate(mov):
        m = np.zeros((64, 64), bool); m[mv] = ok & (own == a_i)
        if not m.any():
            continue
        l2, _ = ndimage.label(ndimage.binary_dilation(m, EIGHT), structure=EIGHT)
        l2 = np.where(m, l2, 0)
        mm = l2 == (np.bincount(l2.ravel())[1:].argmax() + 1)
        if lo * idents[j]["area"] < mm.sum() and 2 * mm.sum() <= idents[j]["area"]:
            out[j] = (float(UU[mm].mean()), float(VV[mm].mean()))
    return out


def read_frame(x, ag, P, idents, place_px, ctx, rd=None, dg=None, contact=None):
    """one frame x (64, 64, 3) uint8 and its unobserved pixels ag (the dilated segmenter tokens of the raw frame, or the
    hidden tokens of its view, view.py) -> pos (K, 2), app (K, 3), area (K,) (0 = unobserved), refined agent mask. rd / dg:
    readable flags (Kt,) and SeeThrough digits (Kt, 6) of the token entities (memory_entities.py order) in this frame, or
    None (places then only in full view). contact = (raw frame, its dilated agent mask) when x is a view: the returned
    agent mask (contact, rest tests) is then refined on the raw frame. Offline table and closed loop use this function."""
    K = len(idents)
    pos = np.zeros((K, 2), np.float32); app = np.zeros((K, 3), np.float32); area = np.zeros(K, np.int16)
    mov, protos, tch = ctx["mov"], ctx["protos"], ctx["tch"]
    ag_r, ch = refine_agent(x, ag, P)                                           # objects seen inside the mask count as seen
    # MOVERS are read on the raw frame (agent colours through the transparent arm, as without views): a view exemplar shows
    # what recurs at a token, and a moving object never recurs exactly (cube-triple with view-read movers: visible .8 -> .6)
    xm, chm = (contact[0], refine_agent(contact[0], contact[1], P)[1]) if contact is not None else (x, ch)
    if mov and chm.any():
        lab, n = ndimage.label(ndimage.binary_dilation(chm, EIGHT), structure=EIGHT)
        lab = np.where(chm, lab, 0)
        mv = chm & ~np.isin(lab, np.unique(lab[place_px & (lab > 0)]))          # instances touching a place belong to it
        if mv.any():
            px = xm[mv].astype(np.float32)
            c = (px / (px.sum(-1, keepdims=True) + 1.0))[:, :2]
            d = np.linalg.norm(c[:, None] - protos[None], axis=-1)
            own = d.argmin(1)
            ok = d.min(1) < np.linalg.norm(c - tch[mv], axis=-1)                  # nearer the prototype than the typical colour
            for a_i, j in enumerate(mov):
                m = np.zeros((64, 64), bool); m[mv] = ok & (own == a_i)
                if not m.any():
                    continue
                l2, n2 = ndimage.label(ndimage.binary_dilation(m, EIGHT), structure=EIGHT)
                l2 = np.where(m, l2, 0)
                mm = l2 == (np.bincount(l2.ravel())[1:].argmax() + 1)             # the largest component of the object's pixels
                if 2 * mm.sum() <= idents[j]["area"]:
                    continue                                                      # observed only when most of it is visible
                pos[j] = (UU[mm].mean(), VV[mm].mean()); app[j] = idents[j]["colour"]; area[j] = int(mm.sum())
    for j, d_ in enumerate(idents):
        if d_["anchor"] != "location":
            continue
        if not ag[d_["disc"]].any():                                              # seen in full (the whole dilated mask hides)
            ap_, pp = place_reading(x, ch, d_)                                    # rounded as the lookup values
            app[j], pos[j] = snap_reading(np.round(ap_, 5), np.round(pp, 3), d_); area[j] = int(d_["disc"].sum())
        elif rd is not None and d_.get("see") and (rd[d_["tokens"]] >= 1).all():    # under the agent: its tokens' SeeThrough codes
            hit = d_["see"].get(np.asarray(dg[d_["tokens"]]).tobytes())
            if hit is not None:
                app[j] = hit[0]; pos[j] = hit[1]; area[j] = 1
    if contact is not None:
        ag_r = refine_agent(contact[0], contact[1], P)[0]
    return pos, app, area, ag_r


def place_states(a, src, frames, P, idents):
    """discrete places: the full-view readings of a place (TRAIN discovery frames) split by 2-means along their principal
    axis; when bimodal (Ashman D > 2) and each side is dominated by one exact reading (more than half of the side), every
    reading snaps to its side (appearance = mean of the side, position = the place centre), so one state is one exact value
    (a light, a button). Exact states make rests, prototypes, canonical values and goal tests agree. -> per place (D,
    dominant share of each side, snapped)."""
    from utils import two_means_threshold
    pl = [j for j, d in enumerate(idents) if d["anchor"] == "location"]
    vals = {j: [] for j in pl}; poss = {j: [] for j in pl}
    for c0 in range(0, len(frames), max(1, len(frames) // 8000)):
        xx, aa = src.get(frames[c0:c0 + 1]); x, ag = xx[0], aa[0]
        ch = None
        for j in pl:
            if ag[idents[j]["disc"]].any():
                continue
            if ch is None:
                ch = refine_agent(x, ag, P)[1]
            ap_, pp = place_reading(x, ch, idents[j])
            vals[j].append(ap_); poss[j].append(pp)
    out = {}
    for j in pl:
        X = np.asarray(vals[j], np.float64); Xp = np.asarray(poss[j], np.float64)
        if len(X) < 4:
            continue
        mu = X.mean(0); U = np.linalg.svd(X - mu, full_matrices=False)[2][0]
        th, D, _ = two_means_threshold((X - mu) @ U)
        lo, hi = (X - mu) @ U <= th, (X - mu) @ U > th
        dom = [float(np.unique(np.round(X[s_], 5), axis=0, return_counts=True)[1].max() / s_.sum()) if s_.any() else 0.0 for s_ in (lo, hi)]
        discrete = bool(D > 2 and min(dom) > 0.5)
        out[int(j)] = {"D": round(float(D), 2), "dominant_share": [round(v, 3) for v in dom], "snapped": discrete}
        if discrete:                                                             # a state = (side appearance, centre)
            c_ = np.round(np.asarray(idents[j]["centre"], np.float64), 3)
            idents[j]["snap"] = (mu, U, th, np.round(X[lo].mean(0), 5), np.round(X[hi].mean(0), 5), c_, c_)
    for j in pl:                                                                 # lookup readings snap the same way
        if idents[j].get("snap") is not None and idents[j].get("see"):
            idents[j]["see"] = {k: tuple(tuple(np.asarray(u_).tolist()) for u_ in snap_reading(v[0], v[1], idents[j]))
                                for k, v in idents[j]["see"].items()}
    return out


def snap_reading(app_, pos_, d):
    """(appearance, position) reading -> the state (appearance, position) of its side for a two-state place (place_states),
    unchanged otherwise."""
    sn = d.get("snap")
    if sn is None:
        return np.asarray(app_), np.asarray(pos_)
    mu, U, th = sn[:3]
    side = 1 if float((np.asarray(app_, np.float64) - mu) @ U) > th else 0
    return sn[3 + side], (sn[5 + side] if len(sn) > 5 else np.asarray(pos_))


def write_chunk(args):
    """worker: entity rows for a list of episodes -> (s0, e0, pos, app, area, agent) per episode."""
    cache, run, split, eps, idents_pk, place_px, w, P, tokens, view = args
    idents = pickle.loads(idents_pk)
    src = source(Path(cache), Path(run), split, view)
    K = len(idents)
    ctx = reader_context(idents, P)
    if tokens:                                                                  # SeeThrough codes of the token entities
        rd = np.load(Path(tokens) / f"entities_{split}.npz")["area"]; dg = np.load(Path(tokens) / f"app_{split}.npy", mmap_mode="r")
    out = []
    for s0, e0 in eps:
        T = e0 - s0 + 1
        pos = np.zeros((T, K, 2), np.float32); app = np.zeros((T, K, 3), np.float32); area = np.zeros((T, K), np.int16)
        agp = np.zeros((T, 64, 8), np.uint8)
        idx = np.arange(s0, e0 + 1)
        X, AG = src.get(idx)
        XR, AR = src.raw(idx) if view else (None, None)
        RD, DG = (rd[s0:e0 + 1], np.asarray(dg[s0:e0 + 1])) if tokens else (None, None)
        for i in range(T):
            pos[i], app[i], area[i], ag_r = read_frame(X[i], AG[i], P, idents, place_px, ctx,
                                                       RD[i] if tokens else None, DG[i] if tokens else None,
                                                       contact=(XR[i], AR[i]) if view else None)
            agp[i] = np.packbits(ag_r, axis=-1)
        out.append((s0, e0, pos, app, area, agp))
    return out


def source(cache: Path, run: Path, split: str, view=None):
    """frames of a split as the rules see them (view.Source): raw frames + agent mask, or views when view = (exemplar
    table, dir of the SeeThrough codes see_codes_{split}.npy / see_prob_{split}.npy)."""
    from view import Source
    obs = np.load(cache / f"{split}_observations.npy", mmap_mode="r")
    seg = np.load(run / f"seg_{split}.npy", mmap_mode="r")
    if not view:
        return Source(obs, seg)
    table, see = Path(view[0]), Path(view[1])
    return Source(obs, seg, table, np.load(see / f"see_codes_{split}.npy", mmap_mode="r"), np.load(see / f"see_prob_{split}.npy", mmap_mode="r"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="front-end dir (seg_{split}.npy)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--discover-episodes", type=int, default=200)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, default=None,
                    help="memory_entities.py output (token entities with SeeThrough codes): places are also read under the agent")
    ap.add_argument("--discover-only", action="store_true", help="stop after discovery (discover report + place map)")
    ap.add_argument("--view", type=Path, default=None,
                    help="view.py output (view_table.npz): every rule runs on views (agent removed, view.py) instead of raw frames")
    ap.add_argument("--see", type=Path, default=None, help="dir with see_codes_{split}.npy / see_prob_{split}.npy (default --tokens)")
    ap.add_argument("--change-places", action="store_true", help="PLACES (iii): change instances join the place candidates (ablation)")
    ap.add_argument("--keep-fragments", action="store_true",
                    help="ablation: keep raw places that are fragments of a view place (before 2026-10-11)")
    ap.add_argument("--view-discovery", action="store_true",
                    help="with --view: discover every object on the views (before 2026-10-10) instead of raw frames + the views' discrete places")
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    rep = {"run": str(a.run), "cache": str(a.cache), "levels": a.levels, "view": str(a.view) if a.view else None}
    view = (str(a.view / "view_table.npz"), str(a.see or a.tokens)) if a.view else None
    src = source(a.cache, a.run, "train", view)                                  # reading source (views: places under the agent)
    # DISCOVERY on raw frames: views read places under the agent and add only the DISCRETE places that raw frames never show
    # agent-free (2026-10-10: discovery on views split scene objects, 6 identities for 5 with the drawer 88 -> 246 px, and
    # lowered cube visibility; raw discovery = v1 on cube-triple / scene, the views add the small boards' hidden lights)
    src_d = source(a.cache, a.run, "train", None) if (view and not a.view_discovery) else src
    st, en = episode_bounds(a.cache, "train", a.discover_episodes)
    frames = np.concatenate([np.arange(s0, e0 + 1, a.stride) for s0, e0 in zip(st, en)])
    tkey, trgb, seen = typical_colours(src_d, frames, a.levels)
    samp = np.sort(frames[np.random.default_rng(0).permutation(len(frames))[:3000]])
    xs, ags = src_d.get(samp)
    d_, c_ = material(xs, quant_key(xs, a.levels), tkey[None], trgb[None])
    sel = d_ & ~ags
    thc, Dc = split2(np.log(c_[sel] + 1e-5), np.log(0.03))
    P = {"L": a.levels, "tkey": tkey, "trgb": trgb, "tau_c": float(np.exp(thc))}
    rep["material"] = {"tau_chroma": P["tau_c"], "chroma_D": Dc, "pixels_seen_free": seen}
    print(json.dumps(rep), flush=True)
    idents, place_px, w = discover(a, src_d, frames, P, rep)
    raw_places = [j for j, d in enumerate(idents) if d["anchor"] == "location"]
    add_idx = []
    if src_d is not src:                                                          # discrete places seen only through the views
        rep_v = {}
        idv, _, _ = discover(a, src, frames, dict(P), rep_v)
        # a raw place that is a FRAGMENT of a place the views see whole (a view place within w of it with more than twice its
        # area) takes the view's pixels: a light under the agent's rest pose is seen agent-free only in pieces on raw frames
        # (2026-10-11 puzzle-4x5 light 3: 2 raw px, one of them background, vs 8 in the view; state contrast .17 vs ~.27 for the
        # other lights, readings that flickered with the arm, the most timeouts of the learned loop). On the five dev boards
        # only that place changes.
        frag = []
        vloc = [d for d in idv if d["anchor"] == "location"]
        for j in (raw_places if (vloc and not a.keep_fragments) else []):
            dd = np.array([np.linalg.norm(d["centre"] - idents[j]["centre"]) for d in vloc]); i = int(dd.argmin())
            if dd[i] <= w and vloc[i]["disc"].sum() > 2 * idents[j]["disc"].sum():
                frag.append({"raw": np.round(idents[j]["centre"], 1).tolist(), "raw_area": int(idents[j]["disc"].sum()),
                             "view": np.round(vloc[i]["centre"], 1).tolist(), "view_area": int(vloc[i]["disc"].sum())})
                # the whole view place (disc, centre and its per-pixel weights w: replacing only disc and centre left 2
                # weights for 8 pixels, a crash in see_lookup)
                idents[j] = {k_: (v_.copy() if isinstance(v_, np.ndarray) else v_) for k_, v_ in vloc[i].items()}
        if frag:
            place_px = np.zeros((64, 64), bool)
            for d in idents:
                if d["anchor"] == "location":
                    place_px |= d["disc"]
        rc = [idents[j]["centre"] for j in raw_places]
        cand = [d for d in idv if d["anchor"] == "location" and all(np.linalg.norm(d["centre"] - c) > w for c in rc)]
        st_v = place_states(a, src, frames, P, cand) if cand else {}
        added = [d for j_, d in enumerate(cand) if st_v.get(j_, {}).get("snapped")]
        for d in added:
            d.pop("snap", None)
            place_px = place_px | d["disc"]
        add_idx = list(range(len(idents), len(idents) + len(added)))
        idents = idents + added
        rep["view_places"] = {"fragments_refined": frag,
                              "candidates": [np.round(d["centre"], 1).tolist() for d in cand], "added": [np.round(d["centre"], 1).tolist() for d in added],
                              "view_discovery": {k: rep_v[k] for k in ("places",) if k in rep_v}}
    if not a.keep_fragments:                                                     # F1b: grow places seen only by an edge
        place_px = grow_fragments(src_d, frames, idents, w, place_px, rep, P)
    if a.tokens is not None:
        rep["see_lookup_states"] = see_lookup(a, src, frames, P, idents)
    # snapping from agent-free raw readings (exact) for raw places, from the views for the places only the views show
    pst = place_states(a, src_d, frames, P, [idents[j] for j in raw_places])
    rep["place_states"] = {int(raw_places[k]): v for k, v in pst.items()}
    if add_idx:
        pst_v = place_states(a, src, frames, P, [idents[j] for j in add_idx])
        rep["place_states"].update({int(add_idx[k]): v for k, v in pst_v.items()})
    K = len(idents)
    rep["identities"] = [{"anchor": d["anchor"], "centre": None if d["anchor"] == "mover" else np.round(d["centre"], 2).tolist(),
                          "proto": None if d["anchor"] != "mover" else np.round(d["proto"], 4).tolist()} for d in idents]
    print(json.dumps({"identities": K, "movers": sum(d["anchor"] == "mover" for d in idents), "places": sum(d["anchor"] == "location" for d in idents),
                      "object_width": w, "min": round((time.time() - t0) / 60, 1)}), flush=True)
    vt = dict(np.load(a.view / "view_table.npz")) if a.view else {}              # the closed loop builds the same views
    np.savez(a.out / "front.npz", tkey=tkey, trgb=trgb, tau_c=P["tau_c"], levels=a.levels, place_px=place_px, w=w, agent_bin=P["agent_bin"],
             **({"view_keys": vt["keys"], "view_patches": vt["patches"]} if vt else {}))
    (a.out / "idents.pkl").write_bytes(pickle.dumps({"idents": idents}))
    if a.discover_only:
        save_json(a.out / "objects_report.json", rep)
        print(json.dumps({k: rep[k] for k in ("places", "movers")}, default=float), flush=True)
        return
    from multiprocessing import get_context
    pk = pickle.dumps(idents)
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        s_, e_ = episode_bounds(a.cache, split, n_ep)
        n = int(e_[-1] + 1)
        eps = list(zip(s_.tolist(), e_.tolist()))
        nch = max(1, min(len(eps), a.workers * 4))
        jobs = [(str(a.cache), str(a.run), split, eps[i::nch], pk, place_px, w, P, str(a.tokens) if a.tokens else "", view) for i in range(nch)]
        pos = np.zeros((n, K, 2), np.float32); app = np.zeros((n, K, 3), np.float32); area = np.zeros((n, K), np.int16)
        agent = np.zeros((n, 64, 8), np.uint8)
        with get_context("spawn").Pool(a.workers) as pool:
            for res in pool.imap_unordered(write_chunk, jobs):
                for s0, e0, p_, ap_, ar_, ag_ in res:
                    pos[s0:e0 + 1] = p_; app[s0:e0 + 1] = ap_; area[s0:e0 + 1] = ar_; agent[s0:e0 + 1] = ag_
        np.savez(a.out / f"entities_{split}.npz", pos=pos, app=app, area=area, agent=agent, processed=np.ones(n, bool))
        print(json.dumps({"table": split, "frames": n, "min": round((time.time() - t0) / 60, 1)}), flush=True)
    disc = {"objects": K, "source": "method/objects.py", "groups": [[i] for i in range(K)],
            "table": [{"cluster": i, "spread_median": float(w / np.sqrt(6))} for i in range(K)],
            "identities": [{"anchor": "location" if d["anchor"] == "location" else "colour",
                            "centre": None if d["anchor"] == "mover" else np.round(d["centre"], 2).tolist(),
                            "radius": None if d["anchor"] == "mover" else round(max(w / 2, 1.0), 2)} for d in idents],
            "object_width": w}
    disc["privileged_diagnostic"] = privileged(a, K)
    rep["privileged_diagnostic"] = disc["privileged_diagnostic"]
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / "discover.json").write_text(json.dumps(disc, indent=1, default=float) + "\n")
    save_json(a.out / "objects_report.json", rep)
    print(json.dumps({"identities": K, "privileged": rep["privileged_diagnostic"], "minutes": rep["minutes"]}, default=float), flush=True)


def privileged(a, K):
    out = []
    z = np.load(a.out / "entities_val.npz")
    n = len(z["processed"])
    q = np.load(a.cache / "val_qpos.npy", mmap_mode="r")
    if "cube" in a.cache.name:
        slices = [s for s in (14, 21, 28, 35) if s + 3 <= q.shape[1]]
        xyz = np.stack([np.asarray(q[:n, s:s + 3]) for s in slices], 1)
        half = n // 2
        for k in range(K):
            vis = z["area"][:, k] >= 1
            f = np.c_[np.ones(n), z["pos"][:, k], z["pos"][:, k] ** 2]
            best = None
            for j in range(len(slices)):
                trm, tem = vis & (np.arange(n) < half), vis & (np.arange(n) >= half)
                if trm.sum() < 30 or tem.sum() < 30:
                    continue
                W = np.linalg.lstsq(f[trm], xyz[trm, j, :2], rcond=None)[0]
                err = float(np.median(np.linalg.norm(f[tem] @ W - xyz[tem, j, :2], axis=-1)) * 100)
                if best is None or err < best["median_err_cm"]:
                    best = {"identity": k, "cube": j, "median_err_cm": round(err, 2), "visible_frac": round(float(vis.mean()), 3)}
            out.append(best)
        return out
    refs = {}
    bf = a.cache / "val_button_states.npy"
    if bf.exists():
        bs = np.asarray(np.load(bf, mmap_mode="r")[:n]).astype(np.float64)
        refs.update({f"button{j}": bs[:, j] for j in range(bs.shape[1])})
    if "scene" in a.cache.name:
        refs.update({"drawer": np.asarray(q[:n, 23], np.float64), "window": np.asarray(q[:n, 24], np.float64),
                     "cube_x": np.asarray(q[:n, 14], np.float64), "cube_y": np.asarray(q[:n, 15], np.float64)})
    for k in range(K):
        vis = z["area"][:, k] >= 1
        best = (0.0, None)
        for name, y in refs.items():
            for feat in (z["app"][:, k, 0], z["app"][:, k, 1], z["app"][:, k, 2], z["pos"][:, k, 0], z["pos"][:, k, 1]):
                xv, yv = feat[vis], y[vis]
                if len(xv) < 30 or xv.std() < 1e-6 or yv.std() < 1e-9:
                    continue
                c = abs(np.corrcoef(xv, yv)[0, 1])
                if c > best[0]:
                    best = (c, name)
        out.append({"identity": k, "best_ref": best[1], "abs_corr": round(float(best[0]), 3), "visible_frac": round(float(vis.mean()), 3)})
    return out


if __name__ == "__main__":
    main()
