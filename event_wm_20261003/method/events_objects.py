#!/usr/bin/env python3
"""Component 8 for object entities (method/README.md): events from the entity table of objects.py.

Copy of docs/generic_state_20261007/base_source/u_events.py (sha256 38fab593d2319c7e, = scripts/u_events.py) with two
changes, both needed by tables that mark an identity as unobserved (area 0: hidden by the agent, covered, or less than half
visible), which the reader tables of the base pipeline never did:
  - --per-frame: an identity is a rest observation only on frames where the table shows it (area >= 1); the base code
    took every processed frame, so an unobserved identity (pos 0, app 0) formed rests at the image corner and every
    occlusion became an event (cube-triple TRAIN 2026-10-08: 25.4 events / episode, the acted cube moved in 0.4% of
    them, acted displacement p50 52 px);
  - a table may cover the first episodes of a longer split (terminals are cut to the table length);
  - noise radii (r_pos, r_app) split the differences between consecutive OBSERVATIONS of an identity, across unobserved
    stretches, instead of between consecutive frames where both are visible (identical for always-visible tables);
  - --per-frame: place identities (discover.json anchor "location") count as rest observations wherever the table shows
    them (objects.py reads a place in full view or through the agent with SeeThrough codes); the agent-clear test stays
    for movers;
and one change of rule, --group stab (default; --group overlap = base): chains of overlapping change windows join
consecutive interactions whenever an identity stays hidden across them (puzzle-4x5 VAL 2026-10-08: 96.6% of object change
windows hold exactly one true flip of their light, yet the chains gave 5.4 events / episode for 29.6 presses). An event
is instead the set of changes whose windows contain one moment (greedy minimum stabbing in arrival order: the fewest
moments that explain every change; no parameter); a window holding several of these moments goes to the one nearest its
centre (an identity hidden by the agent is hidden on its way in and out, so its change lies mid-window; assigned to the
first moment it holds, a light covered while the arm left for the next button joined the previous press and moved its
contact moment off the press: puzzle-4x5 VAL presses inside exactly one event core .789 -> .909 with no slack, cube-triple
.711 -> .714, scene .612 -> .629), and the acted identity is chosen when the last change of the group
departs (all of them under way): among the changed identities the agent touches (an agent pixel within half an object
width, the rest-observation definition of contact) the most central one, else the closest. The coarse agent mask covers
several lights at once, so ranking sub-pixel distances picked a neighbour of the pressed light (puzzle VAL with
PRIVILEGED perfect light readings: acted = pressed .40). The other rules below are unchanged. Original description:

Unified backend, step 1: events from the generic entity table -- one rule set for every task family.

Input: entities_{split}.npz (per frame and identity: pos px, app RGB/255, visible area; agent mask) from
the front end (sam2_tracks.py). Per identity and episode:
  1. rest runs: >= m visible frames whose pos stays within r_pos px and app within r_app of the run mean
     (invisible frames -- occlusion, being covered -- do not break a run); a frame counts as a rest
     observation only while no agent pixel is within half an object width (objects rest free of the agent:
     a cube held still in the gripper is in transit);
  2. consecutive rests are one rest unless they differ by more than tol_pos in position or thr_app in
     appearance; otherwise the gap between them is a change of that identity;
       r_pos = noise / motion split (2-means, Ashman's D > 2) of log |d pos| between consecutive sampled
       frames; tol_pos = 2 r_pos (two rests within r_pos of the same place differ by at most 2 r_pos);
       thr_pos = one object width (sqrt(6) x median mask spread), kept for occupancy and prototypes;
       r_app = 2-means split of log |d app| frame-to-frame, used only if bimodal (Ashman's D > 2) --
       otherwise appearance never changes in this task family; thr_app = 2 r_app (as tol_pos = 2 r_pos);
  3. changes of several identities that overlap (within `gap` frames) form one event; the ACTED entity
     is the changing identity closest to the agent at change onset. Per-frame readings also expose
     changes hidden from contact-free rest runs. Geometric centrality breaks contact ties or supplies
     a fallback when the agent is unobserved; the others are effects the world model must predict;
  4. state of identity k = (pos u, pos v, app r, g, b, covered); covered = never visible during a static
     interval (no move in progress) -- an object on top hides it.
Output per split: events_{split}.npz (t_start, t, e, before (N, K, 6), after, target = after[e], seg_start,
knock = several entities moved, episode) and labels_{split}.npz (per frame rest state + validity, for the
reader); report.json with thresholds and PRIVILEGED diagnostics (event match to qpos / button toggles).

--per-frame (self-training rounds, where the reader has read every frame): every frame of the processed
episodes is a candidate observation, not only the sampled ones (rest runs of m sampled frames need >= 21
frames of rest, which play data rarely leaves between two interactions, so consecutive interactions merged:
job 57371 cube-triple 5.8 events per episode against 11.3 true moves, 57419 puzzle-4x5 6.2 against 29.6
presses). An identity is observed while the agent is clear of it (no agent pixel within half an object
width; a place identity also needs most of its disc uncovered). Each change is then timed on the per-frame
reading: departure = last frame still at the before-rest state, arrival = first frame from which the reading
stays at the after-rest state; changes whose [departure, arrival] windows overlap (slack --frame-gap) form one
event. Before / after states are read on the correct side of the event (last rest observation before it,
first after it). Visible per-frame endpoint readings take precedence over interpolated rest labels,
which can span an unobserved change or refer to a later interaction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import two_means_threshold


def rest_runs(pos, app, vis, a, b, r_pos, r_app, m):
    """Rest runs of one identity in frames [a, b): [start, end, sum_pos, sum_app, n_visible]."""
    runs, cur, cand = [], None, None

    def near(run, p, c):
        return np.hypot(*(p - run[2] / run[4])) <= r_pos and np.abs(c - run[3] / run[4]).max() <= r_app

    for t in range(a, b):
        if not vis[t]:
            continue
        p, c = pos[t], app[t]
        if cur is not None and near(cur, p, c):
            cur[1] = t; cur[2] = cur[2] + p; cur[3] = cur[3] + c; cur[4] += 1; cand = None
            continue
        if cand is not None and near(cand, p, c):
            cand[1] = t; cand[2] = cand[2] + p; cand[3] = cand[3] + c; cand[4] += 1
            if cand[4] >= m:
                if cur is not None:
                    runs.append(cur)
                cur, cand = cand, None
        else:
            cand = [t, t, p.astype(np.float64).copy(), c.astype(np.float64).copy(), 1]
    if cur is not None:
        runs.append(cur)
    return runs


def bimodal_split(x):
    """2-means split of x; (threshold, D) with D = Ashman's separation; inf threshold if not bimodal."""
    thr, sep, _ = two_means_threshold(x)
    return (thr if sep > 2.0 else np.inf), sep


def stab_groups(changes, gap=0):
    """changes (t0, t1, k, episode, ...) -> groups (lists of changes), per episode: the moments of a greedy minimum stabbing
    in arrival order (the earliest remaining arrival; windows starting by then (+ gap) share it), then each change goes to
    the moment, among those its window holds, nearest its window centre (module doc)."""
    groups, by_ep = [], {}
    for c in changes:
        by_ep.setdefault(c[3], []).append(c)
    for ep_ in sorted(by_ep):
        win = {c: (min(c[0], c[1]), max(c[0], c[1])) for c in by_ep[ep_]}       # a change met within one frame: t0 = t1 + 1
        rem = sorted(by_ep[ep_], key=lambda c: win[c][1])
        stabs = []
        while rem:
            stab = win[rem[0]][1]
            stabs.append(stab)
            rem = [c for c in rem if win[c][0] > stab + gap]
        S_ = np.array(stabs); members = [[] for _ in stabs]
        for c in by_ep[ep_]:
            lo, hi = win[c]
            inside = np.flatnonzero((S_ >= lo - gap) & (S_ <= hi))
            members[int(inside[np.argmin(np.abs(S_[inside] - (lo + hi) / 2))])].append(c)
        groups += [g for g in members if g]
    groups.sort(key=lambda g: min(c[0] for c in g))
    return groups


def transition(pos, app, r0, r1, r_pos, r_app, obs=None):
    """Per-frame timing of the change between consecutive rest runs r0, r1 of one identity (rest_runs records):
    arrival = first frame from which the reading stays within the noise radii of r1's state up to r1's start;
    departure = last frame before the arrival still within them of r0's state. -> (departure + 1, arrival).
    obs (n,) bool: only these frames are readings (--effector-lag: a reading in the effector's shadow is the old state
    read late; timing the change from it put the pressed light's change where its reading caught up, after the press, as
    a separate one-frame event: puzzle-4x5 VAL, 99.6% of the events with no simulator flip in their core)."""
    s = np.arange(r0[1], r1[0] + 1)
    if obs is not None:
        s = s[obs[s] | (s == r0[1]) | (s == r1[0])]

    def at(r):
        return (np.linalg.norm(pos[s] - r[2] / r[4], axis=-1) <= r_pos) & (np.abs(app[s] - r[3] / r[4]).max(-1) <= r_app)

    stay = np.flip(np.cumprod(np.flip(at(r1)))).astype(bool)
    arr = int(s[stay.argmax()]) if stay.any() else int(r1[0])
    pre = s[(s < arr) & at(r0)]
    dep = int(pre.max()) if len(pre) else int(r0[1])
    return dep + 1, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, required=True, help="front-end dir (entities_*.npz, discover.json)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--no-contact-free", dest="contact_free", action="store_false",
                    help="ablation: count rest observations also while the agent touches the identity")
    ap.add_argument("--thresholds-from", type=Path, default=None,
                    help="events dir whose report.json fixes r_pos / tol_pos / r_app / thr_app (self-training rounds keep "
                         "the change definitions of the front-end round)")
    ap.add_argument("--per-frame", action="store_true",
                    help="every frame of the processed episodes is a candidate observation (reader tables); changes are "
                         "timed on the per-frame reading and grouped by overlapping transition windows (see module doc)")
    ap.add_argument("--frame-gap", type=int, default=0, help="--per-frame: slack in frames between overlapping transition windows "
                    "(0: windows that share a frame; 2 merged the next press whenever one of its lights was covered right "
                    "after the previous arrival)")
    ap.add_argument("--group", choices=("stab", "overlap"), default="stab",
                    help="stab (default): an event = the changes whose windows contain the earliest remaining arrival (greedy minimum "
                         "stabbing: the fewest moments that explain every change), acted entity at the contact frame = the latest "
                         "departure of the group; overlap: the base u_events chains of overlapping windows")
    ap.add_argument("--effector", type=Path, default=None,
                    help="effector3.py output (track_{split}.npy; effector.py eff_{split}.npy): the effector point replaces the arm mask for contact (acted = the "
                         "identity not known to stay that is nearest the effector at the contact moment) and an identity within "
                         "half an object width of the effector is not observed (a light under the finger reads its old state)")
    ap.add_argument("--effector-lag", type=int, default=0,
                    help="with --effector: an entity the effector came within one object width of stays UNOBSERVED for this many "
                         "frames (the reading under the arm lags: a pressed light read its old state for 7-11 frames, p90 19-27); "
                         "the 90th percentile of within-event arrival spreads (t - t_core1) of a first pass without it")
    ap.add_argument("--effector-offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DU", "DV"),
                    help="with --effector: added to the effector point (effector_calib.py offset: the contact point is not where the "
                         "commanded translation is most visible; scene +4 to +7 px, cube +16 to +20 px below it)")
    ap.add_argument("--shadow-places-only", action="store_true",
                    help="with --effector-lag: only places (read through the arm) are shadowed; movers keep the agent-clear test")
    ap.add_argument("--mover-rest", choices=("effector", "agent"), default="effector",
                    help="--per-frame with --effector: a MOVER reading is a rest observation when the effector contact point is farther "
                         "than one object width (effector, 2026-10-11) or when no agent pixel is within half an object width (agent, "
                         "before). The agent mask is coarse (scene: ~38%% of the image with the arm's shadow) and the play arm hovers "
                         "near the cube between manipulations: scene VAL true cube rest intervals with a rest run .19 (agent) vs .76 "
                         "(effector), cube-triple .62-.76 vs .92-.97, runs that are mostly carried frames 0 vs <= .010 (PRIVILEGED check, "
                         "scratchpad/rest_criteria.py); scene cube moves found by an event were .33")
    ap.add_argument("--no-continuity", dest="continuity", action="store_false",
                    help="ablation: no state continuity between consecutive events (before 2026-10-11)")
    ap.add_argument("--acted-among", choices=("all", "not-staying"), default="all",
                    help="with --effector: acted = the entity nearest the effector among all entities, or among those not known to stay")
    ap.add_argument("--contact", choices=("departure", "closest"), default="departure",
                    help="--group stab with an effector: the contact moment of an event = the latest departure of its changes "
                         "(departure) or the effector's closest approach to the changed entities between the latest departure "
                         "and the first arrival (closest)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    disc = json.loads((a.entities / "discover.json").read_text())
    thr_pos = float(np.sqrt(6) * np.median([t["spread_median"] for t in disc["table"]]))
    E = {s: np.load(a.entities / f"entities_{s}.npz") for s in ("train", "val")}
    # appearance noise / change thresholds from train (frame-to-frame changes of visible identities)
    tr = E["train"]
    vis = tr["area"] >= 1
    term_tr = np.load(a.cache / "train_terminals.npy")[:len(vis)]; ep_tr = np.concatenate([[0], np.cumsum(term_tr[:-1])])
    # differences between consecutive OBSERVATIONS of an identity within an episode (an unobserved stretch in between
    # included): with tables that hide identities under the agent, states change across those gaps, so consecutive frames
    # where both are visible hold almost only reading noise and the split fell inside the noise (puzzle VAL: r_app .0007,
    # light flips .3 never between two consecutive visible frames)
    obs_m = vis & tr["processed"][:, None].astype(bool)
    dpos_l, dapp_l = [], []
    for k in range(vis.shape[1]):
        t = np.flatnonzero(obs_m[:, k])
        t0, t1 = t[:-1], t[1:]; same = ep_tr[t0] == ep_tr[t1]; t0, t1 = t0[same], t1[same]
        dpos_l.append(np.linalg.norm(tr["pos"][t1, k] - tr["pos"][t0, k], axis=-1)); dapp_l.append(np.abs(tr["app"][t1, k] - tr["app"][t0, k]).max(-1))
    dpos = np.concatenate(dpos_l) if dpos_l else np.zeros(0)
    if (dpos > 0).sum() >= 10:
        r_pos_log, D0 = bimodal_split(np.log(dpos[dpos > 0] + 1e-6))
        r_pos = float(np.exp(r_pos_log)) if np.isfinite(r_pos_log) else 1.0
    else:                                                                        # only place identities: position never changes
        r_pos, D0 = 1.0, 0.0
    tol_pos = 2.0 * r_pos                      # two rests within r_pos of one place differ by <= 2 r_pos
    # EXACT identities: readings with at most 8 distinct values in TRAIN (snapped two-state places, mover colours; the
    # canonical-value rule of world_model.py): any change counts (noise radius ~0), and the appearance unit of the world
    # model is half the smallest gap between two of its values (a prediction nearer the right value snaps to it). The
    # others share the split of their consecutive-observation differences (with exact identities in it the split fell
    # between light colours: puzzle .48 for flips of .3).
    K_ = vis.shape[1]
    exact = np.zeros(K_, bool); app_unit_id = np.full(K_, np.inf)
    for k in range(K_):
        v_ = np.unique(np.round(tr["app"][obs_m[:, k], k].astype(np.float64), 5), axis=0)
        if 0 < len(v_) <= 8:
            exact[k] = True
            if len(v_) > 1:
                g_ = np.abs(v_[:, None] - v_[None]).max(-1); app_unit_id[k] = 0.5 * g_[g_ > 0].min()
    dapp = np.concatenate([d_ for k, d_ in enumerate(dapp_l) if not exact[k]] or [np.zeros(0)])
    if (dapp > 0).sum() >= 10:
        r_app_log, D1 = bimodal_split(np.log(dapp[dapp > 0] + 1e-6))
        r_app = float(np.exp(r_app_log)) if np.isfinite(r_app_log) else float(np.percentile(dapp, 99))
    else:                                                                        # no inexact identity changes appearance
        r_app, D1 = np.inf, 0.0
    r_app_id = np.where(exact, 1e-6, r_app)
    app_unit_id = np.where(exact, app_unit_id, 2.0 * r_app)
    thr_app = None
    if a.thresholds_from is not None:
        fx = json.loads((a.thresholds_from / "report.json").read_text())
        thr_pos, r_pos, tol_pos, r_app, thr_app = fx["thr_pos"], fx["r_pos"], fx["tol_pos"], fx["r_app"], fx["thr_app"]
        D0 = D1 = D2 = None
    thr_app_id = np.where(r_app_id < 1e-5, 2e-6, thr_app) if thr_app is not None else None
    rep = {"thr_pos": thr_pos, "r_pos": r_pos, "r_pos_D": D0, "tol_pos": tol_pos, "r_app": r_app, "r_app_D": D1, "m": a.m,
           "thresholds_from": str(a.thresholds_from) if a.thresholds_from else None}
    if thr_app is not None:
        rep.update(thr_app=thr_app)
    for split in ("train", "val"):
        z = E[split]
        pos, app, area, processed = z["pos"].astype(np.float64), z["app"].astype(np.float64), z["area"], z["processed"]
        # the agent: packed pixel masks (pixel track) or the end effector as a point (state track, s_entities.py)
        agent = z["agent"] if "agent" in z.files else None
        eff = z["effector"].astype(np.float64) if "effector" in z.files else None
        vis = area >= 1
        n, K = area.shape
        eff_shadow = None
        if a.effector is not None:                                           # effector3.py track_* (effector.py eff_*)
            ef_ = a.effector / f"track_{split}.npy"
            eff = np.asarray(np.load(ef_ if ef_.exists() else a.effector / f"eff_{split}.npy", mmap_mode="r")[:n, :2], np.float64)
            eff = eff + np.asarray(a.effector_offset, np.float64)                   # contact-point calibration (effector_calib.py)
            term_ = np.load(a.cache / f"{split}_terminals.npy")[:n]; ep_ = np.concatenate([[0], np.cumsum(term_[:-1])])
            last_ = np.maximum.accumulate(np.where(vis, np.arange(n)[:, None], -1), axis=0)   # last frame each entity was seen
            posf = np.where((last_ >= 0)[..., None], pos[np.maximum(last_, 0), np.arange(K)[None]], pos)
            near = np.linalg.norm(posf - eff[:, None], axis=-1) <= thr_pos       # the effector at the entity (one object width)
            eff_shadow = near.copy()
            for d_ in range(1, a.effector_lag + 1):                           # ... and the lag of its reading afterwards
                eff_shadow[d_:] |= near[:-d_] & (ep_[d_:] == ep_[:-d_])[:, None]
            if a.shadow_places_only:
                # the shadow is a reading lag of PLACES (read through the arm, SeeThrough); a mover is a rest observation only
                # with the agent clear of it anyway, and shadowing it merged cube-triple interactions (3.9 events / ep for 6.3)
                ids_ = disc.get("identities", [])
                plc_ = np.array([k < len(ids_) and ids_[k].get("anchor") == "location" for k in range(K)])
                eff_shadow[:, ~plc_] = False
        term = np.load(a.cache / f"{split}_terminals.npy")[:n]
        ep_of = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)
        starts = np.r_[0, np.nonzero(term)[0] + 1]; ends = np.r_[np.nonzero(term)[0] + 1, n]
        starts, ends = starts[starts < ends], ends[starts < ends]                 # a terminal last frame adds an empty episode
        eps = [e for e in range(min(len(starts), len(ends))) if processed[starts[e]]]
        frame_ok = processed.copy()                                              # frames that are read
        if a.per_frame:
            for e in eps:
                frame_ok[starts[e]:ends[e]] = True
        # rest observations need the object free of the agent: unobserved while an agent pixel is within half an
        # object width of it (a cube held still in the gripper is in transit, a light under the finger unread)
        vg, ug = np.mgrid[0:64, 0:64]
        if a.per_frame:
            # every frame is read; observed = the agent is clear of the identity (a place also needs most of its disc
            # uncovered, as in the front end)
            ids = disc.get("identities", [])
            place = {k: np.hypot(ug - i["centre"][0], vg - i["centre"][1]) <= max(i["radius"], 1.0)
                     for k, i in enumerate(ids) if k < K and i.get("centre") is not None}
            vis_rest = np.repeat(frame_ok[:, None], K, 1) & vis                 # observed only where the table shows the identity
            # objects.py reads a place only in full view or through the agent (SeeThrough), so its readings need no
            # agent-clear test (which threw away every SeeThrough reading); movers keep it (a cube held still is in transit)
            is_place = np.array([k < len(ids) and ids[k].get("anchor") == "location" for k in range(K)])
            if eff_shadow is not None:
                vis_rest &= ~eff_shadow
            elif eff is not None:
                vis_rest &= np.linalg.norm(pos - eff[:, None], axis=-1) > thr_pos / 2
            eff_movers = a.mover_rest == "effector" and eff is not None
            if eff_movers:                                                       # movers: the contact point one object width away
                far = np.linalg.norm(pos - eff[:, None], axis=-1) > thr_pos
                vis_rest &= far | is_place[None]
            for t in np.nonzero(frame_ok)[0] if agent is not None else []:
                am = np.unpackbits(agent[t], axis=-1)[:, :64].astype(bool)
                if am.any():
                    d = np.hypot(ug[am][:, None] - pos[t, :, 0][None], vg[am][:, None] - pos[t, :, 1][None]).min(0)
                    vis_rest[t] &= (d > thr_pos / 2) | is_place | eff_movers      # object tables: a place reading is agent-free
                    for k, D_ in place.items():
                        if am[D_].mean() >= 0.5 and not is_place[k]:
                            vis_rest[t, k] = False
        else:
            vis_rest = vis.copy()
            if a.contact_free and eff is not None:
                vis_rest &= np.linalg.norm(pos - eff[:, None], axis=-1) > thr_pos / 2
            elif a.contact_free:
                for t in np.nonzero(processed & vis.any(1))[0]:
                    am = np.unpackbits(agent[t], axis=-1)[:, :64].astype(bool)
                    if am.any():
                        d = np.hypot(ug[am][:, None] - pos[t, :, 0][None], vg[am][:, None] - pos[t, :, 1][None]).min(0)
                        vis_rest[t] &= d > thr_pos / 2
        runs = {(e, k): rest_runs(pos[:, k], app[:, k], vis_rest[:, k], starts[e], ends[e], r_pos, r_app_id[k], a.m) for e in eps for k in range(K)}
        if thr_app is None:
            # rest-to-rest appearance change: two rests of one state lie within r_app of it, so they differ by at
            # most 2 r_app (as tol_pos = 2 r_pos). A 2-means split of rest-to-rest differences fails when rests are
            # read steadily: the "no change" mode vanishes and the split falls between weak and strong changes.
            thr_app, D2 = (2.0 * r_app, None) if np.isfinite(r_app) else (np.inf, None)
            rep.update(thr_app=thr_app, exact_app_identities=np.flatnonzero(exact).tolist(), app_unit_id=np.round(app_unit_id, 5).tolist())
            print({"thr_pos": round(thr_pos, 2), "r_pos": round(r_pos, 3), "tol_pos": round(tol_pos, 3), "D_pos": round(D0, 2), "r_app": round(r_app, 4),
                   "thr_app": thr_app, "D_frame": round(D1, 2), "exact_app_identities": int((r_app_id < 1e-5).sum())}, flush=True)
            thr_app_id = np.where(r_app_id < 1e-5, 2e-6, thr_app)
        labp = np.zeros((n, K, 2), np.float32); laba = np.zeros((n, K, 3), np.float32); valid = np.zeros((n, K), bool)
        changes = []                                                             # (t0, t1, k, e, last frame of the before-rest)
        steps, new_val = {}, {}                                                  # STATE_RULE (--group stab): value switches at changes
        for (e, k), rs in runs.items():
            merged = []
            for r_ in rs:
                if merged:
                    pm, am = merged[-1][2] / merged[-1][4], merged[-1][3] / merged[-1][4]
                    if np.hypot(*(r_[2] / r_[4] - pm)) <= tol_pos and np.abs(r_[3] / r_[4] - am).max() <= thr_app_id[k]:
                        merged[-1] = [merged[-1][0], r_[1], merged[-1][2] + r_[2], merged[-1][3] + r_[3], merged[-1][4] + r_[4]]
                        continue
                merged.append(list(r_))
            t0s = []
            for i, r_ in enumerate(merged):
                labp[r_[0]:r_[1] + 1, k] = r_[2] / r_[4]; laba[r_[0]:r_[1] + 1, k] = r_[3] / r_[4]; valid[r_[0]:r_[1] + 1, k] = True
                if i:
                    if a.per_frame:
                        t0, t1 = transition(pos[:, k], app[:, k], merged[i - 1], r_, r_pos, r_app_id[k],
                                            None if eff_shadow is None else ~eff_shadow[:, k])
                    else:
                        t0, t1 = merged[i - 1][1] + 1, r_[0] - 1
                    changes.append((t0, t1, k, e, merged[i - 1][1]))
                    new_val[(t0, t1, k, e)] = np.r_[r_[2] / r_[4], r_[3] / r_[4]]; t0s.append(t0)
            if merged:
                steps[(e, k)] = (np.array([starts[e]] + t0s), np.array([np.r_[r_[2] / r_[4], r_[3] / r_[4]] for r_ in merged]))
        changes.sort()
        np.savez_compressed(a.out / f"changes_{split}.npz", changes=np.array(changes, np.int64).reshape(-1, 5))   # (t0, t1, k, episode, before-rest end)
        # covered: never visible during a static interval (no change of any identity in progress)
        in_ch = np.zeros(n, bool)
        for t0, t1, *_ in changes:
            in_ch[t0:t1 + 1] = True
        chg = np.r_[True, (in_ch[1:] != in_ch[:-1]) | (ep_of[1:] != ep_of[:-1])]
        st = np.nonzero(chg)[0]; ln = np.diff(np.r_[st, n])
        vsum = np.add.reduceat(vis.astype(np.int64), st, axis=0)
        ok = (~in_ch[st]) & (ln >= a.m) & frame_ok[st]
        run_of = np.cumsum(chg) - 1
        cov = (vsum == 0)[run_of]; cov_valid = np.repeat(ok[:, None], K, 1)[run_of]
        # group overlapping changes into events; acted entity by agent contact
        v_, u_ = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        gap = a.frame_gap if a.per_frame else a.gap
        if a.group == "overlap":                                                 # base rule: chains of overlapping windows
            groups, i = [], 0
            while i < len(changes):
                grp = [changes[i]]; t1 = changes[i][1]; j = i + 1
                while j < len(changes) and changes[j][3] == changes[i][3] and changes[j][0] <= t1 + gap:
                    grp.append(changes[j]); t1 = max(t1, changes[j][1]); j += 1
                i = j
                groups.append(grp)
        else:                                                                    # stab: the fewest moments explaining all changes
            groups = stab_groups(changes, gap)
        events = []
        for grp in groups:
            t1 = max(g[1] for g in grp)
            # Rest runs can miss the contacted object's brief state change. Include changes observed
            # at these same event boundaries before attributing the interaction to its cause.
            c0, c1, ep_ = min(g[0] for g in grp), t1, grp[0][3]
            ca = max(g[0] for g in grp) if a.group == "stab" else c0                # contact frame: the moment all changes are under way
            b0, b1 = max(c0 - 1, starts[ep_]), min(c1 + 1, ends[ep_] - 1)
            ks_ = sorted({g[2] for g in grp})
            p_pre = {k: labp[min(g[4] for g in grp if g[2] == k), k] for k in ks_}
            ct = ca                                                              # moment the acted entity is read at
            if a.contact == "closest" and eff is not None and a.group == "stab":
                # CONTACT = the effector's closest approach to the changed entities within the span all windows share
                # [latest departure, first arrival]: the latest departure is the last sight of the old states, which comes
                # long before the contact when the arm covers the changing entities (small boards: the arm body over a
                # corner, puzzle-4x4 corner presses labelled .60 / .43). Only the acted label uses it; the event core
                # stays [latest departure, first arrival] (moving it shrank the cores: VAL presses in one core 4x4 .85 -> .80)
                span = np.arange(ca, max(min(g[1] for g in grp), ca) + 1)
                span = span[np.isfinite(eff[span]).all(1)]
                if len(span):
                    cpt = np.mean([p_pre[k] for k in ks_], axis=0)
                    ct = int(span[np.argmin(np.linalg.norm(eff[span] - cpt, axis=-1))])
            if a.per_frame:
                raw_changed = (np.linalg.norm(pos[b1] - pos[b0], axis=-1) > tol_pos) | (np.abs(app[b1] - app[b0]).max(-1) > thr_app_id)
                readable = vis[b0] & vis[b1] & frame_ok[b0] & frame_ok[b1]
                ks_ = sorted(set(ks_) | set(np.flatnonzero(raw_changed & readable)))
                for k in ks_:
                    if readable[k]:
                        p_pre[k] = pos[b0, k]
            cen = {k: sum(np.hypot(*(p_pre[k] - p_pre[j])) for j in ks_ if j != k) for k in ks_}
            distance = {k: np.inf for k in ks_}
            if eff is not None and np.isfinite(eff[ca]).all():
                distance = {k: float(np.hypot(*(eff[ca] - p_pre[k]))) for k in ks_}
            elif agent is not None:
                am = np.unpackbits(agent[ca], axis=-1)[:, :64].astype(bool)
                if am.any():
                    distance = {k: float(np.min(np.hypot(u_[am] - p_pre[k][0], v_[am] - p_pre[k][1]))) for k in ks_}
            duration = {k: max([g[1] - g[0] for g in grp if g[2] == k] or [0]) for k in ks_}
            if a.group == "stab":                                                # contact is binary: within half an object width
                touch = {k: distance[k] <= thr_pos / 2 for k in ks_}
                acted = min(ks_, key=lambda k: (not touch[k], 0.0 if touch[k] else distance[k], cen[k], -duration[k]))
            else:
                acted = min(ks_, key=lambda k: (distance[k], cen[k], -duration[k]))
            events.append((c0, t1, acted, len(grp), [(g[2], new_val[tuple(g[:4])], g[1], g[0]) for g in grp],
                           [k for k in ks_ if k not in {g[2] for g in grp} and a.per_frame and readable[k]], b1,
                           ca, min(g[1] for g in grp), ct))

        def state_at(t, side=0):
            """Rest state of every identity at frame t: the nearest rest label, or with side -1 / +1 the last one at or
            before t / the first one at or after t (--per-frame: the state before / after an event must not be read
            on the other side of it)."""
            s = np.zeros((K, 6), np.float32)
            e0 = starts[ep_of[t]]; e1 = ends[ep_of[t]]
            for k in range(K):
                vv = np.nonzero(valid[e0:e1, k])[0] + e0
                if len(vv):
                    if side < 0 and (vv <= t).any():
                        tt = vv[vv <= t].max()
                    elif side > 0 and (vv >= t).any():
                        tt = vv[vv >= t].min()
                    else:
                        tt = vv[np.abs(vv - t).argmin()]
                    s[k, :2] = labp[tt, k]; s[k, 2:5] = laba[tt, k]
                cc = np.nonzero(cov_valid[e0:e1, k])[0]
                if len(cc):
                    s[k, 5] = cov[e0 + cc[np.abs(cc + e0 - t).argmin()], k]
                if a.per_frame and frame_ok[t] and vis[t, k]:
                    # A same-frame reading cannot borrow a side effect from another interaction.
                    s[k, :2] = pos[t, k]
                    s[k, 2:5] = app[t, k]
            return s

        ts = np.array([x[0] for x in events]); te = np.array([x[1] for x in events]); ee = np.array([x[2] for x in events])
        side = 1 if a.per_frame else 0
        ep = ep_of[ts]
        before = np.stack([state_at(max(t - 1, starts[e]), -side) for t, e in zip(ts, ep)])
        after = np.stack([state_at(min(t + 1, ends[e] - 1), side) for t, e in zip(te, ep)])
        before_known = np.ones((len(ts), K), bool); after_known = np.ones((len(ts), K), bool)
        if a.group == "stab":
            # STATES BETWEEN EVENTS: the state of an identity before event i is its last observation between the first
            # arrival of event i-1 and the contact moment of event i (its latest departure), after event i its first
            # observation between the first arrival of event i and the contact moment of event i+1 (bounding by the earliest
            # departures instead, against after-states showing the next press, changed no label: puzzle VAL per-light label
            # accuracy over known pairs .9995 -> .9997, but toggled lights known .829 -> .747); a group member
            # after the event is its new rest when it arrives before event i+1 is under way. Otherwise the state is UNKNOWN
            # (before_known / after_known False; the value then follows the rest whose change began at or before that
            # moment). Reading the nearest rest on either side reached across later interactions for identities hidden
            # meanwhile (puzzle VAL: toggled set right in 15% of single-press events), and "unchanged" for a light flipped
            # and flipped back unseen.
            seen = vis & frame_ok[:, None]
            if eff_shadow is not None:                                           # not under the effector nor in its lag
                seen &= ~eff_shadow
            elif eff is not None:                                                # not under the effector
                seen &= np.linalg.norm(pos - eff[:, None], axis=-1) > thr_pos / 2
            obs_t = [np.flatnonzero(seen[:, k]) for k in range(K)]
            core0 = np.array([x[7] for x in events]); core1 = np.array([x[8] for x in events])
            contact = np.array([x[9] for x in events])                         # = core0 unless --contact closest
            for i, (t, e_, ev) in enumerate(zip(ts, ep, events)):
                prv = i - 1 if i > 0 and ep[i - 1] == e_ else None
                nxt = i + 1 if i + 1 < len(ts) and ep[i + 1] == e_ else None
                lo_b = core1[prv] if prv is not None else starts[e_]
                hi_b = core0[i]                                                   # [lo_b, hi_b): before window
                lo_a = core1[i]
                hi_a = core0[nxt] if nxt is not None else ends[e_]                # [lo_a, hi_a): after window
                grp_new = {k: v for k, v, _, _ in ev[4]}
                arr = {k: t_ for k, _, t_, _ in ev[4]}                           # arrival of each group member's new rest
                for k in range(K):
                    st_ = steps.get((e_, k))
                    ob = obs_t[k]
                    j0, j1 = np.searchsorted(ob, lo_b), np.searchsorted(ob, hi_b)
                    if j1 > j0:
                        tb = ob[j1 - 1]; before[i, k, :2] = pos[tb, k]; before[i, k, 2:5] = app[tb, k]
                    else:
                        before_known[i, k] = False
                        if st_ is not None:
                            before[i, k, :5] = st_[1][max(int(np.searchsorted(st_[0], hi_b - 1, side="right")) - 1, 0)]
                    if k in grp_new:
                        after[i, k, :5] = grp_new[k]
                        after_known[i, k] = arr.get(k, hi_a) < hi_a
                        continue
                    j0, j1 = np.searchsorted(ob, lo_a), np.searchsorted(ob, hi_a)
                    if j1 > j0:
                        tf = ob[j0]; after[i, k, :2] = pos[tf, k]; after[i, k, 2:5] = app[tf, k]
                    else:
                        after_known[i, k] = False
                        after[i, k, :5] = before[i, k, :5]
                # ACTED: among the identities the agent touches at the contact moment and that are not known to stay as they
                # were (none: the changed ones) the one nearest to the centroid of the known changes, then the longest change.
                # The changes of an interaction surround its contact point (a pressed light among its toggled neighbours, a
                # moved cube); the coarse arm mask touches much more (scene: cube, drawer and both buttons at a button press).
                # VAL, PRIVILEGED truth: puzzle .867, cube .888, scene .827 (touched without the unchanged test: .878 / .887 /
                # .681; ranking sub-pixel distances to the agent mask: puzzle .26).
                chg = before_known[i] & after_known[i] & ((np.linalg.norm(after[i, :, :2] - before[i, :, :2], axis=-1) > tol_pos)
                                                          | (np.abs(after[i, :, 2:5] - before[i, :, 2:5]).max(-1) > thr_app_id))
                Pk = np.flatnonzero(chg)
                if not len(Pk):
                    Pk = np.array(sorted(grp_new))
                if a.effector is not None:                                       # EFFECTOR contact: the entity at the effector
                    dist_ = np.linalg.norm(before[i, :, :2] - eff[contact[i]], axis=-1)
                    if a.acted_among == "not-staying":
                        # among the entities not known to stay as they were (changed, or unread on one side): a slid window
                        # handle is far from the window centre and nearer a button that did not change (scene window .07)
                        stay = before_known[i] & after_known[i] & ~chg
                        if (~stay).any():
                            dist_ = np.where(stay, np.inf, dist_)
                    ee[i] = int(np.argmin(dist_))
                    continue
                cpt = before[i, Pk, :2].mean(0)
                cands = Pk
                if agent is not None:
                    am = np.unpackbits(agent[core0[i]], axis=-1)[:, :64].astype(bool)
                    if am.any():
                        dk = np.array([np.min(np.hypot(u_[am] - before[i, k, 0], v_[am] - before[i, k, 1])) for k in range(K)])
                        tk = np.flatnonzero((dk <= thr_pos / 2) & ~(before_known[i] & after_known[i] & ~chg))
                        if len(tk):                                              # touched and not known to stay as it was
                            cands = tk
                dur = {k: t_ - s_ for k, _, t_, s_ in ev[4]}
                ee[i] = min(cands, key=lambda k: (float(np.hypot(*(before[i, k, :2] - cpt))), -dur.get(k, 0)))
        seg = starts[ep].copy()
        same = np.r_[False, ep[1:] == ep[:-1]]
        seg[same] = te[:-1][same[1:]] + 1
        seg = np.minimum(seg, ts)
        if a.continuity and len(ee):
            # CONTINUITY (2026-10-11): nothing changes between consecutive events of an episode (an event holds every change),
            # so an entity unknown after event i takes its state before event i+1 when that is known, and one unknown before
            # event i+1 its state after event i (puzzle-3x3: the pressed centre light was unknown after 70% of its TRAIN events,
            # under the resting arm, the top-middle 40%; the WM learned wrong crosses, event exact .51)
            nfill = [0, 0]
            for i in np.flatnonzero(np.r_[ep[1:] == ep[:-1], False]):
                fa = ~after_known[i] & before_known[i + 1]
                after[i, fa, :5] = before[i + 1, fa, :5]; after_known[i, fa] = True; nfill[0] += int(fa.sum())
                fb = ~before_known[i + 1] & after_known[i]
                before[i + 1, fb, :5] = after[i, fb, :5]; before_known[i + 1, fb] = True; nfill[1] += int(fb.sum())
            print({"split": split, "continuity_filled_after_before": nfill}, flush=True)
        moved = (np.linalg.norm(after[..., :2] - before[..., :2], axis=-1) > tol_pos).sum(1)
        target_known = before_known[np.arange(len(ee)), ee] & after_known[np.arange(len(ee)), ee] if len(ee) else np.zeros(0, bool)
        np.savez_compressed(a.out / f"events_{split}.npz", t_start=ts, t=te, e=ee, before=before, after=after, target_known=target_known,
                            target=after[np.arange(len(ee)), ee], seg_start=seg, knock=moved > 1, episode=ep,
                            n_changed=np.array([x[3] for x in events]), thr_pos=thr_pos, tol_pos=tol_pos, thr_app=thr_app,
                            t_core=np.array([[x[7], x[8]] for x in events], np.int64).reshape(-1, 2),   # [latest departure, first arrival]
                            t_contact=np.array([x[9] for x in events], np.int64),   # moment the acted entity is read at
                            effector_offset=np.asarray(a.effector_offset, np.float64), effector_lag=a.effector_lag,
                            before_known=before_known, after_known=after_known, thr_app_id=thr_app_id, app_unit_id=app_unit_id)
        np.savez_compressed(a.out / f"labels_{split}.npz", pos=labp, app=laba, valid=valid, cov=cov, cov_valid=cov_valid)
        r = {"events": int(len(ee)), "per_episode": float(len(ee) / max(1, len(eps))), "episodes": len(eps),
             "changed_entities_pct": np.percentile([x[3] for x in events], [50, 90, 99]).tolist() if events else None,
             "acted_entity_counts": np.bincount(ee, minlength=K).tolist(), "label_valid_frac_processed": float(valid[processed].mean()),
             "covered_frac": float(cov[cov_valid].mean()) if cov_valid.any() else None}
        # PRIVILEGED diagnostics: event times vs qpos object motion / button toggles
        refm = np.zeros(n - 1, bool)
        same_ep = ep_of[1:] == ep_of[:-1]
        if (a.cache / f"{split}_qpos.npy").exists():
            q = np.load(a.cache / f"{split}_qpos.npy", mmap_mode="r")
            for e in eps:
                s0, s1 = starts[e], ends[e]
                refm[s0:s1 - 1] |= np.abs(np.diff(np.asarray(q[s0:s1, 14:]), axis=0)).max(1) > 2e-3
        if (a.cache / f"{split}_button_states.npy").exists():
            b = np.load(a.cache / f"{split}_button_states.npy", mmap_mode="r")
            for e in eps:
                s0, s1 = starts[e], ends[e]
                refm[s0:s1 - 1] |= (np.asarray(b[s0 + 1:s1]) != np.asarray(b[s0:s1 - 1])).any(1)
        refm &= same_ep
        d_ = np.diff(np.r_[0, refm.astype(np.int8), 0]); rs_, re_ = np.nonzero(d_ == 1)[0] + 1, np.nonzero(d_ == -1)[0]
        def one_to_one(R0, R1, slack=5):
            """Recall above counts a reference as found when any event overlaps it, so an event spanning several
            presses / moves still recalls all of them. Here: references covered by exactly one event that itself
            covers exactly one reference."""
            nref = np.array([((R0 <= y + slack) & (R1 >= x - slack)).sum() for x, y in zip(ts, te)])
            hit = [np.nonzero((ts <= y + slack) & (te >= x - slack))[0] for x, y in zip(R0, R1)]
            return {"refs_per_episode": round(len(R0) / max(1, len(eps)), 2),
                    "events_with_0_1_2plus_refs": [int((nref == 0).sum()), int((nref == 1).sum()), int((nref >= 2).sum())],
                    "refs_in_clean_one_to_one": round(float(np.mean([len(h) == 1 and nref[h[0]] == 1 for h in hit])), 3)}

        if len(rs_) and len(ts):
            r["privileged_recall"] = float(np.mean([np.any((ts <= y + 15) & (te >= x - 15)) for x, y in zip(rs_, re_)]))
            r["privileged_precision"] = float(np.mean([np.any((rs_ <= y + 15) & (re_ >= x - 15)) for x, y in zip(ts, te)]))
            r["privileged_ref_events"] = int(len(rs_))
            r["privileged_one_to_one"] = one_to_one(rs_, re_)
            if "cube" in a.cache.name and (a.cache / f"{split}_qpos.npy").exists():
                # picks and places: reference intervals where some cube's net xy displacement >= 2 cm (half a cube)
                sl = [c for c in (14, 21, 28, 35) if c + 3 <= q.shape[1]]
                disp = np.array([max(np.linalg.norm(np.asarray(q[min(y + 1, n - 1), c:c + 2]) - np.asarray(q[x, c:c + 2])) for c in sl) for x, y in zip(rs_, re_)])
                big = disp >= 0.02
                r["privileged_recall_moves_2cm"] = float(np.mean([np.any((ts <= y + 15) & (te >= x - 15)) for x, y in zip(rs_[big], re_[big])])) if big.any() else None
                r["privileged_ref_moves_2cm"] = int(big.sum())
                if big.any():
                    r["privileged_one_to_one_moves_2cm"] = one_to_one(rs_[big], re_[big])
        rep[split] = r
        print(split, json.dumps(r), flush=True)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
