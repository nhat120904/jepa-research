#!/usr/bin/env python3
"""Generic state front end: observation vector -> the unified backend's entity tables, without layout knowledge.

Replaces s_entities.py, which uses the env's documented observation layout (which columns are the robot and which
are object blocks, movable vs fixed objects, a hand-written on-top relation, scene column names). The only
assumptions here, identical for every family:
  - observations are fixed-length vectors from offline play episodes (terminals given);
  - the agent's dimensions change almost every step, object dimensions rarely;
  - the dimensions of one object change together, different objects at different times;
  - an object starts to change when the agent is at a place determined by the object's own state (contact).
Steps (fitted on TRAIN; val and the closed loop apply the fitted maps):
  1. per dimension: noise radius (2-means split of log frame-to-frame changes, else a tenth of the median change)
     and change rate; constant dimensions are dropped;
  2. agent dimensions = the fast cluster of a 2-means split of log change rates; the others are object dimensions;
  3. object dimensions are grouped into entities by co-change: Jaccard similarity of change indicators (dilated by
     +-2 frames), average-linkage clustering cut at the 2-means split of the pairwise similarities;
  4. per entity: discrete dimensions (<= 8 distinct rest values at 1e-5, as the backend's finite support) and
     continuous ones; an entity whose rest state never varies (rest spread < 10% of its overall spread, no discrete
     dimension) is a transient part (e.g. a joint that springs back) and not planning state;
  5. contact model, hard EM: the agent configuration at the onset of an entity's change is predicted from the
     entity's continuous state just before (ridge regression; intercept only for discrete-only entities). Onsets
     within `gap` frames form one interaction, attributed to the entity whose predicted contact configuration is
     nearest the agent's (agent dims explained by the model, weighted by their residual spread); the model is refit on
     attributed onsets; 3 rounds;
  6. (u, v, w) = top-3 principal components of the predicted contact configurations; entity position (u, v) = its
     contact location on a 64-unit canvas, a0 = w; a1, a2 = principal components of its discrete dimensions (at most
     2); covered = 0 (no hand-written relation); effector = the agent configuration projected the same way.
     Continuous dimensions that do not set the contact location (e.g. orientation) are not planning state; the skill
     still reads the full observation;
  7. thresholds: r_pos / r_app noise radii (2-means of frame-to-frame changes, as s_entities.py); thr_app = half the
     smallest gap between rest-value clusters of the appearance fields; touching radius r_touch = 95th percentile of
     the (u, v) distance between the agent and the entity at attributed onsets; thr_pos = 2 r_touch, tol_pos = r_touch.
  Structural corrections after the contact model (repeated until stable): entities touched at the same place are
  merged (parts of one object), entities without a consistent contact place join the agent.
Writes the files of s_entities.py (cache, front/{entities_*.npz, discover.json, layout.json}, thr/report.json) and
front/generic_report.json. PRIVILEGED diagnostics against the documented layout are computed only for that report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sfa_code import two_means_threshold

MAX_DISCRETE = 8


def split_or(x, fallback, min_n=10):
    """2-means split of x if bimodal (Ashman's D > 2), else fallback."""
    if len(x) < min_n:
        return fallback, 0.0
    thr, sep, _ = two_means_threshold(np.asarray(x, np.float64))
    return (thr if sep > 2.0 else fallback), sep


def episode_index(term):
    return np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)


def frame_changes(X, term):
    """|x_t - x_{t-1}| for frames t >= 1 of the same episode; row 0 and episode starts are 0."""
    d = np.zeros_like(X)
    d[1:] = np.abs(X[1:] - X[:-1])
    d[1:][term[:-1].astype(bool)] = 0.0
    return d


def noise_radii(D):
    r = np.zeros(D.shape[1])
    for j in range(D.shape[1]):
        p = D[:, j][D[:, j] > 0]
        p = p[np.random.default_rng(j).permutation(len(p))[:200000]]
        if len(p) < 10:
            r[j] = np.inf
            continue
        lt, _ = split_or(np.log(p + 1e-12), None)
        r[j] = float(np.exp(lt)) if lt is not None else 0.1 * float(np.median(p))
    return r


def average_linkage(S, thr):
    clusters = [[i] for i in range(len(S))]
    while len(clusters) > 1:
        best, pair = -1.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = S[np.ix_(clusters[a], clusters[b])].mean()
                if s > best:
                    best, pair = s, (a, b)
        if best < thr:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return [sorted(c) for c in clusters]


def runs_of(mask):
    """Start / end (exclusive) of True runs of a 1-d bool array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def entity_onsets(moving, term, m=5):
    """Onsets: first moving frame after >= m still frames of the same episode; with the frame where motion ends
    (start of the next >= m still run). moving (n,) bool."""
    ep = episode_index(term)
    still = ~moving
    s0, s1 = runs_of(still)
    long_ = (s1 - s0) >= m
    s0, s1 = s0[long_], s1[long_]
    out = []
    for i in range(len(s0) - 1):
        a, b = s1[i], s0[i + 1]                       # moving interval [a, b)
        if ep[a - 1] == ep[b] and b > a:
            out.append((a, b))
    return np.array(out, np.int64).reshape(-1, 2)


def real_changes(Xs, ons):
    """Which motion intervals change the entity's state: the net change (rest before -> rest after, dims in std
    units) is at least half the largest excursion during the motion (excludes wobble and spring-back) and lies in the
    upper mode of a 2-means split of log net changes when that split is clear (excludes nudges). -> bool, net, exc."""
    if not len(ons):
        return np.zeros(0, bool), np.zeros(0), np.zeros(0)
    net = np.linalg.norm(Xs[ons[:, 1]] - Xs[ons[:, 0] - 1], axis=-1)
    exc = np.array([np.linalg.norm(Xs[a:b + 1] - Xs[a - 1], axis=-1).max() for a, b in ons])
    real = net >= 0.5 * np.maximum(exc, 1e-12)
    pos = net[real & (net > 0)]
    if len(pos) >= 10:
        t_, sep = split_or(np.log(pos), None)
        lo_, hi_ = np.log(pos)[np.log(pos) <= t_] if t_ is not None else [], np.log(pos)[np.log(pos) > t_] if t_ is not None else []
        if t_ is not None and len(lo_) and len(hi_) and hi_.mean() - lo_.mean() >= np.log(3.0):
            real &= net > np.exp(t_)
    return real, net, exc


def ridge(S, Y, lam=1.0):
    """Y ~ [S, 1] W; intercept not penalised. S (n, p) may have p = 0."""
    A = np.c_[S, np.ones(len(S))]
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + R, A.T @ Y)


def predict(W, S):
    return np.c_[S, np.ones(len(S))] @ W


class GenericMap:
    """Fitted maps from an observation vector to entity states; serialised into layout.json."""

    def __init__(self, L):
        self.L = L
        self.agent = np.array(L["agent_dims"]); self.amu = np.array(L["agent_mean"]); self.asd = np.array(L["agent_std"])
        self.sel = np.array(L["contact_dims"]); self.wt = np.array(L["contact_weight"])
        self.P = np.array(L["proj"]); self.pmu = np.array(L["proj_mean"])
        self.ent = L["entities"]
        self.c_uv, self.s_uv = np.array(L["canvas_centre"]), float(L["canvas_scale"])
        self.w_mid = float(L["w_mid"])

    def project(self, C):
        """Weighted contact configurations (n, |sel|) -> (u, v) on the 64-unit canvas; a0 = w in canvas units / 64
        (same geometric scale as u, v), centred at 0.5."""
        z = (C - self.pmu) @ self.P.T
        uv = (z[:, :2] - self.c_uv) * self.s_uv + 32.0
        a0 = (z[:, 2] - self.w_mid) * self.s_uv / 64.0 + 0.5
        return uv, a0

    def __call__(self, obs):
        obs = np.asarray(obs, np.float64)
        n, K = len(obs), len(self.ent)
        S = np.zeros((n, K, 6), np.float32)
        y = (obs[:, self.agent] - self.amu) / self.asd
        eff, _ = self.project(y[:, self.sel] * self.wt)
        for k, e in enumerate(self.ent):
            cont = np.array(e["cont_dims"], np.int64)
            Sk = (obs[:, cont] - np.array(e["cont_mean"])) / np.array(e["cont_std"]) if len(cont) else np.zeros((n, 0))
            C = predict(np.array(e["W"]), Sk)[:, self.sel] * self.wt
            uv, a0 = self.project(C)
            S[:, k, :2] = uv; S[:, k, 2] = a0
            if e["disc_dims"]:
                Z = (obs[:, e["disc_dims"]] - np.array(e["disc_mean"])) @ np.array(e["disc_proj"]).T
                Z = (Z - np.array(e["disc_lo"])) / np.maximum(np.array(e["disc_hi"]) - np.array(e["disc_lo"]), 1e-9)
                S[:, k, 3:3 + Z.shape[1]] = Z
        return S, eff.astype(np.float32)


def state_entities(obs, L):
    """Same signature as s_entities.state_entities (closed loop, --state-layout)."""
    return GenericMap(L)(obs)


def privileged_groups(env, D):
    """PRIVILEGED (diagnostics only): documented layout -> {object name: dims}, agent dims."""
    agent = list(range(19))
    if env.startswith("cube"):
        K = (D - 19) // 9
        return agent, {f"cube{k}": list(range(19 + 9 * k, 28 + 9 * k)) for k in range(K)}
    if env.startswith("puzzle"):
        K = (D - 19) // 4
        return agent, {f"button{k}": list(range(19 + 4 * k, 23 + 4 * k)) for k in range(K)}
    if env.startswith("scene"):
        return agent, {"cube": list(range(19, 28)), "button0": list(range(28, 32)), "button1": list(range(32, 36)),
                       "drawer": [36, 37], "window": [38, 39]}
    return agent, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--train-episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--gap", type=int, default=10, help="onsets within this many frames form one interaction")
    ap.add_argument("--em-rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cache = a.out / "cache" / a.env
    for d in (cache, a.out / "front", a.out / "thr"):
        d.mkdir(parents=True, exist_ok=True)
    data = {}
    for split, name, n_ep in (("train", f"{a.env}.npz", a.train_episodes), ("val", f"{a.env}-val.npz", a.val_episodes)):
        z = np.load(a.data / name)
        term = z["terminals"]
        ends = np.nonzero(term)[0]
        n = int(ends[min(n_ep, len(ends)) - 1] + 1)
        data[split] = {k: np.asarray(z[k][:n]) for k in ("observations", "actions", "terminals")}
        for k in ("observations", "actions", "terminals"):
            np.save(cache / f"{split}_{k}.npy", data[split][k])
        for k in ("qpos", "button_states"):                                      # PRIVILEGED, diagnostics only
            if k in z.files:
                np.save(cache / f"{split}_{k}.npy", np.asarray(z[k][:n]))
    X = data["train"]["observations"].astype(np.float64)
    term = data["train"]["terminals"].astype(bool)
    n, Dobs = X.shape
    rep = {"env": a.env, "obs_dim": Dobs, "train_frames": n}

    # 1-2. noise radii, change rates, agent vs object dimensions
    Dch = frame_changes(X, term)
    r = noise_radii(Dch)
    const = ~np.isfinite(r) | (X.std(0) < 1e-9)
    moving = Dch > r[None]
    rate = moving.mean(0)
    live = np.nonzero(~const)[0]
    # action response: R^2 of the next change x_{t+1} - x_t from (a_t, a_{t-1}) by least squares on TRAIN
    A_ = data["train"]["actions"].astype(np.float64)
    ok = np.r_[False, ~term[:-1]][1:] & np.r_[False, ~term[:-2]][:len(term) - 1] if len(term) > 2 else np.zeros(0, bool)
    t_idx = np.nonzero(ok)[0][1:]                                       # frames t with t-1, t+1 in the same episode
    t_idx = t_idx[np.random.default_rng(0).permutation(len(t_idx))[:300000]]
    F = np.c_[A_[t_idx], A_[t_idx - 1], np.ones(len(t_idx))]
    dX = X[t_idx + 1] - X[t_idx]
    coef = np.linalg.lstsq(F, dX, rcond=None)[0]
    act_r2 = 1 - ((dX - F @ coef) ** 2).sum(0) / np.maximum(((dX - dX.mean(0)) ** 2).sum(0), 1e-18)
    # the agent moves almost every step or responds directly to the action; objects do neither
    score = np.maximum(np.clip(act_r2, 0, 1), rate)
    st, sep_rate = split_or(score[live], 0.5)
    agent = live[score[live] > st]
    objd = live[score[live] <= st]
    rep.update(constant_dims=np.nonzero(const)[0].tolist(), agent_dims=agent.tolist(), object_dims=objd.tolist(),
               agent_split_D=sep_rate, agent_split=st, rate=np.round(rate, 4).tolist(), action_r2=np.round(act_r2, 4).tolist())
    print({"agent": len(agent), "object": len(objd), "constant": int(const.sum()), "split": round(float(st), 3), "D": round(sep_rate, 2)}, flush=True)

    # 3. entities by co-change
    M = moving[:, objd].astype(np.float32)
    cs = np.cumsum(np.r_[np.zeros((1, len(objd)), np.float32), M], 0)
    lo, hi = np.clip(np.arange(n) - 2, 0, n), np.clip(np.arange(n) + 3, 0, n)
    Md = ((cs[hi] - cs[lo]) > 0).astype(np.float32)
    inter = Md.T @ Md
    cnt = np.diag(inter)
    J = inter / np.maximum(cnt[:, None] + cnt[None] - inter, 1.0)
    iu = np.triu_indices(len(objd), 1)
    jthr, sep_j = split_or(J[iu], 0.5)
    groups = [objd[c].tolist() for c in average_linkage(J, jthr)]
    rep.update(jaccard_split=jthr, jaccard_split_D=sep_j, groups_all=groups)
    print({"groups": len(groups), "jaccard_thr": round(float(jthr), 3), "D": round(sep_j, 2)}, flush=True)

    # 4-5. entities (discrete / continuous dims, transient parts) and contact model; then structural corrections:
    #   - entities whose contact places coincide (median distance < 1 contact-noise unit) are parts of one object
    #     and are merged (e.g. a button's state and its spring joint);
    #   - entities without a consistent contact place (median contact distance at their own onsets an outlier, >= 3x
    #     the others) are not objects the agent acts on at a place; their dims join the agent (e.g. a contact sensor);
    #   repeated until nothing changes (at most 4 passes).
    ep = episode_index(term)

    def describe(g):
        g = np.array(g)
        mv = moving[:, g].any(1)
        still = ~mv
        # state dims: discrete ones with >= 2 rest values, continuous ones whose rest value varies (>= 10% of their
        # overall spread); dims that are constant at rest (velocities, spring joints) are not state
        disc, cont, varies, ratio = [], [], [], []
        for j in g:
            xr = X[still, j]
            vals, cnts = np.unique(np.round(xr, 5), return_counts=True)
            if len(vals) <= MAX_DISCRETE:
                if (cnts >= 1e-3 * len(xr)).sum() >= 2:
                    disc.append(int(j)); varies.append(True)
                else:
                    varies.append(False)
            else:
                ratio.append(float(xr.std() / max(X[:, j].std(), 1e-12)))
                if ratio[-1] >= 0.1:
                    cont.append(int(j)); varies.append(True)
                else:
                    varies.append(False)
        c = np.array(cont, np.int64)
        ons = entity_onsets(mv, term)
        sd_ = disc + cont
        Xs = (X[:, sd_] - X[:, sd_].mean(0)) / (X[:, sd_].std(0) + 1e-12) if sd_ else np.zeros((n, 0))
        real, net, exc = real_changes(Xs, ons)
        # an entity whose motions never change its rest state (e.g. a joint that springs back) is not planning state
        transient = (not sd_) or real.sum() < max(10, 0.02 * len(ons))
        return {"dims": g.tolist(), "disc_dims": disc, "cont_dims": cont, "transient": bool(transient),
                "onsets": ons[real], "n_onsets_all": int(len(ons)), "n_real": int(real.sum()),
                "rest_spread_ratio": np.round(ratio, 3).tolist(),
                "cont_mean": X[:, c].mean(0).tolist() if len(c) else [], "cont_std": (X[:, c].std(0) + 1e-9).tolist() if len(c) else []}

    def feats(e, t):
        c = np.array(e["cont_dims"], np.int64)
        return (X[np.ix_(t - 1, c)] - np.array(e["cont_mean"])) / np.array(e["cont_std"]) if len(c) else np.zeros((len(t), 0))

    def contact_em(plan, agent, tag):
        K = len(plan)
        Y = (X[:, agent] - X[:, agent].mean(0)) / (X[:, agent].std(0) + 1e-9)
        on_t = np.concatenate([e["onsets"][:, 0] for e in plan])
        on_k = np.concatenate([np.full(len(e["onsets"]), k) for k, e in enumerate(plan)])
        order = np.argsort(on_t, kind="stable"); on_t, on_k = on_t[order], on_k[order]
        grp = np.zeros(len(on_t), np.int64)
        g0 = 0
        for i in range(1, len(on_t)):
            if on_t[i] - on_t[g0] > a.gap or ep[on_t[i]] != ep[on_t[g0]]:
                g0 = i
            grp[i] = g0
        acted = np.ones(len(on_t), bool)
        hist = []
        for rnd in range(a.em_rounds + 1):
            W = []
            for k in range(K):
                idx = np.nonzero((on_k == k) & acted)[0]
                if len(idx) < 20:
                    idx = np.nonzero(on_k == k)[0]
                W.append(ridge(feats(plan[k], on_t[idx]), Y[on_t[idx]]))
            pred = np.zeros((len(on_t), len(agent)))
            for k in range(K):
                idx = np.nonzero(on_k == k)[0]
                pred[idx] = predict(W[k], feats(plan[k], on_t[idx]))
            res = Y[on_t] - pred
            # entity-balanced pooling: every entity weighs the same, so an entity with many onsets (e.g. a contact
            # sensor that changes at every grasp) cannot dominate; noise scale = median over entities of their
            # residual spread
            wgt = np.zeros(len(on_t))
            sds = []
            for k in range(K):
                m_ = acted & (on_k == k)
                if m_.sum():
                    wgt[m_] = 1.0 / m_.sum()
                if m_.sum() >= 20:
                    sds.append(res[m_].std(0))
            wgt /= wgt.sum()
            ybar = (wgt[:, None] * Y[on_t]).sum(0)
            r2 = 1 - (wgt[:, None] * res ** 2).sum(0) / np.maximum((wgt[:, None] * (Y[on_t] - ybar) ** 2).sum(0), 1e-12)
            sthr, sep_r2 = split_or(np.clip(r2, 0, 1), 0.5)
            sel = np.nonzero(np.clip(r2, 0, 1) > sthr)[0]
            if len(sel) < 2:                                              # no clear split: the best-explained dims
                sel = np.sort(np.argsort(-r2)[:max(2, int((r2 > 0.25).sum()))])
            noise_sd = np.median(np.array(sds), 0) if sds else res[acted].std(0)
            wt = 1.0 / (noise_sd[sel] + 1e-6)
            dist = (((res[:, sel]) * wt) ** 2).sum(1)
            new = np.zeros(len(on_t), bool)
            for g in np.unique(grp):
                idx = np.nonzero(grp == g)[0]
                new[idx[np.argmin(dist[idx])]] = True
            hist.append({"pass": tag, "round": rnd, "contact_dims": agent[sel].tolist(), "r2": np.round(r2, 3).tolist(), "r2_split_D": sep_r2,
                         "acted_changed": int((new != acted).sum()), "interactions": int(len(np.unique(grp))), "onsets": int(len(on_t))})
            print({k_: v for k_, v in hist[-1].items() if k_ != "r2"}, flush=True)
            if rnd < a.em_rounds:
                acted = new
        return dict(W=W, sel=sel, wt=wt, pred=pred, acted=acted, on_t=on_t, on_k=on_k, dist=dist, hist=hist, wgt=wgt)

    structure = []
    agent_extra = []                       # dims of demoted entities: agent-side parts, not part of the contact space
    for pas in range(4):
        ents = [describe(g) for g in groups]
        plan = [e for e in ents if not e["transient"]]
        em = contact_em(plan, agent, pas)
        K = len(plan)
        # contact consistency: median contact distance (noise units) at the entity's own attributed onsets
        med = np.array([np.median(np.sqrt(em["dist"][(em["on_k"] == k) & em["acted"]])) if ((em["on_k"] == k) & em["acted"]).any()
                        else np.inf for k in range(K)])
        # localisation ratio: an entity's contact noise relative to the spread of the agent's configuration around the
        # (entity-balanced) mean contact configuration at the same onsets; an object localises the agent (ratio well
        # below 1), a part that changes wherever the agent is (e.g. a contact sensor) does not. Demote if > 0.5.
        Ysel = (X[:, agent] - X[:, agent].mean(0)) / (X[:, agent].std(0) + 1e-9)
        Ysel = Ysel[:, em["sel"]] * em["wt"]
        ybar = (em["wgt"][:, None] * Ysel[em["on_t"]]).sum(0)
        rho = []
        for k in range(K):
            m_ = (em["on_k"] == k) & em["acted"]
            if not m_.any():
                rho.append(np.inf); continue
            spread = np.median(np.linalg.norm(Ysel[em["on_t"][m_]] - ybar, axis=1))
            rho.append(float(med[k] / max(spread, 1e-9)))
        rho = np.array(rho)
        demote = [k for k in range(K) if rho[k] > 0.5]
        # coinciding contact places
        sub_ = np.sort(rng.permutation(np.arange(1, n))[:20000])
        Cs = [predict(em["W"][k], feats(plan[k], sub_))[:, em["sel"]] * em["wt"] for k in range(K)]
        parent = list(range(K))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        merges = []
        for k in range(K):
            for l in range(k + 1, K):
                if k in demote or l in demote:
                    continue
                dkl = float(np.median(np.linalg.norm(Cs[k] - Cs[l], axis=1)))
                ref = min(med[k], med[l])                              # inf only if neither was ever attributed
                if np.isfinite(ref) and dkl < ref:                         # closer than either entity's own contact noise
                    parent[find(l)] = find(k); merges.append({"pair": [k, l], "median_distance": dkl})
        structure.append({"pass": pas, "entities": [e["dims"] for e in plan], "median_contact_distance": med.round(3).tolist(),
                          "localisation_ratio": np.round(rho, 3).tolist(),
                          "demoted": [plan[k]["dims"] for k in demote], "merged": merges})
        print({"pass": pas, "K": K, "demoted": [plan[k]["dims"] for k in demote], "merges": len(merges)}, flush=True)
        if not demote and not merges:
            break
        agent_extra = sorted(set(agent_extra) | {j for k in demote for j in plan[k]["dims"]})
        roots = {}
        for k in range(K):
            if k not in demote:
                roots.setdefault(find(k), []).extend(plan[k]["dims"])
        groups = [sorted(v) for v in roots.values()] + [e["dims"] for e in ents if e["transient"]]
    else:                                                               # last pass changed the structure: refit once more
        ents = [describe(g) for g in groups]
        plan = [e for e in ents if not e["transient"]]
        em = contact_em(plan, agent, "final")
        K = len(plan)
    rep["structure_passes"] = structure
    rep["entities_all"] = [{k: v for k, v in e.items() if k not in ("onsets", "cont_mean", "cont_std")} | {"n_onsets": len(e["onsets"])} for e in ents]
    rep["agent_dims_final"] = agent.tolist(); rep["agent_extra_dims"] = agent_extra
    amu, asd = X[:, agent].mean(0), X[:, agent].std(0) + 1e-9
    W, sel, wt, pred, acted, on_t, on_k = (em[k_] for k_ in ("W", "sel", "wt", "pred", "acted", "on_t", "on_k"))
    rep["contact_em"] = em["hist"]
    print({"planning_entities": K, "transient": len(ents) - K, "agent_dims": len(agent)}, flush=True)

    # 6. projection to (u, v, w)
    bal = np.concatenate([rng.choice(np.nonzero(acted & (on_k == k))[0], 2000, replace=True)
                          for k in range(K) if (acted & (on_k == k)).any()])
    C = pred[bal][:, sel] * wt
    pmu = C.mean(0)
    U, sv, Vt = np.linalg.svd(C - pmu, full_matrices=False)
    P = Vt[:3] if Vt.shape[0] >= 3 else np.r_[Vt, np.zeros((3 - Vt.shape[0], Vt.shape[1]))]
    rep["proj_explained"] = (sv[:3] ** 2 / (sv ** 2).sum()).round(3).tolist()
    L = {"kind": "generic", "agent_dims": agent.tolist(), "agent_extra_dims": agent_extra, "agent_mean": amu.tolist(), "agent_std": asd.tolist(),
         "contact_dims": sel.tolist(), "contact_weight": wt.tolist(), "proj": P.tolist(), "proj_mean": pmu.tolist(),
         "canvas_centre": [0.0, 0.0], "canvas_scale": 1.0, "w_mid": 0.0, "entities": []}
    for k, e in enumerate(plan):
        ent = {"dims": e["dims"], "cont_dims": e["cont_dims"], "disc_dims": e["disc_dims"], "cont_mean": e["cont_mean"],
               "cont_std": e["cont_std"], "W": W[k].tolist()}
        if e["disc_dims"]:
            still = ~moving[:, e["disc_dims"]].any(1)
            Z = X[still][:, e["disc_dims"]]
            dmu = Z.mean(0)
            _, s2, V2 = np.linalg.svd(Z[rng.permutation(len(Z))[:200000]] - dmu, full_matrices=False)
            nd = int(min(2, (s2 > 1e-6 * s2[0]).sum()))
            proj = V2[:nd]
            q = (Z - dmu) @ proj.T
            ent.update(disc_mean=dmu.tolist(), disc_proj=proj.tolist(), disc_lo=q.min(0).tolist(), disc_hi=q.max(0).tolist())
        L["entities"].append(ent)
    G = GenericMap(L)
    sub = rng.permutation(n)[:200000]
    S0, _ = G(X[sub])
    stillk = np.stack([~moving[sub][:, e["dims"]].any(1) for e in plan], 1)
    uv = S0[..., :2][stillk]
    lo_, hi_ = np.percentile(uv, 1, 0), np.percentile(uv, 99, 0)        # first pass: uv = z + 32 (centre 0, scale 1)
    L["canvas_scale"] = float(48.0 / max((hi_ - lo_).max(), 1e-9))
    L["canvas_centre"] = ((lo_ + hi_) / 2 - 32.0).tolist()
    L["w_mid"] = float(np.median((S0[..., 2][stillk] - 0.5) * 64.0))       # first pass: a0 = z_w / 64 + 0.5
    G = GenericMap(L)

    # 7. thresholds
    S, eff = G(X)
    same = np.r_[False, ~term[:-1]]
    dpos = np.linalg.norm(S[1:, :, :2] - S[:-1, :, :2], axis=-1)[same[1:]].ravel()
    dapp = np.abs(S[1:, :, 2:5] - S[:-1, :, 2:5]).max(-1)[same[1:]].ravel()

    def noise(x, fallback):
        x = x[x > 0]
        if len(x) < 10:
            return fallback
        t_, _ = split_or(np.log(x + 1e-12), None)
        return float(np.exp(t_)) if t_ is not None else fallback

    r_app = noise(dapp, 1e-3)
    gaps = []
    for k, e in enumerate(plan):
        still = ~moving[:, e["dims"]].any(1)
        for f in range(2, 5):
            v = np.sort(S[still, k, f][rng.permutation(int(still.sum()))[:200000]])
            if len(v) < 100 or v[-1] - v[0] < 4 * r_app:
                continue
            cut = np.nonzero(np.diff(v) > 4 * r_app)[0]
            parts = np.split(v, cut + 1)
            meds = [float(np.median(p)) for p in parts if len(p) >= 0.005 * len(v)]
            if len(meds) >= 2:
                gaps.append({"entity": k, "field": f, "gap": float(np.min(np.diff(meds))), "modes": len(meds)})
    thr_app = 0.5 * min(g["gap"] for g in gaps) if gaps else 0.5
    rep["rest_a0_percentiles"] = [np.percentile(S[stills_k, k, 2], [1, 10, 50, 90, 99]).round(4).tolist()
                                  for k, stills_k in enumerate((~moving[:, e["dims"]].any(1) for e in plan))]
    stills = np.stack([~moving[:, e["dims"]].any(1) for e in plan], 1)
    # touching radius: how close (u, v) the agent is to an entity when it starts changing it (attributed onsets);
    # thr_pos = 2 r_touch (an object's extent for the contact-free rest rule and the move-end rule), tol_pos = r_touch
    d_on = np.linalg.norm(eff[on_t[acted]] - S[on_t[acted], on_k[acted], :2], axis=-1)
    # attributed onsets mix contacts (agent at the object) with changes caused from a distance (knocks, pushes by a
    # carried object); the touching radius is the 95th percentile of the contact mode of a 2-means split of log
    # distances when the modes are clear (D > 2, centres >= 3x apart), else of all onsets
    ld = np.log(d_on + 1e-6)
    t_, _ = split_or(ld, None)
    contact = np.ones(len(d_on), bool)
    if t_ is not None and (ld > t_).any() and (ld <= t_).any() and ld[ld > t_].mean() - ld[ld <= t_].mean() >= np.log(3.0):
        contact = ld <= t_
    r_touch = float(np.percentile(d_on[contact], 95))
    rep["touch_distance_percentiles"] = {"all": np.percentile(d_on, [50, 90, 95, 99]).round(3).tolist(),
                                         "contact_mode_fraction": float(contact.mean()),
                                         "per_entity_p95": [float(np.percentile(d_on[on_k[acted] == k], 95)) if (on_k[acted] == k).any() else None
                                                            for k in range(K)],
                                         "random_frames_p5": float(np.percentile(np.linalg.norm(eff[::50, None] - S[::50, :, :2], axis=-1), 5))}
    thr_pos = 2.0 * r_touch
    pair_min = []                                                   # diagnostic only
    for k in range(K):
        for l in range(k + 1, K):
            both = stills[:, k] & stills[:, l] & (np.abs(S[:, k, 2] - S[:, l, 2]) < thr_app)
            if both.sum() >= 50:
                d = np.linalg.norm(S[both, k, :2] - S[both, l, :2], axis=-1)
                pair_min.append({"pair": [k, l], "p1": float(np.percentile(d, 1))})
    r_pos = noise(dpos, 0.02 * thr_pos)
    thr = {"thr_pos": thr_pos, "r_pos": r_pos, "tol_pos": thr_pos / 2, "r_touch": r_touch, "r_app": r_app, "thr_app": thr_app, "m": 5,
           "source": "g_entities.py (generic state front end)"}
    rep.update(thresholds=thr, appearance_gaps=gaps, pair_min=pair_min)
    (a.out / "thr" / "report.json").write_text(json.dumps(thr, indent=1) + "\n")
    L.update(thr_pos=thr_pos, tol_pos=thr_pos / 2, r_pos=r_pos, r_app=r_app, thr_app=thr_app, K=K)
    (a.out / "front" / "layout.json").write_text(json.dumps(L) + "\n")
    disc = {"objects": K, "source": "g_entities (generic state)", "groups": [[k] for k in range(K)],
            "table": [{"cluster": k, "spread_median": thr_pos / np.sqrt(6)} for k in range(K)],
            "identities": [{"anchor": "contact", "types": [k], "centre": None, "radius": None} for k in range(K)]}
    (a.out / "front" / "discover.json").write_text(json.dumps(disc, indent=1) + "\n")
    for split in ("train", "val"):
        o = data[split]["observations"]
        Ssp, effs = G(o)
        np.savez(a.out / "front" / f"entities_{split}.npz", pos=Ssp[..., :2], app=Ssp[..., 2:5],
                 area=np.ones(Ssp.shape[:2], np.int16), effector=effs, processed=np.ones(len(o), bool))
    print(json.dumps(thr), flush=True)

    # PRIVILEGED diagnostics (report only)
    agent_true, objs = privileged_groups(a.env, Dobs)
    pv = {"agent_dims_true_nonconstant": [j for j in agent_true if not const[j]],
          "agent_dims_found_not_true": [int(j) for j in agent if j not in agent_true],
          "true_agent_dims_classified_object": [int(j) for j in objd if j in agent_true]}
    name_of = {j: nm for nm, ds in objs.items() for j in ds}
    pv["groups"] = [{"dims": e["dims"], "true_objects": sorted({name_of.get(j, "agent") for j in e["dims"]}),
                     "transient": e["transient"], "disc": e["disc_dims"], "cont": e["cont_dims"]} for e in ents]
    pv["planning_entity_objects"] = [sorted({name_of.get(j, "agent") for j in e["dims"]}) for e in plan]
    pv["objects_without_planning_entity"] = sorted(set(objs) - {o for g in pv["planning_entity_objects"] for o in g})
    pv["contact_dims_last_round"] = agent[sel].tolist()
    # position quality: true xy of movable objects (first 2 dims of their block) vs entity (u, v), linear R^2
    q = []
    for k, objs_k in enumerate(pv["planning_entity_objects"]):
        if len(objs_k) == 1 and objs_k[0] != "agent":
            blk = objs[objs_k[0]]
            sub_ = rng.permutation(n)[:50000]
            if X[sub_][:, blk[0]].std() > 1e-6 and len(blk) >= 9:
                T = X[sub_][:, blk[:2]]
                A_ = np.c_[S[sub_, k, :2], np.ones(len(sub_))]
                coef = np.linalg.lstsq(A_, T, rcond=None)[0]
                q.append({"entity": k, "object": objs_k[0], "xy_r2": (1 - ((T - A_ @ coef) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)).round(4).tolist()})
    pv["movable_position_r2"] = q
    # true effector-to-object distance (m; observation = 10 x metres) at the attributed onsets of movable objects
    td = []
    for k, objs_k in enumerate(pv["planning_entity_objects"]):
        if len(objs_k) == 1 and objs_k[0] in objs and len(objs[objs_k[0]]) >= 9:
            t_ = on_t[acted & (on_k == k)]
            d_ = np.linalg.norm(X[t_][:, 12:15] - X[t_][:, objs[objs_k[0]][:3]], axis=1) / 10.0
            td.append({"entity": k, "object": objs_k[0], "onsets": int(len(t_)), "dist_m_p10_50_90": np.percentile(d_, [10, 50, 90]).round(4).tolist()})
    pv["true_effector_object_distance_at_onsets"] = td
    rep["PRIVILEGED"] = pv
    (a.out / "front" / "generic_report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print("PRIVILEGED", json.dumps({k: v for k, v in pv.items() if k != "groups"}, default=float), flush=True)


if __name__ == "__main__":
    main()
