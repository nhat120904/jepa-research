#!/usr/bin/env python3
"""State track: OGBench state observations -> the unified backend's entity tables (u_events.py --per-frame input).

The unified backend (u_events / u_wm / u_skill / u_closed_loop) works on K entities x (u, v, a0, a1, a2, covered).
In the state track the entities come from the object-factored observation of the env (its documented layout:
19 proprioceptive dims, then one block per object), not from perception:
  movable object (cube, block of 9)   (u, v) = its xy, a0 = its height, a1 = a2 = 0; covered = another object rests
                                      on it;
  fixed object (button, block of 4)   (u, v) = its location = median effector xy while its own joint is pressed
                                      (from the play data, no labels), a0 = a1 = a2 = its discrete state (0 / 1).
xy is drawn on the backend's 64-unit canvas (u = 32 + 8 x, v = 32 + 8 y in the observation's scaled coordinates,
1 m = 80 units), height as a0 = z / 4 (1 m = 2.5). The agent is the end effector, a point (`effector` track instead
of agent masks): an entity is observed at rest while the effector is farther than half an object width from it,
the same contact-free rule as the pixel track.
Thresholds (written as a report.json for u_events --thresholds-from):
  thr_pos = object width w: movable objects twice their lowest resting height (they rest on the table), fixed objects
            the smallest spacing between their locations; tol_pos = w / 2 (a move is a displacement of more than half
            an object); r_pos, r_app = noise radii (2-means split of log frame-to-frame changes, as u_events.py);
  thr_app = half of the smallest change between distinct rest values of a0 (one object height for stacking, 1 for a
            button state).
Writes OUT/cache/<env> ({split}_observations / actions / terminals .npy for u_skill / u_closed_loop; qpos and button
states too, read only by the PRIVILEGED diagnostics of u_events.py), OUT/front (entities_{split}.npz, discover.json,
layout.json) and OUT/thr/report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sfa_code import two_means_threshold

PROPRIO = 19
SCALE, OFFSET, ZA = 8.0, 32.0, 0.25          # canvas: u = OFFSET + SCALE * x_obs; a0 = ZA * z_obs (obs = 10 x metres)


def layout_of(env: str, obs_dim: int) -> dict:
    """Object blocks of the env's documented state observation."""
    if env.startswith("cube"):
        return {"kind": "movable", "block": 9, "K": (obs_dim - PROPRIO) // 9}
    if env.startswith("puzzle"):
        return {"kind": "fixed", "block": 4, "K": (obs_dim - PROPRIO) // 4}
    raise SystemExit(f"no state layout for {env}")


def effector_uv(obs):
    return OFFSET + SCALE * obs[..., 12:14]


def state_entities(obs, L):
    """obs (n, D) -> entity states (n, K, 6) [u, v, a0, a1, a2, covered] and effector (n, 2), canvas units."""
    n, K, b = len(obs), L["K"], L["block"]
    S = np.zeros((n, K, 6), np.float32)
    blk = obs[:, PROPRIO:PROPRIO + K * b].reshape(n, K, b)
    if L["kind"] == "movable":
        S[..., 0:2] = OFFSET + SCALE * blk[..., 0:2]
        S[..., 2] = ZA * blk[..., 2]
        w, xy, z = L["w_obs"], blk[..., 0:2], blk[..., 2]
        d = np.linalg.norm(xy[:, :, None] - xy[:, None], axis=-1)              # (n, K, K): k vs j
        dz = z[:, None, :] - z[:, :, None]                                      # z_j - z_k
        on_top = (d < w / 2) & (dz > w / 2) & (dz < 1.5 * w)
        on_top[:, np.arange(K), np.arange(K)] = False
        S[..., 5] = on_top.any(-1)
    else:
        S[..., 0:2] = np.asarray(L["loc_uv"], np.float32)[None]
        S[..., 2:5] = blk[..., 1:2]                                             # one-hot (state 0, state 1) -> state
    return S, effector_uv(obs).astype(np.float32)


def noise_radius(x, fallback):
    x = x[x > 0]
    if len(x) < 10:
        return fallback
    thr, sep, _ = two_means_threshold(np.log(x + 1e-9))
    return float(np.exp(thr)) if sep > 2.0 else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dir with <env>.npz and <env>-val.npz (OGBench state datasets)")
    ap.add_argument("--env", required=True, help="e.g. cube-triple-play-v0, puzzle-4x5-play-v0")
    ap.add_argument("--train-episodes", type=int, default=3000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
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
        for k in ("qpos", "button_states"):                                          # PRIVILEGED, diagnostics only
            if k in z.files:
                np.save(cache / f"{split}_{k}.npy", np.asarray(z[k][:n]))
    obs = data["train"]["observations"]
    L = layout_of(a.env, obs.shape[1])
    K, b = L["K"], L["block"]
    blk = obs[:, PROPRIO:PROPRIO + K * b].reshape(len(obs), K, b)
    if L["kind"] == "movable":
        # objects rest on the table: twice the lowest resting height is the object height = width (cubes)
        L["w_obs"] = float(2 * np.percentile(blk[..., 2], 5))
        thr_pos = L["w_obs"] * SCALE
        thr_app = ZA * L["w_obs"] / 2
    else:
        depth = blk[..., 2]
        loc = []
        for k in range(K):
            pressed = depth[:, k] < (np.median(depth[:, k]) + depth[:, k].min()) / 2       # its own joint moved
            loc.append(np.median(effector_uv(obs[pressed]), 0).tolist())
        L["loc_uv"] = loc
        P = np.array(loc)
        dd = np.linalg.norm(P[:, None] - P[None], axis=-1); dd[np.arange(K), np.arange(K)] = np.inf
        thr_pos = float(dd.min())
        thr_app = 0.5
    S, eff = state_entities(obs, L)
    term = data["train"]["terminals"]
    same = np.r_[False, ~term[:-1]]                                              # frame t and t-1 in one episode
    dpos = np.linalg.norm(S[1:, :, :2] - S[:-1, :, :2], axis=-1)[same[1:]].ravel()
    dapp = np.abs(S[1:, :, 2:5] - S[:-1, :, 2:5]).max(-1)[same[1:]].ravel()
    r_pos = noise_radius(dpos, 0.02 * thr_pos)
    r_app = noise_radius(dapp, 0.02 * thr_app * 2)
    rep = {"thr_pos": thr_pos, "r_pos": r_pos, "tol_pos": thr_pos / 2, "r_app": r_app, "thr_app": thr_app, "m": 5,
           "source": "s_entities.py (state track)"}
    (a.out / "thr" / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    L.update(thr_pos=thr_pos, tol_pos=thr_pos / 2, r_pos=r_pos, r_app=r_app, thr_app=thr_app, scale=SCALE, offset=OFFSET, za=ZA)
    (a.out / "front" / "layout.json").write_text(json.dumps(L, indent=1) + "\n")
    disc = {"objects": K, "source": "s_entities (state)", "groups": [[k] for k in range(K)],
            "table": [{"cluster": k, "spread_median": thr_pos / np.sqrt(6)} for k in range(K)],
            "identities": [{"anchor": "colour" if L["kind"] == "movable" else "location", "types": [k], "centre": None, "radius": None}
                           for k in range(K)]}
    (a.out / "front" / "discover.json").write_text(json.dumps(disc, indent=1) + "\n")
    for split in ("train", "val"):
        o = data[split]["observations"]
        S, eff = state_entities(o, L)
        np.savez(a.out / "front" / f"entities_{split}.npz", pos=S[..., :2], app=S[..., 2:5], area=(S[..., 5] < 0.5).astype(np.int16),
                 effector=eff, processed=np.ones(len(o), bool))
        print(split, {"frames": len(o), "episodes": int(data[split]["terminals"].sum()), "K": K,
                      "covered_frac": float(S[..., 5].mean())}, flush=True)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
