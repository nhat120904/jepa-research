#!/usr/bin/env python3
"""Low level L1 (user decision 2026-10-10, after two behaviour-cloned executors failed on puzzle-4x4): an IMAGE-GOAL policy
trained by GCIVL (OGBench's goal-conditioned implicit V-learning) on the play data. Two uses:
  (i)  executor of one planned event, given a SUBGOAL image (subgoal.py: the current frame with the places the event
       changes shown in their target states);
  (ii) with the task's goal image directly: the flat baseline in our code path (V2_PLAN B2 flat-raw, easy-task path D_i).
No labels: (frame, action, later frame) triples of the play data.

Recipe = OGBench GCIVL for pixel tasks (Park et al. 2024, impls/agents/gcivl.py), ported to PyTorch:
  value  V(s, g): two heads on one Impala-small encoder of concat(s, g); expectile .9 regression of
         r + gamma * mask * V_target(s', g), r = success - 1, mask = 1 - success, success = (goal is the current frame);
         the expectile weight follows the sign of the target-net advantage; Polyak .005 target nets;
  actor  its own Impala-small encoder of concat(s, g); AWR with weight min(exp(alpha (V(s', g) - V(s, g))), 100),
         alpha 10, Gaussian with a fixed std (weighted squared error), deterministic mean at test time;
  goals  value: current .2 / same trajectory .5 (geometric, gamma) / random .3; actor: uniform future frame of the same
         trajectory; random-crop augmentation (pad 4) of all images of a batch with p .5; batch 256; Adam 3e-4.
Frames of --episodes TRAIN episodes are held in RAM (random reads of four images per sample from a memory map are slow).

STATE GOALS (--goal state, user decision 2026-10-11, after image subgoals left the executor the weak level on all three
families: cube-triple 0.2 planned events per episode, puzzle 4x4 events as predicted .30-.39): the goal is the ENTITY STATE the
world model predicts after the event (K x 6: position, appearance, covered), not an image, so no subgoal rendering. Inputs: the
current frame (its own Impala-small) and [current belief, goal, goal - current] (positions / 64; an MLP), joined before the
heads. States per frame = the memory of the closed loop: the latest valid reading of the episode (events_objects.py
labels_train.npz), else its first one. success(s, g) = every entity within the event thresholds of the events table
(position within tol_pos, appearance within thr_app_id, the covered bit equal; a hidden goal entity: only being hidden counts,
as world_model.py at_goal), so all frames with the goal's state count, not only the goal's own frame. Goals also include the
state after the NEXT EVENT of the frame (value: current .2 / next event .2 / trajectory .3 / random .3; actor: next event
.5 / uniform future .5): the executor is asked for one event. Random crops shift the states' positions with the image.
Outputs (--out): gcivl.pt (actor, value, config), gcivl_report.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from utils import save_json


def make_nets(act_dim=5, goal_dim=None):
    """goal_dim None: image goals (Impala on concat(s, g)); else state goals of goal_dim = K * 6 (Impala on s + an MLP on
    goal_vector, joined)."""
    import torch
    import torch.nn as nn

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

        def forward(self, x):
            return x + self.c2(torch.relu(self.c1(torch.relu(x))))

    class Impala(nn.Module):                                                    # OGBench impala_small: stacks 16/32/32, 1 block
        def __init__(self, cin=6, out=512):
            super().__init__()
            layers, c = [], cin
            for s in (16, 32, 32):
                layers += [nn.Conv2d(c, s, 3, padding=1), nn.MaxPool2d(3, 2, padding=1), Res(s)]
                c = s
            self.f = nn.Sequential(*layers)
            self.fc = nn.Sequential(nn.ReLU(), nn.Flatten(), nn.Linear(32 * 8 * 8, out), nn.LayerNorm(out), nn.GELU())

        def forward(self, s, g=None):
            return self.fc(self.f(s if g is None else torch.cat([s, g], 1)))

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
                             nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, o))

    class StateEnc(nn.Module):                                                  # state goals: frame features + goal-vector features
        def __init__(self):
            super().__init__()
            self.img = Impala(3)
            self.vec = nn.Sequential(nn.Linear(3 * goal_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU())

        def forward(self, s, g):
            return torch.cat([self.img(s), self.vec(g)], -1)

    def enc():
        return Impala() if goal_dim is None else StateEnc()

    zd = 512 if goal_dim is None else 1024

    class Value(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.h1, self.h2 = enc(), mlp(zd, 1), mlp(zd, 1)

        def forward(self, s, g):
            z = self.enc(s, g)
            return self.h1(z).squeeze(-1), self.h2(z).squeeze(-1)

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.head = enc(), mlp(zd, act_dim)

        def forward(self, s, g):
            return self.head(self.enc(s, g))

    return Value(), Actor()


def to_tensor(x, dev):
    import torch
    return torch.as_tensor(x, device=dev).permute(0, 3, 1, 2).float().div_(255)


POS_SCALE = np.array([1 / 64, 1 / 64, 1, 1, 1, 1], np.float32)


def goal_vector(cur, goal):
    """(..., K, 6) current belief and goal entity states -> (..., 18 K): per entity [current, goal, goal - current], positions
    divided by 64 (the frame side). Training and the closed loop use this function."""
    c = np.asarray(cur, np.float32) * POS_SCALE; g = np.asarray(goal, np.float32) * POS_SCALE
    return np.concatenate([c, g, g - c], -1).reshape(*c.shape[:-2], -1)


def belief_states(events, n, last):
    """per-frame entity states (n, K, 6) of the first n TRAIN frames as the closed loop's memory: the latest valid reading of
    the episode (labels_train.npz pos / app with valid, cov with cov_valid), else the episode's first valid one."""
    L = np.load(events / "labels_train.npz", mmap_mode="r")
    X = np.concatenate([np.asarray(L["pos"][:n]), np.asarray(L["app"][:n]), np.asarray(L["cov"][:n], np.float32)[..., None]], -1)
    V = (np.asarray(L["valid"][:n]), np.asarray(L["cov_valid"][:n]))
    out = X.copy()
    ends = np.unique(last)
    for s0 in np.concatenate([[0], ends[:-1] + 1]):
        e0 = int(last[s0]) + 1
        ar = np.arange(e0 - s0)
        for k in range(X.shape[1]):
            for cols, vm in ((slice(0, 5), V[0][s0:e0, k]), (slice(5, 6), V[1][s0:e0, k])):
                if not vm.any():
                    continue
                idx = np.maximum.accumulate(np.where(vm, ar, -1))
                idx[idx < 0] = int(np.argmax(vm))
                out[s0:e0, k, cols] = X[s0 + idx, k, cols]
    return out


def reached(S, G, tol_pos, thr_app):
    """(B, K, 6) states and goals -> (B,) every entity at its goal within the event thresholds (a hidden goal entity: only
    being hidden counts, as world_model.py at_goal)."""
    dp = np.linalg.norm(S[..., :2] - G[..., :2], axis=-1) <= tol_pos
    da = np.abs(S[..., 2:5] - G[..., 2:5]).max(-1) <= np.asarray(thr_app)[None] + 1e-4
    dc = (S[..., 5] > 0.5) == (G[..., 5] > 0.5)
    return np.where(G[..., 5] > 0.5, dc, dp & da & dc).all(-1)


def crop(x, sh):
    """x (B, C, 64, 64), sh (B, 2) int shifts in [0, 8] -> random crop after replicate padding of 4 (one shift per sample)."""
    import torch
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="replicate")
    idx = torch.arange(64, device=x.device)
    rows = (sh[:, 1, None] + idx[None]).long(); cols = (sh[:, 0, None] + idx[None]).long()
    b = torch.arange(len(x), device=x.device)[:, None, None]
    return xp.permute(0, 2, 3, 1)[b, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=500, help="TRAIN episodes held in RAM")
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--expectile", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--p-aug", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--goal", choices=("image", "state"), default="image", help="state: entity-state goals (module doc)")
    ap.add_argument("--events", type=Path, default=None, help="--goal state: events_objects.py output (labels_train.npz, events_train.npz)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.goal == "state" and a.events is None:
        raise SystemExit("--goal state needs --events")
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch / local/run_stage.ps1")
    import torch

    torch.manual_seed(0); rng = np.random.default_rng(0)
    dev = a.device
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    term = np.load(a.cache / "train_terminals.npy")
    ends = np.flatnonzero(term)[:a.episodes]; n = int(ends[-1] + 1)
    ep_of = np.concatenate([[0], np.cumsum(term[:n - 1])]).astype(np.int64)
    last = ends[ep_of]                                                          # last frame of each frame's episode
    obs = np.ascontiguousarray(np.load(a.cache / "train_observations.npy", mmap_mode="r")[:n])
    act = np.asarray(np.load(a.cache / "train_actions.npy", mmap_mode="r")[:n], np.float32)
    valid = np.flatnonzero(np.arange(n) < last)                                 # t + 1 in the same episode
    state = a.goal == "state"
    extra = {}
    if state:                                                                   # entity states per frame, thresholds, next events
        B = belief_states(a.events, n, last)
        E = np.load(a.events / "events_train.npz")
        tol_pos, thr_app = float(E["tol_pos"]), np.asarray(E["thr_app_id"], np.float64)
        te = np.sort(E["t"][E["t"] < n])
        j_ = np.searchsorted(te, np.arange(n), side="right")
        cand = te[np.minimum(j_, len(te) - 1)]
        nxt = np.where((j_ < len(te)) & (cand <= last), cand, last)              # end of the frame's next event, else its episode's last
        K = B.shape[1]
        extra = {"goal": "state", "goal_dim": int(K * 6), "K": int(K), "tol_pos": tol_pos, "thr_app_id": thr_app.tolist()}
        print({"state_goals": {"K": K, "events": int(len(te)), "tol_pos": tol_pos, "share_frames_with_next_event": round(float((nxt < last).mean()), 3)}}, flush=True)
    print({"frames": n, "episodes": len(ends), "ram_gb": round(obs.nbytes / 1e9, 2), "load_min": round((time.time() - t0) / 60, 1)}, flush=True)

    def goals(t, p_cur, p_traj, geom, p_next=0.0):
        u = rng.random(len(t))
        g = rng.integers(0, n, len(t))                                          # random frame
        if geom:
            off = rng.geometric(1 - a.discount, len(t))
            traj = np.minimum(t + off, last[t])
        else:                                                                   # uniform over (t, last]
            traj = np.minimum(t + 1 + np.floor(rng.random(len(t)) * (last[t] - t)).astype(np.int64), last[t])
        g = np.where(u < p_cur + p_next + p_traj, traj, g)
        if p_next > 0:
            g = np.where(u < p_cur + p_next, nxt[t], g)
        g = np.where(u < p_cur, t, g)
        return g

    value, actor = make_nets(act.shape[1], goal_dim=extra.get("goal_dim"))
    value, actor = value.to(dev), actor.to(dev)
    target = copy.deepcopy(value).requires_grad_(False)
    opt_v = torch.optim.Adam(value.parameters(), lr=a.lr)
    opt_a = torch.optim.Adam(actor.parameters(), lr=a.lr)
    log = []
    for step in range(a.steps):
        t = valid[rng.integers(0, len(valid), a.batch)]
        if state:
            gv = goals(t, 0.2, 0.3, True, p_next=0.2)
            ga = goals(t, 0.0, 0.5, False, p_next=0.5)
            S, S1 = (to_tensor(obs[i], dev) for i in (t, t + 1))
            Bt, Bt1, BV, BA = (B[i].copy() for i in (t, t + 1, gv, ga))
            if rng.random() < a.p_aug:
                sh_ = rng.integers(0, 9, (a.batch, 2))
                sh = torch.as_tensor(sh_, device=dev)
                S, S1 = (crop(x, sh) for x in (S, S1))
                for X_ in (Bt, Bt1, BV, BA):                                    # the states move with the cropped image
                    X_[..., 0] += (4 - sh_[:, 0])[:, None]; X_[..., 1] += (4 - sh_[:, 1])[:, None]
            GV, GV1, GA, GA1 = (torch.as_tensor(goal_vector(c_, g_), device=dev) for c_, g_ in ((Bt, BV), (Bt1, BV), (Bt, BA), (Bt1, BA)))
            succ = torch.as_tensor(reached(B[t], B[gv], tol_pos, thr_app).astype(np.float32), device=dev)
        else:
            gv = goals(t, 0.2, 0.5, True)
            ga = goals(t, 0.0, 1.0, False)
            S, S1, GV, GA = (to_tensor(obs[i], dev) for i in (t, t + 1, gv, ga))
            if rng.random() < a.p_aug:
                sh = torch.as_tensor(rng.integers(0, 9, (a.batch, 2)), device=dev)
                S, S1, GV, GA = (crop(x, sh) for x in (S, S1, GV, GA))
            GV1, GA1 = GV, GA                                                   # an image goal does not depend on the state
            succ = torch.as_tensor((gv == t).astype(np.float32), device=dev)
        A = torch.as_tensor(act[t], device=dev)
        r, mask = succ - 1.0, 1.0 - succ
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            with torch.no_grad():
                nv1, nv2 = (x.float() for x in target(S1, GV1))
                tv1, tv2 = (x.float() for x in target(S, GV))
                adv_t = r + a.discount * mask * torch.minimum(nv1, nv2) - (tv1 + tv2) / 2
            v1, v2 = (x.float() for x in value(S, GV))
        w = torch.abs(a.expectile - (adv_t < 0).float())
        loss_v = (w * (r + a.discount * mask * nv1 - v1) ** 2).mean() + (w * (r + a.discount * mask * nv2 - v2) ** 2).mean()
        opt_v.zero_grad(set_to_none=True); loss_v.backward(); opt_v.step()
        with torch.no_grad():
            for p_, q_ in zip(target.parameters(), value.parameters()):
                p_.mul_(1 - a.tau).add_(q_, alpha=a.tau)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            with torch.no_grad():
                av1, av2 = (x.float() for x in value(S, GA))
                an1, an2 = (x.float() for x in value(S1, GA1))
                adv = (an1 + an2) / 2 - (av1 + av2) / 2
            mu = actor(S, GA).float()
        wa = torch.clamp(torch.exp(adv * a.alpha), max=100.0)
        loss_a = (wa * ((mu - A) ** 2).sum(-1)).mean()
        opt_a.zero_grad(set_to_none=True); loss_a.backward(); opt_a.step()
        if step % 2000 == 0 or step == a.steps - 1:
            log.append({"step": step, "loss_v": round(loss_v.item(), 4), "loss_a": round(loss_a.item(), 4), "v": round(float(v1.detach().mean()), 2),
                        "succ_share": round(float(succ.mean()), 3),
                        "adv_std": round(float(adv.std()), 4), "w_mean": round(float(wa.mean()), 2),
                        "mse": round(float(((mu.detach() - A) ** 2).sum(-1).mean()), 4), "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
        if (step + 1) % 50000 == 0 or step == a.steps - 1:
            torch.save({"actor": actor.state_dict(), "value": value.state_dict(), "act_dim": int(act.shape[1]), "step": step + 1, **extra,
                        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}, a.out / "gcivl.pt")
    save_json(a.out / "gcivl_report.json", {"frames": n, "steps": a.steps, "log": log, "minutes": round((time.time() - t0) / 60, 1)})


if __name__ == "__main__":
    main()
