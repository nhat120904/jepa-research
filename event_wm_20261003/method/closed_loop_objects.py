#!/usr/bin/env python3
"""Component 17, object track (method/README.md): one closed loop for every family on the official OGBench visual tasks.

COVERED: a mover not seen while the agent is clear of where it was is hidden by an object (covered bit 1, as the events'
"never visible in a static interval"); in the goal image an unseen mover is a hidden goal (the planner then only asks for
it to be hidden: world_model.Model.at_goal), an unseen place is unknown.
GOAL IMAGE: a static frame, so a place under the arm is read from the SeeThrough codes of its tokens whatever their
probability (the lookup only holds code tuples seen with a known state): dev puzzle tasks (30 loop seeds, PRIVILEGED check)
goal lights known .81 with p > .5 (min .70), .988 with any looked-up code (min .90), all of them correct.
Entities not seen in the first frame start at their goal state (or their unknown reading) until first seen, so the first
plan does not act on readings of hidden entities. Within an event an entity's change is measured from its first observation
in that event: a value that changed while it was hidden before (or a first sighting) is information, not this event's
change (otherwise events ended early: puzzle dev run 1, 37% of events ended with no press and 1-3 "changed" lights); the
acted entity still ends the event when it reaches its target. Online reading = the offline table rules, frame by frame (objects.read_frame): segmenter agent mask (dilated by one token),
agent colours, movers by colour prototypes, places in full view or through the agent with the SeeThrough codes of their
tokens (the online SeeThrough of the front end). An entity is a REST observation when the table would show it and, for a
mover, no agent pixel lies within half an object width of it (events_objects.py: a cube held still is in transit; a
place reading is agent-free). State of an entity = its last rest observation (memory); at rest = rest observations in
the last m frames. Goal image: the same reading; goal entities without a rest observation in it are UNKNOWN and ignored by
the goal test (world_model.Model.plan(known=...)).
Loop (as closed_loop.py, from u_closed_loop): batched weighted A* over events -> the skill executes the first event (e, x)
-> the event ends when e is at rest again and either changed or reached x, or another entity changed and is at rest
(once e is at rest too, or while e is hidden) -> belief: entities the world model predicts to change that were not observed at rest since the
first observed change take the predicted state when everything observed since agrees with it (a light next to the
pressed one still under the arm kept its old value and the next plan undid the press) -> the plan continues while the
outcomes agree with the prediction, else replan (replanning after every event with a weighted, non-optimal A* made the
first step of each new plan undo the last one: puzzle dev, one light pressed 11 times in a row); per event timeout: replay the mean post-event action of the play data for
--recover-steps, then replan. Change / arrival tests use the event thresholds (tol_pos in position, thr_app in
appearance). No privileged input; simulator states are logged for analysis only.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from utils import save_json

W5 = 5 ** np.arange(6)


class PressController:
    """PRIVILEGED diagnostic arm only (--low scripted, puzzle): scripted press of one button from the simulator's button and
    effector positions (copy of scripts/inventory.PressController). Not part of the method."""

    def __init__(self, env, button, home):
        self.u, self.button, self.pressed, self.home = env.unwrapped, button, False, home
        self.start = self.u._cur_button_states.copy()

    SAFE_Z = 0.32                                                             # travel height above the board

    def act(self):
        u = self.u
        eff = u.compute_ob_info()["proprio/effector_pos"]
        top = u._data.site_xpos[u._button_site_ids[self.button]].copy()
        above, bottom = top + np.array([0, 0, 0.06]), top - np.array([0, 0, 0.022])
        if not self.pressed and (u._cur_button_states != self.start).any():
            self.pressed = True
        safe = self.SAFE_Z - 0.03
        if self.pressed:
            if eff[2] < safe and np.linalg.norm(self.home[:2] - eff[:2]) > 0.03:
                target = np.array([eff[0], eff[1], self.SAFE_Z])                 # lift straight up first
            elif np.linalg.norm(self.home - eff) < 0.03:
                return None
            elif np.linalg.norm(self.home[:2] - eff[:2]) > 0.03:
                target = np.array([self.home[0], self.home[1], self.SAFE_Z])     # travel at the safe height
            else:
                target = self.home                                              # retreat to the start pose, uncovering the board
        elif np.linalg.norm(above[:2] - eff[:2]) > 0.04:
            # travel to the button at the safe height (a low straight path pressed other buttons on the way: dev run,
            # 2-3 timeouts per episode with up to 10 lights flipped)
            target = np.array([eff[0], eff[1], self.SAFE_Z]) if eff[2] < safe else np.array([above[0], above[1], self.SAFE_Z])
        else:
            target = bottom
        diff = target - eff
        n = np.linalg.norm(diff)
        diff = diff if n >= 0.4 else diff / (n + 1e-6) * 0.4
        a = np.zeros(5); a[:3] = diff * 5; a[4] = 1
        return np.clip(a, -1, 1)


def run(a, jobs):
    os.environ.setdefault("LP_NUM_THREADS", "1")
    import gymnasium
    import ogbench  # noqa: F401
    import torch

    from frontend import SeeThrough, Segmenter, dilate_tokens, see_codes
    from objects import read_frame, read_partial_movers, reader_context
    from view import make_view, token_pixels
    from skill import make_skill
    from world_model import Model

    torch.set_num_threads(1)
    dev = a.device
    M = Model(torch.load(a.model, map_location="cpu", weights_only=False), dev)
    M.feas_weight = a.feas_weight                                              # plan cost per -log event support
    if a.feas_weight > 0 and M.event_support is not None and a.feas_margin_q > 0:   # free feasibility cost: typical events (VAL)
        print({"feas_margin": np.round(M.set_feasibility_margin(a.events / "events_val.npz", a.feas_margin_q), 3).tolist()}, flush=True)
    if a.failure_relevance > 0:                                                # failures remembered against relevant entities
        rel_ = M.set_failure_relevance(a.events / "events_val.npz", a.failure_relevance)
        print({"failure_relevant": {int(e_): np.flatnonzero(rel_[e_]).tolist() for e_ in range(K)} if M.relevance is not None else "all"}, flush=True)
    if a.derived_r2 < 1:                                                       # appearances that follow from the other entities
        print({"app_derived": np.flatnonzero(M.set_derived_appearance(a.events, a.derived_r2)).tolist(), "derived_r2": M.derived_r2}, flush=True)
    if a.pred_tol_q > 0:                                                       # prediction-vs-reading appearance tolerance (VAL)
        print({"pred_tol": np.round(M.set_prediction_tolerance(a.events / "events_val.npz", a.pred_tol_q), 4).tolist(),
               "app_tol": np.round(M.app_tol, 4).tolist()}, flush=True)
    K, D = M.K, M.D
    fz = np.load(a.objects / "front.npz")
    P = {"L": int(fz["levels"]), "tkey": fz["tkey"], "trgb": fz["trgb"], "tau_c": float(fz["tau_c"]), "agent_bin": fz["agent_bin"]}
    place_px, w = fz["place_px"], float(fz["w"])
    vkeys, vpts = (fz["view_keys"], fz["view_patches"]) if "view_keys" in fz.files else (None, None)
    idents = pickle.loads((a.objects / "idents.pkl").read_bytes())["idents"]
    ctx = reader_context(idents, P)
    is_place = np.array([d["anchor"] == "location" for d in idents])
    tokens = np.load(a.tokens / "entities_val.npz")["tokens"]
    sk = torch.load(a.front / "seethru_learned.pt", map_location="cpu", weights_only=False)
    see = SeeThrough(sk["nd"], sk["nl"], sk["width"]).to(dev).eval(); see.load_state_dict(sk["model"])
    gk = torch.load(a.front / "segmenter.pt", map_location="cpu", weights_only=False)
    seg = Segmenter(gk["width"]).to(dev).eval(); seg.load_state_dict(gk["seg"])
    gc_actor, Zs, state_goal = None, None, False
    if a.low == "gcivl":                                                     # goal-conditioned executor (gcivl.py)
        from gcivl import goal_vector, make_nets
        gk_ = torch.load(a.gcivl, map_location="cpu", weights_only=False)
        state_goal = gk_.get("goal") == "state"                               # state goals: the WM's predicted next state
        _, gc_actor = make_nets(gk_["act_dim"], goal_dim=gk_.get("goal_dim")); gc_actor.load_state_dict(gk_["actor"]); gc_actor = gc_actor.to(dev).eval()
        if not state_goal:                                                    # image goals: subgoal images (subgoal.py)
            from subgoal import render_state
            Zs = {k_: v_ for k_, v_ in np.load(a.subgoals).items()}
    pk = torch.load(a.skill, map_location="cpu", weights_only=False) if a.skill is not None else {"kind": "none", "action_mean": 0.0, "action_std": 1.0, "hist_gap": 0}
    delta_kind = pk.get("kind") == "delta"                                    # executor on Delta maps (skill_delta.py)
    if delta_kind:
        from goal_maps import delta_mask, render
        from skill_delta import make_delta_skill
        pi = make_delta_skill(chunk=pk["chunk"], in_ch=7 if pk.get("press_point") else 6).to(dev).eval(); pi.load_state_dict(pk["skill"])
        amu, asd, gap = pk["action_mean"], pk["action_std"], 0
        d_tol, d_thr, d_unit = float(pk["tol_pos"]), np.asarray(pk["thr_app_id"]), np.asarray(pk["app_unit_id"])
    elif pk.get("kind") == "none":
        pi, amu, asd, gap = None, 0.0, 1.0, 0
    else:
        pi = make_skill(pk["K"], pk["A"], chunk=pk["chunk"], cond=pk.get("cond", "full"), arch=pk.get("arch", "cnn")).to(dev).eval(); pi.load_state_dict(pk["skill"])
        amu, asd, gap = pk["action_mean"], pk["action_std"], pk["hist_gap"]
    recover = np.zeros(5)
    if a.cache is not None:
        ev = np.load(a.events / "events_train.npz")
        acts = np.load(a.cache / "train_actions.npy", mmap_mode="r")
        idx = (ev["t"][:, None] + 1 + np.arange(5)[None]).ravel()
        recover = np.clip(np.asarray(acts[idx[idx < len(acts)]]).mean(0), -1, 1)
    tol_pos, thr_app = M.sc.tol_pos, M.app_tol                                # event thresholds (per entity for appearance)
    ptol = M.pred_tol                                                          # a WM prediction against a reading: the WM's precision
    timeout = a.timeout
    if timeout < 0:                                                           # data-derived: p99 of TRAIN event segment lengths
        ev_t = np.load(a.events / "events_train.npz")
        timeout = int(np.ceil(np.percentile(ev_t["t"] - ev_t["seg_start"], 99)))
        print({"timeout": timeout}, flush=True)
    grace = a.grace
    if grace < 0:                                                             # p90 of the arrival spread inside TRAIN events
        ev_ = np.load(a.events / "events_train.npz")
        grace = int(np.ceil(np.percentile(ev_["t"] - ev_["t_core"][:, 1], 90)))
    VV, UU = np.mgrid[0:64, 0:64]

    def read(frame, goal=False, static=False, partial=False):
        """-> reading (K, D), rest observation (K,) bool. goal: every place token counts as readable (see module doc).
        static (goal image, first frame): a mover seen is observed even near the agent (nothing is carried in a frame
        that no action has changed; the near-agent test made cube-triple goal cubes "hidden goals")."""
        x = torch.as_tensor(frame, device=dev).permute(2, 0, 1)[None].float().div_(255)
        with torch.no_grad():
            c, p = see_codes(see, x)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                ag = torch.sigmoid(seg(x).float()).cpu().numpy() > 0.5
        agt = dilate_tokens(ag.reshape(1, 256), 1).reshape(16, 16)
        agp = np.repeat(np.repeat(agt, 4, 0), 4, 1)
        xin, agin, contact = frame, agp, None
        if vkeys is not None:                                                 # the view of the offline tables (view.py)
            V, hid = make_view(frame[None], c, np.round(p * 255), agt.reshape(1, 256), vkeys, vpts, force=goal)
            xin, agin, contact = V[0], token_pixels(hid)[0], (frame, agp)
        c, p = c[0][tokens], p[0][tokens]
        rd = np.ones(len(tokens), np.uint8) if goal else (p > 0.5).astype(np.uint8); dg = ((c[:, None] // W5) % 5).astype(np.uint8)
        pos, app, area, ag_r = read_frame(xin, agin, P, idents, place_px, ctx, rd, dg, contact=contact)
        r = np.zeros((K, D), np.float32); r[:, :2] = pos; r[:, 2:5] = app
        rest = area >= 1
        if ag_r.any() and not static:
            d = np.hypot(UU[ag_r][:, None] - pos[:, 0][None], VV[ag_r][:, None] - pos[:, 1][None]).min(0)
            rest &= (d > w / 2) | is_place
        if partial:                                                           # goal image: partly visible movers (objects.py)
            return r, rest, ag_r, read_partial_movers(frame, agp, P, idents, place_px, ctx)
        return r, rest, ag_r

    def hidden(S, rest, ag_r):
        """movers not seen while no agent pixel is within half an object width of where they were last: covered (an object
        hides them; the covered bit of events_objects.py, never visible in a static moment)."""
        d = np.hypot(UU[ag_r][:, None] - S[:, 0][None], VV[ag_r][:, None] - S[:, 1][None]).min(0) if ag_r.any() else np.full(K, np.inf)
        return ~is_place & ~rest & (d > w / 2)

    env = gymnasium.make(a.env)
    u = env.unwrapped
    button_of, oracle = {}, None
    if a.low == "scripted":                                                   # PRIVILEGED: identity -> button (objects_report)
        rep_ = json.loads((a.objects / "objects_report.json").read_text())["privileged_diagnostic"]
        button_of = {r["identity"]: int(r["best_ref"][6:]) for r in rep_ if r.get("best_ref", "") and r["best_ref"].startswith("button")}
    if a.low == "oracle":                                                     # PRIVILEGED: OGBench plan oracles (oracle_low.py)
        from oracle_low import OracleLow
        rep_o = json.loads((a.objects / "objects_report.json").read_text())
        ev_o = np.load(a.events / "events_train.npz")
        slide_reads = {}                                                      # drawer / window: 2-means of their events' after positions
        for q_ in rep_o.get("privileged_diagnostic", []):
            if q_ and q_.get("best_ref") in ("drawer", "window"):
                k_ = int(q_["identity"]); X_ = ev_o["after"][ev_o["e"] == k_, k_, :2]
                if len(X_) >= 2:
                    c_ = X_[[int(np.argmin(X_[:, 1])), int(np.argmax(X_[:, 1]))]].astype(np.float64)
                    for _ in range(20):
                        lb_ = np.argmin(np.linalg.norm(X_[:, None] - c_[None], axis=-1), 1)
                        c_ = np.stack([X_[lb_ == j_].mean(0) if (lb_ == j_).any() else c_[j_] for j_ in range(2)])
                    slide_reads[k_] = c_
        oracle = OracleLow(env, rep_o, w, slide_reads)

    def sim_state(goal=False):
        try:
            s = {}
            if hasattr(u, "_num_cubes"):
                s["cubes"] = np.round(np.stack([u._data.mocap_pos[u._cube_target_mocap_ids[i]] if goal else u._data.joint(f"object_joint_{i}").qpos[:3]
                                                for i in range(u._num_cubes)]), 4).tolist()
            if hasattr(u, "_cur_button_states"):
                s["buttons"] = np.asarray(u._target_button_states if goal else u._cur_button_states).astype(int).tolist()
            if hasattr(u, "_drawer_site_id"):                                   # scene slides
                s["drawer"] = round(float(u._target_drawer_pos if goal else u._data.joint("drawer_slide").qpos[0]), 3)
                s["window"] = round(float(u._target_window_pos if goal else u._data.joint("window_slide").qpos[0]), 3)
            return s or None
        except Exception:
            return None

    def changed(s1, s0):
        return (np.linalg.norm(s1[:, :2] - s0[:, :2], axis=-1) > tol_pos) | (np.abs(s1[:, 2:-1] - s0[:, 2:-1]).max(-1) > thr_app)

    def differs(s_pred, s_obs):
        """a WM prediction against readings, per entity (appearance within the WM's side-effect precision, Model.pred_tol)"""
        return (np.linalg.norm(s_pred[:, :2] - s_obs[:, :2], axis=-1) > tol_pos) | ((np.abs(s_pred[:, 2:-1] - s_obs[:, 2:-1]).max(-1) > ptol) & ~M.app_derived)

    def arrived(s1, e, x):
        x = np.asarray(x)
        return np.hypot(*(s1[e, :2] - x[:2])) <= tol_pos and (M.app_derived[e] or np.abs(s1[e, 2:-1] - x[2:-1]).max() <= ptol[e])

    park = None
    if a.low in ("scripted", "oracle") and a.home_clear and getattr(u, "_arm_sampling_bounds", None) is not None:
        # PRIVILEGED arms: where to wait after an event. Five poses at the top of the arm sampling bounds (back centre, the
        # two sides, the back corners), each reached in a separate env and read by the method's own perception; the pose whose
        # agent leaves the most place pixels clear of it (by half an object width) wins. The episode start pose covered the
        # 3x3 top-middle and centre lights, high-back the 3x3 top row, the start pose the scene buttons.
        b_ = np.asarray(u._arm_sampling_bounds, np.float64); xm, ym, zt = 0.5 * (b_[0, 0] + b_[1, 0]), 0.5 * (b_[0, 1] + b_[1, 1]), b_[1, 2]
        cands = [[b_[0, 0], ym, zt], [xm, b_[1, 1], zt], [xm, b_[0, 1], zt], [b_[0, 0], b_[1, 1], zt], [b_[0, 0], b_[0, 1], zt]]
        e2 = gymnasium.make(a.env); u2 = e2.unwrapped; cover = []
        for tgt in cands:
            ob2, _ = e2.reset(seed=987654, options=dict(task_id=1, render_goal=False))
            for _ in range(80):
                d2 = np.zeros(5); d2[:3] = np.asarray(tgt) - u2.compute_ob_info()["proprio/effector_pos"]
                ob2, *_ = e2.step(np.clip(u2.normalize_action(d2), -1, 1))
            _, _, ag2 = read(ob2)
            near = ndimage.binary_dilation(ag2, iterations=max(1, int(round(w / 2)))) if ag2.any() else ag2
            cover.append(int((near & place_px).sum()))
        e2.close()
        park = np.asarray(cands[int(np.argmin(cover))], np.float64)
        print({"park_pose": np.round(park, 3).tolist(), "place_px_covered": cover}, flush=True)

    episodes = []
    for task, epi in jobs:
        seed = a.seed * 10000 + task * 100 + epi
        ob, info = env.reset(seed=seed, options=dict(task_id=task, render_goal=False))
        if ob.mean() < 20 or info["goal"].mean() < 20:
            raise RuntimeError("rendering looks broken")
        G, known, ag_g, part_g = read(info["goal"], goal=True, static=True, partial=True)
        S, rest, ag0 = read(ob, static=True)                                      # memory: last rest observation
        # a mover unseen in the goal is hidden there, unless its own colour still shows at its current place in the goal image
        # (the arm is transparent: an unmoved cube under the goal arm, scene task 1, colour in .80 of its footprint, read as
        # "hide the cube" before); then its goal is where it is. The agent covering the place is no evidence (scene tasks 4-5:
        # the arm's shadow covered the cube's old place, colour .00-.07, and the cube was in the closed drawer)
        unseen = ~is_place & ~known
        goal_occ, stays = {}, np.zeros(K, bool)
        if a.goal_occlusion:
            gx = info["goal"].astype(np.float32); gch = (gx / (gx.sum(-1, keepdims=True) + 1.0))[..., :2]
            for j_, m_ in enumerate(ctx["mov"]):
                if unseen[m_] and rest[m_]:
                    fp = np.hypot(UU - S[m_, 0], VV - S[m_, 1]) <= max(w / 2, 1.0)
                    col = np.linalg.norm(gch[fp] - ctx["protos"][j_], axis=-1) < np.linalg.norm(gch[fp] - ctx["tch"][fp], axis=-1)
                    stays[m_] = col.mean() >= 0.5
                    goal_occ[int(m_)] = {"agent_frac": round(float(ag_g[fp].mean()), 2), "colour_frac": round(float(col.mean()), 2)}
        G[stays] = S[stays]                                                       # still there: its goal is its current state
        G[unseen & ~stays, -1] = 1.0
        # PARTIAL GOAL HINTS: hidden goal movers whose side face shows in the goal image (less than half visible), ranked along
        # the learned cover direction (farther from the covering side = deeper): their hidden targets keep only chains of that depth
        hint = {int(k_): part_g[k_] for k_ in np.flatnonzero(unseen & ~stays) if a.goal_hints and int(k_) in part_g}
        # the stacking direction is LEARNED: the mean offset of the cover relations (covering minus covered entity), not "up"
        u_cov = np.mean([o_ for _, _, o_ in M.rel_cover], 0) if M.rel_cover else np.zeros(2)
        hidden_depth = None
        if len(hint) and np.linalg.norm(u_cov) > 0:
            hidden_depth = {k_: i_ + 1 for i_, k_ in enumerate(sorted(hint, key=lambda k_: -float(np.dot(hint[k_], u_cov))))}
        known = known | ~is_place
        under_arm = stays
        obs_ev = rest.copy()                                                      # entities observed since the event started
        S = np.where(rest[:, None], S, np.where(known[:, None], G, S))            # unseen at the start: assumed at its goal until seen
        t_plan = time.time()
        plan, pinfo = M.plan(S, G, known=known, max_expansions=a.max_expansions, hidden_depth=hidden_depth)
        rec = {"task": task, "episode": epi, "first_plan": None if plan is None else len(plan), "plan_info": {k: v for k, v in pinfo.items() if k != "best_plan"},
               "goal_hints": {str(k_): [round(v_, 1) for v_ in hint[k_]] for k_ in hint}, "hidden_depth": hidden_depth,
               "plan_sec": round(time.time() - t_plan, 2), "events": [], "replans": 0, "timeouts": 0, "success": False, "steps": 0,
               "goal_known": float(known.mean()), "start_seen": float(rest.mean()), "sim_start": sim_state(), "sim_goal": sim_state(goal=True), "timeout_log": [],
               # the planner's view at the start: belief (unseen entities at their goal), goal reading, seen flags, first plan
               "start_belief": np.round(S, 3).tolist(), "goal_read": np.round(G, 3).tolist(), "start_rest": rest.astype(int).tolist(),
               "goal_occlusion": goal_occ, "goal_stays": np.flatnonzero(under_arm).tolist(),
               "first_plan_events": [[int(e_), np.round(np.asarray(x_), 3).tolist()] for e_, x_ in (plan or pinfo.get("best_plan") or [])]}
        if plan is None:
            plan = pinfo.get("best_plan") or []
        frames, queue = [ob] * (gap + 1), []
        ctl = None
        home = u.compute_ob_info()["proprio/effector_pos"].copy() if a.low in ("scripted", "oracle") else None
        if park is not None:                                                   # PRIVILEGED arms wait at the chosen park pose
            home = park.copy()
        if oracle is not None:
            oracle.set_home(home, park=park)
        start_state, rest_count, k_steps = S.copy(), np.zeros(K, int), 0
        belief0, surprise = S.copy(), np.zeros(K, bool)                           # the belief the plan's next event assumed; first
        #                                                                           in-event observations that contradict it
        t_clear = np.where(rest, 0, -1)                                           # step of each entity's last agent-clear observation
        predicted_next = None
        recent = []                                                               # states left by the last events (plan avoids them)
        failed, oracle_idle = [], 0                                               # (state, e) of events that did not happen
        t_obs = np.where(rest, 0, -1)                                             # step of each entity's last rest observation
        t_ref = None                                                              # step of the first observed change in this event
        t_trig, pred_trig = None, None                                            # event-end condition first met (grace period)
        done = False
        while not done:
            if plan and a.low == "scripted":
                e, x = plan[0]
                if ctl is None and e in button_of:
                    ctl = PressController(env, button_of[e], home)
                action = ctl.act() if ctl is not None else None
                action = np.zeros(5) if action is None else action
                if k_steps == 0:
                    predicted_next = M.step(start_state, [(e, x)])[0]
            elif plan and a.low == "oracle":
                e, x = plan[0]
                if k_steps == 0:                                                   # a new event: a new oracle plan
                    oracle.start(e, x, S)
                    predicted_next = M.step(start_state, [(e, x)])[0]
                action = oracle.act()
                action = np.zeros(5) if action is None else action
            elif plan and a.low == "gcivl":
                e, x = plan[0]
                if k_steps == 0:                                                   # SUBGOAL: event-start frame, places in their next state
                    predicted_next = M.step(start_state, [(e, x)])[0]
                    if not state_goal:
                        # only the entities the event is predicted to change are drawn; the rest of the subgoal is the current frame
                        # (drawing every place from the belief state made the subgoal differ from the frame wherever a reading was
                        # stale: 4x4 loop, planned light pressed .36, vs .88-.90 in the single-event harness)
                        chg_pred = changed(predicted_next, start_state) | ((predicted_next[:, -1] > 0.5) != (start_state[:, -1] > 0.5))
                        # every kind (subgoal.py): discrete places pasted, movers erased and drawn at their next position, continuous
                        # places copied from the nearest bank frame (G4 2026-10-10: cube-triple, scene)
                        subgoal_img = render_state(frames[-1], start_state, predicted_next, chg_pred, Zs)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    if state_goal:                                                 # STATE GOAL: the belief now and the predicted next state
                        xs = torch.as_tensor(frames[-1][None], device=dev).permute(0, 3, 1, 2).float() / 255.0
                        gv_ = torch.as_tensor(goal_vector(S, predicted_next)[None], device=dev)
                        action = np.clip(gc_actor(xs, gv_).float().cpu().numpy()[0], -1, 1)
                    else:
                        xs = torch.as_tensor(np.stack([frames[-1], subgoal_img]), device=dev).permute(0, 3, 1, 2).float() / 255.0
                        action = np.clip(gc_actor(xs[:1], xs[1:]).float().cpu().numpy()[0], -1, 1)
            elif plan:
                e, x = plan[0]
                if not queue:
                    if k_steps == 0:
                        predicted_next = M.step(start_state, [(e, x)])[0]
                    if delta_kind:                                             # what must change: event start -> predicted state
                        ones = np.ones(K, bool)
                        mp = render(start_state, predicted_next, delta_mask(start_state, predicted_next, ones, ones, d_tol, d_thr), d_unit)
                        if pk.get("press_point"):                              # the planned entity's position (press / grasp point)
                            gg = np.arange(64, dtype=np.float32)
                            u0, v0 = float(start_state[e, 0]), float(start_state[e, 1])
                            mp = np.concatenate([mp, np.exp(-((gg[None, :] - u0) ** 2 + (gg[:, None] - v0) ** 2) / 8.0)[None].astype(np.float32)], 0)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                            px = torch.as_tensor(frames[-1], device=dev).permute(2, 0, 1)[None].float() / 255.0
                            ch = pi(px, torch.as_tensor(mp, device=dev)[None]).float().cpu().numpy()[0]
                    else:
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                            # an event's first decision sees the current frame twice: the motion left over from the previous
                            # event says nothing about this event's target (copied, it sent the arm on to wrong buttons)
                            f0 = frames[-1] if (k_steps == 0 and a.fresh_start) else frames[0]
                            px = torch.as_tensor(np.concatenate([f0, frames[-1]], -1), device=dev).permute(2, 0, 1)[None].float() / 255.0
                            ch = pi(px, torch.tensor([e], device=dev), torch.as_tensor(start_state[e][None], device=dev).float(),
                                    torch.as_tensor(np.asarray(x)[None], device=dev).float()).float().cpu().numpy()[0]
                    queue = list(np.clip(ch[: a.exec_steps] * asd + amu, -1, 1))
                action = queue.pop(0)
            else:
                action = np.zeros(5)
            ob, _, term, trunc, info = env.step(action)
            rec["steps"] += 1; k_steps += 1
            frames = frames[1:] + [ob]
            if info["success"]:
                rec["success"] = True
            done = term or trunc or rec["success"]
            r, rest, ag_r = read(ob)
            if a.trace_steps and rec["steps"] <= a.trace_steps and plan:            # DEBUG trace of the planned entity's reading
                e_t = int(plan[0][0])
                d_t = float(np.hypot(UU[ag_r] - r[e_t, 0], VV[ag_r] - r[e_t, 1]).min()) if ag_r.any() else -1.0
                rec.setdefault("trace", []).append([rec["steps"], e_t, int(rest[e_t]), np.round(r[e_t, :5], 3).tolist(), round(d_t, 1), sim_state()])
            S = np.where(rest[:, None], r, S)
            newly = rest & ~obs_ev                                                # first observation within this event: its baseline
            surprise |= newly & changed(S, belief0)
            start_state[newly] = S[newly]; obs_ev |= rest
            S[hidden(S, rest, ag_r) & (S[:, :2].sum(-1) > 0), -1] = 1.0
            if a.confirm == "clear":                                              # a reading confirms only with the agent clear of it
                d_ag = np.hypot(UU[ag_r][:, None] - r[:, 0][None], VV[ag_r][:, None] - r[:, 1][None]).min(0) if ag_r.any() else np.full(K, np.inf)
                conf_ = rest & (d_ag > w / 2)
            else:
                conf_ = rest
            rest_count = np.where(conf_, rest_count + 1, 0)
            t_obs[rest] = rec["steps"]; t_clear[conf_] = rec["steps"]
            end = False
            if plan:
                e, x = plan[0]
                ch_ = changed(S, start_state)
                if t_ref is None and ch_.any():
                    t_ref = rec["steps"]
                settled = rest_count >= a.m
                others = any(ch_[j] and settled[j] for j in range(K) if j != e)
                # other entities changed and settled while the acted one is hidden (a pressed light under the arm): the
                # event is over; the belief below gives the hidden one its predicted state if the seen ones agree (waiting
                # for it to be seen timed out whenever the arm rested over it: dev scripted arm, 2-3 timeouts of 250 steps
                # per failed episode)
                # (with --confirm clear, "hidden" = not confirmable: a light read through the arm counts as rest but can never
                # settle, so the event never ended and timed out: 3x3, the arm cannot wait clear of the board, z <= .25)
                cond = (settled[e] and (ch_[e] or arrived(S, e, x) or others)) or (others and not conf_[e])
                if cond and grace > 0 and t_trig is None:
                    t_trig, pred_trig = rec["steps"], M.step(start_state, [(e, x)])[0]
                if t_trig is not None:                                        # GRACE: judge once every predicted change is seen done
                    seen_ = t_obs >= (t_ref if t_ref is not None else t_trig)
                    done_ = ~changed(pred_trig, start_state) | (seen_ & ~differs(pred_trig, S))   # predicted change seen as predicted
                    cond = rec["steps"] - t_trig >= grace or bool(done_.all())
                if cond:
                    end = True
                    match = float((~differs(predicted_next, S)).mean()) if predicted_next is not None else None
                    # BELIEF: an entity the world model predicts to change but not observed at rest since the first observed
                    # change (still under the agent, e.g. a light next to the pressed one) takes the predicted state, when
                    # everything observed since agrees with the prediction; later observations overwrite it (memory S).
                    # (predicted again from the event's baselines: entities first seen during the event replaced their
                    # memory values after the first prediction)
                    filled, agree, dis = 0, False, []
                    if t_ref is not None:
                        pred_end = M.step(start_state, [(e, x)])[0]
                        seen = (t_clear if a.confirm == "clear" else t_obs) >= t_ref
                        dis = np.flatnonzero(differs(pred_end, S) & seen).tolist()
                        agree = not dis
                        if agree:
                            fill = ~seen & changed(pred_end, start_state)
                            S[fill] = pred_end[fill]; filled = int(fill.sum())
                    rec["events"].append({"e": int(e), "steps": k_steps, "t": rec["steps"], "acted_changed": bool(ch_[e]),
                                          "changed_ids": np.flatnonzero(ch_).tolist(),
                                          "others_changed": int(sum(ch_[j] for j in range(K) if j != e)),
                                          "wm_predicted_changed": int(changed(predicted_next, start_state).sum()) if predicted_next is not None else None,
                                          "wm_prediction_match": match, "belief_filled": filled, "as_predicted": bool(agree),
                                          "predicted_changed_ids": np.flatnonzero(changed(predicted_next, start_state)).tolist() if predicted_next is not None else None,
                                          "disagree_ids": dis, "start_app": np.round(start_state[:, 2], 3).tolist(), "end_app": np.round(S[:, 2], 3).tolist(),
                                          "sim_end": sim_state()})
            # PRIVILEGED oracle only: done and back home with nothing changed for a grace period = the event failed (the
            # oracle cannot do what the world model allowed, e.g. a locked drawer); not waiting for the timeout
            oracle_idle = oracle_idle + 1 if (oracle is not None and plan and action is not None and not np.any(action)) else 0
            fail_fast = oracle is not None and oracle_idle > a.m + max(grace, 0)
            if end or k_steps > timeout or fail_fast:
                if not end:
                    queue = [recover.copy() for _ in range(a.recover_steps)]
                    rec["timeouts"] += 1
                    if plan:
                        rec["timeout_log"].append({"e": int(plan[0][0]), "t": rec["steps"], "sim_end": sim_state(), "fast": bool(fail_fast)})
                        # FAILURE MEMORY: an event on entity e did not happen from this belief, whatever the world model says.
                        # No plan may use an event on e from a state with the same key, anywhere in the search: forbidding only
                        # the first event let plans insert a no-op pair (press a button twice) before the locked drawer
                        # (scene task 2). A different state (one press: unlocked) allows it again. Kept for the episode.
                        if a.failure_memory:
                            for st_ in (start_state, S):
                                kb_ = np.array(st_, np.float32); kb_[:, -1] = kb_[:, -1] > 0.5
                                failed.append((M.canonical(kb_), int(plan[0][0])))
                            recent.clear()                                      # see below
                else:
                    queue = []
                    # an event that ended with NOTHING changed did not happen either (a target differing only in what the
                    # loop ignores was "arrived" at once and replanned every 5 steps: scene c8): failure memory as above
                    if a.failure_memory and plan and not ch_.any():
                        for st_ in (start_state, S):
                            kb_ = np.array(st_, np.float32); kb_[:, -1] = kb_[:, -1] > 0.5
                            failed.append((M.canonical(kb_), int(plan[0][0])))
                        # after a failure the recent states are no longer to avoid: the failure memory stops the repeat, and the
                        # way on may pass a recent state (scene task 2: relocked by a no-op press pair, the drawer failed; the
                        # plan unlock / close / lock went through the avoided unlocked state and no plan was found)
                        recent.clear()
                oracle_idle = 0
                ctl = None
                if end:
                    recent.append(start_state.copy())
                # SURPRISE: an entity first seen during the event (under the agent when the plan was made, e.g. assumed at its
                # goal at the episode start) whose reading is not what the world model predicts from the belief the plan
                # assumed. The baseline update above hides it from the as-predicted test, so the plan ran on to its end with
                # that entity wrong (4x5 pairg: 19 events per solved episode instead of ~15; scripted episodes ended plans
                # one light off, ~10 more presses to fix)
                unexpected = []
                if a.replan_on_surprise and plan and surprise.any():
                    unexpected = np.flatnonzero(surprise & differs(M.step(belief0, [plan[0]])[0], S)).tolist()
                    if end and rec["events"]:
                        rec["events"][-1]["unexpected_ids"] = unexpected
                start_state, k_steps, rest_count = S.copy(), 0, np.zeros(K, int)
                belief0, surprise = S.copy(), np.zeros(K, bool)
                obs_ev = rest.copy(); t_ref = None; t_trig, pred_trig = None, None
                if not done:
                    if end and agree and len(plan) > 1 and not unexpected:      # as predicted: continue the plan
                        plan = plan[1:]
                        rec["events"][-1]["replan_found"] = None
                    else:
                        plan, pinfo = M.plan(S, G, known=known, max_expansions=a.max_expansions, avoid=recent[-3:], forbid=failed, hidden_depth=hidden_depth)
                        if a.log_plans:                                         # DEBUG: every replan's events and the belief
                            rec.setdefault("plans", []).append({"t": rec["steps"], "found": plan is not None, "forbid": sorted({int(fe) for _, fe in failed}), "forbid_states": [[int(fe), np.round(fs, 3).tolist()] for fs, fe in failed[-4:]],
                                "plan": [[int(e_), np.round(np.asarray(x_), 3).tolist()] for e_, x_ in (plan or pinfo.get("best_plan") or [])],
                                "belief": np.round(S, 3).tolist()})
                        if end:
                            rec["events"][-1]["replan_found"] = plan is not None
                        if plan is None:
                            plan = pinfo.get("best_plan") or []
                        rec["replans"] += 1
        if oracle is not None:                                                  # events the oracle could / could not map
            rec["oracle_events"] = {"mapped": oracle.mapped, "unmapped": oracle.unmapped, "slides": oracle.log}
            oracle.mapped = oracle.unmapped = 0; oracle.log = []
        episodes.append(rec)
        save_json(a.out / "episodes" / f"task{task}_episode{epi}.json", rec)
        print(json.dumps({k: rec[k] for k in ("task", "episode", "first_plan", "success", "steps", "replans", "timeouts", "goal_known")}), flush=True)
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="visual-cube-triple-v0, visual-puzzle-4x5-v0, visual-scene-v0")
    ap.add_argument("--model", type=Path, required=True, help="world_model.py --stage h output (model.pt)")
    ap.add_argument("--skill", type=Path, default=None, help="skill.py / skill_delta.py checkpoint (--low skill)")
    ap.add_argument("--front", type=Path, required=True, help="front-end dir (seethru_learned.pt, segmenter.pt)")
    ap.add_argument("--objects", type=Path, required=True, help="objects.py output (front.npz, idents.pkl)")
    ap.add_argument("--tokens", type=Path, required=True, help="memory_entities.py output (token selection for SeeThrough codes)")
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=250,
                    help="steps before an event is abandoned; -1 = the 99th percentile of the TRAIN event segments (end of the previous "
                         "event to the end of this one; puzzle-4x5: 120) - a fixed 250 spent 45%% of the budget of failed learned episodes")
    ap.add_argument("--exec-steps", type=int, default=4)
    ap.add_argument("--no-fresh-start", dest="fresh_start", action="store_false",
                    help="feed the frame history also at an event's first skill query")
    ap.add_argument("--recover-steps", type=int, default=8)
    ap.add_argument("--trace-steps", type=int, default=0, help="debug: record the planned entity's reading for the first N steps")
    ap.add_argument("--log-plans", action="store_true", help="debug: record every replan (events, forbidden entities, belief)")
    ap.add_argument("--no-goal-hints", dest="goal_hints", action="store_false",
                    help="ablation: no partial goal readings ranking hidden goal movers in a stack (before 2026-10-11)")
    ap.add_argument("--failure-relevance", type=float, default=0.3,
                    help="a failed event stays forbidden until an entity RELEVANT to it changes: mean added pairwise-support cost "
                         "(nats) of randomizing that entity in genuine VAL events >= this (Model.set_failure_relevance); 0 = all entities")
    ap.add_argument("--derived-r2", type=float, default=0.5,
                    help="an entity appearance predictable from the other entities (kNN R^2 on VAL >= this) is derived: goal and "
                         "prediction tests ignore it (world_model.Model.set_derived_appearance); 1 = off (before 2026-10-11)")
    ap.add_argument("--pred-tol-q", type=float, default=0.75,
                    help="appearance tolerance of WM predictions vs readings: this quantile of the WM's VAL side-effect error per entity "
                         "(world_model.Model.set_prediction_tolerance); 0 = the event thresholds (before 2026-10-11)")
    ap.add_argument("--feas-margin-q", type=float, default=0.9,
                    help="only the event cost above this quantile of genuine VAL event costs (per acted entity) enters plans; 0 = all of it")
    ap.add_argument("--feas-weight", type=float, default=1.0,
                    help="A* cost per -log(event support) of each planned event (0: support only abstains below its threshold)")
    ap.add_argument("--home-start-pose", dest="home_clear", action="store_false",
                    help="PRIVILEGED scripted arm: return to the episode's start pose (before 2026-10-11) instead of high and back")
    ap.add_argument("--no-goal-occlusion", dest="goal_occlusion", action="store_false",
                    help="ablation: every mover unseen in the goal image is hidden there, even when its colour shows at its place (before 2026-10-11)")
    ap.add_argument("--no-failure-memory", dest="failure_memory", action="store_false",
                    help="ablation: plans may start again with an event that changed nothing from the same belief (before 2026-10-11)")
    ap.add_argument("--no-replan-on-surprise", dest="replan_on_surprise", action="store_false",
                    help="ablation: keep following the plan when an entity first seen during an event contradicts the belief the "
                         "plan assumed (the old behaviour)")
    ap.add_argument("--max-expansions", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grace", type=int, default=0,
                    help="frames to keep reading after the event-end condition before judging the outcome (stops early once every "
                         "entity predicted to change has been observed); -1 = the 90th percentile of the arrival spread inside TRAIN "
                         "events (lagged lights: 19-23 frames on puzzle). The oracle 4x4 loop judged events while lights were still "
                         "unread and replanned (2-3x the planned events)")
    ap.add_argument("--confirm", choices=("rest", "clear"), default="rest",
                    help="clear: a reading settles an event or contradicts the world model only while no agent pixel is within half "
                         "an object width of the entity (a light under the gripper reads its old state for 7-11 frames; the oracle "
                         "4x4 loop timed out in all 12 failures with 2-3x the planned events, outcomes as predicted in 34%%)")
    ap.add_argument("--gcivl", type=Path, default=None, help="gcivl.py checkpoint (--low gcivl)")
    ap.add_argument("--subgoals", type=Path, default=None, help="subgoal.py output subgoals.npz (--low gcivl with an image-goal GCIVL; not used with state goals)")
    ap.add_argument("--low", choices=("skill", "scripted", "gcivl", "oracle"), default="skill",
                    help="scripted: PRIVILEGED diagnostic arm (puzzle), a scripted press of the planned light's button; oracle: PRIVILEGED diagnostic arm, OGBench plan oracles for every event kind (oracle_low.py)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    jobs = [(t, e) for t in a.tasks for e in range(a.episode_start, a.episode_start + a.episodes)]
    t0 = time.time()
    episodes = run(a, jobs)
    state_goal = False
    if a.low == "gcivl":
        import torch
        state_goal = torch.load(a.gcivl, map_location="cpu", weights_only=False).get("goal") == "state"
    summary = {"arm": ("PRIVILEGED DIAGNOSTIC: scripted low level" if a.low == "scripted" else "PRIVILEGED DIAGNOSTIC: OGBench plan-oracle low level" if a.low == "oracle" else
                       ("METHOD: object entities -> event WM + A* + state-goal GCIVL on the predicted next state (no privileged input)" if state_goal else "METHOD: object entities -> event WM + A* + image-goal GCIVL on subgoal images (no privileged input)") if a.low == "gcivl" else
                       "METHOD: object entities -> event WM + A* + event skill (no privileged input)"), "env": a.env,
               "success": float(np.mean([e["success"] for e in episodes])),
               "by_task": {t: float(np.mean([e["success"] for e in episodes if e["task"] == t])) for t in a.tasks},
               "first_plan_found": float(np.mean([e["first_plan"] is not None for e in episodes])),
               "minutes": round((time.time() - t0) / 60, 1), "args": {k: str(v) for k, v in vars(a).items()}}
    save_json(a.out / "closed_loop.json", {"summary": summary, "episodes": episodes})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
