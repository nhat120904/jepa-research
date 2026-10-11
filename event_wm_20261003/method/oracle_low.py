"""PRIVILEGED DIAGNOSTIC low level (closed_loop_objects.py --low oracle): OGBench's plan oracles (ogbench.manipspace.oracles
.plan, the controllers that generated the play data) execute one planned event (entity e, target state x), so the closed
loop measures the HIGH LEVEL (perception, events, world model, planner, belief) on cube and scene, as the scripted press does
on puzzle. Not part of the method: it reads simulator state and camera geometry.

Entity -> simulator object: objects_report.json privileged_diagnostic best_ref when it names one (scene: cube_y, window,
drawer, button0/1; puzzle: buttonN), else (cube-triple movers) the block whose projection into the 64x64 'front_pixels' frame
is nearest the entity's believed position. Target: a block goes to the table point (z = 0.02) under the target pixel, or
on top of another block when the target pixel is within half an object width of that block's projection; a drawer / window
goes to the slide end (closed or open, the two values of the env's tasks) whose handle displacement in pixels is nearest the
displacement the event asks for; a button is pressed. Oracle noise 0. Events it cannot map are reported (unmapped)."""

from __future__ import annotations

import mujoco
import numpy as np

SLIDES = {"drawer": ("drawer_slide", (0.0, -0.16), "_drawer_site_id"), "window": ("window_slide", (0.0, 0.2), "_window_site_id")}


class OracleLow:
    def __init__(self, env, report, w, slide_reads=None):
        self.env, self.u, self.w = env, env.unwrapped, float(w)
        self.slide_reads = slide_reads or {}                                    # entity -> (2, 2) read clusters of a slide
        pd = [q for q in report.get("privileged_diagnostic", []) if q]
        self.ref = {int(q["identity"]): str(q.get("best_ref")) for q in pd if q.get("best_ref")}
        self.block_of = {int(q["identity"]): int(q["cube"]) for q in pd if q.get("cube") is not None}   # cube-triple: colour -> block
        self.cid = mujoco.mj_name2id(self.u._model, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
        self.ctl, self.unmapped, self.mapped, self.home, self.log, self.at_home = None, 0, 0, None, [], False

    def set_home(self, home, park=None):
        """where the arm waits after each event: high and back (the arm sampling bounds' lowest x, centre y, highest z), out of
        the camera's view of the objects (the oracle's own last pose is random and an arm left over the moved object kept it
        unread: cube-triple, events not ending in 564 steps; the episode's start pose covered the scene buttons, 0.4 px).
        `home` (the start pose) is kept only when the env has no sampling bounds."""
        if park is not None:                                                  # the loop's chosen park pose (closed_loop_objects.py)
            self.home = np.asarray(park, np.float64); return
        b = getattr(self.u, "_arm_sampling_bounds", None)
        self.home = np.array([b[0][0], 0.5 * (b[0][1] + b[1][1]), b[1][2]], np.float64) if b is not None else np.asarray(home, np.float64)

    # camera geometry of the observation frame (64 x 64): pixel (u, v) = column, row, pixel centres at integers
    def project(self, p):
        pos, R = self.u._data.cam_xpos[self.cid], self.u._data.cam_xmat[self.cid].reshape(3, 3)
        f = 32.0 / np.tan(np.deg2rad(float(self.u._model.cam_fovy[self.cid])) / 2)
        pc = R.T @ (np.asarray(p, np.float64) - pos)
        return np.array([32.0 + f * pc[0] / -pc[2] - 0.5, 32.0 - f * pc[1] / -pc[2] - 0.5])

    def unproject(self, uv, z):
        pos, R = self.u._data.cam_xpos[self.cid], self.u._data.cam_xmat[self.cid].reshape(3, 3)
        f = 32.0 / np.tan(np.deg2rad(float(self.u._model.cam_fovy[self.cid])) / 2)
        d = R @ np.array([(uv[0] + 0.5 - 32.0) / f, -(uv[1] + 0.5 - 32.0) / f, -1.0])
        t = (z - pos[2]) / d[2]
        return pos + t * d

    def blocks(self, info):
        n = 0
        while f"privileged/block_{n}_pos" in info:
            n += 1
        return n

    def start(self, e, x, S):
        """build the oracle for event (e, x) from belief S; False if the event cannot be mapped."""
        from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
        from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
        from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
        from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle
        u = self.u
        self.ctl, self.at_home = None, False
        info = dict(u.compute_ob_info())
        ref, x = self.ref.get(int(e), ""), np.asarray(x, np.float64)
        ctl = None
        if ref.startswith("button") and hasattr(u, "_button_site_ids"):
            info["privileged/target_button_top_pos"] = u._data.site_xpos[u._button_site_ids[int(ref[6:])]].copy()
            ctl = ButtonPlanOracle(env=self.env, noise=0.0)
        elif ref in SLIDES:
            jn, ends, site = SLIDES[ref]
            jid = mujoco.mj_name2id(u._model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            q = float(u._data.joint(jn).qpos[0]); axis = u._data.xaxis[jid].copy()
            h = u._data.site_xpos[getattr(u, site)].copy()
            want = x[:2] - np.asarray(S[e, :2], np.float64)                     # pixel displacement the event asks for
            disp = [self.project(h + (qe - q) * axis) - self.project(h) for qe in ends]
            dirn = disp[1] - disp[0]                                            # closed -> open, in pixels (handle)
            if int(e) in self.slide_reads:
                # the two read clusters of the place (its events' after positions): the one further along closed -> open is
                # "open"; the end = the cluster nearest the asked target (a partly opened window read past its open end,
                # so the asked displacement pointed the wrong way: scene task 1, window toggled open / closed 12 times)
                c = self.slide_reads[int(e)]
                c_open, c_closed = (c[1], c[0]) if float((c[1] - c[0]) @ dirn) > 0 else (c[0], c[1])
                qe = ends[1] if np.linalg.norm(x[:2] - c_open) < np.linalg.norm(x[:2] - c_closed) else ends[0]
            else:
                # the end the displacement POINTS to: the read place moves less than the handle (window: 5.7 px read vs 15 px)
                qe = ends[1] if float(want @ dirn) > 0 else ends[0]
            self.log.append({"e": int(e), "ref": ref, "q": round(q, 3), "target_q": qe, "want_px": np.round(want, 1).tolist(),
                             "end_px": [np.round(d_, 1).tolist() for d_ in disp]})
            info[f"privileged/target_{ref}_handle_pos"] = h + (qe - q) * axis
            ctl = (DrawerPlanOracle if ref == "drawer" else WindowPlanOracle)(env=self.env, noise=0.0)
        elif self.blocks(info):
            nb = self.blocks(info)
            P = np.stack([info[f"privileged/block_{i}_pos"] for i in range(nb)])
            if ref.startswith("cube") and nb == 1:
                b = 0
            elif int(e) in self.block_of:                                      # the identity's colour names the block (a stack
                b = self.block_of[int(e)]                                       # projects its blocks onto nearly one pixel)
            else:
                b = int(np.argmin([np.linalg.norm(self.project(p) - S[e, :2]) for p in P]))
            if x[-1] > 0.5:                                                    # the event hides the block: not mapped
                self.unmapped += 1
                return False
            tgt = self.unproject(x[:2], 0.02)
            for i in range(nb):                                                 # onto another block's top
                if i != b and np.linalg.norm(self.project(P[i] + [0, 0, 0.02]) - x[:2]) <= self.w / 2:
                    tgt = P[i] + np.array([0.0, 0.0, 0.04])
            info["privileged/target_block"] = b
            info["privileged/target_block_pos"] = tgt
            info["privileged/target_block_yaw"] = info[f"privileged/block_{b}_yaw"]
            ctl = CubePlanOracle(env=self.env, noise=0.0)
        if ctl is None:
            self.unmapped += 1
            return False
        ctl.reset(None, info)
        self.ctl = ctl; self.mapped += 1
        return True

    def act(self):
        """normalised action of the running oracle; once it is done, a move back to the home pose; None when home (or no
        oracle runs)."""
        info = self.u.compute_ob_info()
        if self.ctl is not None and not self.ctl.done:
            return np.clip(self.ctl.select_action(None, info), -1, 1)
        if self.home is None:
            return None
        d = self.home - info["proprio/effector_pos"]
        if self.at_home or np.linalg.norm(d) < 0.02:                            # home: idle until the next event (the arm then
            self.at_home = True                                                 # drifts around the 2 cm mark under zero actions)
            return None
        a = np.zeros(5); a[:3] = d; a[4] = -float(info["proprio/gripper_opening"][0])   # open the gripper (it may hold a locked handle)
        return np.clip(self.u.normalize_action(a), -1, 1)
