#!/usr/bin/env python3
"""Cube closed loop on the official visual-cube-triple tasks.

--perception privileged (direction D): current and goal cube positions come from the simulator.
--perception reader (direction A, label-free): the CNN object reader (train_cube_reader.py) reads
    every object's pixel position from the current frame and from the goal image; the planner and
    skill are the pixel-state versions trained on cube_events_px.py events.
--perception oracle_px (PRIVILEGED attribution): the label-free pixel planner and skill, but the
    planning state and goal are the simulator positions projected to pixels (cube_projection.py) with
    the true coverage bit, and move ends use the privileged rule -- i.e. perfect perception.
Privileged planner + pixel skill (PRIVILEGED attribution, needs --project): the metre plan's predicted
    landing position of the moved cube is projected to pixels and given to the label-free skill.
Loop: plan moves (cube_planner.plan_moves) -> execute the first move (k, q) with the pixel skill ->
detect the end of the move -> replan. Timeout per move: replay the mean post-move action, then replan.
  privileged: a move ends when cube k has moved > 2 cm (or another cube has: knock) and every cube
              then stays still (speed < 2 mm/step) for 5 steps (as in job 57118).
  reader, --end-rule rest (default): the play-data event detector run online -- object k (or another
              object: knock) is seen by the colour track at rest (m = 5 visible frames within r = 1 px)
              more than the object-width threshold away from where it started. A carried cube is
              hidden by the gripper, so a pause in mid-air does not end the move.
  reader, --end-rule still (job 57124): reader displacement > threshold, then every object still
              (per-step reader displacement below the 99th pct at rest on val) for 5 steps.
--belief (reader mode with the coverage bit): the state used for planning is a belief updated at
every move end, as the puzzle's WM event-consistency filter: the moved object takes the reader's
position; coverage is predicted by the WM from the previous belief and the executed move, and is
overridden to 'uncovered' when the colour track saw the object's top face in the last 5 frames
(direct evidence); a covered object keeps its believed position (it cannot move while covered; the
reader localises hidden objects poorly, job 57136). After a timeout the WM prediction is not used.
Planner constraints in reader mode (label-free, from play data): no target closer to another object
than the 1st percentile of that distance over play moves (occupied place); with the coverage bit, a
covered object is never moved (play data never moves one); without it, affordance >= 1/(2K).
Logged for analysis only, never used for decisions in reader mode: simulator cube positions at every
move end and at the end of the episode (placement error in metres).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import save_json
from cube_planner import buffer_positions, load_cube_planner, plan_moves
from train_cube_skill import make_cube_skill


def cube_xyz(u, K):
    return np.stack([u._data.joint(f"object_joint_{i}").qpos[:3].copy() for i in range(K)]).astype(np.float32)


def run(a, jobs):
    os.environ.setdefault("LP_NUM_THREADS", "1")
    import gymnasium
    import ogbench  # noqa: F401
    import torch

    torch.set_num_threads(1)
    dev = "cuda"
    wm, aff, pk = load_cube_planner(a.planner, dev)
    K = pk["K"]
    sk = torch.load(a.skill, map_location="cpu", weights_only=False)
    use_sup = bool(sk.get("support", False))
    pi = make_cube_skill(K, chunk=sk["chunk"], lo=sk.get("lo", np.array([0.2, -0.4], np.float32)),
                         hi=sk.get("hi", np.array([0.9, 0.4], np.float32)), support=use_sup).to(dev).eval()
    pi.load_state_dict(sk["skill"])
    amu, asd, gap = sk["action_mean"], sk["action_std"], sk["hist_gap"]
    skill_px = float(np.max(np.asarray(sk.get("hi", [0.9])))) > 2                  # skill targets in pixels
    px_planner = "hi" in pk and float(np.max(np.asarray(pk["hi"]))) > 2            # planner state in pixels
    if a.project is not None:
        from cube_projection import feats

        zp = np.load(a.project)
        W, o2c = zp["W"], zp["o2c"]
        c2o = np.argsort(o2c)

        def proj(x):
            return (feats(x) @ W).astype(np.float32)

        def oracle_state(xyz):
            """(K, 3) metres in cube order -> object-order (u, v, covered) with the true coverage bit."""
            cov = np.zeros(K, np.float32)
            for i in range(K):
                for j in range(K):
                    if i != j and xyz[j, 2] > xyz[i, 2] + 0.02 and np.linalg.norm(xyz[i, :2] - xyz[j, :2]) < 0.03:
                        cov[i] = 1.0
            st = np.concatenate([proj(xyz), cov[:, None]], -1)[o2c]
            return st if pk.get("n_bin", 0) else st[:, :2]
    recover = np.zeros(5)
    if a.events is not None and a.cache is not None:
        # mean action over the 5 frames after each play move: release + lift-off (label-free)
        ev = np.load(a.events / "cube_events_train.npz")
        acts = np.load(a.cache / "train_actions.npy", mmap_mode="r")
        idx = (ev["t"][:, None] + 1 + np.arange(5)[None]).ravel()
        recover = np.clip(np.asarray(acts[idx[idx < len(acts)]]).mean(0), -1, 1)
    if px_planner:
        plan_tol = float(pk["tol"])
        key_res = plan_tol / 4
        # with a coverage bit the covered-object rule replaces the affordance gate (job 57128: the gate at
        # 1/(2K) also excluded free cubes and forced detours); without it, 1/(2K)
        aff_min = (0.0 if pk.get("n_bin", 0) else 1.0 / (2 * K)) if a.aff_min is None else a.aff_min
        evt = np.load(a.events / "cube_events_train.npz")
        dd = np.linalg.norm(evt["before"][..., :2] - evt["target_xy"][:, None], axis=-1)
        dd[np.arange(len(evt["k"])), evt["k"]] = np.inf
        occ_min = float(np.percentile(dd.min(1), 1))
    else:
        plan_tol, key_res, occ_min = a.plan_tol, 0.01, 0.0
        aff_min = 0.02 if a.aff_min is None else a.aff_min
    if a.perception == "reader":
        from train_cube_reader import make_cube_reader

        rk = torch.load(a.reader, map_location="cpu", weights_only=False)
        use_cov = bool(rk.get("coverage", False))
        reader = make_cube_reader(K, coverage=use_cov).to(dev).eval()
        reader.load_state_dict(rk["reader"])
        if use_cov != bool(pk.get("n_bin", 0)):
            raise SystemExit("reader coverage head and planner binary dims disagree")
        move_thr = plan_tol
        # stillness: 99th pct of the reader's per-step displacement at rest on val (label-free)
        pred = np.load(a.reader.parent / "reader_val_pred.npy")
        lab = np.load(a.events / "labels_val.npz")
        same = lab["valid"][1:] & lab["valid"][:-1] & (np.linalg.norm(lab["pos"][1:] - lab["pos"][:-1], axis=-1) < 1e-6)
        still_thr = float(np.percentile(np.linalg.norm(pred[1:] - pred[:-1], axis=-1)[same], 99))
        from cube_discover import Perception

        colour = Perception.load(a.discover / "perception.npz", dev)

        def colour_obs(frame):
            with torch.no_grad():
                uv, m = colour.objects(torch.as_tensor(frame, device=dev)[None])
            return uv[0].cpu().numpy(), m[0].cpu().numpy() >= 3

        def observe(frame):
            """-> (K, 2) pixel positions, or (K, 3) with the coverage probability."""
            with torch.no_grad():
                x = torch.as_tensor(frame, device=dev).permute(2, 0, 1)[None].float() / 255.0
                if not use_cov:
                    return reader(x)[0].float().cpu().numpy()
                pos, cl = reader(x, with_cov=True)
                return np.concatenate([pos[0].float().cpu().numpy(), torch.sigmoid(cl[0]).cpu().numpy()[:, None]], -1)
    else:
        move_thr, still_thr = 0.02, 2e-3                         # privileged move-end rule (job 57118)
    rest_rule = a.perception == "reader" and a.end_rule == "rest"
    use_belief = bool(a.belief and rest_rule and pk.get("n_bin", 0))
    pk_args = dict(tol=plan_tol, key_res=key_res, aff_min=aff_min, occ_min=occ_min, n_bin=int(pk.get("n_bin", 0)))
    PD = 2 if a.perception == "reader" else 3                    # dims used for move detection
    env = gymnasium.make("visual-cube-triple-v0")
    episodes = []
    for task, epi in jobs:
        seed = a.seed * 10000 + task * 100 + epi
        ob, info = env.reset(seed=seed, options=dict(task_id=task, render_goal=False))
        if ob.mean() < 20 or info["goal"].mean() < 20:
            raise RuntimeError("rendering looks broken")
        u = env.unwrapped
        goal_true = np.stack([u._data.mocap_pos[u._cube_target_mocap_ids[i]].copy() for i in range(K)]).astype(np.float32)
        if a.perception == "reader":
            goal, s = observe(info["goal"]), observe(ob)
            s_p = s
        elif a.perception == "oracle_px":
            s = cube_xyz(u, K)
            s_p, goal = oracle_state(s), oracle_state(goal_true)
        else:
            goal, s = goal_true, cube_xyz(u, K)
            s_p = s
        hist = [s]
        belief, vis_hist = s_p.copy(), []
        k, q = -1, None                                   # no move executed yet
        s_plan_cur, sup, sup_key, tgt_key = s_p.copy(), K, None, None
        buffers = buffer_positions(pk["ground_xy"], np.concatenate([s_p[:, :2], goal[:, :2]]), n=6, min_dist=2 * plan_tol, seed=seed)
        t_plan = time.time()
        plan, pinfo = plan_moves(s_p, goal, wm, aff, buffers, dev, **pk_args)
        rec = {"task": task, "episode": epi, "first_plan": None if plan is None else len(plan), "moves": [], "replans": 0,
               "success": False, "steps": 0, "plan_nodes": pinfo["nodes"], "plan_sec": round(time.time() - t_plan, 2),
               "init": np.round(s_p, 3).tolist(), "goal": np.round(goal, 3).tolist(),
               "init_true": np.round(cube_xyz(u, K), 3).tolist(), "goal_true": np.round(goal_true, 3).tolist(),
               "plan": None if plan is None else [(k, [round(x, 3) for x in q]) for k, q in plan], "timeouts": 0,
               "move_thr": move_thr, "still_thr": still_thr, "aff_min": aff_min, "occ_min": occ_min,
               "end_rule": "rest" if rest_rule else "still", "belief": use_belief}
        frames, queue = [ob] * (gap + 1), []
        move_start, still, k_steps = s.copy(), 0, 0
        seen = [[] for _ in range(K)]                    # colour-track visible positions since the move start
        done = False
        while not done:
            if plan:
                k, q = plan[0]
                if use_sup and sup_key != (rec["replans"], k):
                    # support object = the one the WM predicts this move will cover (from the planning state)
                    sp = s_plan_cur.copy(); sp[:, 2] = sp[:, 2] > 0.5
                    with torch.no_grad():
                        pr = wm(torch.as_tensor(sp[None], device=dev), torch.tensor([k], device=dev),
                                torch.tensor([q], device=dev, dtype=torch.float32))[0].cpu().numpy()
                    newly = (pr[:, 2] > 0.5) & (sp[:, 2] < 0.5)
                    newly[k] = False
                    sup = int(np.argmin(np.where(newly, np.linalg.norm(pr[:, :2] - np.array(q), axis=-1), np.inf))) if newly.any() else K
                    sup_key = (rec["replans"], k)
                if skill_px and not px_planner:
                    # privileged metre plan + pixel skill: project the predicted landing position of cube k
                    if tgt_key != (rec["replans"], k):
                        with torch.no_grad():
                            pr = wm(torch.as_tensor(s_plan_cur[None], device=dev), torch.tensor([k], device=dev),
                                    torch.tensor([q], device=dev, dtype=torch.float32))[0, k].cpu().numpy()
                        q_sk, k_sk, tgt_key = proj(pr[None])[0].tolist(), int(c2o[k]), (rec["replans"], k)
                else:
                    q_sk, k_sk = q, k
                if not queue:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        px = torch.as_tensor(np.concatenate([frames[0], frames[-1]], -1), device=dev).permute(2, 0, 1)[None].float() / 255.0
                        ch = pi(px, torch.tensor([k_sk], device=dev), torch.tensor([q_sk], device=dev, dtype=torch.float32),
                                torch.tensor([sup], device=dev) if use_sup else None).float().cpu().numpy()[0]
                    queue = list(np.clip(ch[: a.exec_steps] * asd + amu, -1, 1))
                action = queue.pop(0)
            else:
                action = np.zeros(5)
            ob, _, term, trunc, info = env.step(action)
            rec["steps"] += 1
            k_steps += 1
            frames = frames[1:] + [ob]
            if info["success"]:
                rec["success"] = True
            done = term or trunc or rec["success"]
            s_now = observe(ob) if a.perception == "reader" else cube_xyz(u, K)
            hist = (hist + [s_now])[-3:]
            moved = np.linalg.norm(s_now[:, :PD] - move_start[:, :PD], axis=-1)
            speed = np.linalg.norm(s_now[:, :PD] - s[:, :PD], axis=-1).max()
            still = still + 1 if speed < still_thr else 0
            s = s_now
            end = False
            if rest_rule:
                cuv, cvis = colour_obs(ob)
                vis_hist = (vis_hist + [cvis])[-5:]
                rest_moved = np.zeros(K, bool)
                for j in range(K):
                    if cvis[j]:
                        seen[j] = (seen[j] + [cuv[j]])[-5:]
                    if len(seen[j]) == 5:
                        c = np.mean(seen[j], 0)
                        rest_moved[j] = (np.linalg.norm(np.array(seen[j]) - c, axis=-1).max() <= 1.0
                                         and np.linalg.norm(c - move_start[j, :2]) > move_thr)
            if plan:
                k, q = plan[0]
                if rest_rule:
                    knock = any(rest_moved[j] for j in range(K) if j != k)
                    finished = rest_moved[k] or knock
                else:
                    knock = any(moved[j] > move_thr for j in range(K) if j != k)
                    finished = (moved[k] > move_thr or knock) and still >= 5
                if finished:
                    end = True
                    rec["moves"].append({"k": int(k), "q": [round(float(x), 3) for x in q],
                                         "final": [round(float(x), 3) for x in s[k]],
                                         "err": round(float(np.linalg.norm(s[k, :2] - np.array(q))), 3),
                                         "knock": bool(knock), "steps": k_steps,
                                         "true_final": np.round(cube_xyz(u, K), 3).tolist()})
            if end or k_steps > a.timeout:
                if not end:
                    queue = [recover.copy() for _ in range(a.recover_steps)]
                    rec["timeouts"] += 1
                else:
                    queue = []
                s_plan = np.mean(hist, 0)
                s_det = s_plan.copy()
                if a.perception == "oracle_px":
                    s_plan = oracle_state(cube_xyz(u, K))
                if use_belief:
                    new = s_plan.copy()
                    seen_top = np.any(vis_hist, 0) if vis_hist else np.zeros(K, bool)
                    pred_c = None
                    if end:
                        with torch.no_grad():
                            pred = wm(torch.as_tensor(belief[None], device=dev), torch.tensor([k], device=dev),
                                      torch.tensor([q], device=dev, dtype=torch.float32))[0].cpu().numpy()
                        pred_c = pred[:, 2] > 0.5
                    for j in range(K):
                        if seen_top[j]:
                            new[j, 2] = 0.0
                        elif pred_c is not None:
                            new[j, 2] = float(pred_c[j])
                        else:
                            new[j, 2] = float(s_plan[j, 2] > 0.5)
                        if new[j, 2] > 0.5 and j != k:
                            new[j, :2] = belief[j, :2]
                    belief = s_plan = new
                    if rec["moves"] and end:
                        rec["moves"][-1]["belief"] = np.round(belief, 2).tolist()
                move_start, k_steps = (s_det if a.perception == "oracle_px" else s_plan).copy(), 0
                seen = [[] for _ in range(K)]
                if not done:
                    buffers = buffer_positions(pk["ground_xy"], np.concatenate([s_plan[:, :2], goal[:, :2]]), n=6,
                                               min_dist=2 * plan_tol, seed=seed + rec["replans"])
                    plan, _ = plan_moves(s_plan, goal, wm, aff, buffers, dev, **pk_args)
                    rec["replans"] += 1
                    s_plan_cur = s_plan.copy()
        rec["final_state"] = np.round(s, 3).tolist()
        rec["final_err_true_m"] = np.round(np.linalg.norm(cube_xyz(u, K) - goal_true, axis=-1), 3).tolist()
        rec["plan_empty_at_end"] = not plan
        episodes.append(rec)
        print(json.dumps({x: rec[x] for x in ("task", "episode", "first_plan", "success", "steps", "replans", "timeouts")}), flush=True)
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", type=Path, required=True)
    ap.add_argument("--skill", type=Path, required=True)
    ap.add_argument("--perception", choices=["privileged", "reader", "oracle_px"], default="privileged")
    ap.add_argument("--project", type=Path, default=None, help="cube_projection.py output (PRIVILEGED attribution arms)")
    ap.add_argument("--reader", type=Path, default=None, help="cube_reader.pt (reader perception)")
    ap.add_argument("--discover", type=Path, default=None, help="cube_discover.py dir (colour track for --end-rule rest)")
    ap.add_argument("--end-rule", choices=["rest", "still"], default="rest")
    ap.add_argument("--belief", action="store_true", help="WM-consistent belief update at move ends (see docstring)")
    ap.add_argument("--aff-min", type=float, default=None, help="default: 1/(2K) with reader perception, 0.02 privileged")
    ap.add_argument("--low", default="learned")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--episode-start", type=int, default=0, help="run episodes [start, start + episodes)")
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--plan-tol", type=float, default=0.04, help="privileged perception only (metres)")
    ap.add_argument("--timeout", type=int, default=250)
    ap.add_argument("--exec-steps", type=int, default=4)
    ap.add_argument("--recover-steps", type=int, default=8)
    ap.add_argument("--events", type=Path, default=None, help="events dir (recovery action; labels for reader stillness)")
    ap.add_argument("--cache", type=Path, default=None, help="cache/<env> (recovery action)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    if a.perception == "reader" and (a.reader is None or a.events is None or a.discover is None):
        raise SystemExit("--perception reader needs --reader, --events and --discover")
    if a.perception == "oracle_px" and (a.project is None or a.events is None):
        raise SystemExit("--perception oracle_px needs --project and --events")
    jobs = [(t, e) for t in a.tasks for e in range(a.episode_start, a.episode_start + a.episodes)]
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        chunks = [jobs[i::a.workers] for i in range(a.workers)]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            episodes = [r for part in pool.starmap(run, [(a, c) for c in chunks if c]) for r in part]
    else:
        episodes = run(a, jobs)
    episodes.sort(key=lambda r: (r["task"], r["episode"]))
    mv = [m for e in episodes for m in e["moves"]]
    summary = {"arm": {"privileged": "PRIVILEGED object state (direction D)", "reader": "label-free object reader (direction A)",
                       "oracle_px": "PRIVILEGED perfect perception projected to pixels"}[a.perception]
               + " + learned WM/affordance/search + pixel skill" + (" [skill from --skill, targets projected]" if a.project else ""),
               "success": float(np.mean([e["success"] for e in episodes])),
               "by_task": {t: float(np.mean([e["success"] for e in episodes if e["task"] == t])) for t in a.tasks},
               "first_plan_found": float(np.mean([e["first_plan"] is not None for e in episodes])),
               "stuck_with_empty_plan": int(sum((not e["success"]) and e["plan_empty_at_end"] for e in episodes)),
               "moves": len(mv), "knock_frac": float(np.mean([m["knock"] for m in mv])) if mv else None,
               "minutes": round((time.time() - t0) / 60, 1), "args": {k: str(v) for k, v in vars(a).items()}}
    save_json(a.out / "cube_closed_loop.json", {"summary": summary, "episodes": episodes})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
