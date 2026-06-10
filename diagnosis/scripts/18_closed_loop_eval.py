"""Closed-loop Metaworld planning success rate — L2 cost vs the grounded cost.

Protocol replicates the upstream JEPA-WMs paper's Metaworld evaluation
(configs/evals/simu_env_planning/mw/*: goal frame from the scripted expert,
CEM-L2 planner with horizon 6 / 300 samples / 15 iterations / var_scale 1.0,
execute num_act_stepped=3 model-steps (15 raw actions) per replan,
max_episode_steps 100, success = the simulator's flag), with the deviations
stated in the output: 1-frame planning context (upstream: 2), alpha=0 (no
proprio term in the cost), fewer episodes, and contact tasks the paper does not
evaluate closed-loop (its Metaworld tables cover Reach / Reach-Wall only — both
free-space, so MW-Reach here is the sanity anchor against the paper and the
contact tasks are the new experiment the boundary fix targets).

Arms (paired: same env seeds, same CEM noise seeds):
    l2    — upstream planning objective (latent MSE to goal)
    hdyn  — + the grounded object-dynamics term integrated along the rollout
            (models/probes.grounded_dynamics_cost; needs --probe and --dyn-head)

    python scripts/18_closed_loop_eval.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --probe checkpoints/object_probe_dino_wm_metaworld.pt \
        --dyn-head checkpoints/object_dynamics_dino_wm_metaworld.pt \
        --tasks mw-reach mw-push mw-pick-place --episodes 16

Output: results/metaworld_closed_loop.csv (+ per-episode rows).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from models.probes import load_probe, load_dynamics_head, grounded_dynamics_cost  # noqa: E402
from planning.cem_planner import cem_plan  # noqa: E402

FRAMESKIP = 5          # metaworld frameskip the checkpoints were trained with
RAW_A = 4


def make_env(task: str, seed: int, img_size: int = 224):
    """Metaworld V3 goal-observable env with the upstream camera setup."""
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

    env_id = task.split("-", 1)[-1] + "-v3-goal-observable"
    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=seed)
    env.seeded_rand_vec = False
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]       # upstream corner2 tweak
    env.render_mode = "rgb_array"
    env.camera_name = "corner2"
    env.width = env.height = img_size
    return env


def expert_policy(task: str):
    from metaworld import policies

    special = {"mw-peg-insert-side": "SawyerPegInsertionSideV3Policy"}
    if task in special:
        return getattr(policies, special[task])()
    name = "Sawyer" + "".join(w.capitalize() for w in task.split("-")[1:]) + "V3Policy"
    return getattr(policies, name)()


def render(env) -> np.ndarray:
    frame = env.render()
    if frame is None or frame.sum() == 0:
        raise RuntimeError("Metaworld render returned an empty frame")
    # NOTE: the upstream wrapper flips vertically, but that compensated *their*
    # renderer stack; gymnasium 1.3 / mujoco 3.9 already returns right-side-up
    # (verified visually, results/logs/render_raw.png) — flipping here fed the
    # encoder an upside-down scene and zeroed the smoke success rate.
    return frame.copy()


def rollout_expert(task: str, seed: int, goal_steps: int | None, max_steps: int = 200):
    """Roll the scripted expert; return (goal_frame, goal_state, init_state,
    success_step). goal_steps=None → goal at the expert's success frame."""
    env = make_env(task, seed)
    obs, _ = env.reset()
    init_state = obs.copy()
    pol = expert_policy(task)
    succ_step = None
    goal_frame, goal_state = None, None
    for t in range(1, max_steps + 1):
        obs, _, _, _, info = env.step(pol.get_action(obs))
        if succ_step is None and info.get("success", 0) > 0.5:
            succ_step = t
        if goal_steps is not None and t == goal_steps:
            goal_frame, goal_state = render(env), obs.copy()
        if goal_steps is None and succ_step is not None:
            goal_frame, goal_state = render(env), obs.copy()
            break
    if goal_frame is None:                            # expert never succeeded /
        goal_frame, goal_state = render(env), obs.copy()  # or goal_steps > rollout
    env.close()
    return goal_frame, goal_state, init_state, succ_step


@torch.no_grad()
def encode_frame(adapter, frame: np.ndarray, proprio: np.ndarray, device):
    vis = torch.from_numpy(frame.copy()).permute(2, 0, 1).float()[None, None]   # (1,1,C,H,W)
    prop = torch.from_numpy(proprio.astype(np.float32))[None, None]             # (1,1,P)
    z = adapter.encode(vis.to(device), prop.to(device) if adapter.uses_proprio() else None)
    return z[0, 0]                                                              # (V,H,W,D)


def run_episode(arm, task, seed, adapter, device, *, probe, dyn_head, s_z, s_g,
                beta, cem_kw, horizon, num_act_stepped, max_episode_steps):
    env = make_env(task, seed)
    obs, _ = env.reset()
    goal_frame, goal_state, init_state, expert_succ = rollout_expert(
        task, seed, goal_steps=None)
    if not np.allclose(obs, init_state, atol=1e-6):
        print(f"  [warn] init-state mismatch (seed {seed}); proceeding")

    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    success, steps = False, 0
    while steps < max_episode_steps:
        frame = render(env)
        z_t = encode_frame(adapter, frame, obs[:4], device)
        prop = torch.from_numpy(obs[:4].astype(np.float32)).to(device)
        tcf = None
        if arm == "hdyn":
            tcf = grounded_dynamics_cost(probe, dyn_head, adapter, z_t, z_goal,
                                         s_z=s_z, s_g=s_g, beta=beta)
        plan = cem_plan(
            adapter, z_t, z_goal, horizon=horizon, action_dim=RAW_A * FRAMESKIP,
            num_act_stepped=num_act_stepped, proprio_t=prop,
            generator=torch.Generator(device=device).manual_seed(seed * 1000 + steps),
            traj_cost_fn=tcf, **cem_kw)
        raw = plan.reshape(-1, RAW_A).cpu().numpy()
        for a in raw:
            obs, _, _, _, info = env.step(np.clip(a, -1, 1))
            steps += 1
            if info.get("success", 0) > 0.5:
                success = True
            if steps >= max_episode_steps:
                break
        if success:
            break
    final_dist = float(np.linalg.norm(obs - goal_state))
    env.close()
    return {"task": task, "arm": arm, "seed": seed, "success": int(success),
            "steps": steps, "final_state_dist": final_dist,
            "state_dist_success": int(final_dist < 0.3),
            "expert_success_step": expert_succ}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--dyn-head", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mw-reach", "mw-push", "mw-pick-place"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--s-z", type=float, default=169.31)   # pool scales (scripts/15 log)
    ap.add_argument("--s-g", type=float, default=0.1276)
    ap.add_argument("--horizon", type=int, default=6)      # upstream mw config
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--cem-num-samples", type=int, default=300)
    ap.add_argument("--cem-iterations", type=int, default=15)
    ap.add_argument("--var-scale", type=float, default=1.0)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--arms", nargs="+", default=["l2", "hdyn"])
    ap.add_argument("--out", default="results/metaworld_closed_loop.csv")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    probe, _ = load_probe(args.probe, device)
    dyn_head, dyn_meta = load_dynamics_head(args.dyn_head, device)
    cem_kw = dict(num_samples=args.cem_num_samples, iterations=args.cem_iterations,
                  num_elites=10, var_scale=args.var_scale,
                  max_norms=[1.0], max_norm_dims=[list(range(RAW_A * FRAMESKIP))])
    print(f"protocol: H={args.horizon} nas={args.num_act_stepped} "
          f"samples={args.cem_num_samples} var={args.var_scale} "
          f"max_steps={args.max_episode_steps} episodes={args.episodes} "
          f"arms={args.arms} (deviations vs paper: ctxt=1, alpha=0)", flush=True)
    print(f"dyn head: cf_corr={dyn_meta.get('cf_corr'):.3f} beta={args.beta} "
          f"s_z={args.s_z} s_g={args.s_g}", flush=True)

    import pandas as pd
    rows = []
    for task in args.tasks:
        for ep in range(args.episodes):
            seed = 10_000 + ep
            for arm in args.arms:
                t0 = time.time()
                try:
                    r = run_episode(arm, task, seed, adapter, device,
                                    probe=probe, dyn_head=dyn_head,
                                    s_z=args.s_z, s_g=args.s_g, beta=args.beta,
                                    cem_kw=cem_kw, horizon=args.horizon,
                                    num_act_stepped=args.num_act_stepped,
                                    max_episode_steps=args.max_episode_steps)
                except Exception as e:  # noqa: BLE001 — keep the sweep alive
                    print(f"  [error] {task} ep{ep} {arm}: {e}", flush=True)
                    continue
                r["minutes"] = round((time.time() - t0) / 60, 2)
                rows.append(r)
                print(f"  {task:16s} ep{ep:02d} {arm:5s} success={r['success']} "
                      f"steps={r['steps']:3d} dist={r['final_state_dist']:.3f} "
                      f"({r['minutes']:.1f} min)", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)   # checkpoint as we go

        d = pd.DataFrame(rows)
        for arm in args.arms:
            sel = d[(d.task == task) & (d.arm == arm)]
            if len(sel):
                print(f"== {task} {arm}: success {sel.success.mean():.2%} "
                      f"({int(sel.success.sum())}/{len(sel)}), "
                      f"state-dist<0.3 {sel.state_dist_success.mean():.2%}", flush=True)

    print(f"\nWrote {args.out} ({len(rows)} episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
