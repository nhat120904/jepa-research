"""Closed-loop Metaworld planning success rate — L2 cost vs the grounded cost.

Protocol replicates the upstream JEPA-WMs paper's Metaworld evaluation, read
off the shipped config (base_configs/mw/reach-wall_L2_cem_sourcexp_H6_nas3_
ctxt2.yaml): goal frame = the scripted expert's final frame, CEM-L2 planner
with horizon 6 / 300 samples / 15 iterations / var_scale 1.0, execute
num_act_stepped=3 model-steps (15 raw actions) per replan, max_episode_steps
100, one zero-action warmup step after reset (env wrapper's reset_warmup),
horizon shrunk to the remaining model-steps near episode end, success = the
simulator's flag. Upstream mw uses `alpha: 0` — no proprio term in the COST
(the unroll context still carries proprio; dino_wm's predictor requires it:
424 = 384 visual + 20 proprio + 20 action). α=0 is the default here
(--alpha restores the term). Remaining deviations: fewer episodes, and contact tasks
the paper does not evaluate closed-loop (its Metaworld tables cover Reach /
Reach-Wall only — both free-space, so MW-Reach here is the sanity anchor
against the paper and the contact tasks are the new experiment the boundary
fix targets).

Arms (paired: same env seeds, same CEM noise seeds):
    l2    — upstream planning objective: latent MSE to goal (+ α·proprio-feature
            MSE if --alpha > 0; upstream mw uses α=0)
    hdyn  — + the grounded object-dynamics term integrated along the rollout
            (per-dim normalised, weighted --beta; needs --probe and --dyn-head)

Implementation notes: one env per (task, seed) with the rand_vec frozen after the
first reset, reused for the expert goal rollout and both arms (repeated MuJoCo
renderer creation crashed the process natively on Windows); the proprio term
needs the unroll's proprio predictions, which `_PlanAdapter` captures per rollout.

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
from models.probes import load_probe, load_dynamics_head  # noqa: E402
from planning.cem_planner import cem_plan  # noqa: E402

FRAMESKIP = 5          # metaworld frameskip the checkpoints were trained with
RAW_A = 4


def make_env(task: str, seed: int, img_size: int = 224, cam_tweak: bool = True):
    """Metaworld V3 goal-observable env with the upstream camera setup.

    After the first reset we freeze the rand_vec so every subsequent reset of
    THIS instance reproduces the same initial state — the expert goal rollout
    and both planning arms then share one env (and one MuJoCo renderer)."""
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

    env_id = task.split("-", 1)[-1] + "-v3-goal-observable"
    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=seed)
    env.seeded_rand_vec = False
    if cam_tweak:
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]   # upstream wrapper's corner2 tweak
    env.render_mode = "rgb_array"
    env.camera_name = "corner2"
    env.width = env.height = img_size
    # CRITICAL: the env constructed its offscreen renderer at the default
    # 480x480 — assigning env.width afterwards does NOT resize it, and the
    # encoder then sees 480px frames while the checkpoints were trained on
    # native 224px renders (measured: 8.5x one-step pred error, 3.3x latent
    # NN distance). Upstream's MetaWorldWrapper.init_renderer() re-creates
    # the renderer at img_size; mirror it exactly.
    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
    env.mujoco_renderer = MujocoRenderer(
        env.model,
        env.data,
        env.mujoco_renderer.default_cam_config,
        width=img_size,
        height=img_size,
        max_geom=env.mujoco_renderer.max_geom,
        camera_id=None,
        camera_name=env.camera_name,
    )
    obs0, _ = env.reset()
    env._freeze_rand_vec = True                      # later resets: same init
    return env, obs0


class _PlanAdapter:
    """Pass-through planning adapter that captures the unroll's PROPRIO
    predictions so the cost can apply the upstream α·proprio-feature term
    (`planning_objective.alpha: 0.1` in the mw configs). Only the methods
    cem_plan touches are forwarded."""

    def __init__(self, base, residual_head=None):
        self.base = base
        self.spec = base.spec
        self.device = base.device
        self.last_proprio = None                     # (B, H+1, ...) features
        # When set, the rollout is corrected per step: ẑ_{t+1}=F(z,a)+Δ(z,a)
        # (option C). Δ corrects the visual latent only; proprio (reliable) is
        # propagated uncorrected. Requires the per-step loop below.
        self.residual_head = residual_head

    # CEM batches of 300 unrolls peak at ~11.6 of 12.2 GB VRAM — any other GPU
    # user (a notebook kernel, the desktop) then crashes the run natively.
    # Chunking the unroll batch is mathematically identical and buys headroom.
    _CHUNK = int(os.environ.get("CAI_JEPA_PLAN_CHUNK", "150"))

    def predict_rollout(self, z_t, actions, proprio_t=None):
        B = actions.shape[0]
        if B <= self._CHUNK:
            return self._predict_rollout_chunk(z_t, actions, proprio_t)
        outs, props = [], []
        for s in range(0, B, self._CHUNK):
            e = min(B, s + self._CHUNK)
            outs.append(self._predict_rollout_chunk(
                z_t[s:e], actions[s:e],
                proprio_t[s:e] if proprio_t is not None else None))
            props.append(self.last_proprio)
        self.last_proprio = (torch.cat(props, dim=0)
                             if props[0] is not None else None)
        return torch.cat(outs, dim=0)

    def _predict_rollout_chunk(self, z_t, actions, proprio_t=None):
        if self.residual_head is not None:
            return self._corrected_rollout_chunk(z_t, actions, proprio_t)
        from einops import rearrange
        from tensordict.tensordict import TensorDict

        b = self.base
        B, H, _ = actions.shape
        z_t = z_t.to(b.device, dtype=torch.float32)
        a = actions.to(b.device, dtype=torch.float32).reshape(B, -1, b.spec.action_dim)
        a = b.normalize_action(a).reshape(B, -1, b._model_action_dim)
        act_suffix = rearrange(a, "b t a -> t b a")
        z_ctxt_visual = z_t.unsqueeze(1)
        if b.spec.uses_proprio and proprio_t is not None:
            prop_feat = b.encode_proprio_features(proprio_t.reshape(B, 1, -1))
            ctxt = TensorDict({"visual": z_ctxt_visual, "proprio": prop_feat}, batch_size=[])
            pred = b.encpred.unroll(ctxt, act_suffix=act_suffix)
            self.last_proprio = rearrange(pred["proprio"], "t b ... -> b t ...")
            return rearrange(pred["visual"], "t b ... -> b t ...")
        self.last_proprio = None
        pred = b.encpred.unroll(z_ctxt_visual, act_suffix=act_suffix)
        return rearrange(pred, "t b ... -> b t ...")

    def _corrected_rollout_chunk(self, z_t, actions, proprio_t=None):
        """Recursive corrected rollout: ẑ_{t+1} = F(z_t,a_t) + Δ(z_t,a_t), one
        model-step at a time so the correction (and the predicted proprio
        context) feed back. Mirrors the frozen path's normalisation exactly."""
        from einops import rearrange
        from tensordict.tensordict import TensorDict

        b, rh = self.base, self.residual_head
        B, H, _ = actions.shape
        z_cur = z_t.to(b.device, dtype=torch.float32)
        a = actions.to(b.device, dtype=torch.float32).reshape(B, -1, b.spec.action_dim)
        a = b.normalize_action(a).reshape(B, H, b._model_action_dim)   # model-space per step
        uses_p = b.spec.uses_proprio and proprio_t is not None
        prop_feat = (b.encode_proprio_features(proprio_t.reshape(B, 1, -1))
                     if uses_p else None)                              # (B, 1, ...) batch-first
        outs, props = [z_cur], []
        for t in range(H):
            a_t = a[:, t]                                              # (B, model_action_dim)
            act_suffix = a_t.unsqueeze(0)                             # (1, B, A)
            if uses_p:
                ctxt = TensorDict({"visual": z_cur.unsqueeze(1), "proprio": prop_feat},
                                  batch_size=[])
                pred = b.encpred.unroll(ctxt, act_suffix=act_suffix)
                z_next = pred["visual"][-1]                           # frozen one-step (B,V,H,W,D)
                prop_bt = rearrange(pred["proprio"], "t b ... -> b t ...")
                prop_feat = prop_bt[:, -1:]                          # next ctxt proprio (uncorrected)
                props.append(prop_bt[:, -1])
            else:
                pred = b.encpred.unroll(z_cur.unsqueeze(1), act_suffix=act_suffix)
                z_next = pred[-1]
            z_next = z_next + rh(z_cur, a_t)                          # CORRECT the visual latent
            z_cur = z_next
            outs.append(z_cur)
        self.last_proprio = (torch.stack(props, dim=1) if props else None)   # (B, H, ...)
        return torch.stack(outs, dim=1)                              # (B, H+1, V,H,W,D)

    def predict(self, z_t, a_t, proprio_t=None):
        return self.base.predict(z_t, a_t, proprio_t=proprio_t)

    def normalize_action(self, a):
        return self.base.normalize_action(a)

    def action_dim(self):
        return self.base.action_dim()

    def uses_proprio(self):
        return self.base.uses_proprio()


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
    # The training data IS vertically flipped relative to today's corner2
    # render: pixel-calibrating against dataset init frames gives MSE 71.6 for
    # flipud(render) under the wrapper camera vs ~3600+ for every unflipped
    # candidate (results/logs/camera_calib7). The upstream wrapper's [::-1]
    # flip is part of the data-generation pipeline, so the planner must feed
    # the encoder flipped frames too. (The earlier "right-side-up" check that
    # removed this flip was made while the renderer silently used the default
    # free camera — wrong camera, wrong conclusion.)
    return frame[::-1].copy()


def rollout_expert(env, init_obs: np.ndarray, task: str, max_steps: int = 100):
    """Roll the scripted expert on the SHARED env (already reset); return
    (goal_frame, goal_state, first_success_step) at the expert's FINAL frame.

    Upstream takes `expert_obses[-1]` — the expert runs the whole episode and
    keeps refining after the success flag first fires (flag = entering the
    5 cm radius; the final pose is ~1 cm from target). Breaking at first
    success put our goal frame right at the radius edge, so a planner that
    faithfully reached it could still sit outside the env's 5 cm success
    check (measured: ee 2-4 cm with success=0, systematically)."""
    obs = init_obs
    pol = expert_policy(task)
    succ_step = None
    for t in range(1, max_steps + 1):
        obs, _, _, _, info = env.step(pol.get_action(obs))
        if succ_step is None and info.get("success", 0) > 0.5:
            succ_step = t
    return render(env), obs.copy(), succ_step


@torch.no_grad()
def encode_frame(adapter, frame: np.ndarray, proprio: np.ndarray, device):
    vis = torch.from_numpy(frame.copy()).permute(2, 0, 1).float()[None, None]   # (1,1,C,H,W)
    prop = torch.from_numpy(proprio.astype(np.float32))[None, None]             # (1,1,P)
    z = adapter.encode(vis.to(device), prop.to(device) if adapter.uses_proprio() else None)
    return z[0, 0]                                                              # (V,H,W,D)


_COST_DEBUG = os.environ.get("CAI_JEPA_COST_DEBUG")
_cost_dbg_n = 0


def make_traj_cost(arm, plan_adapter, base, probe, dyn_head, z_t, z_goal,
                   goal_prop_feat, *, alpha, beta, s_g, ee_probe=None,
                   lambda_app=0.0, lambda_obj=0.0):
    """Upstream objective (visual MSE + α·proprio-feature MSE) plus a grounded
    object term. ``hdyn``: final object-at-goal (scoring only). ``hexp``: dense
    APPROACH (predicted ee → current object, drives contact) + dense MANIPULATE
    (object → goal, denser selection signal) — the exploration fix (option B,
    docs/plans/2026-06-18-grounded-exploration-design.md). All per-dim normalised
    on the object scale s_g so the weights are comparable to β."""
    g_goal = g_init = None
    if arm in ("hdyn", "hexp"):
        with torch.no_grad():
            g_goal = probe(z_goal.unsqueeze(0))
            g_init = probe(z_t.unsqueeze(0))
    if arm == "hexp" and ee_probe is None:
        raise ValueError("hexp arm needs --ee-probe")
    s_g_dim = s_g / np.sqrt(probe.out_dim)

    def cost(pred, actions, z_goal_):
        global _cost_dbg_n
        B = pred.shape[0]
        c = ((pred[:, -1].reshape(B, -1) - z_goal_.reshape(1, -1)) ** 2).mean(-1)
        lp = plan_adapter.last_proprio
        if goal_prop_feat is not None and lp is not None:
            c = c + alpha * ((lp[:, -1].reshape(B, -1)
                              - goal_prop_feat.reshape(1, -1)) ** 2).mean(-1)
        if arm == "hdyn":
            obj = g_init.expand(B, -1).clone()
            H = pred.shape[1] - 1
            for t in range(H):
                a = actions[:, t].reshape(B, -1, base.action_dim())
                a = base.normalize_action(a).reshape(B, -1)
                obj = obj + dyn_head(pred[:, t], a)
            c = c + beta * (((obj - g_goal) / s_g_dim) ** 2).mean(-1)
        elif arm == "hexp":
            H = pred.shape[1] - 1
            obj = g_init.expand(B, -1).clone()
            app = torch.zeros(B, device=pred.device)
            man = torch.zeros(B, device=pred.device)
            for t in range(H):
                ee = ee_probe(pred[:, t])                          # (B,3) predicted ee
                app = app + (((ee - obj) / s_g_dim) ** 2).mean(-1)  # ee → current object
                a = actions[:, t].reshape(B, -1, base.action_dim())
                a = base.normalize_action(a).reshape(B, -1)
                obj = obj + dyn_head(pred[:, t], a)               # integrate object motion
                man = man + (((obj - g_goal) / s_g_dim) ** 2).mean(-1)  # dense object → goal
            app, man = app / H, man / H
            if _COST_DEBUG and _cost_dbg_n < 8:
                print(f"    [cost] visual={c.mean():.4f} app={app.mean():.4f} "
                      f"(x{lambda_app}={lambda_app*app.mean():.4f}) "
                      f"man={man.mean():.4f} (x{lambda_obj}={lambda_obj*man.mean():.4f})",
                      flush=True)
                _cost_dbg_n += 1
            c = c + lambda_app * app + lambda_obj * man
        return c

    return cost


def run_episode(arm, task, seed, env, init_state, goal_frame, goal_state,
                expert_succ, adapter, plan_adapter, device, *, probe, dyn_head,
                s_g, alpha, beta, cem_kw, horizon, num_act_stepped,
                max_episode_steps, proprio_ctxt, ee_probe=None,
                lambda_app=0.0, lambda_obj=0.0, cost_arm=None):
    cost_arm = cost_arm or arm
    obs, _ = env.reset()
    if not np.allclose(obs, init_state, atol=1e-5):
        print(f"  [warn] init-state mismatch after reset (seed {seed})")
    # upstream reset_warmup: one zero-action step before the first observation
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))

    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    goal_prop_feat = None
    if alpha > 0 and adapter.uses_proprio():
        with torch.no_grad():
            goal_prop_feat = adapter.encode_proprio_features(
                torch.from_numpy(goal_state[:4].astype(np.float32))[None, None].to(device))
    success, steps = False, 0
    while steps < max_episode_steps:
        frame = render(env)
        z_t = encode_frame(adapter, frame, obs[:4], device)
        prop = None
        if proprio_ctxt:
            prop = torch.from_numpy(obs[:4].astype(np.float32)).to(device)
        tcf = make_traj_cost(cost_arm, plan_adapter, adapter, probe, dyn_head,
                             z_t, z_goal, goal_prop_feat,
                             alpha=alpha, beta=beta, s_g=s_g, ee_probe=ee_probe,
                             lambda_app=lambda_app, lambda_obj=lambda_obj)
        # upstream shrinks the plan to the remaining model-steps near the end
        plan_h = min(horizon, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan(
            plan_adapter, z_t, z_goal, horizon=plan_h, action_dim=RAW_A * FRAMESKIP,
            num_act_stepped=min(num_act_stepped, plan_h), proprio_t=prop,
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
    return {"task": task, "arm": arm, "seed": seed, "success": int(success),
            "steps": steps,
            "final_state_dist": float(np.linalg.norm(obs - goal_state)),
            "ee_dist": float(np.linalg.norm(obs[:3] - goal_state[:3])),
            "state_dist_success": int(np.linalg.norm(obs - goal_state) < 0.3),
            "expert_success_step": expert_succ}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--dyn-head", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mw-reach", "mw-push", "mw-pick-place"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="proprio-feature cost weight (upstream mw config: 0)")
    ap.add_argument("--no-proprio-ctxt", dest="proprio_ctxt", action="store_false",
                    help="drop proprio from the unroll context (NOTE: dino_wm's "
                         "predictor needs it — 384 visual + 20 proprio + 20 action "
                         "= 424; upstream always carries proprio in the obs td)")
    ap.add_argument("--beta", type=float, default=0.1, help="grounded-term weight (hdyn)")
    ap.add_argument("--ee-probe", default=None,
                    help="ee-probe ckpt (scripts/19); required for the hexp arm")
    ap.add_argument("--lambda-app", type=float, default=1.0,
                    help="hexp APPROACH (ee→object) weight")
    ap.add_argument("--lambda-obj", type=float, default=1.0,
                    help="hexp MANIPULATE (dense object→goal) weight")
    ap.add_argument("--s-g", type=float, default=0.1276)   # pool scale (scripts/15 log)
    ap.add_argument("--horizon", type=int, default=6)      # upstream mw config
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--cem-num-samples", type=int, default=300)
    ap.add_argument("--cem-iterations", type=int, default=15)
    ap.add_argument("--var-scale", type=float, default=1.0)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--arms", nargs="+", default=["l2", "hdyn"],
                    help="l2/hdyn/hexp use the frozen predictor; suffix 'c' "
                         "(l2c/hdync) uses the corrected predictor F+Δ (--residual-head)")
    ap.add_argument("--residual-head", default=None,
                    help="residual corrective predictor ckpt (scripts/20); "
                         "required for any 'c'-suffixed arm")
    ap.add_argument("--out", default="results/metaworld_closed_loop.csv")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    plan_adapter = _PlanAdapter(adapter)
    # Corrected predictor adapter for 'c'-suffixed arms (option C).
    plan_adapter_c = None
    if any(a.endswith("c") for a in args.arms):
        from models.heads import load_residual_head
        if not args.residual_head:
            raise SystemExit("--residual-head is required for a 'c'-suffixed arm")
        rh, rh_meta = load_residual_head(args.residual_head, device)
        plan_adapter_c = _PlanAdapter(adapter, residual_head=rh)
        print(f"residual head: obj err corrected/frozen="
              f"{rh_meta.get('corrected_obj_mse', float('nan'))/max(rh_meta.get('frozen_obj_mse', 1), 1e-9):.3f}x "
              f"lambda_obj_train={rh_meta.get('lambda_obj')}", flush=True)
    probe, _ = load_probe(args.probe, device)
    dyn_head, dyn_meta = load_dynamics_head(args.dyn_head, device)
    ee_probe = None
    if "hexp" in args.arms:
        if not args.ee_probe:
            raise SystemExit("--ee-probe is required when the hexp arm is selected")
        ee_probe, ee_meta = load_probe(args.ee_probe, device)
        print(f"ee probe: v1_median={ee_meta.get('v1_median')} "
              f"lambda_app={args.lambda_app} lambda_obj={args.lambda_obj}", flush=True)
    cem_kw = dict(num_samples=args.cem_num_samples, iterations=args.cem_iterations,
                  num_elites=10, var_scale=args.var_scale,
                  max_norms=[1.0], max_norm_dims=[list(range(RAW_A * FRAMESKIP))])
    print(f"protocol: H={args.horizon} nas={args.num_act_stepped} "
          f"samples={args.cem_num_samples} var={args.var_scale} "
          f"max_steps={args.max_episode_steps} episodes={args.episodes} "
          f"arms={args.arms} alpha={args.alpha} proprio_ctxt={args.proprio_ctxt} "
          f"(upstream mw parity: rgb-only ctxt, alpha=0, warmup step, "
          f"horizon shrink)", flush=True)
    print(f"dyn head: cf_corr={dyn_meta.get('cf_corr'):.3f} beta={args.beta} "
          f"s_g={args.s_g}", flush=True)

    import pandas as pd
    # Resume: the sweep python dies natively now and then (MuJoCo/driver on
    # Windows, no traceback). Rows already in --out are kept and their
    # (task, seed, arm) cells skipped, so an outer retry loop can relaunch
    # this script until the sweep is complete.
    rows = []
    done_pairs = set()
    if Path(args.out).exists():
        prev = pd.read_csv(args.out)
        # The rand_vec is random per env creation (only frozen within one
        # env), so a half-done (task, seed) pair cannot be completed against
        # the same init after a crash — drop partial pairs and redo them
        # whole, keeping the comparison paired.
        cells = {(t, int(s)): set(g.arm)
                 for (t, s), g in prev.groupby(["task", "seed"])}
        done_pairs = {k for k, arms in cells.items() if set(args.arms) <= arms}
        keep = prev[[((r.task, int(r.seed)) in done_pairs) for r in prev.itertuples()]]
        dropped = len(prev) - len(keep)
        rows = keep.to_dict("records")
        print(f"resume: {len(rows)} episodes kept from {args.out}"
              + (f" ({dropped} partial-pair rows redone)" if dropped else ""),
              flush=True)
    for task in args.tasks:
        for ep in range(args.episodes):
            seed = 10_000 + ep
            if (task, seed) in done_pairs:
                continue
            try:
                env, init_state = make_env(task, seed)
                goal_frame, goal_state, expert_succ = rollout_expert(env, init_state, task)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {task} ep{ep} env/expert: {e}", flush=True)
                continue
            for arm in args.arms:
                t0 = time.time()
                # 'c'-suffixed arms use the corrected predictor; the cost is the
                # base arm (l2c -> L2 cost, hdync -> hdyn cost).
                corrected = arm.endswith("c")
                pa = plan_adapter_c if corrected else plan_adapter
                cost_arm = arm[:-1] if corrected else arm
                try:
                    r = run_episode(arm, task, seed, env, init_state, goal_frame,
                                    goal_state, expert_succ, adapter, pa,
                                    device, probe=probe, dyn_head=dyn_head,
                                    s_g=args.s_g, alpha=args.alpha, beta=args.beta,
                                    cem_kw=cem_kw, horizon=args.horizon,
                                    num_act_stepped=args.num_act_stepped,
                                    max_episode_steps=args.max_episode_steps,
                                    proprio_ctxt=args.proprio_ctxt,
                                    ee_probe=ee_probe, lambda_app=args.lambda_app,
                                    lambda_obj=args.lambda_obj, cost_arm=cost_arm)
                except Exception as e:  # noqa: BLE001 — keep the sweep alive
                    print(f"  [error] {task} ep{ep} {arm}: {e}", flush=True)
                    continue
                r["minutes"] = round((time.time() - t0) / 60, 2)
                rows.append(r)
                print(f"  {task:16s} ep{ep:02d} {arm:5s} success={r['success']} "
                      f"steps={r['steps']:3d} dist={r['final_state_dist']:.3f} "
                      f"ee={r['ee_dist']:.3f} ({r['minutes']:.1f} min)", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)   # checkpoint as we go
            env.close()

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
