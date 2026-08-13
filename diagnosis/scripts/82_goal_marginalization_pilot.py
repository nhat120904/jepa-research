"""Gate 2 (goal-marginalization design, 2026-08-11) — causal arm-nuisance pilot.

Under perfect (oracle) latent dynamics through the real frozen encoder, same
CEM budget as scripts/30 `--cost l2`, compare three constructions of `z_goal`
on the SAME episode / SAME expert-derived goal state:

  baseline           - encode(expert's final frame). Reproduces the existing
                        l2 oracle arm exactly (push should still be ~0/16).
  arm_marginalized   - restore to the goal snapshot K times, each time step
                        the SIMULATOR forward with small random raw actions
                        (arm/gripper moves locally, object should not), render
                        + encode, then average the K latents. No hand-written
                        physics state anywhere - only real env.step() calls.
  noise_matched_control - same K-average, but the K "perturbed" latents are
                        the baseline latent plus i.i.d. Gaussian noise whose
                        per-episode scale matches the empirical RMS of the
                        arm_marginalized residuals. Same total noise energy,
                        unstructured instead of arm-structured. If this arm
                        matches arm_marginalized, generic averaging explains
                        any gain, not arm-nuisance removal.

See docs/plans/2026-08-11-goal-marginalization-design.md for the full
protocol, decision rule, and why K/n_pert/scale are locked before running.

    python scripts/82_goal_marginalization_pilot.py \
        --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
        --tasks mw-push --episodes 16 --seed0 90000 \
        --out results/goal_marginalization_mw-push_seed90000_n16.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util as _ilu
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_cl = _load("closed_loop_eval", "18_closed_loop_eval.py")
_or = _load("oracle_ceiling", "29_oracle_ceiling.py")
_lo = _load("latent_oracle", "30_latent_oracle.py")
make_env, rollout_expert = _cl.make_env, _cl.rollout_expert
render, encode_frame = _cl.render, _cl.encode_frame
FRAMESKIP, RAW_A = _cl.FRAMESKIP, _cl.RAW_A
snapshot, restore = _or.snapshot, _or.restore
encode_batch = _lo.encode_batch
build_oracle_cost = _lo.build_oracle_cost
cem_plan_latent = _lo.cem_plan_latent

ARMS = ["baseline", "arm_marginalized", "noise_matched_control"]
OBJ_CLEAN_THRESHOLD_M = 0.01   # 1 cm; 1/5 of the 5 cm success radius


def _stable_seed(base: int, key: tuple[object, ...]) -> int:
    digest = hashlib.blake2b("|".join(map(str, key)).encode("utf-8"), digest_size=8).digest()
    return (base + int.from_bytes(digest, "little")) % (2**63 - 1)


@torch.no_grad()
def build_zgoal_variants(env, adapter, snap_goal, goal_frame, goal_state, device, *,
                         seed, K, n_pert, pert_sigma, pert_bound):
    """Return dict arm -> (z_goal (frame-shaped tensor), diagnostics dict)."""
    z_base = encode_frame(adapter, goal_frame, goal_state[:4], device)

    frames_k, props_k, obj_disp = [], [], []
    for k in range(K):
        restore(env, snap_goal)
        rng_k = np.random.default_rng(_stable_seed(seed, ("arm_pert", k)))
        obs = None
        for _ in range(n_pert):
            a = np.clip(rng_k.normal(0.0, pert_sigma, RAW_A), -pert_bound, pert_bound)
            obs, _, _, _, _ = env.step(a)
        frames_k.append(render(env))
        props_k.append(obs[:4].astype(np.float32))
        obj_disp.append(float(np.linalg.norm(obs[OBJECT_SLICE] - goal_state[OBJECT_SLICE])))
    restore(env, snap_goal)   # leave env clean for the caller's subsequent env.reset()

    z_k = encode_batch(adapter, frames_k, props_k, device)          # (K, *frame)
    z_arm_marg = z_k.mean(dim=0)

    resid = (z_k - z_base.unsqueeze(0)).reshape(K, -1)
    sigma = float(resid.float().std(unbiased=True).clamp_min(1e-8))
    rng_noise = np.random.default_rng(_stable_seed(seed, ("noise", 0)))
    noise = rng_noise.normal(0.0, sigma, size=(K,) + tuple(z_base.shape)).astype(np.float32)
    z_noise_k = z_base.unsqueeze(0) + torch.from_numpy(noise).to(z_base.device)
    z_noise_marg = z_noise_k.mean(dim=0)

    diag = {
        "K": K, "obj_disp_median_m": float(np.median(obj_disp)),
        "obj_disp_max_m": float(np.max(obj_disp)),
        "arm_marginalization_clean": int(np.median(obj_disp) <= OBJ_CLEAN_THRESHOLD_M),
        "noise_sigma": sigma,
    }
    return {"baseline": z_base, "arm_marginalized": z_arm_marg,
            "noise_matched_control": z_noise_marg}, diag


def run_episode_for_zgoal(env, adapter, device, z_goal, arm_name, task, seed, *,
                          plan_h, num_act_stepped, max_episode_steps, cem_kw, strict,
                          goal_state, expert_succ):
    goal_obj = goal_state[OBJECT_SLICE]
    cost_fn = build_oracle_cost("l2", z_goal)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))
    success, last_success, steps = False, False, 0
    rng = np.random.default_rng(_stable_seed(seed, (arm_name, "plan")))
    while steps < max_episode_steps:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_latent(env, adapter, z_goal, device, plan_h=plan_h_eff, rng=rng,
                               cost_fn=cost_fn, **cem_kw)
        for a in plan[: num_act_stepped * FRAMESKIP]:
            obs, _, _, _, info = env.step(np.clip(a, -1, 1))
            steps += 1
            last_success = info.get("success", 0) > 0.5
            if last_success:
                success = True
            if steps >= max_episode_steps:
                break
        if success and not strict:
            break
    return {"task": task, "arm": arm_name, "seed": seed,
            "success": int(success), "success_end": int(last_success), "steps": steps,
            "obj_goal_dist": float(np.linalg.norm(obs[OBJECT_SLICE] - goal_obj)),
            "expert_success_step": expert_succ}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="dino_wm_metaworld")
    ap.add_argument("--tasks", nargs="+", default=["mw-push"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=90000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--cem-num-samples", type=int, default=100)
    ap.add_argument("--cem-iterations", type=int, default=6)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=8, help="perturbation/noise samples averaged")
    ap.add_argument("--n-pert", type=int, default=10, help="raw steps per perturbation rollout")
    ap.add_argument("--pert-sigma", type=float, default=0.15)
    ap.add_argument("--pert-bound", type=float, default=0.3)
    ap.add_argument("--strict-success", action="store_true")
    ap.add_argument("--out", default="results/goal_marginalization_pilot.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()

    cem_kw = dict(num_samples=args.cem_num_samples, iterations=args.cem_iterations,
                  elite_frac=args.elite_frac, var0=args.var0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "arm", "seed", "success", "success_end", "steps", "obj_goal_dist",
              "expert_success_step", "obj_disp_median_m", "obj_disp_max_m",
              "arm_marginalization_clean"]
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for task in args.tasks:
            for e in range(args.episodes):
                seed = args.seed0 + e
                t0 = time.time()
                env, init_state = make_env(task, seed)
                goal_frame, goal_state, expert_succ = rollout_expert(env, init_state, task)
                snap_goal = snapshot(env)
                zvars, diag = build_zgoal_variants(
                    env, adapter, snap_goal, goal_frame, goal_state, device,
                    seed=seed, K=args.K, n_pert=args.n_pert,
                    pert_sigma=args.pert_sigma, pert_bound=args.pert_bound)
                for arm_name in ARMS:
                    r = run_episode_for_zgoal(
                        env, adapter, device, zvars[arm_name], arm_name, task, seed,
                        plan_h=args.horizon, num_act_stepped=args.num_act_stepped,
                        max_episode_steps=args.max_episode_steps, cem_kw=cem_kw,
                        strict=args.strict_success, goal_state=goal_state,
                        expert_succ=expert_succ)
                    r.update(obj_disp_median_m=diag["obj_disp_median_m"],
                             obj_disp_max_m=diag["obj_disp_max_m"],
                             arm_marginalization_clean=diag["arm_marginalization_clean"])
                    w.writerow(r); f.flush(); rows.append(r)
                    print(f"  {task:12s} seed={seed} {arm_name:24s} "
                          f"success_end={r['success_end']} obj_goal={r['obj_goal_dist']:.3f} "
                          f"clean={diag['arm_marginalization_clean']} "
                          f"disp_med={diag['obj_disp_median_m']*100:.2f}cm", flush=True)
                env.close()
                print(f"  -- episode wall time {(time.time()-t0)/60:.1f} min", flush=True)

    print("\n=== goal-marginalization pilot: per-arm summary ===")
    for task in args.tasks:
        for arm_name in ARMS:
            tr = [r for r in rows if r["task"] == task and r["arm"] == arm_name]
            s = sum(r["success_end"] for r in tr)
            print(f"  {task:12s} {arm_name:24s} success_end {s}/{len(tr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
