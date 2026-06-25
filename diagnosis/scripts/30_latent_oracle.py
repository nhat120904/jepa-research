"""Latent oracle — separates the PREDICTOR from the ENCODER representation.

The state oracle (scripts/29) removes both the predictor F and the encoder from
the planning loop, so a success there only says "the gap is the model" without
saying which half. This script isolates the predictor by giving the planner
PERFECT latent dynamics THROUGH THE REAL FROZEN ENCODER:

  * Encoder: the real frozen WM encoder (so the representation is exactly the one
    the WM planner uses — NOT idealised).
  * Dynamics: perfect. For each CEM action candidate we step the SIMULATOR, render
    the resulting frame, and ENCODE it — the true latent the encoder would produce
    for that action. The learned predictor F is never called.
  * Cost: the SAME upstream L2-in-latent distance to z_goal that scripts/18 uses
    (mean squared error on the final latent; α=0, no proprio term).
  * Planner / horizon / num_act_stepped / strict success: identical to scripts/18.

The ONLY thing swapped vs the WM `l2` arm is F → perfect latent dynamics. So:

  latent-oracle SUCCEEDS (while the F-based WM arms fail)
      → the encoder + the L2-in-latent cost are sufficient; the predictor F is the
        bottleneck. This is the predictor-vs-encoder separation, and it licenses
        the GT-supervised predictor fix (scripts/26 λ_msk>0) as the paper's method.
  latent-oracle FAILS (even with perfect latent dynamics)
      → the bottleneck is the ENCODER representation or the L2-in-DINO-latent cost
        geometry (the goal-latent distance has no usable minimum at task success),
        NOT the predictor. A predictor fix cannot beat baseline; the solution must
        target the representation / cost.

Read together with scripts/29: state-oracle PASS + latent-oracle FAIL pins the
blame on the latent geometry / cost (planner and dynamics are both fine).

Because the upstream cost scores only the FINAL latent, each candidate needs just
one render + one encode (the encode is batched across the CEM population on GPU);
the renders are the per-step cost. Needs the real encoder, hence a GPU.

    python scripts/30_latent_oracle.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --tasks mw-reach mw-push mw-pick-place \
        --episodes 16 --out results/metaworld_latent_oracle.csv
"""

from __future__ import annotations

import argparse
import csv
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


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_cl = _load("closed_loop_eval", "18_closed_loop_eval.py")
_or = _load("oracle_ceiling", "29_oracle_ceiling.py")
make_env, rollout_expert = _cl.make_env, _cl.rollout_expert
render, encode_frame = _cl.render, _cl.encode_frame
FRAMESKIP, RAW_A = _cl.FRAMESKIP, _cl.RAW_A
snapshot, restore = _or.snapshot, _or.restore
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


@torch.no_grad()
def encode_batch(adapter, frames, proprios, device):
    """Encode a population of final frames in one GPU call. frames: list of
    (H,W,C) uint8-ish arrays; proprios: list of (4,) arrays. -> (B,V,H,W,D)."""
    vis = torch.stack([torch.from_numpy(f.copy()).permute(2, 0, 1).float() for f in frames])
    vis = vis.unsqueeze(1)                                          # (B,1,C,H,W)
    prop = None
    if adapter.uses_proprio():
        prop = torch.stack([torch.from_numpy(p.astype(np.float32)) for p in proprios]).unsqueeze(1)
        prop = prop.to(device)
    z = adapter.encode(vis.to(device), prop)
    return z[:, 0]                                                  # (B,V,H,W,D)


def roll_final_frame(env, snap, plan_raw):
    """Roll one candidate raw-action sequence on the sim from `snap`; return the
    FINAL rendered frame + its proprio (obs[:4]). Sim is rewound by the caller."""
    restore(env, snap)
    last = None
    for a in plan_raw:
        last, _, _, _, _ = env.step(np.clip(a, -1, 1))
    return render(env), last[:4].astype(np.float32), last


def cem_plan_latent(env, adapter, z_goal, device, *, plan_h, num_samples, iterations,
                    elite_frac, var0, rng):
    plan_raw_len = plan_h * FRAMESKIP
    dim = plan_raw_len * RAW_A
    mean = np.zeros(dim); var = np.full(dim, var0)
    n_elite = max(2, int(num_samples * elite_frac))
    snap = snapshot(env)
    zg = z_goal.reshape(-1)
    for _ in range(iterations):
        samples = np.clip(mean[None] + np.sqrt(var)[None] * rng.standard_normal((num_samples, dim)),
                          -1.0, 1.0)
        frames, props = [], []
        for i in range(num_samples):
            fr, pr, _ = roll_final_frame(env, snap, samples[i].reshape(plan_raw_len, RAW_A))
            frames.append(fr); props.append(pr)
        z_fin = encode_batch(adapter, frames, props, device)        # (B,V,H,W,D) — TRUE latent
        costs = ((z_fin.reshape(num_samples, -1) - zg[None]) ** 2).mean(-1).cpu().numpy()
        elites = samples[np.argsort(costs)[:n_elite]]
        mean = elites.mean(0); var = elites.var(0) + 1e-4
    restore(env, snap)
    return mean.reshape(plan_raw_len, RAW_A)


def run_episode(task, seed, env, goal_frame, goal_state, expert_succ, adapter, device, *,
                plan_h, num_act_stepped, max_episode_steps, cem_kw, strict):
    goal_obj = goal_state[OBJECT_SLICE]
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))                     # upstream reset_warmup
    success, last_success, steps = False, False, 0
    rng = np.random.default_rng(seed)
    while steps < max_episode_steps:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_latent(env, adapter, z_goal, device, plan_h=plan_h_eff, rng=rng, **cem_kw)
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
    return {"task": task, "arm": "latent_oracle", "seed": seed, "success": int(success),
            "success_end": int(last_success), "steps": steps,
            "final_state_dist": float(np.linalg.norm(obs - goal_state)),
            "ee_dist": float(np.linalg.norm(obs[:3] - goal_state[:3])),
            "obj_goal_dist": float(np.linalg.norm(obs[OBJECT_SLICE] - goal_obj)),
            "expert_success_step": expert_succ}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mw-reach", "mw-push", "mw-pick-place"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--cem-num-samples", type=int, default=100)
    ap.add_argument("--cem-iterations", type=int, default=6)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--strict-success", action="store_true")
    ap.add_argument("--out", default="results/metaworld_latent_oracle.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    cem_kw = dict(num_samples=args.cem_num_samples, iterations=args.cem_iterations,
                  elite_frac=args.elite_frac, var0=args.var0)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "arm", "seed", "success", "success_end", "steps",
              "final_state_dist", "ee_dist", "obj_goal_dist", "expert_success_step"]
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for task in args.tasks:
            for e in range(args.episodes):
                seed = args.seed0 + e
                t0 = time.time()
                env, init_state = make_env(task, seed)
                goal_frame, goal_state, expert_succ = rollout_expert(env, init_state, task)
                r = run_episode(task, seed, env, goal_frame, goal_state, expert_succ,
                                adapter, device, plan_h=args.horizon,
                                num_act_stepped=args.num_act_stepped,
                                max_episode_steps=args.max_episode_steps, cem_kw=cem_kw,
                                strict=args.strict_success)
                env.close()
                w.writerow(r); f.flush(); rows.append(r)
                print(f"  {task:16s} ep{e:02d} latent_oracle success={r['success']} "
                      f"end={r['success_end']} obj_goal={r['obj_goal_dist']:.3f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n=== latent oracle: success by task ===")
    for task in args.tasks:
        tr = [r for r in rows if r["task"] == task]
        s = sum(r["success"] for r in tr); se = sum(r["success_end"] for r in tr)
        print(f"  {task:16s} success {s}/{len(tr)}   success_end {se}/{len(tr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
