"""Oracle planning ceiling — the decisive go/no-go for the model-side fix.

Every world-model arm in scripts/18 is 0/16 on the contact tasks (mw-push,
mw-pick-place). Before betting the paper on a *predictor* fix we must know WHERE
the 0/16 comes from:

  * If goal-conditioned MPC with PERFECT dynamics + PERFECT object state ALSO
    fails → the bottleneck is the planning protocol (CEM budget / horizon /
    goal-distance cost / success radius), not the learned model. No predictor
    fix can beat the baseline until that is repaired, and the paper's "solution"
    must target planning/representation, not the predictor.
  * If the oracle SUCCEEDS while the WM arms fail → contact tasks are solvable
    under this exact protocol and the gap is the model's. There is real headroom
    to close, and the GT-supervised predictor fix (scripts/26 with λ_msk>0) is
    the right lever.

This is an APPLES-TO-APPLES ceiling for the scripts/18 planner: same env
(make_env, frozen rand_vec), same expert goal (rollout_expert's final frame →
goal object position), same horizon / num_act_stepped / max_episode_steps and
the same strict end-of-episode success judging. The ONLY thing swapped is the
forward model: instead of unrolling the latent predictor and scoring an
L2-in-latent cost, CEM rolls the MuJoCo simulator itself and scores the TRUE
object→goal distance (+ a hand→object approach term, mirroring hexp). No
encoder, no adapter, no checkpoints — pure privileged MPC.

    python scripts/29_oracle_ceiling.py --config configs/diagnostic_metaworld.yaml \
        --tasks mw-reach mw-push mw-pick-place --episodes 16 \
        --out results/metaworld_oracle_ceiling.csv

Output: results/metaworld_oracle_ceiling.csv (per-episode rows; success =
any-step latch, success_end = env flag at episode end, matching scripts/18).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the EXACT env construction + expert goal protocol from the WM eval so the
# ceiling is comparable arm-for-arm. The module name starts with a digit, so load
# it by path (no model/adapter is touched — only make_env / rollout_expert).
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "closed_loop_eval", str(ROOT / "scripts" / "18_closed_loop_eval.py"))
_cl = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cl)
make_env, rollout_expert = _cl.make_env, _cl.rollout_expert
FRAMESKIP, RAW_A = _cl.FRAMESKIP, _cl.RAW_A
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402

HAND_SLICE = slice(0, 3)


def snapshot(env) -> dict:
    """Full MuJoCo physics snapshot so a CEM sample can be rolled and rewound.
    qpos/qvel are the dynamic state; mocap_pos/quat + ctrl are the Sawyer
    controller targets the next env.step() integrates from; curr_path_length is
    metaworld's internal step counter (drives truncation/success bookkeeping)."""
    d = env.data
    return {
        "qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
        "mocap_pos": d.mocap_pos.copy(), "mocap_quat": d.mocap_quat.copy(),
        "ctrl": d.ctrl.copy(),
        "act": d.act.copy() if d.act.size else None,
        "cpl": int(getattr(env, "curr_path_length", 0)),
    }


def restore(env, s: dict) -> None:
    import mujoco
    d = env.data
    d.qpos[:] = s["qpos"]; d.qvel[:] = s["qvel"]
    d.mocap_pos[:] = s["mocap_pos"]; d.mocap_quat[:] = s["mocap_quat"]
    d.ctrl[:] = s["ctrl"]
    if s["act"] is not None:
        d.act[:] = s["act"]
    if hasattr(env, "curr_path_length"):
        env.curr_path_length = s["cpl"]
    mujoco.mj_forward(env.model, env.data)


def _obj_hand(obs: np.ndarray):
    return obs[OBJECT_SLICE], obs[HAND_SLICE]


def roll_cost(env, snap, plan_raw: np.ndarray, goal_obj: np.ndarray, w_hand: float):
    """Roll one candidate raw-action sequence on the sim from `snap`; cost is the
    final object→goal distance plus a dense hand→object approach term (so CEM gets
    signal before the object first moves). Sim is rewound by the caller."""
    restore(env, snap)
    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else None
    last = None
    for a in plan_raw:
        obs, _, _, _, _ = env.step(np.clip(a, -1, 1))
        last = obs
    obj, hand = _obj_hand(last)
    return float(np.linalg.norm(obj - goal_obj) + w_hand * np.linalg.norm(hand - obj))


def cem_plan_sim(env, goal_obj, *, plan_h, num_samples, iterations, elite_frac,
                 var0, w_hand, rng):
    plan_raw_len = plan_h * FRAMESKIP
    dim = plan_raw_len * RAW_A
    mean = np.zeros(dim, dtype=np.float64)
    var = np.full(dim, var0, dtype=np.float64)
    n_elite = max(2, int(num_samples * elite_frac))
    snap = snapshot(env)
    for _ in range(iterations):
        samples = mean[None] + np.sqrt(var)[None] * rng.standard_normal((num_samples, dim))
        samples = np.clip(samples, -1.0, 1.0)
        costs = np.empty(num_samples)
        for i in range(num_samples):
            costs[i] = roll_cost(env, snap, samples[i].reshape(plan_raw_len, RAW_A),
                                 goal_obj, w_hand)
        elites = samples[np.argsort(costs)[:n_elite]]
        mean = elites.mean(0)
        var = elites.var(0) + 1e-4
    restore(env, snap)                       # rewind to the real current state
    return mean.reshape(plan_raw_len, RAW_A)


def run_episode(task, seed, env, init_state, goal_state, expert_succ, *,
                plan_h, num_act_stepped, max_episode_steps, cem_kw, strict):
    goal_obj = goal_state[OBJECT_SLICE]
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))          # upstream reset_warmup
    success, last_success, steps = False, False, 0
    rng = np.random.default_rng(seed)
    while steps < max_episode_steps:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_sim(env, goal_obj, plan_h=plan_h_eff, rng=rng, **cem_kw)
        exec_raw = plan[: num_act_stepped * FRAMESKIP]
        for a in exec_raw:
            obs, _, _, _, info = env.step(np.clip(a, -1, 1))
            steps += 1
            last_success = info.get("success", 0) > 0.5
            if last_success:
                success = True
            if steps >= max_episode_steps:
                break
        if success and not strict:
            break
    return {"task": task, "arm": "oracle", "seed": seed, "success": int(success),
            "success_end": int(last_success), "steps": steps,
            "final_state_dist": float(np.linalg.norm(obs - goal_state)),
            "ee_dist": float(np.linalg.norm(obs[:3] - goal_state[:3])),
            "obj_goal_dist": float(np.linalg.norm(obs[OBJECT_SLICE] - goal_obj)),
            "expert_success_step": expert_succ}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)   # accepted for parity; unused (no model)
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
    ap.add_argument("--w-hand", type=float, default=0.5,
                    help="dense hand→object approach weight in the sim cost")
    ap.add_argument("--strict-success", action="store_true")
    ap.add_argument("--out", default="results/metaworld_oracle_ceiling.csv")
    args = ap.parse_args()

    cem_kw = dict(num_samples=args.cem_num_samples, iterations=args.cem_iterations,
                  elite_frac=args.elite_frac, var0=args.var0, w_hand=args.w_hand)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
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
                r = run_episode(task, seed, env, init_state, goal_state, expert_succ,
                                plan_h=args.horizon, num_act_stepped=args.num_act_stepped,
                                max_episode_steps=args.max_episode_steps, cem_kw=cem_kw,
                                strict=args.strict_success)
                env.close()
                w.writerow(r); f.flush(); rows.append(r)
                print(f"  {task:16s} ep{e:02d} oracle success={r['success']} "
                      f"end={r['success_end']} obj_goal={r['obj_goal_dist']:.3f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    print("\n=== oracle ceiling: success by task ===")
    for task in args.tasks:
        tr = [r for r in rows if r["task"] == task]
        s = sum(r["success"] for r in tr); se = sum(r["success_end"] for r in tr)
        print(f"  {task:16s} success {s}/{len(tr)}   success_end {se}/{len(tr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
