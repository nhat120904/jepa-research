"""Paired ACID-style planning baseline under learned and oracle dynamics.

For every held-out environment seed this runner evaluates the unchanged upstream
terminal latent-MSE cost and the ACID-augmented cost with identical CEM noise.
``learned`` uses the released predictor. ``oracle`` snapshot/restores MuJoCo and
encodes every true model-step boundary, so its trajectory dynamics are perfect
while its terminal cost remains the same frozen-encoder latent MSE.

Model/MuJoCo execution is heavy: run only through the Slurm wrapper.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from models.heads.acid_idm import (  # noqa: E402
    action_consistency_cost,
    acid_cost,
    load_acid_idm,
)
from planning.cem_planner import cem_plan  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_cl = _load("acid_closed_loop_helpers", "18_closed_loop_eval.py")
_or = _load("acid_oracle_helpers", "29_oracle_ceiling.py")
make_env, rollout_expert = _cl.make_env, _cl.rollout_expert
render, encode_frame, PlanAdapter = _cl.render, _cl.encode_frame, _cl._PlanAdapter
FRAMESKIP, RAW_A = _cl.FRAMESKIP, _cl.RAW_A
snapshot, restore = _or.snapshot, _or.restore


class ACIDPoolCost:
    """Stateful trajectory cost; records adaptive-scale diagnostics per CEM pool."""

    def __init__(self, idm, lambda_acid: float):
        self.idm = idm
        self.lambda_acid = float(lambda_acid)
        self.diagnostics: list[dict[str, float]] = []

    @torch.no_grad()
    def __call__(self, trajectory, actions, z_goal):
        bsz = trajectory.shape[0]
        goal = ((trajectory[:, -1].reshape(bsz, -1) - z_goal.reshape(1, -1)) ** 2).mean(-1)
        # The null is exactly the upstream terminal objective and skips the IDM.
        if self.lambda_acid == 0.0:
            return goal
        consistency = action_consistency_cost(trajectory, actions, self.idm)
        total, diag = acid_cost(goal, consistency, lambda_acid=self.lambda_acid)
        self.diagnostics.append({k: float(v.detach().cpu()) for k, v in diag.items()})
        return total

    def summary(self) -> dict[str, float]:
        if not self.diagnostics:
            return {
                "mean_sigma_goal": float("nan"),
                "mean_sigma_acid": float("nan"),
                "mean_acid_weight": 0.0 if self.lambda_acid == 0 else float("nan"),
                "n_cost_pools": 0,
            }
        return {
            "mean_sigma_goal": float(np.mean([d["sigma_goal"] for d in self.diagnostics])),
            "mean_sigma_acid": float(np.mean([d["sigma_acid"] for d in self.diagnostics])),
            "mean_acid_weight": float(np.mean([d["acid_weight"] for d in self.diagnostics])),
            "n_cost_pools": len(self.diagnostics),
        }


@torch.no_grad()
def encode_batch(adapter, frames, proprios, device):
    vis = torch.stack(
        [torch.from_numpy(frame.copy()).permute(2, 0, 1).float() for frame in frames]
    ).unsqueeze(1)
    prop = None
    if adapter.uses_proprio():
        prop = torch.stack(
            [torch.from_numpy(np.asarray(p, dtype=np.float32)) for p in proprios]
        ).unsqueeze(1).to(device)
    return adapter.encode(vis.to(device), prop)[:, 0]


def _episode_result(task, seed, dynamics, arm, success, success_end, steps, obs,
                    goal_state, expert_succ, cost_recorder):
    row = {
        "task": task,
        "seed": seed,
        "dynamics": dynamics,
        "arm": arm,
        "success": int(success),
        "success_end": int(success_end),
        "steps": steps,
        "final_state_dist": float(np.linalg.norm(obs - goal_state)),
        "ee_dist": float(np.linalg.norm(obs[:3] - goal_state[:3])),
        "obj_goal_dist": float(
            np.linalg.norm(obs[OBJECT_SLICE] - goal_state[OBJECT_SLICE])
        ),
        "expert_success_step": expert_succ,
        "lambda_acid": cost_recorder.lambda_acid,
    }
    row.update(cost_recorder.summary())
    return row


def run_learned(task, seed, env, goal_frame, goal_state, expert_succ, adapter,
                device, idm, *, lambda_acid, horizon, num_act_stepped,
                max_episode_steps, cem_kw, strict):
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    plan_adapter = PlanAdapter(adapter)
    recorder = ACIDPoolCost(idm, lambda_acid)
    success = success_end = False
    steps = 0
    while steps < max_episode_steps:
        z_t = encode_frame(adapter, render(env), obs[:4], device)
        prop = torch.from_numpy(obs[:4].astype(np.float32)).to(device)
        plan_h = min(horizon, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan(
            plan_adapter, z_t, z_goal,
            horizon=plan_h, action_dim=RAW_A * FRAMESKIP,
            num_act_stepped=min(num_act_stepped, plan_h), proprio_t=prop,
            generator=torch.Generator(device=device).manual_seed(seed * 1000 + steps),
            traj_cost_fn=recorder, **cem_kw,
        )
        for action in plan.reshape(-1, RAW_A).cpu().numpy():
            obs, _, _, _, info = env.step(np.clip(action, -1, 1))
            steps += 1
            success_end = bool(info.get("success", 0) > 0.5)
            success = success or success_end
            if steps >= max_episode_steps:
                break
        if success and not strict:
            break
    return _episode_result(
        task, seed, "learned", "acid" if lambda_acid else "terminal",
        success, success_end, steps, obs, goal_state, expert_succ, recorder,
    )


@torch.no_grad()
def oracle_population_trajectory(env, snap, adapter, z_init, samples, device):
    """True latent trajectory for a candidate pool, with one batch encode per step."""
    n, horizon, action_dim = samples.shape
    if action_dim != RAW_A * FRAMESKIP:
        raise ValueError("oracle planner action dimension must be RAW_A*FRAMESKIP")
    frames_by_step = [[] for _ in range(horizon)]
    props_by_step = [[] for _ in range(horizon)]
    for i in range(n):
        restore(env, snap)
        for t in range(horizon):
            obs = None
            for action in samples[i, t].reshape(FRAMESKIP, RAW_A):
                obs, _, _, _, _ = env.step(np.clip(action, -1, 1))
            frames_by_step[t].append(render(env))
            props_by_step[t].append(obs[:4].astype(np.float32))
    encoded = [
        encode_batch(adapter, frames_by_step[t], props_by_step[t], device)
        for t in range(horizon)
    ]
    z0 = z_init.unsqueeze(0).expand(n, *z_init.shape)
    restore(env, snap)
    return torch.stack([z0, *encoded], dim=1)


def cem_plan_oracle_acid(env, adapter, z_init, z_goal, device, recorder, *,
                         horizon, num_samples, iterations, num_elites,
                         var_scale, rng):
    action_dim = RAW_A * FRAMESKIP
    mean = np.zeros((horizon, action_dim), dtype=np.float64)
    std = np.full_like(mean, var_scale)
    snap = snapshot(env)
    for _ in range(iterations):
        samples = mean[None] + std[None] * rng.standard_normal(
            (num_samples, horizon, action_dim)
        )
        samples = np.clip(samples, -1.0, 1.0)
        samples[0] = mean
        trajectory = oracle_population_trajectory(
            env, snap, adapter, z_init, samples, device
        )
        actions = torch.as_tensor(samples, dtype=torch.float32, device=device)
        costs = recorder(trajectory, actions, z_goal).detach().cpu().numpy()
        elites = samples[np.argsort(costs)[:num_elites]]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-4
    restore(env, snap)
    return mean


def run_oracle(task, seed, env, goal_frame, goal_state, expert_succ, adapter,
               device, idm, *, lambda_acid, horizon, num_act_stepped,
               max_episode_steps, cem_kw, strict):
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    recorder = ACIDPoolCost(idm, lambda_acid)
    rng = np.random.default_rng(seed)
    success = success_end = False
    steps = 0
    while steps < max_episode_steps:
        z_t = encode_frame(adapter, render(env), obs[:4], device)
        plan_h = min(horizon, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_oracle_acid(
            env, adapter, z_t, z_goal, device, recorder,
            horizon=plan_h, rng=rng, **cem_kw,
        )
        for action in plan[:num_act_stepped].reshape(-1, RAW_A):
            obs, _, _, _, info = env.step(np.clip(action, -1, 1))
            steps += 1
            success_end = bool(info.get("success", 0) > 0.5)
            success = success or success_end
            if steps >= max_episode_steps:
                break
        if success and not strict:
            break
    return _episode_result(
        task, seed, "oracle", "acid" if lambda_acid else "terminal",
        success, success_end, steps, obs, goal_state, expert_succ, recorder,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--idm", required=True)
    ap.add_argument("--dynamics", nargs="+", choices=["learned", "oracle"],
                    default=["learned", "oracle"])
    ap.add_argument("--tasks", nargs="+", default=["mw-push", "mw-pick-place"])
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=22000)
    ap.add_argument("--lambda-acid", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--cem-num-samples", type=int, default=100)
    ap.add_argument("--cem-iterations", type=int, default=6)
    ap.add_argument("--num-elites", type=int, default=10)
    ap.add_argument("--var-scale", type=float, default=1.0)
    ap.add_argument("--strict-success", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.lambda_acid < 0:
        raise SystemExit("--lambda-acid must be non-negative")
    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "8")))
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    idm, idm_meta = load_acid_idm(args.idm, device)
    if idm_meta.get("model") != args.model:
        raise ValueError(
            f"IDM was trained for {idm_meta.get('model')}, not {args.model}"
        )
    if int(idm_meta.get("frames_per_step", -1)) != int(adapter.frames_per_step):
        raise ValueError("IDM/checkpoint frames_per_step mismatch")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if out.exists():
        rows = pd.read_csv(out).to_dict("records")
    required = {(d, a) for d in args.dynamics for a in ("terminal", "acid")}
    completed = set()
    if rows:
        frame = pd.DataFrame(rows)
        for (task, seed), group in frame.groupby(["task", "seed"]):
            if required <= set(zip(group["dynamics"], group["arm"])):
                completed.add((str(task), int(seed)))
        rows = [r for r in rows if (str(r["task"]), int(r["seed"])) in completed]

    learned_kw = dict(
        num_samples=args.cem_num_samples, iterations=args.cem_iterations,
        num_elites=args.num_elites, var_scale=args.var_scale,
        max_norms=[1.0], max_norm_dims=[list(range(RAW_A * FRAMESKIP))],
    )
    oracle_kw = dict(
        num_samples=args.cem_num_samples, iterations=args.cem_iterations,
        num_elites=args.num_elites, var_scale=args.var_scale,
    )
    print(
        f"ACID approximation={idm_meta.get('architecture')} model={args.model} "
        f"lambda={args.lambda_acid} manifest_hash={idm_meta.get('split_manifest_sha256')} "
        f"seeds={args.seed0}:{args.seed0 + args.episodes - 1}", flush=True,
    )
    for task in args.tasks:
        for episode in range(args.episodes):
            seed = args.seed0 + episode
            if (task, seed) in completed:
                continue
            env, init_state = make_env(task, seed)
            goal_frame, goal_state, expert_succ = rollout_expert(env, init_state, task)
            pair_rows = []
            for dynamics in args.dynamics:
                for arm, lam in (("terminal", 0.0), ("acid", args.lambda_acid)):
                    start = time.time()
                    common = dict(
                        task=task, seed=seed, env=env, goal_frame=goal_frame,
                        goal_state=goal_state, expert_succ=expert_succ,
                        adapter=adapter, device=device, idm=idm, lambda_acid=lam,
                        horizon=args.horizon, num_act_stepped=args.num_act_stepped,
                        max_episode_steps=args.max_episode_steps,
                        strict=args.strict_success,
                    )
                    if dynamics == "learned":
                        row = run_learned(**common, cem_kw=learned_kw)
                    else:
                        row = run_oracle(**common, cem_kw=oracle_kw)
                    row["minutes"] = round((time.time() - start) / 60, 3)
                    row["idm_architecture"] = idm_meta.get("architecture")
                    row["idm_split_sha256"] = idm_meta.get("split_manifest_sha256")
                    pair_rows.append(row)
                    print(
                        f"{task} seed={seed} {dynamics}/{arm}: "
                        f"success_end={row['success_end']} obj={row['obj_goal_dist']:.3f} "
                        f"({row['minutes']:.2f} min)", flush=True,
                    )
            rows.extend(pair_rows)
            pd.DataFrame(rows).to_csv(out, index=False)
            env.close()
    print(f"wrote {out} ({len(rows)} paired rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
