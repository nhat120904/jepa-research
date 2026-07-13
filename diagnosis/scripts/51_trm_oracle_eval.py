"""Evaluate a TRM-style terminal selector with simulator-exact latent dynamics.

This runner intentionally imports (rather than modifies) the established oracle
dynamics harness in ``30_latent_oracle.py``.  Encoder, simulator rollout,
candidate sampler, CEM update, horizon, action execution, goal construction, and
success evaluation therefore remain fixed; only the terminal scalar is TRM
replacement or per-population standardized TRM+L2 hybrid.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util as ilu
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from models.heads.trajectory_reachability import (  # noqa: E402
    load_trajectory_reachability_metric,
    trm_terminal_cost,
)
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def _load_oracle_harness():
    spec = ilu.spec_from_file_location("latent_oracle_for_trm", ROOT / "scripts" / "30_latent_oracle.py")
    module = ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_episode(task, seed, env, goal_frame, goal_state, expert_success_step,
                adapter, metric, device, *, mode, hybrid_weight, plan_h,
                num_act_stepped, max_episode_steps, cem_kw, strict, harness):
    z_goal = harness.encode_frame(adapter, goal_frame, goal_state[:4], device)

    @torch.no_grad()
    def cost_fn(z_final):
        return trm_terminal_cost(
            metric, z_final, z_goal, mode=mode, hybrid_weight=hybrid_weight
        )

    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(harness.RAW_A))
    success = False
    success_end = False
    steps = 0
    rng = np.random.default_rng(seed)
    while steps < max_episode_steps:
        plan_h_eff = min(
            plan_h, max(1, -(-(max_episode_steps - steps) // harness.FRAMESKIP))
        )
        plan = harness.cem_plan_latent(
            env, adapter, z_goal, device, plan_h=plan_h_eff, rng=rng,
            cost_fn=cost_fn, **cem_kw,
        )
        for action in plan[:num_act_stepped * harness.FRAMESKIP]:
            obs, _, _, _, info = env.step(np.clip(action, -1, 1))
            steps += 1
            success_end = bool(info.get("success", 0) > 0.5)
            success = success or success_end
            if steps >= max_episode_steps:
                break
        if success and not strict:
            break
    goal_obj = goal_state[OBJECT_SLICE]
    arm = "latent_oracle_trm_replacement"
    if mode == "hybrid":
        arm = f"latent_oracle_trm_hybrid_w{hybrid_weight:g}"
    return {
        "task": task,
        "arm": arm,
        "seed": seed,
        "success": int(success),
        "success_end": int(success_end),
        "steps": steps,
        "final_state_dist": float(np.linalg.norm(obs - goal_state)),
        "ee_dist": float(np.linalg.norm(obs[:3] - goal_state[:3])),
        "obj_goal_dist": float(np.linalg.norm(obs[OBJECT_SLICE] - goal_obj)),
        "expert_success_step": expert_success_step,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trm-head", required=True)
    ap.add_argument("--mode", choices=("replacement", "hybrid"), required=True)
    ap.add_argument("--hybrid-weight", type=float, default=1.0)
    ap.add_argument("--tasks", nargs="+", default=("mw-push", "mw-pick-place"))
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=30000,
                    help="fresh simulator seeds, disjoint from development/confirmatory runs")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--cem-num-samples", type=int, default=100)
    ap.add_argument("--cem-iterations", type=int, default=6)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--strict-success", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.hybrid_weight < 0:
        raise SystemExit("--hybrid-weight must be nonnegative")
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    metric, metadata = load_trajectory_reachability_metric(args.trm_head, device)
    if metadata.get("model") != args.model:
        raise SystemExit(
            f"TRM checkpoint model mismatch: {metadata.get('model')} != {args.model}"
        )
    expected_gap = args.horizon * 5  # upstream MetaWorld harness FRAMESKIP is locked to five.
    if int(metadata.get("max_gap", -1)) != expected_gap:
        raise SystemExit(
            f"horizon mismatch: head max_gap={metadata.get('max_gap')} but evaluation "
            f"horizon implies {expected_gap}; retrain a horizon-matched head"
        )
    if metadata.get("test_used_for_selection") is not False:
        raise SystemExit("TRM checkpoint lacks the held-out-selection guarantee")
    harness = _load_oracle_harness()
    if harness.FRAMESKIP != 5:
        raise SystemExit(f"unexpected oracle harness FRAMESKIP={harness.FRAMESKIP}")

    cem_kw = {
        "num_samples": args.cem_num_samples,
        "iterations": args.cem_iterations,
        "elite_frac": args.elite_frac,
        "var0": args.var0,
        "planner": "cem",
        "mppi_beta": 5.0,
    }
    print(
        f"TRM-style model={args.model} head={args.trm_head} mode={args.mode} "
        f"hybrid_weight={args.hybrid_weight} head_seed={metadata.get('head_seed')} "
        f"split_sha256={metadata.get('manifest_sha256')} seed0={args.seed0}",
        flush=True,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task", "arm", "seed", "success", "success_end", "steps",
        "final_state_dist", "ee_dist", "obj_goal_dist", "expert_success_step",
        "model", "head_seed", "manifest_sha256", "trm_head",
    ]
    rows = []
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in args.tasks:
            for episode in range(args.episodes):
                seed = args.seed0 + episode
                start = time.time()
                env, init_state = harness.make_env(task, seed)
                goal_frame, goal_state, expert_step = harness.rollout_expert(
                    env, init_state, task
                )
                row = run_episode(
                    task, seed, env, goal_frame, goal_state, expert_step,
                    adapter, metric, device, mode=args.mode,
                    hybrid_weight=args.hybrid_weight, plan_h=args.horizon,
                    num_act_stepped=args.num_act_stepped,
                    max_episode_steps=args.max_episode_steps, cem_kw=cem_kw,
                    strict=args.strict_success, harness=harness,
                )
                env.close()
                row.update({
                    "model": args.model,
                    "head_seed": metadata.get("head_seed"),
                    "manifest_sha256": metadata.get("manifest_sha256"),
                    "trm_head": str(args.trm_head),
                })
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(
                    f"{task:16s} ep{episode:02d} {row['arm']} "
                    f"success={row['success']} obj_goal={row['obj_goal_dist']:.3f} "
                    f"minutes={(time.time()-start)/60:.1f}", flush=True,
                )
    for task in args.tasks:
        selected = [row for row in rows if row["task"] == task]
        print(f"{task}: success={sum(row['success'] for row in selected)}/{len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
