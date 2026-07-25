"""Shared-noise branched-CEM audit under simulator-perfect dynamics.

This experiment separates two ways a representation-induced cost can hurt CEM:

1. immediate selection: on the *same* initial candidate population, the proxy
   selects candidates with worse simulator-state cost than the state oracle;
2. proposal feedback: after the first elite refit, proxy- and oracle-driven CEM
   visit different candidate distributions, even when they use common Gaussian
   noise at every iteration.

At each MPC snapshot every branch starts from the same zero-mean, ``var0``
Gaussian.  Iteration zero is therefore candidate-for-candidate identical across
branches.  Later iterations use the same standard-normal ``eps`` tensor, but
each branch applies it to its own refitted mean and variance.  Every candidate
is rolled in MuJoCo from the same snapshot, encoded by the real frozen encoder,
and scored by *all* of: state-probe cost, simulator-state cost, and latent L2.
The learned dynamics predictor is never called.

The carrier branch only chooses which branch mean is executed to obtain the
next MPC snapshot.  It does not alter the paired within-snapshot comparison.
For the cleanest contact-state audit use ``--carrier true_state``.

Heavy execution requires a GPU compute node; do not run this script on a login
node.  It writes an aligned gzip candidate CSV, an iteration CSV, and a compact
carrier-episode CSV.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters import build_adapter  # noqa: E402
from models.probes import load_probe  # noqa: E402
from stratification.metaworld_regimes import EE_SLICE, OBJECT_SLICE  # noqa: E402


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lo = _load("latent_oracle_shared_branch", "30_latent_oracle.py")
make_env, rollout_expert, encode_frame = _lo.make_env, _lo.rollout_expert, _lo.encode_frame
snapshot, restore = _lo.snapshot, _lo.restore
roll_final_frame, encode_batch = _lo.roll_final_frame, _lo.encode_batch
build_oracle_cost = _lo.build_oracle_cost
FRAMESKIP, RAW_A = _lo.FRAMESKIP, _lo.RAW_A

ALL_COSTS = ("stateprobe", "true_state", "l2")

CANDIDATE_FIELDS = [
    "task", "seed", "replan", "iter", "generating_branch", "candidate",
    "action_hash", "noise_hash", "action_l2", "action_linf",
    "stateprobe_cost", "true_state_cost", "l2_cost",
    "obj_goal_dist", "ee_goal_dist", "hand_obj_dist",
    "obj_decode_error_cm", "ee_decode_error_cm", "stateprobe_optimism_m",
    "success_any", "success_end",
    "elite_by_stateprobe", "elite_by_true_state", "elite_by_l2",
    "argmin_by_stateprobe", "argmin_by_true_state", "argmin_by_l2",
]

ITERATION_FIELDS = [
    "task", "seed", "replan", "iter", "generating_branch", "n_candidates",
    "n_elite", "noise_hash", "identical_to_first_branch",
    "pre_mean_l2", "pre_var_mean", "paired_action_rmse_vs_first_branch",
    "post_mean_l2", "post_var_mean", "post_mean_l2_vs_first_branch",
    "population_obj_decode_error_mean_cm", "population_obj_decode_error_median_cm",
    "population_obj_decode_within5",
    "own_elite_obj_decode_error_mean_cm", "own_elite_obj_decode_error_median_cm",
    "own_elite_obj_decode_within5", "true_elite_obj_decode_error_mean_cm",
    "true_elite_obj_decode_error_median_cm", "true_elite_obj_decode_within5",
    "own_selected_true_state_cost", "best_true_state_cost", "own_selected_true_regret",
    "stateprobe_selected_true_state_cost", "stateprobe_selected_true_regret",
    "true_selected_stateprobe_cost", "best_stateprobe_cost",
    "stateprobe_true_spearman", "stateprobe_true_topk_overlap",
    "coverage_success_any", "coverage_success_end", "n_success_any", "n_success_end",
]

EPISODE_FIELDS = [
    "task", "seed", "carrier", "success", "success_end", "steps", "replans",
    "final_obj_goal_dist", "final_ee_goal_dist", "expert_success_step",
]


def _hash_array(value: np.ndarray) -> str:
    canonical = np.asarray(value, dtype="<f4", order="C")
    return hashlib.blake2b(canonical.tobytes(), digest_size=8).hexdigest()


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks without scipy; ties are uncommon but handled exactly."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ai = set(np.argsort(a, kind="mergesort")[:k].tolist())
    bi = set(np.argsort(b, kind="mergesort")[:k].tolist())
    return len(ai & bi) / k


def _finite_summary(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    return float(values.mean()), float(np.median(values)), float(np.mean(values < 5.0))


@torch.no_grad()
def _score_population(
    env,
    snap,
    samples: np.ndarray,
    adapter,
    device,
    stateprobe_fn,
    l2_fn,
    probe,
    ee_probe,
    goal_obj: np.ndarray,
    goal_ee: np.ndarray,
    task: str,
    w_hand: float,
):
    """Roll once, then dual-score every candidate with proxy and sim truth."""
    frames, props, raws, success_any, success_end = [], [], [], [], []
    for sample in samples:
        frame, prop, raw, succ_any, succ_end = roll_final_frame(
            env, snap, sample, return_success=True)
        frames.append(frame)
        props.append(prop)
        raws.append(raw)
        success_any.append(succ_any)
        success_end.append(succ_end)
    z_fin = encode_batch(adapter, frames, props, device)
    raw = np.asarray(raws, dtype=np.float32)
    obj, ee = raw[:, OBJECT_SLICE], raw[:, EE_SLICE]
    obj_goal = np.linalg.norm(obj - goal_obj[None], axis=-1)
    ee_goal = np.linalg.norm(ee - goal_ee[None], axis=-1)
    hand_obj = np.linalg.norm(ee - obj, axis=-1)
    true_cost = ee_goal if task.startswith("mw-reach") else obj_goal + w_hand * hand_obj
    stateprobe_cost = stateprobe_fn(z_fin).detach().cpu().numpy().astype(float)
    l2_cost = l2_fn(z_fin).detach().cpu().numpy().astype(float)
    decoded_obj = probe(z_fin).detach().cpu().numpy()
    decoded_ee = ee_probe(z_fin).detach().cpu().numpy()
    obj_decode_cm = 100.0 * np.linalg.norm(decoded_obj - obj, axis=-1)
    ee_decode_cm = 100.0 * np.linalg.norm(decoded_ee - ee, axis=-1)
    return {
        "stateprobe": stateprobe_cost,
        "true_state": true_cost.astype(float),
        "l2": l2_cost,
        "obj_goal": obj_goal,
        "ee_goal": ee_goal,
        "hand_obj": hand_obj,
        "obj_decode_cm": obj_decode_cm,
        "ee_decode_cm": ee_decode_cm,
        "stateprobe_optimism": true_cost - stateprobe_cost,
        "success_any": np.asarray(success_any, dtype=bool),
        "success_end": np.asarray(success_end, dtype=bool),
    }


def branched_cem(
    env,
    adapter,
    device,
    z_goal,
    goal_obj,
    goal_ee,
    task,
    *,
    probe,
    ee_probe,
    branches,
    plan_h,
    num_samples,
    iterations,
    elite_frac,
    var0,
    w_hand,
    rng,
    seed,
    replan,
    candidate_writer,
    iteration_writer,
):
    """Run paired branches from one snapshot and return each branch mean plan."""
    plan_raw_len = plan_h * FRAMESKIP
    dim = plan_raw_len * RAW_A
    n_elite = max(2, int(num_samples * elite_frac))
    branch_state = {
        name: {
            "mean": np.zeros(dim, dtype=np.float64),
            "var": np.full(dim, var0, dtype=np.float64),
        }
        for name in branches
    }
    eps = rng.standard_normal((iterations, num_samples, dim))
    stateprobe_fn = build_oracle_cost(
        "stateprobe", z_goal, probe=probe, ee_probe=ee_probe, w_hand=w_hand)
    l2_fn = build_oracle_cost("l2", z_goal)
    snap = snapshot(env)

    try:
        for iteration in range(iterations):
            generated = {}
            first_branch = branches[0]
            for branch in branches:
                state = branch_state[branch]
                samples_flat = np.clip(
                    state["mean"][None]
                    + np.sqrt(state["var"])[None] * eps[iteration],
                    -1.0,
                    1.0,
                )
                generated[branch] = samples_flat

            # Iteration zero must be exactly shared, not merely statistically paired.
            if iteration == 0:
                for branch in branches[1:]:
                    if not np.array_equal(generated[first_branch], generated[branch]):
                        raise RuntimeError("iteration-0 branch populations are not identical")

            first_samples = generated[first_branch]
            first_post_mean = None
            scored_first = None
            for branch in branches:
                state = branch_state[branch]
                samples_flat = generated[branch]
                samples = samples_flat.reshape(num_samples, plan_raw_len, RAW_A)
                # Avoid duplicate MuJoCo/encoder work for the exactly shared iter-0 pool.
                if iteration == 0 and scored_first is not None:
                    scored = scored_first
                else:
                    scored = _score_population(
                        env, snap, samples, adapter, device, stateprobe_fn, l2_fn,
                        probe, ee_probe, goal_obj, goal_ee, task, w_hand)
                    if iteration == 0:
                        scored_first = scored

                orders = {
                    cost: np.argsort(scored[cost], kind="mergesort")
                    for cost in ALL_COSTS
                }
                elites = {cost: order[:n_elite] for cost, order in orders.items()}
                own_elite = elites[branch]
                post_mean = samples_flat[own_elite].mean(axis=0)
                post_var = samples_flat[own_elite].var(axis=0) + 1e-4
                paired_rmse = float(np.sqrt(np.mean((samples_flat - first_samples) ** 2)))
                if first_post_mean is None:
                    first_post_mean = post_mean
                post_mean_delta = float(np.linalg.norm(post_mean - first_post_mean))
                noise_hash = _hash_array(eps[iteration])

                true_order = orders["true_state"]
                stateprobe_order = orders["stateprobe"]
                own_order = orders[branch]
                pop_err = _finite_summary(scored["obj_decode_cm"])
                own_err = _finite_summary(scored["obj_decode_cm"][own_elite])
                true_err = _finite_summary(scored["obj_decode_cm"][elites["true_state"]])
                iteration_writer.writerow({
                    "task": task,
                    "seed": seed,
                    "replan": replan,
                    "iter": iteration,
                    "generating_branch": branch,
                    "n_candidates": num_samples,
                    "n_elite": n_elite,
                    "noise_hash": noise_hash,
                    "identical_to_first_branch": int(np.array_equal(samples_flat, first_samples)),
                    "pre_mean_l2": float(np.linalg.norm(state["mean"])),
                    "pre_var_mean": float(state["var"].mean()),
                    "paired_action_rmse_vs_first_branch": paired_rmse,
                    "post_mean_l2": float(np.linalg.norm(post_mean)),
                    "post_var_mean": float(post_var.mean()),
                    "post_mean_l2_vs_first_branch": post_mean_delta,
                    "population_obj_decode_error_mean_cm": pop_err[0],
                    "population_obj_decode_error_median_cm": pop_err[1],
                    "population_obj_decode_within5": pop_err[2],
                    "own_elite_obj_decode_error_mean_cm": own_err[0],
                    "own_elite_obj_decode_error_median_cm": own_err[1],
                    "own_elite_obj_decode_within5": own_err[2],
                    "true_elite_obj_decode_error_mean_cm": true_err[0],
                    "true_elite_obj_decode_error_median_cm": true_err[1],
                    "true_elite_obj_decode_within5": true_err[2],
                    "own_selected_true_state_cost": float(scored["true_state"][own_order[0]]),
                    "best_true_state_cost": float(scored["true_state"][true_order[0]]),
                    "own_selected_true_regret": float(
                        scored["true_state"][own_order[0]] - scored["true_state"][true_order[0]]),
                    "stateprobe_selected_true_state_cost": float(
                        scored["true_state"][stateprobe_order[0]]),
                    "stateprobe_selected_true_regret": float(
                        scored["true_state"][stateprobe_order[0]]
                        - scored["true_state"][true_order[0]]),
                    "true_selected_stateprobe_cost": float(
                        scored["stateprobe"][true_order[0]]),
                    "best_stateprobe_cost": float(scored["stateprobe"][stateprobe_order[0]]),
                    "stateprobe_true_spearman": _spearman(
                        scored["stateprobe"], scored["true_state"]),
                    "stateprobe_true_topk_overlap": _topk_overlap(
                        scored["stateprobe"], scored["true_state"], n_elite),
                    "coverage_success_any": int(scored["success_any"].any()),
                    "coverage_success_end": int(scored["success_end"].any()),
                    "n_success_any": int(scored["success_any"].sum()),
                    "n_success_end": int(scored["success_end"].sum()),
                })

                for candidate in range(num_samples):
                    action = samples[candidate]
                    candidate_writer.writerow({
                        "task": task,
                        "seed": seed,
                        "replan": replan,
                        "iter": iteration,
                        "generating_branch": branch,
                        "candidate": candidate,
                        "action_hash": _hash_array(action),
                        "noise_hash": noise_hash,
                        "action_l2": float(np.linalg.norm(action)),
                        "action_linf": float(np.abs(action).max()),
                        "stateprobe_cost": float(scored["stateprobe"][candidate]),
                        "true_state_cost": float(scored["true_state"][candidate]),
                        "l2_cost": float(scored["l2"][candidate]),
                        "obj_goal_dist": float(scored["obj_goal"][candidate]),
                        "ee_goal_dist": float(scored["ee_goal"][candidate]),
                        "hand_obj_dist": float(scored["hand_obj"][candidate]),
                        "obj_decode_error_cm": float(scored["obj_decode_cm"][candidate]),
                        "ee_decode_error_cm": float(scored["ee_decode_cm"][candidate]),
                        "stateprobe_optimism_m": float(
                            scored["stateprobe_optimism"][candidate]),
                        "success_any": int(scored["success_any"][candidate]),
                        "success_end": int(scored["success_end"][candidate]),
                        **{
                            f"elite_by_{cost}": int(candidate in set(elites[cost].tolist()))
                            for cost in ALL_COSTS
                        },
                        **{
                            f"argmin_by_{cost}": int(candidate == orders[cost][0])
                            for cost in ALL_COSTS
                        },
                    })

                state["mean"], state["var"] = post_mean, post_var
    finally:
        restore(env, snap)

    return {
        branch: branch_state[branch]["mean"].reshape(plan_raw_len, RAW_A)
        for branch in branches
    }


def run_episode(
    task,
    seed,
    env,
    goal_frame,
    goal_state,
    expert_succ,
    adapter,
    device,
    *,
    probe,
    ee_probe,
    branches,
    carrier,
    plan_h,
    num_act_stepped,
    max_episode_steps,
    max_replans,
    num_samples,
    iterations,
    elite_frac,
    var0,
    w_hand,
    strict,
    candidate_writer,
    iteration_writer,
):
    goal_obj = goal_state[OBJECT_SLICE].astype(np.float32)
    goal_ee = goal_state[EE_SLICE].astype(np.float32)
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))
    rng = np.random.default_rng(seed)
    success, last_success, steps, replan = False, False, 0, 0
    while steps < max_episode_steps and replan < max_replans:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plans = branched_cem(
            env, adapter, device, z_goal, goal_obj, goal_ee, task,
            probe=probe, ee_probe=ee_probe, branches=branches, plan_h=plan_h_eff,
            num_samples=num_samples, iterations=iterations, elite_frac=elite_frac,
            var0=var0, w_hand=w_hand, rng=rng, seed=seed, replan=replan,
            candidate_writer=candidate_writer, iteration_writer=iteration_writer,
        )
        plan = plans[carrier]
        replan += 1
        for action in plan[: num_act_stepped * FRAMESKIP]:
            obs, _, _, _, info = env.step(np.clip(action, -1, 1))
            steps += 1
            last_success = bool(info.get("success", 0) > 0.5)
            success = success or last_success
            if steps >= max_episode_steps:
                break
        if success and not strict:
            break
    return {
        "task": task,
        "seed": seed,
        "carrier": carrier,
        "success": int(success),
        "success_end": int(last_success),
        "steps": steps,
        "replans": replan,
        "final_obj_goal_dist": float(np.linalg.norm(obs[OBJECT_SLICE] - goal_obj)),
        "final_ee_goal_dist": float(np.linalg.norm(obs[EE_SLICE] - goal_ee)),
        "expert_success_step": expert_succ,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mw-push"])
    ap.add_argument("--probe", required=True, help="object probe checkpoint")
    ap.add_argument("--ee-probe", required=True, help="end-effector probe checkpoint")
    ap.add_argument(
        "--branches", nargs="+", choices=ALL_COSTS,
        default=["stateprobe", "true_state"],
        help="independently refitted CEM branches; every candidate is scored by all costs",
    )
    ap.add_argument(
        "--carrier", choices=ALL_COSTS, default="true_state",
        help="branch mean executed only to obtain the next shared MPC snapshot",
    )
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=54000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--max-replans", type=int, default=34)
    ap.add_argument("--cem-num-samples", type=int, default=100)
    ap.add_argument("--cem-iterations", type=int, default=6)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--w-hand", type=float, default=0.5)
    ap.add_argument("--strict-success", action="store_true")
    ap.add_argument("--out-prefix", default="results/shared_population_branch")
    args = ap.parse_args()

    # Preserve order for a stable first-branch reference while rejecting duplicates.
    branches = list(dict.fromkeys(args.branches))
    if args.carrier not in branches:
        raise SystemExit("--carrier must be one of --branches")
    if len(branches) < 2:
        raise SystemExit("at least two distinct --branches are required")
    if not 0.0 < args.elite_frac <= 1.0:
        raise SystemExit("--elite-frac must be in (0, 1]")

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    probe, probe_meta = load_probe(args.probe, device)
    ee_probe, ee_meta = load_probe(args.ee_probe, device)
    print(
        f"model={args.model} branches={branches} carrier={args.carrier} "
        f"obj_probe={probe_meta.get('v1_median')} ee_probe={ee_meta.get('v1_median')}",
        flush=True,
    )

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = prefix.with_name(prefix.name + "_candidates.csv.gz")
    iteration_path = prefix.with_name(prefix.name + "_iterations.csv")
    episode_path = prefix.with_name(prefix.name + "_episodes.csv")
    metadata_path = prefix.with_name(prefix.name + "_metadata.json")
    candidate_tmp = Path(str(candidate_path) + ".tmp")
    iteration_tmp = Path(str(iteration_path) + ".tmp")
    episode_tmp = Path(str(episode_path) + ".tmp")

    metadata = {
        "script": Path(__file__).name,
        "config": args.config,
        "model": args.model,
        "tasks": args.tasks,
        "probe": args.probe,
        "ee_probe": args.ee_probe,
        "branches": branches,
        "carrier": args.carrier,
        "episodes": args.episodes,
        "seed0": args.seed0,
        "horizon": args.horizon,
        "num_act_stepped": args.num_act_stepped,
        "max_episode_steps": args.max_episode_steps,
        "max_replans": args.max_replans,
        "cem_num_samples": args.cem_num_samples,
        "cem_iterations": args.cem_iterations,
        "elite_frac": args.elite_frac,
        "var0": args.var0,
        "w_hand": args.w_hand,
        "strict_success": args.strict_success,
        "common_random_numbers": "same eps tensor at every iteration; exact same population at iter=0",
        "dynamics": "MuJoCo simulator; learned predictor bypassed",
    }

    try:
        with gzip.open(candidate_tmp, "wt", newline="") as candidate_file, open(
            iteration_tmp, "w", newline=""
        ) as iteration_file, open(episode_tmp, "w", newline="") as episode_file:
            candidate_writer = csv.DictWriter(candidate_file, fieldnames=CANDIDATE_FIELDS)
            iteration_writer = csv.DictWriter(iteration_file, fieldnames=ITERATION_FIELDS)
            episode_writer = csv.DictWriter(episode_file, fieldnames=EPISODE_FIELDS)
            candidate_writer.writeheader()
            iteration_writer.writeheader()
            episode_writer.writeheader()
            for task in args.tasks:
                for episode in range(args.episodes):
                    seed = args.seed0 + episode
                    started = time.time()
                    env, init_state = make_env(task, seed)
                    try:
                        goal_frame, goal_state, expert_succ = rollout_expert(
                            env, init_state, task)
                        row = run_episode(
                            task, seed, env, goal_frame, goal_state, expert_succ,
                            adapter, device, probe=probe, ee_probe=ee_probe,
                            branches=branches, carrier=args.carrier, plan_h=args.horizon,
                            num_act_stepped=args.num_act_stepped,
                            max_episode_steps=args.max_episode_steps,
                            max_replans=args.max_replans,
                            num_samples=args.cem_num_samples,
                            iterations=args.cem_iterations,
                            elite_frac=args.elite_frac, var0=args.var0,
                            w_hand=args.w_hand, strict=args.strict_success,
                            candidate_writer=candidate_writer,
                            iteration_writer=iteration_writer,
                        )
                    finally:
                        env.close()
                    episode_writer.writerow(row)
                    candidate_file.flush()
                    iteration_file.flush()
                    episode_file.flush()
                    print(
                        f"{task} ep={episode:02d} carrier={args.carrier} "
                        f"success={row['success']} replans={row['replans']} "
                        f"obj={row['final_obj_goal_dist']:.3f} "
                        f"minutes={(time.time() - started) / 60:.1f}",
                        flush=True,
                    )
        candidate_tmp.replace(candidate_path)
        iteration_tmp.replace(iteration_path)
        episode_tmp.replace(episode_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        for tmp in (candidate_tmp, iteration_tmp, episode_tmp):
            if tmp.exists():
                tmp.unlink()

    print(f"wrote {candidate_path}", flush=True)
    print(f"wrote {iteration_path}", flush=True)
    print(f"wrote {episode_path}", flush=True)
    print(f"wrote {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
