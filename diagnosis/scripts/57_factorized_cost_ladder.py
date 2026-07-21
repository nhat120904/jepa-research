"""Closed-loop factorized cost ladder under simulator-perfect candidate dynamics.

Each arm changes only which object/hand coordinates enter the existing shaped
terminal cost. Candidate dynamics always come from MuJoCo and the frozen encoder
is used only to decode the non-privileged channels. Heavy execution is Slurm-only.
"""

from __future__ import annotations

import argparse
import csv
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
from planning.factorized_state_cost import ARMS, factorized_state_cost  # noqa: E402
from stratification.metaworld_regimes import EE_SLICE, OBJECT_SLICE  # noqa: E402


def _load(modname: str, filename: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / filename))
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lo = _load("latent_oracle_factorized", "30_latent_oracle.py")
make_env, rollout_expert, encode_frame = _lo.make_env, _lo.rollout_expert, _lo.encode_frame
snapshot, restore = _lo.snapshot, _lo.restore
roll_final_frame, encode_batch = _lo.roll_final_frame, _lo.encode_batch
FRAMESKIP, RAW_A = _lo.FRAMESKIP, _lo.RAW_A


@torch.no_grad()
def score_candidates(
    env,
    snap,
    samples: np.ndarray,
    adapter,
    device,
    probe,
    ee_probe,
    decoded_goal_object: torch.Tensor,
    true_goal_object: np.ndarray,
    arm: str,
    w_hand: float,
) -> np.ndarray:
    frames, proprios, raw_states = [], [], []
    for sample in samples:
        frame, proprio, raw = roll_final_frame(env, snap, sample)
        frames.append(frame)
        proprios.append(proprio)
        raw_states.append(raw)

    z_final = encode_batch(adapter, frames, proprios, device)
    decoded_object = probe(z_final)
    decoded_hand = ee_probe(z_final)
    raw = torch.as_tensor(
        np.asarray(raw_states), device=decoded_object.device, dtype=decoded_object.dtype
    )
    true_object = raw[:, OBJECT_SLICE]
    true_hand = raw[:, EE_SLICE]
    goal = torch.as_tensor(
        true_goal_object, device=decoded_object.device, dtype=decoded_object.dtype
    )
    costs = factorized_state_cost(
        arm,
        decoded_object=decoded_object,
        decoded_hand=decoded_hand,
        true_object=true_object,
        true_hand=true_hand,
        decoded_goal_object=decoded_goal_object,
        true_goal_object=goal,
        w_hand=w_hand,
    )
    return costs.detach().cpu().numpy().astype(float)


def cem_plan_factorized(
    env,
    adapter,
    device,
    probe,
    ee_probe,
    decoded_goal_object,
    true_goal_object,
    arm,
    *,
    plan_h,
    num_samples,
    iterations,
    elite_frac,
    var0,
    w_hand,
    rng,
):
    plan_raw_len = plan_h * FRAMESKIP
    dim = plan_raw_len * RAW_A
    mean = np.zeros(dim, dtype=np.float64)
    var = np.full(dim, var0, dtype=np.float64)
    n_elite = max(2, int(num_samples * elite_frac))
    snap = snapshot(env)
    try:
        for _ in range(iterations):
            samples_flat = np.clip(
                mean[None] + np.sqrt(var)[None] * rng.standard_normal((num_samples, dim)),
                -1.0,
                1.0,
            )
            samples = samples_flat.reshape(num_samples, plan_raw_len, RAW_A)
            costs = score_candidates(
                env,
                snap,
                samples,
                adapter,
                device,
                probe,
                ee_probe,
                decoded_goal_object,
                true_goal_object,
                arm,
                w_hand,
            )
            elites = samples_flat[np.argsort(costs, kind="mergesort")[:n_elite]]
            mean = elites.mean(axis=0)
            var = elites.var(axis=0) + 1e-4
    finally:
        restore(env, snap)
    return mean.reshape(plan_raw_len, RAW_A)


def run_episode(
    model,
    task,
    seed,
    env,
    goal_frame,
    goal_state,
    expert_success_step,
    adapter,
    device,
    probe,
    ee_probe,
    arm,
    *,
    plan_h,
    num_act_stepped,
    max_episode_steps,
    num_samples,
    iterations,
    elite_frac,
    var0,
    w_hand,
    strict,
):
    true_goal_object = goal_state[OBJECT_SLICE].astype(np.float32)
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    with torch.no_grad():
        decoded_goal_object = probe(z_goal.unsqueeze(0)).squeeze(0)

    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))
    rng = np.random.default_rng(seed)
    success, last_success, steps = False, False, 0
    while steps < max_episode_steps:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_factorized(
            env,
            adapter,
            device,
            probe,
            ee_probe,
            decoded_goal_object,
            true_goal_object,
            arm,
            plan_h=plan_h_eff,
            num_samples=num_samples,
            iterations=iterations,
            elite_frac=elite_frac,
            var0=var0,
            w_hand=w_hand,
            rng=rng,
        )
        for action in plan[: num_act_stepped * FRAMESKIP]:
            obs, _, _, _, info = env.step(np.clip(action, -1, 1))
            steps += 1
            last_success = bool(info.get("success", 0) > 0.5)
            success = success or last_success
            if steps >= max_episode_steps:
                break
        if success and not strict:
            break

    final_object = obs[OBJECT_SLICE]
    final_hand = obs[EE_SLICE]
    object_distance = float(np.linalg.norm(final_object - true_goal_object))
    hand_object_distance = float(np.linalg.norm(final_hand - final_object))
    return {
        "model": model,
        "task": task,
        "arm": arm,
        "seed": seed,
        "success": int(success),
        "success_end": int(last_success),
        "steps": steps,
        "obj_goal_dist": object_distance,
        "hand_obj_dist": hand_object_distance,
        "true_shaped_cost": object_distance + w_hand * hand_object_distance,
        "expert_success_step": expert_success_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["mw-push", "mw-pick-place"], required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--ee-probe", required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed0", type=int, default=61000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-act-stepped", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--cem-num-samples", type=int, default=100)
    parser.add_argument("--cem-iterations", type=int, default=6)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--var0", type=float, default=1.0)
    parser.add_argument("--w-hand", type=float, default=0.5)
    parser.add_argument("--strict-success", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not 0.0 < args.elite_frac <= 1.0:
        raise SystemExit("--elite-frac must be in (0, 1]")

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    probe, probe_meta = load_probe(args.probe, device)
    ee_probe, ee_meta = load_probe(args.ee_probe, device)
    print(
        f"model={args.model} task={args.task} arm={args.arm} "
        f"object_probe={probe_meta.get('v1_median')} ee_probe={ee_meta.get('v1_median')}",
        flush=True,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fields = [
        "model", "task", "arm", "seed", "success", "success_end", "steps",
        "obj_goal_dist", "hand_obj_dist", "true_shaped_cost", "expert_success_step",
    ]
    rows = []
    try:
        with open(temporary, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for episode in range(args.episodes):
                seed = args.seed0 + episode
                started = time.time()
                env, initial_state = make_env(args.task, seed)
                try:
                    goal_frame, goal_state, expert_success_step = rollout_expert(
                        env, initial_state, args.task
                    )
                    row = run_episode(
                        args.model,
                        args.task,
                        seed,
                        env,
                        goal_frame,
                        goal_state,
                        expert_success_step,
                        adapter,
                        device,
                        probe,
                        ee_probe,
                        args.arm,
                        plan_h=args.horizon,
                        num_act_stepped=args.num_act_stepped,
                        max_episode_steps=args.max_episode_steps,
                        num_samples=args.cem_num_samples,
                        iterations=args.cem_iterations,
                        elite_frac=args.elite_frac,
                        var0=args.var0,
                        w_hand=args.w_hand,
                        strict=args.strict_success,
                    )
                finally:
                    env.close()
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(
                    f"episode={episode:02d} seed={seed} success_end={row['success_end']} "
                    f"object={row['obj_goal_dist']:.3f} "
                    f"minutes={(time.time() - started) / 60:.1f}",
                    flush=True,
                )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    metadata = vars(args).copy()
    metadata.update(
        {
            "script": Path(__file__).name,
            "dynamics": "MuJoCo simulator; learned predictor bypassed",
            "hand_definition": "end-effector xyz; not gripper aperture",
            "primary_endpoint": "strict success_end",
            "pairing": "same seed and CEM RNG across arms; trajectories diverge after cost-based selection",
        }
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output}; success_end={sum(r['success_end'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
