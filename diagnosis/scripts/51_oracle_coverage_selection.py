"""Candidate coverage-vs-selection audit under perfect latent dynamics.

The latent-oracle planner can fail for two different reasons:

1. **coverage:** its proposal never samples a physically good candidate;
2. **selection:** a good candidate is present, but the latent proxy ranks another
   candidate above it.

This runner uses ``scripts/30_latent_oracle.py``'s opt-in full-population hook.
For every CEM iteration it stores proxy cost, simulator-state task distance, the
state-oracle shaped cost, and exact environment success flags for the *same*
candidates.  The ordinary latent-oracle behavior is unchanged when the hook is
not supplied.

Outputs are one episode CSV, one iteration-summary CSV, and an optional gzip
candidate CSV.  Candidate action sequences are identified by a deterministic
hash and compact norm statistics rather than duplicating every action scalar.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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
from models.probes import load_probe  # noqa: E402
from scripts._coverage_selection_metrics import (  # noqa: E402
    coverage_selection_summary,
    spearman_costs,
    topk_overlap,
)
from stratification.metaworld_regimes import EE_SLICE, OBJECT_SLICE  # noqa: E402


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lo = _load("latent_oracle_coverage", "30_latent_oracle.py")
cem_plan_latent = _lo.cem_plan_latent
build_oracle_cost = _lo.build_oracle_cost
make_env, rollout_expert, encode_frame = _lo.make_env, _lo.rollout_expert, _lo.encode_frame
FRAMESKIP, RAW_A = _lo.FRAMESKIP, _lo.RAW_A

CANDIDATE_FIELDS = [
    "task", "cost", "seed", "replan", "iter", "candidate", "action_hash",
    "action_l2", "action_linf", "proxy_cost", "true_progress_cost",
    "true_shaped_cost", "obj_goal_dist", "ee_goal_dist", "hand_obj_dist",
    "obj_decode_error_cm", "ee_decode_error_cm",
    "decoded_obj_to_probe_goal", "decoded_obj_to_true_goal",
    "decoded_hand_obj_dist", "decoded_stateprobe_cost",
    "stateprobe_optimism_m",
    "success_any", "success_end", "proxy_selected", "true_progress_best",
]


def _action_hash(action_sequence: np.ndarray) -> str:
    canonical = np.asarray(action_sequence, dtype="<f4", order="C")
    return hashlib.blake2b(canonical.tobytes(), digest_size=8).hexdigest()


def run_episode(task, cost_name, seed, env, goal_frame, goal_state, expert_succ,
                adapter, device, *, cost_spec, plan_h, num_act_stepped,
                max_episode_steps, cem_kw, strict, w_hand, topk_frac,
                iteration_writer, candidate_writer, audit_probe=None,
                audit_ee_probe=None):
    goal_obj = goal_state[OBJECT_SLICE].astype(np.float32)
    goal_ee = goal_state[EE_SLICE].astype(np.float32)
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    cost_fn = build_oracle_cost(z_goal=z_goal, **cost_spec)
    decoded_goal_obj = None
    if audit_probe is not None:
        with torch.no_grad():
            decoded_goal_obj = audit_probe(z_goal.unsqueeze(0)).detach().cpu().numpy()[0]
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))
    rng = np.random.default_rng(seed)
    success, last_success, steps, replan = False, False, 0, 0

    def on_candidates(z_fin, raw_final, proxy_cost, action_sequences, iteration,
                      *, success_any, success_end):
        raw = np.asarray(raw_final, dtype=np.float32)
        proxy = np.asarray(proxy_cost, dtype=float)
        obj = raw[:, OBJECT_SLICE]
        ee = raw[:, EE_SLICE]
        obj_dist = np.linalg.norm(obj - goal_obj[None], axis=-1)
        ee_dist = np.linalg.norm(ee - goal_ee[None], axis=-1)
        hand_obj = np.linalg.norm(ee - obj, axis=-1)
        progress = ee_dist if task.startswith("mw-reach") else obj_dist
        shaped = (ee_dist if task.startswith("mw-reach")
                  else obj_dist + w_hand * hand_obj)
        obj_decode_cm = np.full(len(proxy), np.nan, dtype=float)
        ee_decode_cm = np.full(len(proxy), np.nan, dtype=float)
        decoded_obj_probe_goal = np.full(len(proxy), np.nan, dtype=float)
        decoded_obj_true_goal = np.full(len(proxy), np.nan, dtype=float)
        decoded_hand_obj = np.full(len(proxy), np.nan, dtype=float)
        decoded_stateprobe = np.full(len(proxy), np.nan, dtype=float)
        if audit_probe is not None:
            with torch.no_grad():
                decoded_obj = audit_probe(z_fin).detach().cpu().numpy()
            obj_decode_cm = 100.0 * np.linalg.norm(decoded_obj - obj, axis=-1)
            decoded_obj_probe_goal = np.linalg.norm(
                decoded_obj - decoded_goal_obj[None], axis=-1)
            decoded_obj_true_goal = np.linalg.norm(
                decoded_obj - goal_obj[None], axis=-1)
            if audit_ee_probe is not None:
                with torch.no_grad():
                    decoded_ee = audit_ee_probe(z_fin).detach().cpu().numpy()
                ee_decode_cm = 100.0 * np.linalg.norm(decoded_ee - ee, axis=-1)
                decoded_hand_obj = np.linalg.norm(decoded_ee - decoded_obj, axis=-1)
                decoded_stateprobe = (
                    decoded_obj_probe_goal + w_hand * decoded_hand_obj)
        summary = coverage_selection_summary(
            proxy, progress, success_any, success_end, topk_frac=topk_frac)
        k = int(summary["topk"])
        summary.update(
            task=task,
            cost=cost_name,
            seed=seed,
            replan=replan,
            iter=int(iteration),
            true_progress_definition=("ee_goal" if task.startswith("mw-reach")
                                      else "object_goal"),
            true_shaped_definition=("ee_goal" if task.startswith("mw-reach")
                                    else "object_goal_plus_hand_approach"),
            proxy_shaped_spearman=spearman_costs(proxy, shaped),
            proxy_shaped_topk_overlap=topk_overlap(proxy, shaped, k),
            best_true_shaped=float(shaped.min()),
            selected_true_shaped=float(shaped[int(summary["selected_index"])]),
            selected_shaped_regret=float(
                shaped[int(summary["selected_index"])] - shaped.min()),
        )
        iteration_writer.writerow(summary)

        if candidate_writer is not None:
            selected = int(summary["selected_index"])
            best = int(np.argmin(progress))
            flat_actions = action_sequences.reshape(len(action_sequences), -1)
            for i in range(len(proxy)):
                candidate_writer.writerow({
                    "task": task, "cost": cost_name, "seed": seed,
                    "replan": replan, "iter": int(iteration), "candidate": i,
                    "action_hash": _action_hash(action_sequences[i]),
                    "action_l2": float(np.linalg.norm(flat_actions[i])),
                    "action_linf": float(np.abs(flat_actions[i]).max()),
                    "proxy_cost": float(proxy[i]),
                    "true_progress_cost": float(progress[i]),
                    "true_shaped_cost": float(shaped[i]),
                    "obj_goal_dist": float(obj_dist[i]),
                    "ee_goal_dist": float(ee_dist[i]),
                    "hand_obj_dist": float(hand_obj[i]),
                    "obj_decode_error_cm": float(obj_decode_cm[i]),
                    "ee_decode_error_cm": float(ee_decode_cm[i]),
                    "decoded_obj_to_probe_goal": float(decoded_obj_probe_goal[i]),
                    "decoded_obj_to_true_goal": float(decoded_obj_true_goal[i]),
                    "decoded_hand_obj_dist": float(decoded_hand_obj[i]),
                    "decoded_stateprobe_cost": float(decoded_stateprobe[i]),
                    "stateprobe_optimism_m": float(
                        shaped[i] - decoded_stateprobe[i]),
                    "success_any": int(success_any[i]),
                    "success_end": int(success_end[i]),
                    "proxy_selected": int(i == selected),
                    "true_progress_best": int(i == best),
                })

    while steps < max_episode_steps:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_latent(
            env, adapter, z_goal, device, plan_h=plan_h_eff, rng=rng,
            cost_fn=cost_fn, on_candidates=on_candidates, **cem_kw)
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
        "task": task, "cost": cost_name, "seed": seed,
        "success": int(success), "success_end": int(last_success), "steps": steps,
        "replans": replan,
        "final_state_dist": float(np.linalg.norm(obs - goal_state)),
        "ee_dist": float(np.linalg.norm(obs[EE_SLICE] - goal_ee)),
        "obj_goal_dist": float(np.linalg.norm(obs[OBJECT_SLICE] - goal_obj)),
        "expert_success_step": expert_succ,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mw-push"])
    ap.add_argument("--cost", choices=["l2", "stateprobe", "gobj", "straight"], default="l2")
    ap.add_argument("--encoder-lora", default=None,
                    help="encoder-LoRA ckpt (hys_h0/scripts/04 or scripts/38); injected "
                         "into the adapter so the audit scores the SAME reshaped latents "
                         "the cost was trained on")
    ap.add_argument("--projector", default=None,
                    help="straightening-projector ckpt (hys_h0/scripts/02); --cost straight")
    ap.add_argument("--probe", default=None)
    ap.add_argument("--ee-probe", default=None)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=40000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--cem-num-samples", type=int, default=100)
    ap.add_argument("--cem-iterations", type=int, default=6)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--w-hand", type=float, default=0.5)
    ap.add_argument("--s-g", type=float, default=0.1276)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--gamma-l2", type=float, default=1.0)
    ap.add_argument("--topk-frac", type=float, default=0.1)
    ap.add_argument("--strict-success", action="store_true")
    ap.add_argument("--dump-candidates", action="store_true",
                    help="write aligned per-candidate gzip CSV (off by default)")
    ap.add_argument("--out-prefix", default="results/oracle_coverage_selection")
    args = ap.parse_args()

    if args.cost in ("gobj", "stateprobe") and not args.probe:
        raise SystemExit(f"--cost {args.cost} requires --probe")
    if args.cost == "stateprobe" and not args.ee_probe:
        raise SystemExit("--cost stateprobe requires --ee-probe")

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    probe = ee_probe = None
    if args.probe:
        probe, _ = load_probe(args.probe, device)
    if args.ee_probe:
        ee_probe, _ = load_probe(args.ee_probe, device)
    if args.encoder_lora:
        from models.heads.lora_encoder import load_encoder_lora
        _inj, _lmeta = load_encoder_lora(adapter, args.encoder_lora, device)
        print(f"encoder LoRA: {args.encoder_lora} (r={_lmeta.get('r')}, "
              f"gate={_lmeta.get('meta', {}).get('gate')})", flush=True)

    projector = None
    if args.cost == "straight":
        if not args.projector:
            raise SystemExit("--projector is required for --cost straight")
        from models.heads.straightening_projector import load_projector
        projector, pmeta = load_projector(args.projector, device)
        print(f"straightening projector: {args.projector} (gate={pmeta.get('gate')}, "
              f"final_val_curv={pmeta.get('final_eval', {}).get('curv_all')})", flush=True)
    cost_spec = dict(
        cost=args.cost, probe=probe, ee_probe=ee_probe, metric=None,
        repr_adapter=None, projector=projector, w_hand=args.w_hand, s_g=args.s_g,
        beta=args.beta, gamma_l2=args.gamma_l2,
    )
    cem_kw = dict(
        num_samples=args.cem_num_samples, iterations=args.cem_iterations,
        elite_frac=args.elite_frac, var0=args.var0, planner="cem", mppi_beta=5.0,
    )

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ep_path = prefix.with_name(prefix.name + "_episodes.csv")
    it_path = prefix.with_name(prefix.name + "_iterations.csv")
    cand_path = prefix.with_name(prefix.name + "_candidates.csv.gz")
    ep_tmp, it_tmp = Path(str(ep_path) + ".tmp"), Path(str(it_path) + ".tmp")
    cand_tmp = Path(str(cand_path) + ".tmp")
    episode_fields = [
        "task", "cost", "seed", "success", "success_end", "steps", "replans",
        "final_state_dist", "ee_dist", "obj_goal_dist", "expert_success_step",
    ]
    iteration_fields = [
        "task", "cost", "seed", "replan", "iter", "n_candidates", "topk",
        "true_progress_definition", "true_shaped_definition",
        "coverage_success_any", "coverage_success_end", "n_success_any", "n_success_end",
        "selected_success_any", "selected_success_end", "best_true_progress",
        "selected_true_progress", "selected_physical_regret", "proxy_true_spearman",
        "proxy_true_topk_overlap", "selected_index", "best_true_shaped",
        "selected_true_shaped", "selected_shaped_regret", "proxy_shaped_spearman",
        "proxy_shaped_topk_overlap",
    ]

    cand_handle = None
    try:
        with open(ep_tmp, "w", newline="") as ep_f, open(it_tmp, "w", newline="") as it_f:
            ep_writer = csv.DictWriter(ep_f, fieldnames=episode_fields)
            it_writer = csv.DictWriter(it_f, fieldnames=iteration_fields)
            ep_writer.writeheader(); it_writer.writeheader()
            cand_writer = None
            if args.dump_candidates:
                cand_handle = gzip.open(cand_tmp, "wt", newline="")
                cand_writer = csv.DictWriter(cand_handle, fieldnames=CANDIDATE_FIELDS)
                cand_writer.writeheader()
            for task in args.tasks:
                for episode in range(args.episodes):
                    seed = args.seed0 + episode
                    t0 = time.time()
                    env, init_state = make_env(task, seed)
                    try:
                        goal_frame, goal_state, expert_succ = rollout_expert(env, init_state, task)
                        row = run_episode(
                            task, args.cost, seed, env, goal_frame, goal_state, expert_succ,
                            adapter, device, cost_spec=cost_spec, plan_h=args.horizon,
                            num_act_stepped=args.num_act_stepped,
                            max_episode_steps=args.max_episode_steps, cem_kw=cem_kw,
                            strict=args.strict_success, w_hand=args.w_hand,
                            topk_frac=args.topk_frac, iteration_writer=it_writer,
                            candidate_writer=cand_writer, audit_probe=probe,
                            audit_ee_probe=ee_probe,
                        )
                    finally:
                        env.close()
                    ep_writer.writerow(row); ep_f.flush(); it_f.flush()
                    if cand_handle is not None:
                        cand_handle.flush()
                    print(
                        f"{task} ep{episode:02d} cost={args.cost} "
                        f"success_end={row['success_end']} obj={row['obj_goal_dist']:.3f} "
                        f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if cand_handle is not None:
            cand_handle.close(); cand_handle = None
        ep_tmp.replace(ep_path); it_tmp.replace(it_path)
        if args.dump_candidates:
            cand_tmp.replace(cand_path)
    finally:
        if cand_handle is not None:
            cand_handle.close()
    print(f"wrote {ep_path} and {it_path}" +
          (f" and {cand_path}" if args.dump_candidates else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
