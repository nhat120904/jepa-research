"""Mine grouped CEM candidate subsets for selection-aware encoder training.

Unlike the older elite buffers, this preserves candidates from the *same* CEM
population and always includes both the proxy argmin and simulator-cost argmin.
That grouping is required by pairwise/listwise objectives and by selected-regret
validation.  Heavy execution belongs on a GPU Slurm node.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
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
from models.probes import load_probe  # noqa: E402
from stratification.metaworld_regimes import EE_SLICE, OBJECT_SLICE  # noqa: E402


def _load(modname: str, filename: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / filename))
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lo = _load("selection_mining_latent_oracle", "30_latent_oracle.py")
make_env, rollout_expert, encode_frame = _lo.make_env, _lo.rollout_expert, _lo.encode_frame
cem_plan_latent, build_oracle_cost = _lo.cem_plan_latent, _lo.build_oracle_cost
FRAMESKIP, RAW_A = _lo.FRAMESKIP, _lo.RAW_A


def registered_subset(proxy: np.ndarray, truth: np.ndarray, rng: np.random.Generator,
                      top_proxy: int, top_true: int, random_count: int):
    """Return unique indices plus role masks for the pre-registered subset."""
    n = len(proxy)
    p = np.argsort(proxy, kind="mergesort")[:min(top_proxy, n)]
    t = np.argsort(truth, kind="mergesort")[:min(top_true, n)]
    r = rng.choice(n, size=min(random_count, n), replace=False)
    selected = np.unique(np.concatenate([p, t, r])).astype(np.int64)
    return selected, np.isin(selected, p), np.isin(selected, t), np.isin(selected, r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="dino_wm_metaworld")
    ap.add_argument("--task", default="mw-push")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=62000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--cem-num-samples", type=int, default=100)
    ap.add_argument("--cem-iterations", type=int, default=6)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--top-proxy", type=int, default=10)
    ap.add_argument("--top-true", type=int, default=10)
    ap.add_argument("--random-count", type=int, default=10)
    ap.add_argument("--w-hand", type=float, default=0.5)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--ee-probe", required=True)
    ap.add_argument("--out", default="results/selection_populations_dino_push.pt")
    args = ap.parse_args()

    if args.task != "mw-push":
        raise SystemExit("the pre-registered pilot is mw-push only")
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    probe, _ = load_probe(args.probe, device)
    ee_probe, _ = load_probe(args.ee_probe, device)

    rows = {key: [] for key in (
        "frames", "prop", "true_cost", "proxy_cost", "seed", "replan", "iteration",
        "group_id", "candidate", "ep_idx", "role_proxy", "role_true", "role_random",
        "success_any", "success_end")}
    ep_goal_frames, ep_goal_prop, ep_goal_obj = [], [], []
    group_counter = 0

    for episode in range(args.episodes):
        seed = args.seed0 + episode
        env, init_state = make_env(args.task, seed)
        goal_frame, goal_state, _ = rollout_expert(env, init_state, args.task)
        z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
        goal_obj = goal_state[OBJECT_SLICE].astype(np.float32)
        proxy_fn = build_oracle_cost(
            "stateprobe", z_goal, probe=probe, ee_probe=ee_probe, w_hand=args.w_hand)
        ep_goal_frames.append(np.asarray(goal_frame, dtype=np.uint8))
        ep_goal_prop.append(goal_state[:4].astype(np.float32))
        ep_goal_obj.append(goal_obj)

        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(np.zeros(RAW_A))
        planner_rng = np.random.default_rng(seed)
        subset_rng = np.random.default_rng(seed + 99173)
        steps = replans = 0
        success_any_episode = False
        t0 = time.time()
        while steps < args.max_episode_steps:
            plan_h_eff = min(
                args.horizon,
                max(1, -(-(args.max_episode_steps - steps) // FRAMESKIP)),
            )

            def on_candidates(z_fin, raw_final, proxy_cost, action_sequences, iteration,
                              *, success_any, success_end, frames, props,
                              _seed=seed, _replan=replans, _ep=episode):
                nonlocal group_counter
                raw = np.asarray(raw_final, dtype=np.float32)
                obj, ee = raw[:, OBJECT_SLICE], raw[:, EE_SLICE]
                truth = (np.linalg.norm(obj - goal_obj[None], axis=-1)
                         + args.w_hand * np.linalg.norm(ee - obj, axis=-1))
                idx, rp, rt, rr = registered_subset(
                    np.asarray(proxy_cost), truth, subset_rng,
                    args.top_proxy, args.top_true, args.random_count)
                n = len(idx)
                rows["frames"].append(np.stack([frames[i] for i in idx]).astype(np.uint8))
                rows["prop"].append(np.stack([props[i] for i in idx]).astype(np.float32))
                rows["true_cost"].append(truth[idx].astype(np.float32))
                rows["proxy_cost"].append(np.asarray(proxy_cost, dtype=np.float32)[idx])
                rows["seed"].append(np.full(n, _seed, dtype=np.int64))
                rows["replan"].append(np.full(n, _replan, dtype=np.int32))
                rows["iteration"].append(np.full(n, iteration, dtype=np.int16))
                rows["group_id"].append(np.full(n, group_counter, dtype=np.int64))
                rows["candidate"].append(idx.astype(np.int16))
                rows["ep_idx"].append(np.full(n, _ep, dtype=np.int16))
                rows["role_proxy"].append(rp)
                rows["role_true"].append(rt)
                rows["role_random"].append(rr)
                rows["success_any"].append(np.asarray(success_any, dtype=bool)[idx])
                rows["success_end"].append(np.asarray(success_end, dtype=bool)[idx])
                group_counter += 1

            plan = cem_plan_latent(
                env, adapter, z_goal, device, plan_h=plan_h_eff,
                num_samples=args.cem_num_samples, iterations=args.cem_iterations,
                elite_frac=args.elite_frac, var0=args.var0, rng=planner_rng,
                cost_fn=proxy_fn, on_candidates=on_candidates)
            for action in plan[: args.num_act_stepped * FRAMESKIP]:
                obs, _, _, _, info = env.step(np.clip(action, -1, 1))
                success_any_episode |= bool(info.get("success", 0) > 0.5)
                steps += 1
                if steps >= args.max_episode_steps:
                    break
            replans += 1
        env.close()
        print(f"episode={episode} seed={seed} replans={replans} groups={group_counter} "
              f"success_any={int(success_any_episode)} minutes={(time.time()-t0)/60:.1f}",
              flush=True)

    tensor_keys = ("frames", "prop", "true_cost", "proxy_cost", "seed", "replan",
                   "iteration", "group_id", "candidate", "ep_idx", "role_proxy",
                   "role_true", "role_random", "success_any", "success_end")
    out = {key: torch.from_numpy(np.concatenate(rows[key], axis=0)) for key in tensor_keys}
    out.update({
        "ep_goal_frames": torch.from_numpy(np.stack(ep_goal_frames)),
        "ep_goal_prop": torch.from_numpy(np.stack(ep_goal_prop)),
        "ep_goal_obj": torch.from_numpy(np.stack(ep_goal_obj)),
        "metadata": vars(args) | {
            "format_version": 1,
            "n_rows": int(out["true_cost"].numel()),
            "n_groups": int(group_counter),
            "train_seeds": list(range(args.seed0, args.seed0 + 6)),
            "val_seeds": list(range(args.seed0 + 6, args.seed0 + 8)),
        },
    })
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(out, temporary)
    os.replace(temporary, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(out["metadata"], indent=2) + "\n")
    print(f"saved {path}: rows={out['metadata']['n_rows']} groups={group_counter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
