"""Oracle-dynamics planning gate for a trained selection-aware encoder."""

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
from models.heads.encoder_adaptation import load_trainable_encoder_state  # noqa: E402
from models.heads.lora_encoder import inject_encoder_lora  # noqa: E402
from models.heads.lora_predictor import LoRALinear  # noqa: E402
from models.heads.selection_cost import build_selection_cost_from_checkpoint  # noqa: E402
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def _load(modname: str, filename: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / filename))
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lo = _load("selection_eval_latent_oracle", "30_latent_oracle.py")
make_env, rollout_expert, encode_frame = _lo.make_env, _lo.rollout_expert, _lo.encode_frame
cem_plan_latent, build_oracle_cost = _lo.cem_plan_latent, _lo.build_oracle_cost
FRAMESKIP, RAW_A = _lo.FRAMESKIP, _lo.RAW_A


def load_adaptation(adapter, checkpoint, device):
    if checkpoint["adaptation"] == "last_blocks":
        return load_trainable_encoder_state(
            adapter, checkpoint["encoder"], checkpoint["last_blocks"], device)
    injected = inject_encoder_lora(
        adapter, r=checkpoint["lora_r"], alpha=checkpoint["lora_alpha"])
    encoder = adapter.wm.encoder
    live = {name: module for name, module in encoder.named_modules()
            if isinstance(module, LoRALinear)}
    missing = sorted(set(checkpoint["encoder"]) - set(live))
    if missing:
        raise RuntimeError(f"LoRA checkpoint mismatch: {missing[:5]}")
    with torch.no_grad():
        for name, state in checkpoint["encoder"].items():
            live[name].A.copy_(state["A"].to(device))
            live[name].B.copy_(state["B"].to(device))
    return injected, {"n_lora_modules": len(injected)}


def run_episode(task, seed, adapter, head, device, args):
    env, init_state = make_env(task, seed)
    goal_frame, goal_state, expert_succ = rollout_expert(env, init_state, task)
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    if task == "mw-push":
        zg = z_goal.unsqueeze(0)

        @torch.no_grad()
        def cost_fn(z_fin):
            return head(z_fin, zg.expand(z_fin.shape[0], *([-1] * (zg.ndim - 1))))
        cost_name = "selection"
    elif task == "mw-reach":
        cost_fn = build_oracle_cost("l2", z_goal)
        cost_name = "l2_preservation"
    else:
        raise ValueError("pilot evaluation supports only mw-push and mw-reach")

    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))
    success_any = success_end = False
    steps = 0
    rng = np.random.default_rng(seed)
    while steps < args.max_episode_steps:
        plan_h_eff = min(
            args.horizon, max(1, -(-(args.max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_latent(
            env, adapter, z_goal, device, plan_h=plan_h_eff,
            num_samples=args.cem_num_samples, iterations=args.cem_iterations,
            elite_frac=args.elite_frac, var0=args.var0, rng=rng, cost_fn=cost_fn)
        for action in plan[: args.num_act_stepped * FRAMESKIP]:
            obs, _, _, _, info = env.step(np.clip(action, -1, 1))
            success_end = bool(info.get("success", 0) > 0.5)
            success_any |= success_end
            steps += 1
            if steps >= args.max_episode_steps:
                break
    env.close()
    return {
        "task": task, "cost": cost_name, "eval_seed": seed,
        "success": int(success_any), "success_end": int(success_end), "steps": steps,
        "obj_goal_dist": float(np.linalg.norm(
            obs[OBJECT_SLICE] - goal_state[OBJECT_SLICE])),
        "ee_goal_dist": float(np.linalg.norm(obs[:3] - goal_state[:3])),
        "expert_success_step": expert_succ,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mw-push", "mw-reach"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=63000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--cem-num-samples", type=int, default=100)
    ap.add_argument("--cem-iterations", type=int, default=6)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    adapter = build_adapter(checkpoint["model"], device=str(device)).eval()
    _, adaptation_meta = load_adaptation(adapter, checkpoint, device)
    head = build_selection_cost_from_checkpoint(checkpoint, device)
    tag = f"{checkpoint['adaptation']}_{checkpoint['objective']}_s{checkpoint['seed']}"
    print(f"checkpoint={args.checkpoint} tag={tag} adaptation_meta={adaptation_meta} "
          f"val={checkpoint['val']}", flush=True)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arm", "train_seed", "task", "cost", "eval_seed", "success",
              "success_end", "steps", "obj_goal_dist", "ee_goal_dist",
              "expert_success_step", "val_selected_regret", "val_mae"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in args.tasks:
            for episode in range(args.episodes):
                t0 = time.time()
                row = run_episode(task, args.seed0 + episode, adapter, head, device, args)
                row.update({
                    "arm": f"{checkpoint['adaptation']}_{checkpoint['objective']}",
                    "train_seed": checkpoint["seed"],
                    "val_selected_regret": checkpoint["val"]["selected_regret"],
                    "val_mae": checkpoint["val"]["mae"],
                })
                writer.writerow(row)
                handle.flush()
                print(f"{tag} {task} ep={episode:02d} end={row['success_end']} "
                      f"obj={row['obj_goal_dist']:.3f} ee={row['ee_goal_dist']:.3f} "
                      f"minutes={(time.time()-t0)/60:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
