"""Latent oracle — ENSEMBLE / disagreement-penalised cost (Phase G).

Phase 0-F closed every SINGLE post-hoc cost on the frozen (or lightly LoRA-reshaped)
latent: CEM, searching the cost MINIMUM over ~100×6 candidates, finds whatever
residual-error pocket the cost has (reward-hacking, not missing information). A
learned φ readout, even encoder-LoRA-reshaped, still gates push at frozen-noise
(seed sweep {5,0,2,1,1}/16, mean 1.8).

This script attacks the reward-hacking mechanism directly instead of trying to make
one readout more accurate. It ensembles the K independently-trained encoder-LoRA+φ
seeds we already have on disk and adds a DISAGREEMENT penalty to the cost:

    cost(z) = mean_k ‖φ_k(z)_obj − φ_k(goal)_obj‖² / s_g²   (consensus goal vote)
            + λ · Var_k[ ‖φ_k(z)_obj − φ_k(goal)_obj‖ ] / s_g²  (spread = distrust)

    Each seed is scored against its OWN goal readout so per-seed calibration bias
    cancels (see ``ensemble_cost``).

Each seed reshapes the encoder differently, so its object readout aliases a
different set of states. A genuine goal-adjacent state is in-distribution for all K
seeds → they agree → low penalty. A CEM-mined exploit pocket fools ONE seed's φ →
the K estimates scatter → the penalty inflates the cost → CEM avoids it. The
mechanism only works if the aliasing is seed-SPECIFIC; if all K seeds share the same
blind spot (inherited from the common frozen DINOv2 backbone) the disagreement stays
flat and the ensemble buys nothing. Either way the gate answers the question.

The dynamics stay PERFECT and identical to scripts/30 (sim render + encode; F never
called); the ONLY change is a multi-encoder consensus cost. Because each candidate
frame is encoded once PER SEED, this is ~K× the encode cost of scripts/30.

    python scripts/39_latent_oracle_ensemble.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld \
        --encoder-loras checkpoints/encoder_lora_dino_wm_metaworld.pt \
                        checkpoints/encoder_lora_dino_wm_metaworld_s1.pt ... \
        --phi-adapters  checkpoints/phi_enclora_dino_wm_metaworld.pt \
                        checkpoints/phi_enclora_dino_wm_metaworld_s1.pt ... \
        --lambda-dis 1.0 --tasks mw-push --episodes 16 --strict-success \
        --out results/latent_oracle_phiens_l1.csv
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


# Reuse the exact latent-oracle machinery (perfect-dynamics rollout + encode) so this
# arm is directly comparable to scripts/30 — only the cost differs.
_lo = _load("latent_oracle", "30_latent_oracle.py")
make_env, rollout_expert = _lo.make_env, _lo.rollout_expert
encode_frame, encode_batch = _lo.encode_frame, _lo.encode_batch
roll_final_frame = _lo.roll_final_frame
snapshot, restore = _lo.snapshot, _lo.restore
FRAMESKIP, RAW_A, OBJECT_SLICE = _lo.FRAMESKIP, _lo.RAW_A, _lo.OBJECT_SLICE


def ensemble_cost(E: torch.Tensor, E_goal: torch.Tensor, s_g: float,
                  lam: float) -> torch.Tensor:
    """Pure cost math (no env/model), so it is unit-testable offline.

    ``E``      : (K, B, obj) object estimates, K seeds × B candidates.
    ``E_goal`` : (K, obj)    each seed's object estimate at the goal frame.
    returns    : (B,) cost = mean per-seed goal-distance² + λ·disagreement, in
                 (metres / s_g)² units so ``lam`` is scale-free.

    Each seed is scored against ITS OWN goal estimate (``d_k = ‖φ_k(z)−φ_k(goal)‖``),
    not a shared consensus vector: a seed with a constant readout bias carries that
    bias in both terms of ``d_k`` so it cancels — otherwise the seeds' calibration
    scatter would put a spurious disagreement floor at the true goal. Then:

      * consensus  = ``mean_k d_k²`` — the averaged "how far are we" vote. To drive it
                     to zero CEM must satisfy MOST seeds, not exploit one.
      * disagree   = ``Var_k[d_k]``  — seeds disagreeing on the distance flags an
                     exploit pocket (one seed fooled to ~0 while others read far).
    At a true goal every ``d_k≈0`` so both terms vanish."""
    d = (E - E_goal[:, None, :]).norm(dim=-1) / s_g        # (K, B) per-seed goal dist
    consensus = (d ** 2).mean(0)                           # (B,)
    disag = d.var(0, unbiased=False)                       # (B,)
    return consensus + lam * disag


@torch.no_grad()
def _ensemble_estimates(adapters, phis, frames, props, device):
    """Encode ONE candidate batch through every seed and read its object subspace.
    Returns E: (K, B, obj)."""
    ests = []
    for adapter, phi in zip(adapters, phis):
        z = encode_batch(adapter, frames, props, device)   # (B,V,H,W,D) TRUE latent
        ests.append(phi(z)[:, : phi.obj_dim])              # (B, obj)
    return torch.stack(ests, dim=0)


@torch.no_grad()
def _ensemble_goal(adapters, phis, goal_frame, goal_prop, device):
    """Each seed's object estimate at the goal frame. Returns E_goal: (K, obj)."""
    gs = []
    for adapter, phi in zip(adapters, phis):
        zg = encode_frame(adapter, goal_frame, goal_prop, device)
        gs.append(phi(zg.unsqueeze(0))[:, : phi.obj_dim][0])   # (obj,)
    return torch.stack(gs, dim=0)


def cem_plan_ensemble(env, adapters, phis, E_goal, device, *, plan_h, num_samples,
                      iterations, elite_frac, var0, rng, s_g, lam):
    """CEM identical to scripts/30's cem_plan_latent, but the cost is the ensemble
    consensus + disagreement penalty (each candidate encoded through all K seeds)."""
    plan_raw_len = plan_h * FRAMESKIP
    dim = plan_raw_len * RAW_A
    mean = np.zeros(dim); var = np.full(dim, var0)
    n_elite = max(2, int(num_samples * elite_frac))
    snap = snapshot(env)
    for _ in range(iterations):
        samples = np.clip(mean[None] + np.sqrt(var)[None] * rng.standard_normal((num_samples, dim)),
                          -1.0, 1.0)
        frames, props = [], []
        for i in range(num_samples):
            fr, pr, _ = roll_final_frame(env, snap, samples[i].reshape(plan_raw_len, RAW_A))
            frames.append(fr); props.append(pr)
        E = _ensemble_estimates(adapters, phis, frames, props, device)   # (K,B,obj)
        costs = ensemble_cost(E, E_goal, s_g, lam).detach().cpu().numpy()
        elite_idx = np.argsort(costs)[:n_elite]
        elites = samples[elite_idx]
        mean = elites.mean(0); var = elites.var(0) + 1e-4
    restore(env, snap)
    return mean.reshape(plan_raw_len, RAW_A)


def run_episode(task, seed, env, goal_frame, goal_state, expert_succ, adapters, phis,
                device, *, plan_h, num_act_stepped, max_episode_steps, cem_kw, strict,
                s_g, lam):
    goal_obj = goal_state[OBJECT_SLICE]
    E_goal = _ensemble_goal(adapters, phis, goal_frame, goal_state[:4], device)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))                     # upstream reset_warmup
    success, last_success, steps = False, False, 0
    rng = np.random.default_rng(seed)
    while steps < max_episode_steps:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_ensemble(env, adapters, phis, E_goal, device, plan_h=plan_h_eff,
                                 rng=rng, s_g=s_g, lam=lam, **cem_kw)
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
    return {"task": task, "arm": f"latent_oracle_phiens_l{lam:g}", "seed": seed,
            "success": int(success), "success_end": int(last_success), "steps": steps,
            "final_state_dist": float(np.linalg.norm(obs - goal_state)),
            "ee_dist": float(np.linalg.norm(obs[:3] - goal_state[:3])),
            "obj_goal_dist": float(np.linalg.norm(obs[OBJECT_SLICE] - goal_obj)),
            "expert_success_step": expert_succ}


def _build_ensemble(model, encoder_loras, phi_adapters, device):
    """One adapter PER seed (each holds its own encoder-LoRA), paired with its φ head.
    Separate adapters because load_encoder_lora injects in place — a single adapter
    can only carry one LoRA weight-set at a time."""
    from models.heads.lora_encoder import load_encoder_lora
    from models.heads.action_repr_adapter import load_repr_adapter
    if len(encoder_loras) != len(phi_adapters):
        raise SystemExit("--encoder-loras and --phi-adapters must be parallel lists")
    adapters, phis = [], []
    for lora_ckpt, phi_ckpt in zip(encoder_loras, phi_adapters):
        adapter = build_adapter(model, device=str(device)).eval()
        _, lmeta = load_encoder_lora(adapter, lora_ckpt, device)
        phi, _ = load_repr_adapter(phi_ckpt, device)
        adapters.append(adapter); phis.append(phi)
        print(f"  seed: encoder_lora={Path(lora_ckpt).name} (r={lmeta['r']}) "
              f"phi={Path(phi_ckpt).name} (obj_dim={phi.obj_dim})", flush=True)
    return adapters, phis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--encoder-loras", nargs="+", required=True,
                    help="encoder-LoRA ckpts, one per ensemble member (scripts/38)")
    ap.add_argument("--phi-adapters", nargs="+", required=True,
                    help="repr-adapter (phi) ckpts, PARALLEL to --encoder-loras")
    ap.add_argument("--lambda-dis", type=float, default=1.0,
                    help="disagreement-penalty weight (0 = pure consensus mean)")
    ap.add_argument("--s-g", type=float, default=0.1276,
                    help="object scale (matches scripts/18/30)")
    ap.add_argument("--tasks", nargs="+", default=["mw-push"])
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
    ap.add_argument("--out", default="results/metaworld_latent_oracle_ensemble.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"latent-oracle ENSEMBLE cost: K={len(args.encoder_loras)} seeds, "
          f"lambda_dis={args.lambda_dis} s_g={args.s_g}", flush=True)
    adapters, phis = _build_ensemble(args.model, args.encoder_loras, args.phi_adapters, device)

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
                                adapters, phis, device, plan_h=args.horizon,
                                num_act_stepped=args.num_act_stepped,
                                max_episode_steps=args.max_episode_steps, cem_kw=cem_kw,
                                strict=args.strict_success, s_g=args.s_g, lam=args.lambda_dis)
                env.close()
                w.writerow(r); f.flush(); rows.append(r)
                print(f"  {task:16s} ep{e:02d} phiens[l{args.lambda_dis:g}] "
                      f"success={r['success']} end={r['success_end']} "
                      f"obj_goal={r['obj_goal_dist']:.3f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    print(f"\n=== latent oracle ENSEMBLE [lambda={args.lambda_dis:g}]: success by task ===")
    for task in args.tasks:
        tr = [r for r in rows if r["task"] == task]
        s = sum(r["success"] for r in tr); se = sum(r["success_end"] for r in tr)
        print(f"  {task:16s} success {s}/{len(tr)}   success_end {se}/{len(tr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
