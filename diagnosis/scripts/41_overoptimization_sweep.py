"""E0 — cost-overoptimization sweep: Goodhart curves for latent planning.

The world-model analogue of Gao et al.'s reward-model overoptimization study
(docs/plans/2026-07-06-overoptimization-sweep-design.md). Phase A showed a single
point of the phenomenon (CEM-elite decode 91.5%→24%); this script measures the whole
curve: as CEM optimization pressure grows, does the PROXY cost keep improving while
the TRUE task outcome stalls or degrades?

Harness: the latent oracle (scripts/30) — perfect dynamics (sim-step → render →
encode), only the cost is learned. Any proxy/true divergence is therefore the cost
under optimization, not predictor rollout error. Two instruments:

  * WITHIN-SEARCH curves (results/overopt_curves.csv): `cem_plan_latent`'s
    `on_elites` hook records, per CEM iteration of every replan, the elites' proxy
    cost, the TRUE object→goal distance of the candidate the planner most believes
    in (argmin proxy), the true-state analogue of the stateprobe cost (so proxy and
    true are unit-comparable), and the object-probe decode error on the elites
    (pocket depth vs pressure).
  * ACROSS-BUDGET outcomes (results/overopt_episodes.csv): closed-loop episode
    results per (cost, task, iterations, samples) budget cell — "does more search
    help?".

Predictions and the decision rule are in the design doc; the short version:
push×stateprobe proxy↓/true↛ with decode error ↑ = reward-hacking confirmed as a
monotone function of pressure; reach = the honest-cost control; push×l2 = the
no-signal (geometry) failure mode for contrast.

    python scripts/41_overoptimization_sweep.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --tasks mw-push \
        --probe checkpoints/spatial_object_probe_dino_wm_metaworld_offpolicy.pt \
        --ee-probe checkpoints/ee_probe_dino_wm_metaworld_offpolicy.pt \
        --costs stateprobe l2 --iters-grid 2 6 12 24 --episodes 8 --strict-success
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
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
from stratification.metaworld_regimes import OBJECT_SLICE, EE_SLICE  # noqa: E402


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_lo = _load("latent_oracle", "30_latent_oracle.py")
cem_plan_latent, build_oracle_cost = _lo.cem_plan_latent, _lo.build_oracle_cost
make_env, rollout_expert, encode_frame = _lo.make_env, _lo.rollout_expert, _lo.encode_frame
FRAMESKIP, RAW_A = _lo.FRAMESKIP, _lo.RAW_A


def run_episode(task, seed, env, goal_frame, goal_state, expert_succ, adapter, device, *,
                cost, cost_spec, iterations, samples, elite_frac, var0, plan_h,
                num_act_stepped, max_episode_steps, strict, w_hand, probe, curve_rows):
    """One closed-loop latent-oracle episode at a fixed CEM budget; the on_elites
    hook appends per-iteration Goodhart rows to ``curve_rows``."""
    goal_obj = goal_state[OBJECT_SLICE].astype(np.float32)
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    cost_fn = build_oracle_cost(z_goal=z_goal, **cost_spec)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))            # upstream reset_warmup
    rng = np.random.default_rng(seed)
    success, last_success, steps = False, False, 0
    replan_i = 0

    base_key = dict(task=task, seed=seed, cost=cost, iterations=iterations, samples=samples)

    def on_elites(z_elite, raw_elite, cost_elite, it):
        # elites arrive cost-ascending: index 0 = the candidate the planner most
        # believes in. raw_elite are TRUE final sim states of each candidate.
        obj = np.stack([r[OBJECT_SLICE] for r in raw_elite]).astype(np.float32)
        ee = np.stack([r[EE_SLICE] for r in raw_elite]).astype(np.float32)
        d_obj = np.linalg.norm(obj - goal_obj[None], axis=-1)          # true obj→goal
        d_app = np.linalg.norm(ee - obj, axis=-1)                      # true hand→obj
        true_sp = d_obj + w_hand * d_app                               # true stateprobe analogue
        row = dict(base_key, replan=replan_i, iter=int(it), n_elite=len(raw_elite),
                   proxy_min=float(cost_elite[0]), proxy_med=float(np.median(cost_elite)),
                   true_obj_min_believed=float(d_obj[0]), true_obj_med=float(np.median(d_obj)),
                   true_sp_min_believed=float(true_sp[0]), true_sp_med=float(np.median(true_sp)))
        if probe is not None:                                          # pocket depth
            with torch.no_grad():
                dec = probe(z_elite).detach().cpu().numpy()
            derr = np.linalg.norm(dec - obj, axis=-1)
            row["decode_err_med_cm"] = float(100 * np.median(derr))
            row["decode_err_believed_cm"] = float(100 * derr[0])
        curve_rows.append(row)

    while steps < max_episode_steps:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_latent(env, adapter, z_goal, device, plan_h=plan_h_eff,
                               num_samples=samples, iterations=iterations,
                               elite_frac=elite_frac, var0=var0, rng=rng,
                               cost_fn=cost_fn, on_elites=on_elites)
        replan_i += 1
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
    return dict(base_key, success=int(success), success_end=int(last_success), steps=steps,
                final_state_dist=float(np.linalg.norm(obs - goal_state)),
                ee_dist=float(np.linalg.norm(obs[:3] - goal_state[:3])),
                obj_goal_dist=float(np.linalg.norm(obs[OBJECT_SLICE] - goal_obj)),
                expert_success_step=expert_succ)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mw-push"])
    ap.add_argument("--costs", nargs="+", default=["stateprobe", "l2"],
                    choices=["stateprobe", "l2", "gobj"],
                    help="latent-oracle costs to sweep (scripts/30 build_oracle_cost)")
    ap.add_argument("--iters-grid", nargs="+", type=int, default=[2, 6, 12, 24],
                    help="CEM iteration budgets (optimization-pressure axis)")
    ap.add_argument("--samples-grid", nargs="+", type=int, default=[100],
                    help="CEM population sizes (best-of-n pressure axis)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--strict-success", action="store_true")
    ap.add_argument("--probe", default=None,
                    help="object-probe ckpt — required for stateprobe/gobj; also used "
                         "(when given) for the decode-error mechanism metric on ALL arms")
    ap.add_argument("--ee-probe", default=None, help="ee-probe ckpt (stateprobe cost)")
    ap.add_argument("--w-hand", type=float, default=0.5)
    ap.add_argument("--s-g", type=float, default=0.1276)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--gamma-l2", type=float, default=1.0)
    ap.add_argument("--out-episodes", default="results/overopt_episodes.csv")
    ap.add_argument("--out-curves", default="results/overopt_curves.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()

    probe = ee_probe = None
    if args.probe:
        from models.probes import load_probe
        probe, pmeta = load_probe(args.probe, device)
        print(f"object probe: {args.probe} (v1_median={pmeta.get('v1_median')})", flush=True)
    if args.ee_probe:
        from models.probes import load_probe
        ee_probe, emeta = load_probe(args.ee_probe, device)
        print(f"ee probe: {args.ee_probe} (v1_median={emeta.get('v1_median')})", flush=True)
    if "stateprobe" in args.costs and (probe is None or ee_probe is None):
        raise SystemExit("--costs stateprobe needs --probe and --ee-probe")
    if "gobj" in args.costs and probe is None:
        raise SystemExit("--costs gobj needs --probe")

    def cost_spec_for(cost):
        return dict(cost=cost, probe=probe, ee_probe=ee_probe, metric=None,
                    repr_adapter=None, w_hand=args.w_hand, s_g=args.s_g,
                    beta=args.beta, gamma_l2=args.gamma_l2)

    # Cells per (task, seed) pair — cheap budgets first so a crash costs the least.
    budgets = sorted((s, i) for s in args.samples_grid for i in args.iters_grid)
    cells = [(s, i, c) for (s, i) in budgets for c in args.costs]

    # Resume: the rand_vec is random per env creation (frozen only within one
    # instance), so a half-done (task, seed) pair cannot be completed against the
    # same init after a crash — drop partial pairs and redo them whole (paired
    # comparison preserved; same convention as scripts/18).
    ep_rows, curve_rows_all, done_pairs = [], [], set()
    want_cells = {(c, i, s) for (s, i, c) in cells}
    if Path(args.out_episodes).exists():
        prev = pd.read_csv(args.out_episodes)
        have = {(t, int(sd)): {(r.cost, int(r.iterations), int(r.samples))
                               for r in g.itertuples()}
                for (t, sd), g in prev.groupby(["task", "seed"])}
        done_pairs = {k for k, cs in have.items() if want_cells <= cs}
        keep = prev[[(r.task, int(r.seed)) in done_pairs for r in prev.itertuples()]]
        ep_rows = keep.to_dict("records")
        if Path(args.out_curves).exists():
            pc = pd.read_csv(args.out_curves)
            curve_rows_all = pc[[(r.task, int(r.seed)) in done_pairs
                                 for r in pc.itertuples()]].to_dict("records")
        print(f"resume: {len(done_pairs)} complete pairs kept "
              f"({len(prev) - len(ep_rows)} partial-pair episode rows redone)", flush=True)

    Path(args.out_episodes).parent.mkdir(parents=True, exist_ok=True)
    print(f"grid: tasks={args.tasks} costs={args.costs} iters={args.iters_grid} "
          f"samples={args.samples_grid} episodes={args.episodes} strict={args.strict_success}",
          flush=True)

    for task in args.tasks:
        for ep in range(args.episodes):
            seed = args.seed0 + ep
            if (task, seed) in done_pairs:
                continue
            try:
                env, init_state = make_env(task, seed)
                goal_frame, goal_state, expert_succ = rollout_expert(env, init_state, task)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {task} ep{ep} env/expert: {e}", flush=True)
                continue
            pair_ep, pair_curves = [], []
            ok = True
            for samples, iters, cost in cells:
                t0 = time.time()
                try:
                    r = run_episode(task, seed, env, goal_frame, goal_state, expert_succ,
                                    adapter, device, cost=cost,
                                    cost_spec=cost_spec_for(cost),
                                    iterations=iters, samples=samples,
                                    elite_frac=args.elite_frac, var0=args.var0,
                                    plan_h=args.horizon,
                                    num_act_stepped=args.num_act_stepped,
                                    max_episode_steps=args.max_episode_steps,
                                    strict=args.strict_success, w_hand=args.w_hand,
                                    probe=probe, curve_rows=pair_curves)
                except Exception as e:  # noqa: BLE001 — keep the sweep alive
                    print(f"  [error] {task} ep{ep} {cost} it={iters} n={samples}: {e}",
                          flush=True)
                    ok = False
                    break
                r["minutes"] = round((time.time() - t0) / 60, 2)
                pair_ep.append(r)
                print(f"  {task:14s} ep{ep:02d} {cost:10s} it={iters:2d} n={samples:3d} "
                      f"end={r['success_end']} obj={r['obj_goal_dist']:.3f} "
                      f"ee={r['ee_dist']:.3f} ({r['minutes']:.1f} min)", flush=True)
            env.close()
            if not ok:
                continue                       # pair incomplete — redo whole next run
            ep_rows.extend(pair_ep)
            curve_rows_all.extend(pair_curves)
            pd.DataFrame(ep_rows).to_csv(args.out_episodes, index=False)
            pd.DataFrame(curve_rows_all).to_csv(args.out_curves, index=False)

    d = pd.DataFrame(ep_rows)
    if len(d):
        print("\n=== across-budget outcomes (success_end / median obj_goal_dist) ===")
        for (task, cost, it, n), g in d.groupby(["task", "cost", "iterations", "samples"]):
            print(f"  {task:14s} {cost:10s} it={it:2d} n={n:3d}  "
                  f"end={int(g.success_end.sum())}/{len(g)}  "
                  f"obj_med={g.obj_goal_dist.median():.3f}  ee_med={g.ee_dist.median():.3f}",
                  flush=True)
    print(f"\nwrote {args.out_episodes} ({len(ep_rows)} episodes), "
          f"{args.out_curves} ({len(curve_rows_all)} curve rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
