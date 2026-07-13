"""E1 — amortized GC-IDM control + search-dose response (remove the adversary).

E0 (scripts/41) showed the Goodhart signature of CEM against a learned cost under
perfect dynamics. E1 tests the corollary (docs/plans/2026-07-06-e1-amortized-control-
design.md): if test-time search is the adversary, a search-free amortized controller
built from the SAME frozen representation + probe should do no worse — and re-adding
search in increasing doses should corrupt its plans.

Arms (mw-push, latent-oracle env, paired seeds):
  * gcidm          — pure amortized: every model step, encode → probe →
                     Δobj request (one typical-scale step toward goal) →
                     h_inv(z_t, Δobj) → execute the 20-dim raw chunk. No cost,
                     no search, 3× faster replanning than the CEM protocol.
  * cemseed_it{K}  — CEM (cost=stateprobe) seeded with the gcidm proposal
                     (init_mean + mean-inclusion, scripts/30), K ∈ {0,2,6,12,24}
                     iterations at CEM cadence (commit 3 model steps). K=0 executes
                     the seed with zero search — the same-cadence no-search control.

Crown measurement — seed-vs-chosen (results/e1_seed_vs_chosen.csv): at every replan
of every seeded arm, BOTH the seed plan and the CEM-chosen plan are rolled on the sim
and scored by proxy AND true state. A "corruption event" = search lowered the proxy
while raising the true cost — the per-replan, quantified form of reward-hacking.

    python scripts/43_e1_amortized_control.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --tasks mw-push \
        --probe checkpoints/spatial_object_probe_dino_wm_metaworld_offpolicy.pt \
        --ee-probe checkpoints/ee_probe_dino_wm_metaworld_offpolicy.pt \
        --inverse-head checkpoints/inverse_proposal_dino_wm_metaworld.pt \
        --iters-grid 0 2 6 12 24 --episodes 8 --strict-success
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
from scripts._exploitation_metrics import candidate_alignment  # noqa: E402


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_lo = _load("latent_oracle", "30_latent_oracle.py")
cem_plan_latent, build_oracle_cost = _lo.cem_plan_latent, _lo.build_oracle_cost
make_env, rollout_expert, encode_frame = _lo.make_env, _lo.rollout_expert, _lo.encode_frame
roll_final_frame, encode_batch = _lo.roll_final_frame, _lo.encode_batch
snapshot, restore = _lo.snapshot, _lo.restore
FRAMESKIP, RAW_A = _lo.FRAMESKIP, _lo.RAW_A


def _true_sp(raw_state, goal_obj, w_hand):
    """True-state analogue of the stateprobe cost for one 39-dim sim obs."""
    obj = raw_state[OBJECT_SLICE]; ee = raw_state[EE_SLICE]
    return (float(np.linalg.norm(obj - goal_obj)),
            float(np.linalg.norm(obj - goal_obj) + w_hand * np.linalg.norm(ee - obj)))


@torch.no_grad()
def gcidm_proposal(adapter, inverse_head, probe, z_t, g_goal, *, mode, plan_h, step_scale):
    """One model-step raw-action chunk from the amortized inverse.

    mode='step'    — request one typical-scale object step toward the goal
                     (‖Δ‖ clipped to step_scale); the pure-controller setting.
    mode='horizon' — request (g_goal − g_t)/plan_h (the scripts/18 inv-seed
                     convention); used when seeding a horizon-plan CEM."""
    g_t = probe(z_t.unsqueeze(0))                                  # (1, 3)
    delta = (g_goal - g_t)
    if mode == "step":
        nd = float(delta.norm())
        if nd > step_scale > 0:
            delta = delta * (step_scale / nd)
    else:
        delta = delta / max(plan_h, 1)
    return inverse_head(z_t.unsqueeze(0), delta)[0].detach().cpu().numpy()  # (RAW_A*FRAMESKIP,)


def run_gcidm_episode(task, seed, env, goal_frame, goal_state, expert_succ, adapter,
                      device, *, probe, inverse_head, step_scale, max_episode_steps,
                      strict, w_hand):
    """Pure amortized control: replan every model step, no cost, no search."""
    goal_obj = goal_state[OBJECT_SLICE].astype(np.float32)
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    with torch.no_grad():
        g_goal = probe(z_goal.unsqueeze(0))
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))                    # upstream reset_warmup
    success, last_success, steps = False, False, 0
    while steps < max_episode_steps:
        frame = _lo.render(env)
        z_t = encode_frame(adapter, frame, obs[:4], device)
        chunk = gcidm_proposal(adapter, inverse_head, probe, z_t, g_goal,
                               mode="step", plan_h=1, step_scale=step_scale)
        for a in chunk.reshape(FRAMESKIP, RAW_A):
            obs, _, _, _, info = env.step(np.clip(a, -1, 1))
            steps += 1
            last_success = info.get("success", 0) > 0.5
            if last_success:
                success = True
            if steps >= max_episode_steps:
                break
        if success and not strict:
            break
    d_obj, _ = _true_sp(obs, goal_obj, w_hand)
    return dict(task=task, seed=seed, arm="gcidm", iterations=-1, samples=0,
                success=int(success), success_end=int(last_success), steps=steps,
                ee_dist=float(np.linalg.norm(obs[:3] - goal_state[:3])),
                obj_goal_dist=d_obj, expert_success_step=expert_succ)


def run_seeded_episode(task, seed, env, goal_frame, goal_state, expert_succ, adapter,
                       device, *, probe, ee_probe, inverse_head, iterations, samples,
                       elite_frac, var0, plan_h, num_act_stepped, max_episode_steps,
                       strict, w_hand, step_scale, sv_rows, curve_rows):
    """CEM (stateprobe cost) seeded with the gcidm proposal; iterations=0 executes
    the seed with zero search. Appends seed-vs-chosen rows to ``sv_rows`` and E0-style
    per-iteration Goodhart rows to ``curve_rows``."""
    arm = f"cemseed_it{iterations}"
    goal_obj = goal_state[OBJECT_SLICE].astype(np.float32)
    goal_ee = goal_state[EE_SLICE].astype(np.float32)
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    cost_fn = build_oracle_cost(cost="stateprobe", z_goal=z_goal, probe=probe,
                                ee_probe=ee_probe, w_hand=w_hand)
    with torch.no_grad():
        g_goal = probe(z_goal.unsqueeze(0))
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))
    rng = np.random.default_rng(seed)
    success, last_success, steps = False, False, 0
    replan_i = 0
    base_key = dict(task=task, seed=seed, cost=f"stateprobe_{arm}",
                    iterations=iterations, samples=samples)

    def on_elites(z_elite, raw_elite, cost_elite, it):
        obj = np.stack([r[OBJECT_SLICE] for r in raw_elite]).astype(np.float32)
        ee = np.stack([r[EE_SLICE] for r in raw_elite]).astype(np.float32)
        d_obj = np.linalg.norm(obj - goal_obj[None], axis=-1)
        d_app = np.linalg.norm(ee - obj, axis=-1)
        true_sp = d_obj + w_hand * d_app
        true_task = (np.linalg.norm(ee - goal_ee[None], axis=-1)
                     if task.startswith("mw-reach") else true_sp)
        obj_align = candidate_alignment(cost_elite, d_obj)
        sp_align = candidate_alignment(cost_elite, true_sp)
        task_align = candidate_alignment(cost_elite, true_task)
        row = dict(base_key, replan=replan_i, iter=int(it), n_elite=len(raw_elite),
                   proxy_min=float(cost_elite[0]), proxy_med=float(np.median(cost_elite)),
                   true_obj_min_believed=float(d_obj[0]), true_obj_med=float(np.median(d_obj)),
                   true_sp_min_believed=float(true_sp[0]), true_sp_med=float(np.median(true_sp)),
                   true_obj_best_elite=obj_align["true_best"],
                   selected_true_obj_regret=obj_align["selected_true_regret"],
                   true_sp_best_elite=sp_align["true_best"],
                   selected_true_sp_regret=sp_align["selected_true_regret"],
                   selected_true_sp_rank_frac=sp_align["selected_true_rank_frac"],
                   proxy_truth_inversion_frac=sp_align["proxy_truth_inversion_frac"],
                   proxy_truth_comparable_pairs=int(sp_align["n_comparable_pairs"]),
                   true_task_definition=("ee_goal" if task.startswith("mw-reach")
                                         else "state_oracle_obj_plus_approach"),
                   true_task_min_believed=task_align["selected_true"],
                   true_task_best_elite=task_align["true_best"],
                   selected_true_task_regret=task_align["selected_true_regret"],
                   selected_true_task_rank_frac=task_align["selected_true_rank_frac"],
                   proxy_true_task_inversion_frac=task_align["proxy_truth_inversion_frac"])
        with torch.no_grad():
            dec = probe(z_elite).detach().cpu().numpy()
        derr = np.linalg.norm(dec - obj, axis=-1)
        row["decode_err_med_cm"] = float(100 * np.median(derr))
        row["decode_err_believed_cm"] = float(100 * derr[0])
        curve_rows.append(row)

    def _score_plan(plan_flat, plan_raw_len):
        """Roll one full-horizon plan on the sim; return (proxy, true_obj, true_sp)."""
        snap = snapshot(env)
        fr, pr, raw = roll_final_frame(env, snap, plan_flat.reshape(plan_raw_len, RAW_A))
        restore(env, snap)
        z_fin = encode_batch(adapter, [fr], [pr], device)
        proxy = float(cost_fn(z_fin).detach().cpu().numpy()[0])
        t_obj, t_sp = _true_sp(raw, goal_obj, w_hand)
        return proxy, t_obj, t_sp

    while steps < max_episode_steps:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan_raw_len = plan_h_eff * FRAMESKIP
        frame = _lo.render(env)
        z_t = encode_frame(adapter, frame, obs[:4], device)
        chunk = gcidm_proposal(adapter, inverse_head, probe, z_t, g_goal,
                               mode="horizon", plan_h=plan_h_eff, step_scale=step_scale)
        seed_flat = np.clip(np.tile(chunk, plan_h_eff), -1.0, 1.0)   # (plan_raw_len*RAW_A,)

        if iterations == 0:
            plan = seed_flat.reshape(plan_raw_len, RAW_A)
            s_proxy, s_obj, s_sp = _score_plan(seed_flat, plan_raw_len)
            c_proxy, c_obj, c_sp = s_proxy, s_obj, s_sp
        else:
            plan = cem_plan_latent(env, adapter, z_goal, device, plan_h=plan_h_eff,
                                   num_samples=samples, iterations=iterations,
                                   elite_frac=elite_frac, var0=var0, rng=rng,
                                   cost_fn=cost_fn, on_elites=on_elites,
                                   init_mean=seed_flat)
            s_proxy, s_obj, s_sp = _score_plan(seed_flat, plan_raw_len)
            c_proxy, c_obj, c_sp = _score_plan(plan.reshape(-1), plan_raw_len)
        sv_rows.append(dict(task=task, seed=seed, arm=arm, iterations=iterations,
                            replan=replan_i,
                            seed_proxy=s_proxy, seed_true_obj=s_obj, seed_true_sp=s_sp,
                            chosen_proxy=c_proxy, chosen_true_obj=c_obj, chosen_true_sp=c_sp,
                            corruption=int(c_proxy < s_proxy and c_sp > s_sp)))
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
    d_obj, _ = _true_sp(obs, goal_obj, w_hand)
    return dict(task=task, seed=seed, arm=arm, iterations=iterations, samples=samples,
                success=int(success), success_end=int(last_success), steps=steps,
                ee_dist=float(np.linalg.norm(obs[:3] - goal_state[:3])),
                obj_goal_dist=d_obj, expert_success_step=expert_succ)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mw-push"])
    ap.add_argument("--iters-grid", nargs="+", type=int, default=[0, 2, 6, 12, 24],
                    help="search doses for the seeded arms (0 = execute seed, no search)")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--strict-success", action="store_true")
    ap.add_argument("--skip-gcidm", action="store_true",
                    help="run only the seeded dose-response arms")
    ap.add_argument("--probe", required=True, help="spatial object-probe ckpt (scripts/22)")
    ap.add_argument("--ee-probe", required=True, help="ee-probe ckpt (stateprobe cost)")
    ap.add_argument("--inverse-head", required=True,
                    help="inverse action-proposal ckpt (scripts/28)")
    ap.add_argument("--w-hand", type=float, default=0.5)
    ap.add_argument("--step-scale", type=float, default=None,
                    help="‖Δobj‖ request for the pure gcidm arm (default: the head's "
                         "training obj_scale)")
    ap.add_argument("--out-episodes", default="results/e1_episodes.csv")
    ap.add_argument("--out-curves", default="results/e1_curves.csv")
    ap.add_argument("--out-seedvs", default="results/e1_seed_vs_chosen.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    from models.probes import load_probe, load_inverse_head
    probe, pmeta = load_probe(args.probe, device)
    ee_probe, emeta = load_probe(args.ee_probe, device)
    inverse_head, imeta = load_inverse_head(args.inverse_head, device)
    step_scale = args.step_scale if args.step_scale is not None else float(imeta.get("obj_scale", 0.028))
    print(f"object probe: {args.probe} (v1_median={pmeta.get('v1_median')})", flush=True)
    print(f"ee probe: {args.ee_probe} (v1_median={emeta.get('v1_median')})", flush=True)
    print(f"inverse head: {args.inverse_head} (val_action_mse={imeta.get('val_action_mse'):.4f} "
          f"baseline={imeta.get('baseline_action_mse'):.4f} obj_scale={step_scale:.4f})", flush=True)

    arms = ([] if args.skip_gcidm else ["gcidm"]) + [f"cemseed_it{k}" for k in args.iters_grid]

    # Resume — drop partial (task, seed) pairs, redo whole (scripts/18 convention).
    ep_rows, curve_rows_all, sv_rows_all, done_pairs = [], [], [], set()
    want_arms = set(arms)
    if Path(args.out_episodes).exists():
        prev = pd.read_csv(args.out_episodes)
        have = {(t, int(s)): set(g.arm) for (t, s), g in prev.groupby(["task", "seed"])}
        done_pairs = {k for k, a in have.items() if want_arms <= a}
        ep_rows = prev[[(r.task, int(r.seed)) in done_pairs
                        for r in prev.itertuples()]].to_dict("records")
        for p, dst in ((args.out_curves, curve_rows_all), (args.out_seedvs, sv_rows_all)):
            if Path(p).exists():
                d = pd.read_csv(p)
                dst.extend(d[[(r.task, int(r.seed)) in done_pairs
                              for r in d.itertuples()]].to_dict("records"))
        print(f"resume: {len(done_pairs)} complete pairs kept", flush=True)

    Path(args.out_episodes).parent.mkdir(parents=True, exist_ok=True)
    print(f"arms={arms} tasks={args.tasks} episodes={args.episodes} samples={args.samples} "
          f"strict={args.strict_success}", flush=True)

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
            pair_ep, pair_curves, pair_sv = [], [], []
            ok = True
            for arm in arms:
                t0 = time.time()
                try:
                    if arm == "gcidm":
                        r = run_gcidm_episode(
                            task, seed, env, goal_frame, goal_state, expert_succ,
                            adapter, device, probe=probe, inverse_head=inverse_head,
                            step_scale=step_scale,
                            max_episode_steps=args.max_episode_steps,
                            strict=args.strict_success, w_hand=args.w_hand)
                    else:
                        k = int(arm.rsplit("it", 1)[1])
                        r = run_seeded_episode(
                            task, seed, env, goal_frame, goal_state, expert_succ,
                            adapter, device, probe=probe, ee_probe=ee_probe,
                            inverse_head=inverse_head, iterations=k,
                            samples=args.samples, elite_frac=args.elite_frac,
                            var0=args.var0, plan_h=args.horizon,
                            num_act_stepped=args.num_act_stepped,
                            max_episode_steps=args.max_episode_steps,
                            strict=args.strict_success, w_hand=args.w_hand,
                            step_scale=step_scale, sv_rows=pair_sv,
                            curve_rows=pair_curves)
                except Exception as e:  # noqa: BLE001 — keep the sweep alive
                    print(f"  [error] {task} ep{ep} {arm}: {e}", flush=True)
                    ok = False
                    break
                r["minutes"] = round((time.time() - t0) / 60, 2)
                pair_ep.append(r)
                print(f"  {task:12s} ep{ep:02d} {arm:12s} end={r['success_end']} "
                      f"obj={r['obj_goal_dist']:.3f} ee={r['ee_dist']:.3f} "
                      f"({r['minutes']:.1f} min)", flush=True)
            env.close()
            if not ok:
                continue
            ep_rows.extend(pair_ep)
            curve_rows_all.extend(pair_curves)
            sv_rows_all.extend(pair_sv)
            pd.DataFrame(ep_rows).to_csv(args.out_episodes, index=False)
            if curve_rows_all:
                pd.DataFrame(curve_rows_all).to_csv(args.out_curves, index=False)
            if sv_rows_all:
                pd.DataFrame(sv_rows_all).to_csv(args.out_seedvs, index=False)

    d = pd.DataFrame(ep_rows)
    if len(d):
        print("\n=== E1 outcomes by arm ===")
        for (task, arm), g in d.groupby(["task", "arm"]):
            print(f"  {task:12s} {arm:12s} end={int(g.success_end.sum())}/{len(g)}  "
                  f"obj_med={g.obj_goal_dist.median():.3f}  ee_med={g.ee_dist.median():.3f}",
                  flush=True)
    sv = pd.DataFrame(sv_rows_all)
    if len(sv):
        print("\n=== seed-vs-chosen (per-replan) ===")
        for (arm,), g in sv.groupby(["arm"]):
            n = len(g)
            better_proxy = (g.chosen_proxy < g.seed_proxy).mean()
            worse_true = (g.chosen_true_sp > g.seed_true_sp).mean()
            print(f"  {arm:12s} n={n:4d}  search-beats-seed(proxy)={better_proxy:.2f}  "
                  f"worse-true={worse_true:.2f}  corruption={g.corruption.mean():.2f}",
                  flush=True)
    print(f"\nwrote {args.out_episodes} / {args.out_curves} / {args.out_seedvs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
