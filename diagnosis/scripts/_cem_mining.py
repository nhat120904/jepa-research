"""Shared CEM elite-mining utility — the foundation for Track 1 (adversarial cost
hardening) and Track 2's adversarial loss term
(docs/plans/2026-07-01-adversarial-cost-and-repr-design.md).

Phase-3 3b (results/oracle_ladder_cost_report.md) showed a real off-policy-ROBUST
object probe (78→92% <5cm on RANDOM off-policy frames) still re-gates push 1/16.
The reconciliation: CEM doesn't sample uniformly off-policy — it specifically
SEARCHES FOR the cost minimum, so if the cost has any residual-error pocket, CEM
adversarially finds and converges its population onto exactly that pocket. Both the
Track-1 DAgger loop (scripts/35 diagnose -> scripts/22/19 --extra-buffer harden ->
re-gate) and Track-2's adversarial loss term (models/heads/action_repr_adapter.py)
need exactly these EXPLOITED frames, not the random distribution
`_offpolicy_frames.py` samples.

`mine_cem_frames` re-runs the exact latent-oracle planner (scripts/30's
`cem_plan_latent`, unmodified except for its `on_elites` hook) over real episodes
and records every CEM elite candidate the planner actually keeps at each
iteration — the population its search converges around, i.e. the frames the
deployed cost is trusted on.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(modname: str, fname: str):
    spec = _ilu.spec_from_file_location(modname, str(ROOT / "scripts" / fname))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


@torch.no_grad()
def mine_cem_frames(adapter, device, *, cost_spec_kwargs, tasks, episodes, seed0=10000,
                    horizon=6, num_act_stepped=3, max_episode_steps=100, cem_kw=None,
                    strict=True, verbose=True, keep_frames=False):
    """Run the latent-oracle planner with the given cost and record every CEM
    elite's (true latent, true object/ee, the episode's goal latent + goal object,
    cost) — the distribution the cost is trusted on.

    ``cost_spec_kwargs`` matches ``30_latent_oracle.build_oracle_cost`` (minus
    ``z_goal``, supplied per-episode); must include ``cost=...``.

    Returns CPU tensors/arrays: ``z``/``z_goal`` (N, *frame), ``obj``/``ee``/
    ``goal_obj`` (N, 3), ``cost`` (N,), plus parallel lists ``task``/``seed``/
    ``iter`` (N,). ``z_goal``/``goal_obj`` are broadcast — the same value repeated
    for every elite mined within one episode.

    ``keep_frames=True`` additionally records the raw pixels needed by
    encoder-level training (scripts/38), where buffered latents go STALE the moment
    the encoder moves: ``frames`` (N, H, W, C uint8) elite renders, ``prop`` (N, 4)
    their proprio, ``ep_idx`` (N,) an episode index into ``ep_goal_frames``
    (n_ep, H, W, C uint8) / ``ep_goal_prop`` (n_ep, 4) so goal frames are stored
    once per episode, not per row.
    """
    lo = _load("latent_oracle", "30_latent_oracle.py")
    cem_plan_latent, build_oracle_cost = lo.cem_plan_latent, lo.build_oracle_cost
    make_env, rollout_expert, encode_frame = lo.make_env, lo.rollout_expert, lo.encode_frame
    FRAMESKIP, RAW_A = lo.FRAMESKIP, lo.RAW_A
    from stratification.metaworld_regimes import OBJECT_SLICE, EE_SLICE

    _cem_defaults = dict(num_samples=100, iterations=6, elite_frac=0.1, var0=1.0)
    _cem_defaults.update(cem_kw or {})
    cem_kw = _cem_defaults
    buf = {"z": [], "obj": [], "ee": [], "z_goal": [], "goal_obj": [], "cost": [],
          "task": [], "seed": [], "iter": [],
          "frames": [], "prop": [], "ep_idx": [], "ep_goal_frames": [], "ep_goal_prop": []}
    ep_counter = 0

    for task in tasks:
        for e in range(episodes):
            seed = seed0 + e
            env, init_state = make_env(task, seed)
            goal_frame, goal_state, _ = rollout_expert(env, init_state, task)
            z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
            goal_obj = goal_state[OBJECT_SLICE].astype(np.float32)
            cost_fn = build_oracle_cost(z_goal=z_goal, **cost_spec_kwargs)
            zg_cpu = z_goal.detach().cpu()
            ep_i = ep_counter; ep_counter += 1
            if keep_frames:
                buf["ep_goal_frames"].append(np.asarray(goal_frame, dtype=np.uint8))
                buf["ep_goal_prop"].append(goal_state[:4].astype(np.float32))

            def _record(z_elite, raw_elite, cost_elite, it, _t=task, _s=seed,
                        _g=goal_obj, _zg=zg_cpu, _ep=ep_i):
                n = len(raw_elite)
                buf["z"].append(z_elite.detach().cpu())
                buf["obj"].append(np.stack([r[OBJECT_SLICE] for r in raw_elite]).astype(np.float32))
                buf["ee"].append(np.stack([r[EE_SLICE] for r in raw_elite]).astype(np.float32))
                buf["z_goal"].append(_zg.unsqueeze(0).expand(n, *([-1] * _zg.dim())))
                buf["goal_obj"].append(np.tile(_g[None], (n, 1)))
                buf["cost"].append(np.asarray(cost_elite, dtype=np.float32))
                buf["task"].extend([_t] * n); buf["seed"].extend([_s] * n)
                buf["iter"].extend([it] * n)
                buf["ep_idx"].extend([_ep] * n)

            if keep_frames:
                def on_elites(z_elite, raw_elite, cost_elite, it, frames_elite=None,
                              _rec=_record):
                    _rec(z_elite, raw_elite, cost_elite, it)
                    buf["frames"].append(np.stack(frames_elite).astype(np.uint8))
                    buf["prop"].append(np.stack([r[:4] for r in raw_elite]).astype(np.float32))
            else:
                on_elites = _record

            obs, _ = env.reset()
            obs, _, _, _, _ = env.step(np.zeros(RAW_A))          # upstream reset_warmup
            rng = np.random.default_rng(seed)
            steps = 0; success = False
            while steps < max_episode_steps:
                plan_h_eff = min(horizon, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
                plan = cem_plan_latent(env, adapter, z_goal, device, plan_h=plan_h_eff, rng=rng,
                                       cost_fn=cost_fn, on_elites=on_elites, **cem_kw)
                for a in plan[: num_act_stepped * FRAMESKIP]:
                    obs, _, _, _, info = env.step(np.clip(a, -1, 1))
                    steps += 1
                    if info.get("success", 0) > 0.5:
                        success = True
                    if steps >= max_episode_steps:
                        break
                if success and not strict:
                    break
            env.close()
            if verbose:
                print(f"  mine {task:16s} seed={seed} rows={sum(len(c) for c in buf['cost'])}",
                      flush=True)

    out = {
        "z": torch.cat(buf["z"], 0),
        "obj": torch.tensor(np.concatenate(buf["obj"], 0)),
        "ee": torch.tensor(np.concatenate(buf["ee"], 0)),
        "z_goal": torch.cat(buf["z_goal"], 0),
        "goal_obj": torch.tensor(np.concatenate(buf["goal_obj"], 0)),
        "cost": np.concatenate(buf["cost"], 0),
        "task": buf["task"], "seed": buf["seed"], "iter": buf["iter"],
    }
    if keep_frames:
        out["frames"] = torch.from_numpy(np.concatenate(buf["frames"], 0))       # (N,H,W,C) u8
        out["prop"] = torch.from_numpy(np.concatenate(buf["prop"], 0))           # (N,4)
        out["ep_idx"] = torch.tensor(buf["ep_idx"], dtype=torch.long)            # (N,)
        out["ep_goal_frames"] = torch.from_numpy(np.stack(buf["ep_goal_frames"]))  # (n_ep,H,W,C)
        out["ep_goal_prop"] = torch.from_numpy(np.stack(buf["ep_goal_prop"]))    # (n_ep,4)
    return out
