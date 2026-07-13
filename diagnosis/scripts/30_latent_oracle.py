"""Latent oracle — separates the PREDICTOR from the ENCODER representation.

The state oracle (scripts/29) removes both the predictor F and the encoder from
the planning loop, so a success there only says "the gap is the model" without
saying which half. This script isolates the predictor by giving the planner
PERFECT latent dynamics THROUGH THE REAL FROZEN ENCODER:

  * Encoder: the real frozen WM encoder (so the representation is exactly the one
    the WM planner uses — NOT idealised).
  * Dynamics: perfect. For each CEM action candidate we step the SIMULATOR, render
    the resulting frame, and ENCODE it — the true latent the encoder would produce
    for that action. The learned predictor F is never called.
  * Cost: the SAME upstream L2-in-latent distance to z_goal that scripts/18 uses
    (mean squared error on the final latent; α=0, no proprio term).
  * Planner / horizon / num_act_stepped / strict success: identical to scripts/18.

The ONLY thing swapped vs the WM `l2` arm is F → perfect latent dynamics. So:

  latent-oracle SUCCEEDS (while the F-based WM arms fail)
      → the encoder + the L2-in-latent cost are sufficient; the predictor F is the
        bottleneck. This is the predictor-vs-encoder separation, and it licenses
        the GT-supervised predictor fix (scripts/26 λ_msk>0) as the paper's method.
  latent-oracle FAILS (even with perfect latent dynamics)
      → the bottleneck is the ENCODER representation or the L2-in-DINO-latent cost
        geometry (the goal-latent distance has no usable minimum at task success),
        NOT the predictor. A predictor fix cannot beat baseline; the solution must
        target the representation / cost.

Read together with scripts/29: state-oracle PASS + latent-oracle FAIL pins the
blame on the latent geometry / cost (planner and dynamics are both fine).

Because the upstream cost scores only the FINAL latent, each candidate needs just
one render + one encode (the encode is batched across the CEM population on GPU);
the renders are the per-step cost. Needs the real encoder, hence a GPU.

    python scripts/30_latent_oracle.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --tasks mw-reach mw-push mw-pick-place \
        --episodes 16 --out results/metaworld_latent_oracle.csv
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


_cl = _load("closed_loop_eval", "18_closed_loop_eval.py")
_or = _load("oracle_ceiling", "29_oracle_ceiling.py")
make_env, rollout_expert = _cl.make_env, _cl.rollout_expert
render, encode_frame = _cl.render, _cl.encode_frame
FRAMESKIP, RAW_A = _cl.FRAMESKIP, _cl.RAW_A
snapshot, restore = _or.snapshot, _or.restore
from stratification.metaworld_regimes import OBJECT_SLICE  # noqa: E402


def build_oracle_cost(cost, z_goal, *, probe=None, metric=None, ee_probe=None,
                      repr_adapter=None, w_hand=0.5, s_g=0.1276, beta=0.1, gamma_l2=1.0):
    """Build the latent-oracle scoring fn ``(z_fin (B,*frame)) -> cost (B,)``.

    The latent oracle gives PERFECT latent dynamics; the only thing swapped here
    is the *cost*, which the oracle ladder localised as the contact-task wall:

      * ``l2``    — the upstream ``mean‖z_fin − z_goal‖²`` (the null; push/pick 0/16).
      * ``gobj``  — read the object straight off the true encoded final latent and
                    drive ``g(z_fin) → g(z_goal)``; mirrors the ``gobj`` arm in
                    ``scripts/18`` (γ·L2 + β·mean((Δg/s_g_dim)²)). Tests whether the
                    object READOUT is precise enough to plan with under perfect
                    dynamics. ``--gamma-l2 0`` = pure-readout (no L2 anchor).
      * ``metric``/``advmetric`` — a learned latent metric ``d_θ(z_fin, z_goal)``
                    (scripts/33). ``advmetric`` is the same scoring function under a
                    distinct arm name for CSV traceability — it is meant to be paired
                    with a ``--metric-head`` checkpoint hardened against mined
                    CEM-exploited negatives (Track 1 Phase B; see
                    ``scripts/_cem_mining.py``), vs a plain ``metric`` checkpoint.
      * ``stateprobe`` — the EXACT state-oracle cost (scripts/29 ``roll_cost``:
                    ``‖obj−goal‖ + w_hand·‖hand−obj‖``) but with object AND hand read
                    from PROBES instead of sim truth. The fair probe-analogue of the
                    state-oracle (push 16/16): same structure incl. the hand-APPROACH
                    term plain ``gobj`` lacked, so a failure isolates the probe readout,
                    not the missing approach shaping.
      * ``phi``   — Track 2 (heavy fix): ``models/heads/action_repr_adapter.py``'s
                    learned φ = A(z), scored as ``‖φ_obj(z_fin)−φ_obj(z_goal)‖²/s_g²
                    + β·‖φ_extra(z_fin)−φ_extra(z_goal)‖²/s_extra²`` (the two-term
                    "squared norms over scale norms" pattern also used by ``gobj``/
                    ``boundary_aware_cost`` — φ's grounded object subspace is in
                    metres, its free "extra" subspace has its own learned scale
                    stored in the checkpoint, so they must not be summed unscaled).

    Phase-0/3 verdict (2026-07, ``results/oracle_ladder_cost_report.md``): every
    cost above through ``stateprobe`` — including an OFF-POLICY-ROBUST probe —
    plateaus at push 0–2/16 vs the state-oracle's 16/16. CEM exploits whatever
    residual-error pockets a post-hoc frozen-encoder cost has. ``advmetric``/``phi``
    are the two follow-up tracks: harden the cost against its own exploited
    negatives (Track 1), or reshape the representation so the pockets can't form
    (Track 2, ``scripts/37_train_repr_adapter.py``)."""
    zg = z_goal.reshape(-1)

    if cost == "l2":
        def fn(z_fin):
            B = z_fin.shape[0]
            return ((z_fin.reshape(B, -1) - zg[None]) ** 2).mean(-1)
        return fn

    if cost == "gobj":
        if probe is None:
            raise ValueError("--cost gobj needs --probe")
        s_g_dim = s_g / np.sqrt(probe.out_dim)
        with torch.no_grad():
            g_goal = probe(z_goal.unsqueeze(0))                    # (1, out_dim)

        @torch.no_grad()
        def fn(z_fin):
            B = z_fin.shape[0]
            c = ((z_fin.reshape(B, -1) - zg[None]) ** 2).mean(-1)
            obj_term = (((probe(z_fin) - g_goal) / s_g_dim) ** 2).mean(-1)
            return gamma_l2 * c + beta * obj_term
        return fn

    if cost == "stateprobe":
        if probe is None or ee_probe is None:
            raise ValueError("--cost stateprobe needs --probe and --ee-probe")
        with torch.no_grad():
            g_goal = probe(z_goal.unsqueeze(0))                    # (1, 3) object@goal

        @torch.no_grad()
        def fn(z_fin):
            obj = probe(z_fin)                                     # (B, 3) object xyz
            hand = ee_probe(z_fin)                                 # (B, 3) ee xyz
            d_obj = (obj - g_goal).norm(dim=-1)                    # ‖obj − goal‖
            d_app = (hand - obj).norm(dim=-1)                      # ‖hand − obj‖ (approach)
            return d_obj + w_hand * d_app                          # exact scripts/29 form
        return fn

    if cost in ("metric", "advmetric"):
        if metric is None:
            raise ValueError(f"--cost {cost} needs --metric-head")
        zg_frame = z_goal.unsqueeze(0)                             # (1, *frame)

        @torch.no_grad()
        def fn(z_fin):
            B = z_fin.shape[0]
            return metric(z_fin, zg_frame.expand(B, *([-1] * (zg_frame.dim() - 1))))
        return fn

    if cost == "phi":
        if repr_adapter is None:
            raise ValueError("--cost phi needs --repr-adapter")
        obj_dim = repr_adapter.obj_dim
        s_extra = float(repr_adapter.extra_scale)
        with torch.no_grad():
            phi_goal = repr_adapter(z_goal.unsqueeze(0))            # (1, phi_dim)
            obj_goal, extra_goal = phi_goal[:, :obj_dim], phi_goal[:, obj_dim:]

        @torch.no_grad()
        def fn(z_fin):
            phi = repr_adapter(z_fin)
            obj, extra = phi[:, :obj_dim], phi[:, obj_dim:]
            sq_obj = ((obj - obj_goal) ** 2).sum(-1) / (s_g ** 2)
            sq_extra = ((extra - extra_goal) ** 2).sum(-1) / (s_extra ** 2 + 1e-8)
            return sq_obj + beta * sq_extra
        return fn

    if cost == "phil2":
        # HYBRID (Phase D follow-up): raw L2 + the φ_obj grounding term. Same
        # γ·L2 + β·mean((Δ/s_g)²) structure as ``gobj``, but the object estimate
        # comes from the reshaped-encoder φ head, not a frozen probe. Motivation
        # (Phase D job 24128): L2+encLoRA already solves reach 16/16 but push 0/8,
        # while φ+encLoRA gets push 5/16 but regresses reach to 2/16 (φ's object-
        # centric cost is degenerate on the no-object reach task). This is ONE
        # general cost — no task branching: L2 drives the hand (reach), φ_obj adds
        # the grounded object signal (push), and on reach φ_obj≈const so it can't
        # break the L2 solution. Needs BOTH --repr-adapter and (usually)
        # --encoder-lora so φ reads the same reshaped latent L2 scores.
        if repr_adapter is None:
            raise ValueError("--cost phil2 needs --repr-adapter")
        obj_dim = repr_adapter.obj_dim
        s_g_dim = s_g / np.sqrt(obj_dim)
        with torch.no_grad():
            obj_goal = repr_adapter(z_goal.unsqueeze(0))[:, :obj_dim]   # (1, obj_dim)

        @torch.no_grad()
        def fn(z_fin):
            B = z_fin.shape[0]
            c = ((z_fin.reshape(B, -1) - zg[None]) ** 2).mean(-1)
            obj_term = (((repr_adapter(z_fin)[:, :obj_dim] - obj_goal) / s_g_dim) ** 2).mean(-1)
            return gamma_l2 * c + beta * obj_term
        return fn

    raise ValueError(f"unknown --cost {cost}")


@torch.no_grad()
def encode_batch(adapter, frames, proprios, device):
    """Encode a population of final frames in one GPU call. frames: list of
    (H,W,C) uint8-ish arrays; proprios: list of (4,) arrays. -> (B,V,H,W,D)."""
    vis = torch.stack([torch.from_numpy(f.copy()).permute(2, 0, 1).float() for f in frames])
    vis = vis.unsqueeze(1)                                          # (B,1,C,H,W)
    prop = None
    if adapter.uses_proprio():
        prop = torch.stack([torch.from_numpy(p.astype(np.float32)) for p in proprios]).unsqueeze(1)
        prop = prop.to(device)
    z = adapter.encode(vis.to(device), prop)
    return z[:, 0]                                                  # (B,V,H,W,D)


def roll_final_frame(env, snap, plan_raw):
    """Roll one candidate raw-action sequence on the sim from `snap`; return the
    FINAL rendered frame + its proprio (obs[:4]). Sim is rewound by the caller."""
    restore(env, snap)
    last = None
    for a in plan_raw:
        last, _, _, _, _ = env.step(np.clip(a, -1, 1))
    return render(env), last[:4].astype(np.float32), last


def cem_plan_latent(env, adapter, z_goal, device, *, plan_h, num_samples, iterations,
                    elite_frac, var0, rng, cost_fn, on_elites=None, init_mean=None,
                    planner="cem", mppi_beta=5.0):
    """``on_elites(z_elite, raw_elite, cost_elite, iteration)``, if given, is called
    once per CEM iteration with the surviving elite population: ``z_elite`` (n_elite,
    *frame) TRUE latents, ``raw_elite`` a list of n_elite full sim obs (39-dim
    Metaworld state — object/hand slices included), ``cost_elite`` (n_elite,) their
    cost values, in cost-ascending order. This is what CEM actually TRUSTS and
    converges its search around, as opposed to a random off-policy sample —
    `scripts/_cem_mining.py` uses it to mine the frames a cost gets EXPLOITED on.

    Hooks that additionally declare a ``frames_elite`` keyword receive the elites'
    rendered RGB frames (list of (H,W,C) uint8) — needed by encoder-level training
    (scripts/38), where a buffered latent goes stale the moment the encoder moves.
    Detected by signature, so every existing 4-arg hook keeps working unchanged.

    ``init_mean``, if given, is a flat (plan_h*FRAMESKIP*RAW_A,) raw-action seed for
    the CEM mean (e.g. the amortized inverse proposal, scripts/43). When seeded, the
    mean is also injected into the population each iteration (upstream cem_plan's
    mean-inclusion trick) so the elites can RETAIN the seed if the cost genuinely
    prefers it — search then only replaces the seed when the cost approves the swap.
    Default None preserves the original zero-mean behaviour exactly."""
    plan_raw_len = plan_h * FRAMESKIP
    dim = plan_raw_len * RAW_A
    if init_mean is None:
        mean = np.zeros(dim)
    else:
        mean = np.clip(np.asarray(init_mean, dtype=float).reshape(dim).copy(), -1.0, 1.0)
    var = np.full(dim, var0)
    n_elite = max(2, int(num_samples * elite_frac))
    hook_wants_frames = False
    if on_elites is not None:
        import inspect
        hook_wants_frames = "frames_elite" in inspect.signature(on_elites).parameters
    snap = snapshot(env)
    best_cost, best_sample = None, None       # global-best sample (the plan `shooting` commits to)
    for it in range(iterations):
        samples = np.clip(mean[None] + np.sqrt(var)[None] * rng.standard_normal((num_samples, dim)),
                          -1.0, 1.0)
        if init_mean is not None:
            samples[0] = mean                       # mean-inclusion (upstream trick)
        frames, props, raws = [], [], []
        for i in range(num_samples):
            fr, pr, raw = roll_final_frame(env, snap, samples[i].reshape(plan_raw_len, RAW_A))
            frames.append(fr); props.append(pr); raws.append(raw)
        z_fin = encode_batch(adapter, frames, props, device)        # (B,V,H,W,D) — TRUE latent
        costs = cost_fn(z_fin).detach().cpu().numpy()               # only the COST is swapped
        order = np.argsort(costs)
        elite_idx = order[:n_elite]
        if on_elites is not None:
            if hook_wants_frames:
                on_elites(z_fin[elite_idx], [raws[i] for i in elite_idx],
                          costs[elite_idx], it,
                          frames_elite=[frames[i] for i in elite_idx])
            else:
                on_elites(z_fin[elite_idx], [raws[i] for i in elite_idx],
                          costs[elite_idx], it)
        # global-best sample across all iterations — the plan `shooting` returns,
        # and a useful diagnostic for the others.
        if best_cost is None or costs[order[0]] < best_cost:
            best_cost = float(costs[order[0]]); best_sample = samples[order[0]].copy()
        # The ONLY thing that differs across planners is the distribution update.
        # All three minimise the SAME cost under perfect dynamics, so if all three
        # fail (and their trusted frames decode worse than random), the exploitation
        # is planner-general, not a CEM artefact.
        if planner == "cem":
            elites = samples[elite_idx]
            mean = elites.mean(0); var = elites.var(0) + 1e-4
        elif planner == "mppi":
            # soft reweighting of the WHOLE population (no hard elite cutoff):
            # standardise costs (cost is scale-dependent latent MSE), then
            # w ∝ exp(-beta·z-score); larger beta ⇒ sharper concentration.
            c = (costs - costs.mean()) / (costs.std() + 1e-8)
            w = np.exp(-mppi_beta * c); w /= w.sum()
            mean = (w[:, None] * samples).sum(0)
            var = (w[:, None] * (samples - mean[None]) ** 2).sum(0) + 1e-4
        elif planner == "shooting":
            pass                                # fixed proposal (no refit); commit to global best
        else:
            raise ValueError(f"unknown planner {planner!r}")
    restore(env, snap)
    if planner == "shooting":
        return best_sample.reshape(plan_raw_len, RAW_A)
    return mean.reshape(plan_raw_len, RAW_A)


def run_episode(task, seed, env, goal_frame, goal_state, expert_succ, adapter, device, *,
                plan_h, num_act_stepped, max_episode_steps, cem_kw, strict, cost_spec):
    goal_obj = goal_state[OBJECT_SLICE]
    z_goal = encode_frame(adapter, goal_frame, goal_state[:4], device)
    cost_fn = build_oracle_cost(z_goal=z_goal, **cost_spec)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.zeros(RAW_A))                     # upstream reset_warmup
    success, last_success, steps = False, False, 0
    rng = np.random.default_rng(seed)
    while steps < max_episode_steps:
        plan_h_eff = min(plan_h, max(1, -(-(max_episode_steps - steps) // FRAMESKIP)))
        plan = cem_plan_latent(env, adapter, z_goal, device, plan_h=plan_h_eff, rng=rng,
                               cost_fn=cost_fn, **cem_kw)
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
    _pl = cem_kw.get("planner", "cem")
    arm = f"latent_oracle_{cost_spec['cost']}" + ("" if _pl == "cem" else f"_{_pl}")
    return {"task": task, "arm": arm, "seed": seed,
            "success": int(success),
            "success_end": int(last_success), "steps": steps,
            "final_state_dist": float(np.linalg.norm(obs - goal_state)),
            "ee_dist": float(np.linalg.norm(obs[:3] - goal_state[:3])),
            "obj_goal_dist": float(np.linalg.norm(obs[OBJECT_SLICE] - goal_obj)),
            "expert_success_step": expert_succ}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mw-reach", "mw-push", "mw-pick-place"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--num-act-stepped", type=int, default=3)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--cem-num-samples", type=int, default=100)
    ap.add_argument("--cem-iterations", type=int, default=6)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--var0", type=float, default=1.0)
    ap.add_argument("--planner", choices=["cem", "mppi", "shooting"], default="cem",
                    help="test-time optimizer: cem (elite-refit), mppi (softmax-weighted), "
                         "shooting (fixed proposal, argmin over samples x iters)")
    ap.add_argument("--mppi-beta", type=float, default=5.0,
                    help="MPPI temperature (concentration on standardized cost); higher = sharper")
    ap.add_argument("--strict-success", action="store_true")
    # Phase-0 cost gate: only the COST is swapped vs the L2 null (the dynamics stay
    # perfect). See build_oracle_cost for the decision rule.
    ap.add_argument("--cost",
                    choices=["l2", "gobj", "stateprobe", "metric", "advmetric", "phi", "phil2"],
                    default="l2",
                    help="latent-oracle scoring: l2 (null) / gobj (object readout) / "
                         "stateprobe (state-oracle cost via probes: object + hand-approach) / "
                         "metric (learned d_theta) / advmetric (metric, hardened checkpoint) / "
                         "phi (learned action-repr-adapter, scripts/37/38) / "
                         "phil2 (hybrid L2 + β·φ_obj — general cost: L2 for reach, φ_obj for push)")
    ap.add_argument("--probe", default=None,
                    help="object-probe ckpt (scripts/22); --cost gobj/stateprobe")
    ap.add_argument("--ee-probe", default=None,
                    help="ee-probe ckpt (scripts/19); --cost stateprobe (hand-approach term)")
    ap.add_argument("--w-hand", type=float, default=0.5,
                    help="hand-approach weight for --cost stateprobe (matches scripts/29)")
    ap.add_argument("--metric-head", default=None,
                    help="learned latent-metric ckpt (scripts/33); --cost metric/advmetric")
    ap.add_argument("--repr-adapter", default=None,
                    help="action-repr-adapter ckpt (scripts/37 or scripts/38); --cost phi")
    ap.add_argument("--encoder-lora", default=None,
                    help="encoder-LoRA ckpt (scripts/38, Phase D): injected before any "
                         "encoding, so the whole gate — goal encoding + every CEM "
                         "candidate — runs in the reshaped representation. Composes "
                         "with any --cost (phi = the Phase-D gate; l2 = the 'is plain "
                         "L2 now plannable?' variant)")
    ap.add_argument("--s-g", type=float, default=0.1276,
                    help="object scale for the gobj cost (matches scripts/18)")
    ap.add_argument("--beta", type=float, default=0.1, help="gobj object-term weight")
    ap.add_argument("--gamma-l2", type=float, default=1.0,
                    help="gobj L2-anchor weight (0 = pure object-readout cost)")
    ap.add_argument("--out", default="results/metaworld_latent_oracle.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    if args.encoder_lora:
        from models.heads.lora_encoder import load_encoder_lora
        injected, lmeta = load_encoder_lora(adapter, args.encoder_lora, device)
        print(f"encoder LoRA: {args.encoder_lora} (r={lmeta['r']}, "
              f"{len(injected)} modules, val_obj_median_cm="
              f"{lmeta.get('val_obj_median_cm')})", flush=True)
    probe = metric = ee_probe = repr_adapter = None
    if args.cost in ("gobj", "stateprobe"):
        if not args.probe:
            raise SystemExit(f"--probe is required for --cost {args.cost}")
        from models.probes import load_probe
        probe, pmeta = load_probe(args.probe, device)
        print(f"object probe: {args.probe} (out_dim={probe.out_dim}, "
              f"v1_median={pmeta.get('v1_median')})", flush=True)
    if args.cost == "stateprobe":
        if not args.ee_probe:
            raise SystemExit("--ee-probe is required for --cost stateprobe")
        from models.probes import load_probe
        ee_probe, emeta = load_probe(args.ee_probe, device)
        print(f"ee probe: {args.ee_probe} (out_dim={ee_probe.out_dim}, "
              f"v1_median={emeta.get('v1_median')}) w_hand={args.w_hand}", flush=True)
    if args.cost in ("metric", "advmetric"):
        if not args.metric_head:
            raise SystemExit(f"--metric-head is required for --cost {args.cost}")
        from models.heads.latent_metric import load_latent_metric
        metric, mmeta = load_latent_metric(args.metric_head, device)
        print(f"latent metric [{args.cost}]: {args.metric_head} "
              f"(spearman={mmeta.get('val_spearman')})", flush=True)
    if args.cost in ("phi", "phil2"):
        if not args.repr_adapter:
            raise SystemExit(f"--repr-adapter is required for --cost {args.cost}")
        from models.heads.action_repr_adapter import load_repr_adapter
        repr_adapter, rmeta = load_repr_adapter(args.repr_adapter, device)
        print(f"repr adapter: {args.repr_adapter} (phi_dim={rmeta.get('phi_dim')} "
              f"obj_dim={rmeta.get('obj_dim')} extra_scale={rmeta.get('extra_scale')})",
              flush=True)
    cost_spec = dict(cost=args.cost, probe=probe, metric=metric, ee_probe=ee_probe,
                     repr_adapter=repr_adapter, w_hand=args.w_hand, s_g=args.s_g,
                     beta=args.beta, gamma_l2=args.gamma_l2)
    print(f"latent-oracle cost={args.cost} (s_g={args.s_g} beta={args.beta} "
          f"gamma_l2={args.gamma_l2} w_hand={args.w_hand})", flush=True)
    cem_kw = dict(num_samples=args.cem_num_samples, iterations=args.cem_iterations,
                  elite_frac=args.elite_frac, var0=args.var0,
                  planner=args.planner, mppi_beta=args.mppi_beta)
    print(f"planner={args.planner}" + (f" (mppi_beta={args.mppi_beta})"
          if args.planner == "mppi" else ""), flush=True)
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
                                adapter, device, plan_h=args.horizon,
                                num_act_stepped=args.num_act_stepped,
                                max_episode_steps=args.max_episode_steps, cem_kw=cem_kw,
                                strict=args.strict_success, cost_spec=cost_spec)
                env.close()
                w.writerow(r); f.flush(); rows.append(r)
                print(f"  {task:16s} ep{e:02d} latent_oracle[{args.cost}] "
                      f"success={r['success']} end={r['success_end']} "
                      f"obj_goal={r['obj_goal_dist']:.3f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    print(f"\n=== latent oracle [cost={args.cost} planner={args.planner}]: success by task ===")
    for task in args.tasks:
        tr = [r for r in rows if r["task"] == task]
        s = sum(r["success"] for r in tr); se = sum(r["success_end"] for r in tr)
        print(f"  {task:16s} success {s}/{len(tr)}   success_end {se}/{len(tr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
