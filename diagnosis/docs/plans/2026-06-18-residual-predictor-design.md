# Residual corrective predictor — the predictor-side fix

**Date:** 2026-06-18 · **Status:** design → implementation
**Builds on:** `results/grounded_explore_report.md` (option B null),
`docs/FIX_C1_EXPLAINER.md` §7, `scripts/17` (the dyn-head), `scripts/18`.

## 1. Why this leg exists

The fix ladder eliminated the **planner** as the contact-task bottleneck:
`hdyn` (grounded scoring) and `hexp` (grounded exploration) both leave success at
0/16 — `hexp` is even slightly worse, with the planner acting on the frozen
predictor's *wrong imagined contact dynamics*. Both planner-side levers are
exhausted with the model frozen. The residual lives **inside the predictor's
rollout**: its counterfactual action→object→latent transition at the grasp
boundary disagrees with the simulator, so no cost or search over those rollouts
can plan a contact.

This leg breaks the frozen-everything constraint **minimally**: keep the encoder
and the base predictor frozen; train a small **latent-space residual corrector**
that fixes the object channel *inside* the unroll.

## 2. The model

```
ẑ_{t+1} = F_frozen(z_t, a_t) + Δ_θ(z_t, a_t)
```

- `F_frozen` = the pretrained `EncPredWM` one-step prediction (unchanged).
- `Δ_θ` = a small token-wise, action-conditioned transformer over the latent
  grid (`models/heads/residual_predictor.py`), **zero-initialised** at the output
  so training starts exactly at the frozen predictor (identity correction) and
  only learns the correction.

Trained single-step on the cache (teacher-forced from real `z_t`); applied
**recursively** in the planner's rollout (the empirical multi-step-drift question
is checked in the smoke).

### Loss — object-grounded, not just latent recon

```
L = ‖ẑ_{t+1} − z_{t+1}‖²_latent               (recon: stay near the true next latent)
  + λ_obj · ‖g(ẑ_{t+1}) − obj_{t+1}‖²          (the object channel must be RIGHT)
```

- `z_{t+1}`, `obj_{t+1}` are cached ground truth; `g` = the frozen object probe.
- The recon term alone would mostly fix appearance/arm (BB showed the object is
  only ~9% of the residual); the `λ_obj` term forces the corrected latent to
  **decode to the correct object position** — i.e. the corrected rollout moves
  the object the way the simulator does. This is exactly the cf-corr-0.682 signal
  the dyn-head proved learnable, now baked **into the latent** where both the
  plain L2 cost and the probe can see it.
- Optionally up-weight contact/pre-grasp regime transitions (where the correction
  matters); start uniform, the `λ_obj` term already concentrates capacity where
  the object moves.

## 3. Integration into the closed-loop harness

The base predictor's `unroll` is a black box over H steps, but the correction is
**recursive**, and the predictor **requires proprio in context** (dino_wm: 424 =
384 visual + 20 proprio + 20 action) with `unroll` propagating proprio
internally. So the corrected rollout is a per-step loop that feeds back BOTH the
corrected visual latent and the model's predicted proprio feature:

```
ctxt = {visual: z_t, proprio: encode_proprio(prop_0)}
for t in 0..H-1:
    out   = encpred.unroll(ctxt, act_suffix=a_t)     # frozen 1-step
    z_next = out.visual[-1] + Δ(z_cur, a_t)          # CORRECT the visual only
    ctxt  = {visual: z_next, proprio: out.proprio[-1]}  # proprio uncorrected (it is reliable; BB is the OBJECT)
```

Implemented as a `residual_head` option on `_PlanAdapter.predict_rollout` in
`scripts/18` (per-step loop when set, the batched `unroll` when not), so the
planner, protocol, seeds and pairing are otherwise identical. Cost ≈ H× more
1-step unroll calls per rollout — H≤6, acceptable; smoke measures the real time.

## 4. Experiment — paired A/B (predictor is the variable, cost held at L2)

| arm   | predictor            | cost      | role                        |
|-------|----------------------|-----------|-----------------------------|
| `l2`  | frozen `F`           | visual L2 | baseline (re-run, paired)   |
| `l2c` | corrected `F+Δ`      | visual L2 | **the predictor fix**       |

- Holding the cost at plain L2 isolates the predictor: if the corrected rollout
  moves the object correctly, the L2-to-goal (object-at-goal in the goal latent)
  rewards the grasping action on its own — no grounded cost needed.
- Tasks: `mw-push`, `mw-pick-place`. Start at 8 episodes/task for a first signal
  (corrected rollout is slower), expand to 16 if promising.
- Primary metric: **success rate**. Secondary: paired final state-dist Δ.
- (Stretch arm `hdync` = corrected predictor + grounded cost, only if `l2c`
  shows life.)

## 5. Pre-registered decision rule

- **l2c success > l2 (CI-aware):** the predictor fix works — contact-task success
  beyond the paper, and the BB story closes with a constructive fix. Headline.
- **l2c success ≈ 0 but one-step object error ≪ frozen AND state-dist beats l2:**
  the correction improves the rollout but the grasp needs finer contact geometry
  than the latent resolves → **encoder/representation is the bottleneck** (beyond
  a frozen-checkpoint study). Still a clean, deeper conclusion.
- **corrected one-step object error ≈ frozen:** Δ failed to learn the channel →
  revisit λ_obj / architecture / regime weighting before concluding.

## 6. Sanity gates (smoke, before the sweep)

1. Corrected **one-step** object-decode error < frozen baseline on held-out cache
   (the training must actually reduce object error — printed by `scripts/20`).
2. Corrected **multi-step** (H=6) rollout latent error not worse than frozen
   (no compounding blow-up) — a quick check on cached trajectories.
3. One closed-loop episode runs end-to-end with the corrected adapter; record the
   per-episode wall time to size the sweep.
