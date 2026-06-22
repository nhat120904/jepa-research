# Grounded exploration for closed-loop contact tasks (option B)

**Date:** 2026-06-18 · **Status:** design → implementation
**Owner:** local 12 GB RTX box · **Builds on:** `results/closed_loop_report.md`,
`docs/FIX_C1_EXPLAINER.md` §7, `scripts/18_closed_loop_eval.py`.

## 1. The gap this targets

The closed-loop sweep (2026-06-12) established: on MW-push / MW-pick-place the
grounded object term (`hdyn`) improves final state-distance by a measured
**+0.089 [+0.022, +0.162]** but flips **zero** successes (still 0/16 both arms).
The report localized the residual bottleneck precisely:

> "CEM rarely *samples* an action sequence that creates the contact the grounded
> term could then score. The remaining bottleneck is exploration/imagination at
> the contact boundary (planner-side), not the scoring metric."

So we have a fix that is demonstrably right at the **scoring** level (cf-corr
+0.682, BB −50%) but enters CEM only as a `traj_cost_fn` — it re-ranks the 300
sampled plans. If none of the 300 creates contact, re-ranking cannot help.

**Hypothesis (B):** the *same* grounded channel, used to shape the **search**
rather than only the final score, can drive CEM toward contact-creating plans
and move closed-loop success — or, failing that, give a clean negative that
locates the bottleneck inside the frozen predictor's imagined rollout (a
cost function cannot reach it), which is itself a paper-grade conclusion.

## 2. Why object-dynamics alone cannot explore

`h(z,a) → Δobject` correctly predicts ≈0 object motion for **every** action when
the gripper is far from the object (no action moves an untouched object). So a
search signal built only from `h` (or from the object-at-goal term) is **flat
across samples** until the arm is already at the object — exactly the regime CEM
fails to reach. The missing ingredient is an **approach** signal: get the
predicted end-effector to the object so contact becomes *reachable*; only then
does the object-dynamics term have variance to exploit.

The plain L2 visual cost does pull the arm — but toward the **goal frame's** ee
pose (arm where the object should *end up*), not toward where the object *is
now*. Those coincide in free space and diverge precisely at the grasp boundary.

## 3. Design — the `hexp` arm

Two grounded terms added to the upstream visual cost, both evaluated **densely**
along the imagined rollout (not just at the horizon), both in world-position
units (xyz, metres) on the shared object scale `s_g`:

```
obj(0) = g_obj(z_t)                         # probe on the real current frame
for t in 0..H-1:
    A += || g_ee(pred[:,t]) - obj(t) ||^2    # APPROACH: ee → current object
    obj(t+1) = obj(t) + h(pred[:,t], a_t)    # integrate grounded object motion
    M += || obj(t+1) - g_obj(z_goal) ||^2    # MANIPULATE: object → goal (dense)

cost = MSE_perdim(z_H, z_goal)               # upstream visual term (unchanged)
     + lambda_app * mean_perdim(A / H) / s_g_dim^2
     + lambda_obj * mean_perdim(M / H) / s_g_dim^2
```

- **APPROACH** uses a new probe `g_ee(z) → ee xyz`, trained on the cache exactly
  like the object probe (`EE_SLICE = state[0:3]`). ee is the most salient moving
  thing in the frame, so its held-out decode error is expected tiny (reported as
  the probe's V1). This is the term that creates contact opportunities.
- **MANIPULATE** is the *dense* form of the existing `hdyn` object-at-goal term
  (sum over steps, not final-only), so partial object progress is rewarded and
  provides selection gradient earlier in the CEM iterations.
- The object trajectory `obj(t)` uses the **dyn-head integration** (the grounded
  counterfactual channel), not `probe(pred[:,t])` — the imagined latent is
  BB-blind by construction; `h` is the channel that actually responds to actions.
- `s_g_dim = s_g / sqrt(3)` matches script 18's existing per-dim normalisation so
  the weights are comparable to the `hdyn` `beta`.

This keeps the encoder, predictor, object probe and dyn-head **all frozen** — the
only new artifact is the ee probe (cache-only, ~0.3 M params). Pure planner-side
change, consistent with the "supervise a metric, not the model" thesis.

## 4. Experiment — paired A/B/C

One sweep, arms paired on the same env rand_vec + CEM noise per (task, seed):

| arm    | cost                                            | role                     |
|--------|-------------------------------------------------|--------------------------|
| `l2`   | visual MSE only                                 | baseline (re-run for pairing) |
| `hdyn` | visual + final object-at-goal                   | scoring-only fix (prior result) |
| `hexp` | visual + dense approach + dense object-progress | **the exploration fix**  |

- Tasks: `mw-push`, `mw-pick-place` (the contact tasks, both 0% today). Optional
  `mw-reach` no-harm check at reduced episodes.
- 16 episodes/task, seeds 10000–10015 (matches the prior sweep's protocol).
- Primary metric: **success rate** (sim flag). Secondary: paired final
  state-distance Δ vs `l2` (same bootstrap as the report).
- Output: `results/metaworld_grounded_explore.csv` + a report.

## 5. Decision rule

- **hexp success > 0 and > l2 (CI-aware):** option B works — the grounded channel
  improves *planning success* when used for exploration. This is the missing
  positive result. Promote to the paper's planning leg.
- **hexp success ≈ 0 but state-dist Δ beats hdyn:** exploration helps approach
  but the frozen predictor's imagined contact dynamics still block the grasp →
  bottleneck is *inside* the world model's rollout, not the cost. Clean negative.
- **hexp ≈ hdyn:** the approach signal does not change the search → reconsider
  weighting / contact-prior, or fall back to option A (honest framing).

## 6. Risk / ops

- Same fragile-Windows ops as sweep `18`: run under `run_with_watchdog.ps1`,
  resume from CSV via a `run_sweep_resume`-style loop, chunked unroll
  (`CAI_JEPA_PLAN_CHUNK=150`).
- Two extra probe evals per rollout step — negligible vs the unroll cost.
- Smoke first (1 task, 2 episodes, all arms): verify the three cost terms are
  commensurate (print magnitudes) before committing to the multi-hour sweep.
- Weights `lambda_app`, `lambda_obj` are the only new knobs; calibrate in smoke
  so the grounded terms actually drive elite selection (not swamped by visual).
