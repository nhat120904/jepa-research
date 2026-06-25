# Option D — train the predictor's counterfactual object channel

**Date:** 2026-06-23 · **Status:** design
**Builds on:** `results/spatial_h_report.md` (spatial-h null — scoring/aim are NOT
the wall), `docs/plans/2026-06-18-residual-predictor-design.md` + `scripts/20`
(option C: residual corrector, **gamed the pooled probe**, illusory), `scripts/24`
(V3-spatial — the diagnostic that caught the gaming; the true locus), `scripts/17`
(the dyn-head `h(z,a)→Δobject`, cf-corr **+0.68** — the proven teacher), `scripts/22`
(spatial probe, 2.1 cm), `scripts/23` (rollout fidelity / Test 2), `scripts/18`
(closed-loop, `--residual-head` → `l2c`/`hdync` arms).

## 1. Why this leg exists — every other lever is exhausted and points here

The wall is now pinned to **one place**: the frozen predictor's *counterfactual*
object response is dead. V3-spatial (`scripts/24`) measured it through the
ungameable 2.1 cm spatial probe: from a similar-state anchor, applying different
neighbour actions, the **true** object spread is **2.42 cm** but the predictor's
predicted spread is **0.74 cm, corr −0.015**. The predictor gives ~identical object
predictions for different actions.

Everything around that channel is healthy or exhausted:
- encoder ✓ (object decodable to 2.1 cm), predictor *factual* tracking ✓ (3 cm/6 steps);
- planner scoring ✓ and planner aim ✓ — **spatial-h** just proved 0/32 with a 2 cm
  aim + object-dominant β=5 cost that *did* redirect the arm to the object (ee 6–7×)
  but couldn't convert, because the rollout it plans over mis-imagines contact;
- the signal **is learnable**: the dyn-head reaches cf-corr +0.68 from the *same*
  cross-sample neighbourhood data — the bottleneck was the **training target**, not
  capacity or data.

So the only remaining model-side move is to put the dyn-head's action-discriminative
object response **inside the predictor's latent rollout**, where the planner's cost
and search actually operate.

## 2. Why option C failed, and what D changes

Option C (`scripts/20`) trained `ẑ=F(z,a)+Δ(z,a)` with
`L = ‖ẑ−z_{t+1}‖² + λ_obj‖g_pooled(ẑ)−obj_{t+1}‖²`. Two fatal properties:
1. **Pooled probe** `g_pooled` (6.4 cm, noisy) → Δ could lower the loss by gaming the
   pooled readout without moving the real object channel (Test 2: corrected factual
   tracking 2.6→3.8, *no better* than frozen 2.6→3.2).
2. **Purely factual** — a single `(z,a)→obj_{t+1}` term. Nothing supervises that
   *different* actions from the *same* state produce *different* objects. The dead
   channel is exactly the thing C never trained.

**Option D fixes both:** decode with the **spatial probe** (ungameable), and add the
decisive new term — a **counterfactual distillation loss** that forces the corrected
rollout's action→object response to match the dyn-head's (cf-corr 0.68).

## 3. The model & loss

Reuse `models/heads/residual_predictor.py` (the dyn-head proved a small,
zero-initialised head has enough capacity — this isolates "was it target or
capacity?": it was the target). Encoder + base predictor stay **frozen**; only `Δ_θ`
trains, cache-only.

```
ẑ(z,a) = F_frozen(z,a) + Δ_θ(z,a)
g       = g_spatial   (frozen spatial object probe, scripts/22 — 2.1 cm, ungameable)
h       = frozen dyn-head h(z,a)→Δobject (scripts/17, cf-corr 0.68 — the teacher)

L = ‖ẑ(z,a) − z_{t+1}‖²                                   (recon: keep latent sane)
  + λ_obj · ‖g(ẑ(z,a)) − obj_{t+1}‖²                       (FACTUAL object, spatial probe)
  + λ_cf  · L_cf                                            (THE new term)
```

**`L_cf` (counterfactual distillation), per similar-state neighbourhood** `N(i)`
built with `state_neighbours` (same anchors/radius as `scripts/24`/BB): for anchor
`z_i` and neighbour actions `{a_j}`, the *predicted* object delta read off the
corrected rollout must track the *teacher* object delta:

```
ô_j   = g( F(z_i, a_j) + Δ(z_i, a_j) )            # corrected predicted object
δ̂_j  = ô_j − mean_k ô_k                          # predicted spread about the anchor
δ*_j  = h(z_i, a_j) − mean_k h(z_i, a_k)          # dyn-head (teacher) spread
L_cf  = ‖ δ̂_j − δ*_j ‖²   (+ optional 1−corr(δ̂, δ*) to target the corr metric directly)
```

This bakes the corr-0.68 action→object signal **into the latent** the planner rolls
over. (Teacher = dyn-head because its cf-corr is already proven; an ablation can swap
in *true neighbour outcomes* as the target to check the dyn-head isn't a ceiling.)

New script: **`scripts/25_train_cf_predictor.py`** (extends `scripts/20`'s loop with
`g_spatial`, the dyn-head teacher, and the neighbourhood `L_cf`). Saves
`checkpoints/cf_predictor_<model>.pt` in the residual-head format `scripts/18
--residual-head` already loads.

## 4. Validation — diagnostic-FIRST, the lesson option C taught us

C's closed-loop looked like the **best leg** (push Δ +0.227) and was **illusory**.
So D is gated on the in-loop diagnostics *before* any closed-loop spend:

1. **V3-spatial (`scripts/24`) on the corrected predictor** — the primary gate.
   `--residual-head checkpoints/cf_predictor_<model>.pt` (add the flag to `scripts/24`).
   PASS = predicted cf spread **0.74 → ≥ ~2.0 cm** and **corr −0.015 → ≥ 0.5**.
   This is the exact number that defines the failure; D must move it or it has not
   touched the locus.
2. **Rollout fidelity / Test 2 (`scripts/23`) on the corrected predictor** — guard.
   Factual multi-step tracking must **not regress** (~3 cm/6 steps); a recursively
   applied Δ must not blow up the latent (multi-step drift check).
3. **Only if (1)+(2) pass → closed-loop (`scripts/18`)** `l2c`/`hdync` arms (corrected
   rollout), 16 paired episodes/task on the contact tasks, on the cluster
   (`MUJOCO_GL=egl`, reuse `scripts/slurm_spatial_h.sh` pattern with `--residual-head`).

## 5. Decision / falsification

- **Gate 1 fails** (cf-corr stays ~0 even with explicit distillation): the latent
  representation cannot carry a recursively-usable counterfactual object signal →
  deeper architectural problem (partial predictor unfreeze / a dedicated object
  latent), not a small-head fix.
- **Gates 1–2 pass, closed-loop flips success:** the predictor rollout *was* the
  whole wall — strong result, the model-side fix completes the story.
- **Gates 1–2 pass, closed-loop still 0/16:** the model is now right but CEM still
  never *proposes* a contact → composes with **lever #2** (BC-seeded / sub-goal
  proposal). This is the clean hand-off: D + lever #2 together, each now isolated.

## 6. Risks

- **Multi-step drift.** Δ is trained single-step (teacher-forced) but applied
  recursively in the rollout; small per-step object corrections can compound. Mitigate
  with the recon term, a short multi-step consistency term (2–3 step unroll in
  training), and the `scripts/23` guard before closed-loop.
- **Distillation ceiling.** The dyn-head teacher is itself cf-corr 0.68, not 1.0; the
  corrected latent can be at most as discriminative. Ablation with true-outcome targets
  bounds this.
- **Readout-fixed ≠ rollout-fixed.** D could satisfy gate 1 (the probe reads a
  discriminative object) yet still not produce a *dynamically consistent* contact
  rollout. Gate 2 + closed-loop catch this; if it bites, escalate to partial predictor
  fine-tuning (LoRA-style adapters inside `ViTPredictor`) rather than an additive head.

## 7. Scope & order of work

Breaks "frozen-everything" minimally (only `Δ_θ` trains). Order:
1. `scripts/25_train_cf_predictor.py` (spatial probe + dyn-head teacher + `L_cf`), cache-only.
2. Gate on `scripts/24` (+`--residual-head`) and `scripts/23` — **diagnostic before closed-loop**.
3. If gated-in, closed-loop `l2c`/`hdync` on the cluster.
4. From the result: success → write up; 0/16-despite-gates → run **lever #2** on top.
This is the precisely-motivated, falsifiable shot at contact success; it is also a
clean paper result either way (the fix that moves the diagnostic, with an honest
closed-loop readout).
