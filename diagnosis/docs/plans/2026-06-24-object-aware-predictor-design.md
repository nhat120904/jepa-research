# Object-aware predictor — fix the SCORE side end-to-end (model-side, C-JEPA-style)

**Date:** 2026-06-24 · **Status:** design
**Builds on:** `docs/plans/2026-06-23-predictor-cf-channel-design.md` (Option D —
additive cf head, gate V3-spatial corr **+0.37**, closed-loop CI-positive cost but
**0/32** success, saturates < teacher 0.68 / < gate 0.5; §6 explicitly names the
escalation: *"partial predictor fine-tuning (LoRA-style adapters inside
`ViTPredictor`) rather than an additive head"*), `scripts/25_train_cf_predictor.py`,
`scripts/24` (V3-spatial gate), `scripts/23` (rollout fidelity), `scripts/17`
(dyn-head teacher, cf-corr 0.68), `scripts/22` (spatial probe, 2.1 cm),
`scripts/18 --residual-head` (closed-loop). **External anchor (verified, read
directly):** `world_model/causal-jepa.pdf` — Nam, Le Lidec, Maes, LeCun,
Balestriero, Feb 2026 (arXiv 2602.11389).

## 1. Why this leg exists — the additive head has a ceiling, not a bug

The wall is pinned to the frozen predictor's **counterfactual object channel**:
from a similar-state anchor, applying different neighbour actions, the *true* object
spread is 2.42 cm but the predictor's predicted spread is 0.74 cm (corr −0.015,
`scripts/24`). Option D (additive `Δ_θ(z,a)` + cf-distillation from the dyn-head)
*moved* the diagnostic for the first time (corr **−0.015 → +0.37**, spread
0.74 → 1.71 cm) and gave a CI-positive closed-loop cost (+0.087 pick-place), but:

- it **saturates at corr ~0.37** — below the gate target (0.5) and the teacher
  ceiling (dyn-head cf-corr 0.68). Longer / stronger training (`_strong.pt`,
  λ_cf=30) made it *worse* (corr +0.24, over-fit), confirming the ceiling is the
  **additive-head form**, not the data or the optimiser;
- success stays **0/32**. The corrected *readout* is discriminative, but the
  recursively-rolled latent is not a *dynamically consistent* contact rollout
  (Option D §6, risk "readout-fixed ≠ rollout-fixed").

The root cause is the **training target**: full-latent L2 buries the object's
action-dependence (the bifurcation is ~9 L2 units against a 106-unit residual), so a
small head bolted on top can only distill a teacher, never exceed it. To break the
ceiling we must change the **objective the predictor's own rollout is trained
under**, so action-mediated object dynamics are represented *inside* the latent the
planner searches over.

## 2. The idea (C-JEPA, adapted): make interaction reasoning *necessary*

C-JEPA's core mechanism: train the latent predictor under **object-level masking**.
At each step a subset of object slots is masked across the history window (a minimal
identity anchor kept), and the predictor must recover the masked object's trajectory
**from the other objects' states + auxiliary variables (action, proprio)**. This is
a latent-level intervention with counterfactual-like effect: the objective is
*minimised only if* the predictor models how one entity's future depends on the
others and on the action — exactly the action→object channel that L2 buries. C-JEPA
reports ~20% absolute gain on counterfactual VQA (CLEVRER) and PushT-MPC parity
using ~1% of patch tokens (Sec. 5–6 of the paper). Encoder stays **frozen**; only
the predictor (and a slot grouping) trains — compatible with our frozen-everything
ethos.

We adapt this to our Metaworld setting in **two rungs, cheap-first**, both reusing
the cached latents and the spatial probe (no env, no encoder training).

### Rung A (primary, cheap): partial-unfreeze + masked-object objective
The escalation Option D §6 names, made concrete. Instead of an additive head:

- Insert **LoRA adapters** (rank 8–16) into the `ViTPredictor` blocks (the
  `EncPredWM` predictor reached via the adapter; keep encoder + base weights frozen,
  train only the adapters — ~1–3M params). This lets the **rollout itself** carry an
  action-discriminative object response, not just a post-hoc readout.
- Train with the Option-D losses **plus a masked-object term**: on each
  similar-state neighbourhood, mask the object slice of the predictor's context and
  require the corrected rollout to still predict the *neighbour-specific* object
  delta (read via the frozen spatial probe `g`, 2.1 cm). Masking removes the
  shortcut of copying the object straight through, forcing the action to carry it:
  ```
  L = ‖ẑ(z,a) − z_{t+1}‖²                         (recon, latent sane)
    + λ_obj · ‖g(ẑ(z,a)) − obj_{t+1}‖²            (factual object, spatial probe)
    + λ_cf  · ‖ δ̂_j − δ*_j ‖²                      (cf distillation, dyn-head teacher)
    + λ_msk · ‖g(ẑ(z_mask,a)) − obj_{t+1}‖²        (NEW: object-context masked)
  ```
  where `δ̂_j`/`δ*_j` are the predicted / teacher object spreads about the anchor
  (Option D §3), and `z_mask` zeros (or replaces with a learned mask token) the
  probe-identified object tokens of the context.
- **Multi-step consistency** (guards the recursive drift Option D flagged): add a
  2–3 step teacher-forced unroll term so `Δ` applied recursively does not blow up.

New script `26_train_predictor_lora_cf.py` (extends `scripts/25`: swap the additive
head for LoRA adapters; add the masked-object + multi-step terms). Saves a checkpoint
loadable by the existing `scripts/18 --residual-head` path (or a sibling
`--predictor-lora` flag if the adapter format differs).

### Rung B (deeper, if A still < gate): slot-level object predictor
Full C-JEPA: run **Slot Attention** on the frozen DINO patch latents → N object
slots; train a slot predictor with the **object-level masked latent prediction
objective** (C-JEPA Eq. 5–6), action + proprio as auxiliary conditioning. This is
the principled, JEPA-native fix — the predictor is structurally an
interaction-reasoner — and is the right home if the partial-unfreeze still can't
represent a dynamically-consistent contact rollout. New script
`27_slot_object_predictor.py`. Higher cost (slot grouping + predictor retrain), so
gated behind Rung A's result.

## 3. Validation — diagnostic-first (the lesson option C taught)

Identical gating discipline to Option D; **no closed-loop spend until the in-loop
gates pass**:

1. **V3-spatial (`scripts/24 --residual-head`/`--predictor-lora`)** — primary gate.
   PASS = predicted cf spread 0.74 → **≥ ~2.0 cm** and **corr −0.015 → ≥ 0.5**
   (strictly beating Option D's +0.37). This is the number that *defines* the
   failure; Rung A must clear what the additive head could not.
2. **Rollout fidelity (`scripts/23`)** — guard. Factual multi-step tracking must not
   regress (~3 cm / 6 steps); recursive adapter application must not blow up the
   latent.
3. **Only if (1)+(2) pass → closed-loop (`scripts/18`)**, contact tasks, paired
   against `l2`/`hdyn`. Compose with the planner-side leg
   (`2026-06-24-inverse-action-proposal-design.md`): this leg fixes **scoring**, that
   leg fixes **sampling**.

## 4. Decision / falsification

- **Gate 1 still ~0.37 under partial-unfreeze:** the latent cannot carry a
  recursively-usable counterfactual object signal from this objective → go to Rung B
  (slot predictor) before concluding a deeper architectural limit.
- **Gate 1 ≥ 0.5, rollout holds, closed-loop flips success (composed with proposal):**
  the predictor rollout *was* the model-side half of the wall — the diagnosis+fix
  story is complete.
- **Gates pass, composed run still 0/16:** the corrected rollout still mis-scores
  *proposed* contacts → escalate to Rung B / deeper unfreeze; the planner-side leg
  has already ruled proposal in as healthy.

## 5. Scope, cost, honesty

- Breaks "frozen-everything" minimally: Rung A trains LoRA adapters only (encoder +
  base predictor weights frozen); Rung B adds slot grouping + a predictor. Cache-only,
  no env.
- Metaworld only (needs object state to define/supervise the masked-object target);
  DROID transfer of the *recipe* remains a known honest negative (no object label) —
  this leg does not change that scope.
- Fallbacks if both rungs underdeliver (kept as related-work, lower priority — both
  need encoder training and are toy-validated, so weaker fits than C-JEPA):
  temporal-straightening curvature regulariser (`world_model/temporal-straightening-
  latent-planning.pdf`, geometric conditioning for gradient planning) and
  value-guided IQL latent shaping (`world_model/value-guided-jepa.pdf`).
