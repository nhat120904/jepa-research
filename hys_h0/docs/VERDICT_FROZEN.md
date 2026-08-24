# HyS-JEPA — verdict on the frozen-encoder form

Date: 2026-08-22. DINO-WM MetaWorld `mw-push`, oracle-dynamics CEM, 16 episodes/cell,
seed0=41000, 112 candidate populations per cell, 3 training seeds per arm.
Series v2 (curvature + prediction + reconstruction + VICReg + boundary head).

## The decisive comparison: contact-aligned gating vs a matched random drop

rho_final (Spearman between the proxy cost and true shaped cost over the final CEM population):

| seed | `switch` (drop contact-mode switches) | `random` (drop a matched 6.6% at random) |
|---|---|---|
| 0 | -0.003 [-0.031, +0.026] | **+0.067 [+0.021, +0.105]** |
| 1 | +0.000 [-0.029, +0.029] | +0.018 [-0.015, +0.050] |
| 2 | **+0.036 [+0.007, +0.060]** | **+0.068 [+0.028, +0.105]** |
| mean | **+0.011** | **+0.051** |

Reference: raw latent L2 is -0.085 [-0.135, -0.033], CI-clean negative.

**The random control beats contact-aligned gating on every seed.** This is not a null; it is a
reversal. Knowing where contact actually begins and ends is worth less than dropping the same
number of transitions arbitrarily.

## Consistent with every upstream measurement

`switch` and `random` were already indistinguishable before the planner saw them (3 seeds each):

| metric | `switch` | `random` |
|---|---|---|
| val curvature | 0.515 | 0.502 |
| object decode (median) | 6.58 - 6.86 cm | 6.33 - 6.85 cm |
| boundary AUC | 0.752 - 0.788 | 0.735 - 0.781 |

## What survives

Something real is happening, but it is not the proposed mechanism:

1. **A learned bottleneck removes CEM's anti-alignment.** Raw latent L2 has rho_final CI-clean
   negative (-0.085): under search the cost becomes actively anti-correlated with task progress.
   Every projector arm removes that, and `random` turns it CI-clean positive (mean +0.051).
2. **Dropping ~6.6% of transitions from the curvature loss helps.** Which 6.6% does not matter.
   This is a regularisation effect, not a contact effect.
3. **The `off` control (no curvature term at all) also removes the anti-alignment.** So even the
   straightening objective itself has not demonstrated a planning benefit over a plain learned
   bottleneck.

## What is refuted, in this setting

- **Contact-aligned gating.** `random` >= `switch` on all three seeds and on every upstream metric.
- **The boundary head's motivation.** Global straightening keeps boundary AUC at 0.764; it does
  not blur the contact event, so there is nothing for the head to protect. `off` (no straightening)
  has the *lowest* AUC of all arms.
- **The earlier promising signal.** v1 `switch` seed 0 gave rho_final +0.071 CI-clean positive.
  With the same gate in v2, seed 0 gives -0.003. That was seed noise, exactly the Phase-D pattern.

## What is NOT refuted

The full method with **encoder + predictor fine-tuning** has not been tested. The mechanism that
could change under fine-tuning is specific: a projector can only select a subspace of frozen
DINOv2 features, so if temporal straightness and object precision are in tension inside that
space it must trade one for the other -- which is what the object-decode regression from 2.9 cm
to 6.6-7.6 cm shows. A fine-tuned encoder can create features rather than select them.

Two cautions against over-weighting that hope:
- The `off`-control equivalence is not explained by information loss (`off` loses none and still
  matches), so extra encoder capacity does not obviously address it.
- Phase D already ran encoder-LoRA on this exact arena: push {5,0,2,1,1}/16, CI [0, 3.7], a
  5-seed null. What is new here would be the objective, not the capacity.

The premise remains confirmed and is independent of all of this: the physical object trajectory
kinks 2.4x more at contact-mode switches (+0.0413, CI [+0.0349, +0.0481]), and the frozen latent
is more curved than chance (1.096 vs 1.000) while the physics it encodes is straight (0.03-0.32).
The premise being true did not make the mechanism work.
