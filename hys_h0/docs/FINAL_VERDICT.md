# HyS-JEPA — final verdict (frozen AND fine-tuned forms)

Date: 2026-08-24. `mw-push`, oracle-dynamics CEM, 16 episodes/cell, 112 populations/cell,
3 training seeds, DINO-WM MetaWorld. Encoder-LoRA r=16, straightening + prediction +
reconstruction anchor, no boundary head (dropped per pre-registered plan).

## rho_final, encoder-LoRA form (the pre-registered primary comparison)

| seed | `switch` (contact-aligned) | `random` (matched control) |
|---|---|---|
| 0 | **-0.175** [-0.228, -0.116] CI-clean NEG | **-0.182** [-0.225, -0.138] CI-clean NEG |
| 1 | +0.075 [+0.030, +0.119] CI-clean POS | -0.049 [-0.112, +0.011] |
| 2 | **-0.153** [-0.214, -0.093] CI-clean NEG | -0.009 [-0.066, +0.040] |
| mean | **-0.084** | **-0.080** |

Reference: raw latent L2 (frozen encoder) rho_final = -0.085, CI-clean negative.

**Both arms average out to the same anti-alignment as the unmodified baseline.** Neither
`switch` nor `random` fixes what plain latent L2 was already broken at. Seed 0 and seed 2 are
CI-clean NEGATIVE for `switch` -- fine-tuning did not just fail to help, on 2 of 3 seeds it
reproduced the exact failure mode straightening was meant to cure. `switch` and `random` do not
separate: -0.084 vs -0.080, well inside each arm's own seed-to-seed spread (~0.13).

## Reading this against the frozen-projector result

| form | switch mean | random mean | separation |
|---|---|---|---|
| frozen projector | +0.011 | +0.051 | random beats switch on all 3 seeds |
| encoder-LoRA | -0.084 | -0.080 | indistinguishable, both back at baseline-negative |

The frozen form at least removed the anti-alignment (both arms near/above zero). The
fine-tuned form did not preserve that: rho_final is negative again, in the same range as
doing nothing. Fine-tuning the encoder did not carry forward the one thing the frozen
projector series had established.

## Why: representation collapse under CEM search

This is consistent with the representation health numbers already measured
(`ENC_HEALTH.json`): straightening arms held effective rank 2.0-5.7 out of a 314-dimensional
frozen reference. A representation collapsed to ~3-5 usable dimensions cannot support a cost
function that orders 100+ CEM candidates reliably -- there is not enough geometric room. The
frozen-projector series showed the same collapse (rank ~4) but the object-decode readout still
tracked reasonably (5.3-6.8 cm); apparently even that was not sufficient for the cost to survive
adversarial search, and the fine-tuned encoder made the collapse worse in the region CEM
actually explores, not better.

## Overall status of HyS-JEPA as a planning-cost method

| claim | status |
|---|---|
| Frozen latent trajectory is more curved than chance while physics is straight | CONFIRMED |
| Physical object trajectory kinks 2.4x more at contact-mode switches | CONFIRMED |
| Contact-aligned gating beats a matched random drop (frozen projector) | REFUTED (random won, 3/3 seeds) |
| Contact-aligned gating beats a matched random drop (encoder fine-tune) | REFUTED (indistinguishable, both negative) |
| Straightening + fine-tuning produces a usable CEM planning cost | REFUTED (rho_final back to baseline-negative) |
| Boundary head has a motivating problem to solve | REFUTED (global straightening kept boundary AUC 0.764) |

Both premises that motivated HyS-JEPA are true. Neither the frozen-projector form nor the
fine-tuned-encoder form of the proposed fix produces a cost that survives CEM search better than
the unmodified latent-L2 baseline it was meant to replace. This is the last open branch of the
mechanism (frozen vs. fine-tuned) called for in the original design, and it closes the same way
the frozen branch did.

## What is not closed

- Only `mw-push`, only DINO-WM. Not tested on `mw-pick-place` or the JEPA-WM checkpoint.
- Only r=16 LoRA; the capacity probe (r=64) was never run.
- 3 seeds is the statistical floor established by this program (Phase D needed 5 to see
  a high-variance outlier settle). The negative here is consistent across seeds in sign for
  `random`, mixed for `switch` -- more seeds would sharpen but are unlikely to reverse a mean
  this close to the frozen baseline itself.
