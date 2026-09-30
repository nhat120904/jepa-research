# Label regime A: labels on executed segments only (PushT, offline, pre-registered)

Pinned 2026-09-29, before any code or data for it. User decision of 2026-09-29: run this screen and
`FAST_POLICY_HEADROOM_PROTOCOL.md` in parallel, and fix the paper framing on their results.

## Why

Under branched labels (every sibling labelled, `CTA_E2E_PROTOCOL.md`), CTA and D_direct differ offline by about
0.2 retained gap. Closed-loop gain is roughly Δretention × headroom. On PushT with the released policy,
headroom cannot exceed 1 − P0 ≈ 38 pp, and even a perfect world model turns CTA into FULL (offline 0.80,
closed loop +10 pp [−2, +22] on n = 100). So CTA − DIRECT stays below what 600 roots can resolve. The lever
that could separate them is Δretention, and it depends on how labels are obtained, not on the arena.

Hypothesis: when labels exist only on segments the robot actually executed, CTA has a structural advantage.
- Its reader maps an outcome code to progress, so factual outcomes are enough to train it.
- Its world model learns action sensitivity from unlabelled action-diverse data.
- A direct scorer D(C, A, g) must learn action sensitivity from the labels themselves. With one labelled action
  per state it can fit the label from C and ignore A. UF-OPS (arXiv 2603.10282) reports this failure for one
  verifier variant.

## Data (no new collection, no new roots)

Cached features of the round-1 collection: training roots 30250–31049, offline dev roots 2000–2099.
The dev roots were already used for debugging, so this is a screen, not a confirmation.

## Label regimes

| Regime | Label used in training | Loss |
|---|---|---|
| **factual (primary, gating)** | Candidate 0 only (the executed chunk): y = cov8[d, 0] − cov8[d−1, 0] within a root, for d ≥ 1. The other 7 siblings carry no label | Huber, δ = 1, on y standardised with training mean and sd |
| success (secondary, reported) | Candidate 0 only: the P0 episode success of the root, from the collection shard JSON `outcomes` | BCE with logits |
| branched (reference) | Existing runs (round 1 and the parameter-matched DIRECT job); not rerun | pairwise rank on cov8 |

Decision 0 of each root enters no label loss. It still enters reconstruction and world-model training.

## What each model sees

Everything not listed is the round-1 configuration (`CTA_DEBUG_LOG.md`): M = 16, FSQ (8, 8, 4), `--sat 1.0`,
`--wm-contrast 1.0`, λ = 0, no stage 3, no reader adaptation, seed 0, the same step counts.

| Model | Unlabelled (all 8 siblings) | Labelled (candidate 0 only) |
|---|---|---|
| Codec (encoder + decoder) | conditional reconstruction of future tokens | reader loss backpropagates into the encoder |
| Reader D(C, S, g) | — | factual or success loss on candidate 0's code |
| FULL (tier 1) | — | same loss on candidate 0's actual future |
| DIRECT D(C, A, g) | — | same loss on (C, A₀). Architecture: the parameter-matched configuration of the DIRECT job running on 2026-09-29 (id recorded in the ledger at submission) |
| **DIRECT+aux** | a head on the DIRECT trunk predicts the future-token change of every sibling from (C, A_k), with the FutureDecoder target and weight 1 | same loss as DIRECT |
| World model + prior | stage 2 on source codes of all siblings | none |

DIRECT+aux is the decisive control. It gets the same unlabelled data as CTA and lacks only the action-free
bottleneck. If it matches CTA, the advantage comes from auxiliary data and not from the code.

## Evaluation

The existing ladder on dev decisions, labelled with all 8 cov8 values. Tiers: full, code, pred, pred_soft, direct, direct_aux.
- Within-bank Spearman and retained gap, with root-clustered 95% intervals.
- Paired differences: root bootstrap (10,000) of the per-root retained-gap numerator difference over the common denominator.
- **The primary tier is pred_soft, fixed now.**

## Pre-registered decision (factual regime only)

**PASS** if both conditions hold:
1. gap(pred_soft) − max(gap(direct), gap(direct_aux)) ≥ 0.25, and the paired 95% CI lower bound is > 0.10.
   The paired CI is taken against the baseline with the higher point estimate.
2. gap(pred_soft) ≥ 0.30. Below that, a relative win still leaves too little to steer with.

Otherwise **STOP A**.

After reading, there is no retuning of loss, δ, steps or tier on these dev roots.

| Pattern | Reading |
|---|---|
| PASS | The factual-label claim is viable. Next step decided with the user: dev closed loop on 2100–2199, then sealed 3000–3399 with 3 seeds |
| pred_soft high, direct_aux close | The advantage comes from unlabelled data, not the code; the code claim fails |
| every tier, including full, below 0.2 | Factual labels cannot resolve siblings for any model (the C2 lesson); A is dead |
| full high, code or pred low | Codec or world model loses it; no debug rounds are budgeted here |

## Reported, not gating

- Spearman per tier.
- Code diagnostics (perplexity, bank distinctness, action bits, R²).
- The success-regime ladder.
- The branched reference numbers alongside.

## Compute and implementation

- Code:
  - `scripts/cta_train.py --labels {branched,factual,success}`. The default `branched` must reproduce the current path.
  - `Split` loads root success from the shard JSONs.
  - A DIRECT+aux model.
  - Unit tests for the label construction, in particular that no sibling label other than candidate 0 is ever read.
- Jobs:
  - a CPU smoke;
  - two trainings (factual, success): 1× MIG 3g.40gb, 8 CPU, 128 GB, limit 3:00 each, afterok on the smoke.
- About 4 GPU-h.
