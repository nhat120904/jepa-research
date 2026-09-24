# CTA end to end: build the whole method, then debug with the attribution ladder (PushT)

Pinned 2026-09-24, before any CTA data exists. This replaces the sequential gate D → E plan
(`GATE_D_CODEC_PROTOCOL.md`), by the user's decision of 2026-09-24. There are no go/no-go gates in front of implementation.
Numbers are reported with CIs, and next steps are decided with the user.

## What is built

The method follows `RESEARCH_DESIGN.md` §3–4, with the §5a amendment: the reader is trained on the task's labels.

| Part | Definition |
|---|---|
| Context C | Decision frame: frozen DINOv2 ViT-S/14 16×16 tokens, PCA-128 fitted on training frames. Previous frame: 8×8 pooled tokens. Agent positions now and one step before |
| Future τ (training only) | Frames after steps 2, 4 and 6 (8×8), the end frame after step 8 (16×16), and the end agent positions |
| Source encoder q(S \| C, τ) | M = 16 segment queries cross-attend to τ and C (3 layers). FSQ levels (8, 8, 4), so V = 256 and 128 bits per candidate |
| Reader D(C, S, goal) | 4-layer Transformer. Never sees actions. Score = mean over the 16 dev goal frames. Loss: pairwise ranking on cov8 (§5a) |
| Distortion | A decoder reconstructs the future tokens from (C, S). The loss is normalised by the copy-the-decision-frame baseline, weight 1 |
| World model p(S \| C, A) | 4-layer encoder over C and the 8 action tokens, then a 4-layer autoregressive decoder, categorical over V. Never sees τ, the goal, or a label |
| Prior r(S \| C) | The same model without actions. Gives the rate estimate, and the WM gap to it gives the bits the actions add |

Training runs in stages:
1. Codec: encoder, reader and decoder on actual futures.
2. WM and prior on stop-gradient source codes.
3. Optional: alternating co-design with λ > 0.

An optional fourth step fine-tunes the reader on predicted codes.

Code: `ti_wm/cta.py`, `scripts/cta_train.py`, `ti_wm/cta_runtime.py`.

## Condition 1: baselines from the start

All baselines are trained on the same data, the same batches, the same steps and the same ranking loss as the reader.

- **FULL (tier 1):** a reader on the uncompressed actual future. It is the upper bound that the code must approach.
- **DIRECT:** D_direct(C, A, goal), 8 layers. If it matches CTA, the code S contributes nothing on PushT.
- **Closed loop:** P0, FULL8, CODE8 (tier 2) and CTA8 (tier 3), plus DIRECT8.
  - PHYS8 is added on the sealed roots.
  - CTA8 and DIRECT8 are scored from (state, actions) before any candidate is simulated.

## Condition 2: roots

| Use | Roots |
|---|---|
| Training | 30250–31049 (800 episodes) |
| Offline dev ladder | 2000–2099 |
| Closed-loop dev (all debugging) | 2100–2299 |
| **Sealed confirmation** | **3000–3399**, untouched until the configuration is locked |

Roots already used elsewhere:
- 1000–1999: gates A–C2, S1 held-out and W2.
- 5000–5031: dev goal frames.
- 20000–20999: C2 training.
- 30000–30249: S1 training.
- 39990–39999: smokes.

Every closed-loop job runs a preflight: the scores computed from raw frames must reproduce the training job's cached-feature dev scores.

## Condition 3: simple before complex

- **Round 0:** λ = 0. No stage 3 and no reader adaptation.
- λ > 0 (stage 3) and reader adaptation are debugging options. Each is measured against round 0; neither is a requirement for the pipeline to work.

## Condition 4: bounded debugging, result fixed now

- At most **3 debug rounds** after round 0, on dev roots only.
- Before a round, `docs/CTA_DEBUG_LOG.md` records:
  - the ladder numbers that motivate it;
  - the single planned change;
  - what it should move.
- Results are appended afterwards.
- The configuration is locked on the dev closed-loop result CTA8 − P0. Ties are broken by the offline `pred` retained gap.
- **Result reported, fixed now:** sealed roots 3000–3399, the locked configuration trained with seeds 0, 1 and 2.
  - **Primary:** CTA8 − P0, using each root's success averaged over seeds, with a root-paired bootstrap.
  - **Secondary:** CTA8 − DIRECT8, CTA8 retention vs FULL8, and CTA8 − PHYS8. P0 and PHYS8 are run once, since they do not depend on the model.
- The number of sealed roots and arms per seed is fixed from the dev discordance and the GPU budget before the sealed set is opened.
- If it exceeds the budget, the user decides.

## Ladder (offline, dev roots 2000–2099)

Per tier (full, code, pred, pred_soft, pred_s4, direct), the ladder reports:
- within-bank Spearman vs cov8;
- retained gap vs the cov8 oracle;
- the rate of choosing the default candidate.

It also reports:
- agreement of choices between tiers;
- source and predicted code perplexity, and the fraction of distinct codes within a bank;
- WM and prior cross-entropy, action information in bits, token accuracy and exact-code rate;
- R² of the distortion relative to the copy baseline.

A tier that drops sharply from the one above it is the part to fix.

## Logging

Weights & Biases project `cta-pusht`, group = round tag (`CTA_TAG`).
- Training: step-wise losses, pairwise accuracies, code perplexity and learning rate, plus the dev ladder every 2,000 steps and at the end.
- Closed loop: per-root success and running success per arm.
- Aggregation: tables of paired differences and per-arm decision statistics.

Runs go online when a key and the network are available, and offline otherwise. Sync offline runs from the login node:

```bash
wandb sync /mnt/data/nhatnc129/jepa/trajectory_innovation/<run>/wandb/offline-run-*
```

The JSON reports in each run directory remain the record.

## Compute

Round 0 needs about 8 MIG GPU-h:
- collection 18 × ~6 min;
- encoding ~0.3 h;
- training ~1.5 h;
- closed loop 4 × ~70 min.

Accounting since 2026-09-22 shows 22.6 GPU-h used of the 60 GPU-h the user approved.
