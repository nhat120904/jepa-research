# One controlled target correction for the segment arms (final Scrub test of this version)

Locked 2026-09-15, before any training or result. This is the single target correction
allowed after re-evaluation 52642. No other architecture, feature, label or arena change is
part of this test.

## Hypothesis being corrected

In the 52634 segment arms, the target encoder's reconstruction anchor predicted
`fut_visual.mean(dim=(2, 3))`: the per-frame DINO feature averaged over cameras and
patches. That anchor asks the summary to keep global feature dynamics, but it never asks it
to keep the spatial structure that contact location and coverage depend on.

The correction asks the summary to keep camera/patch structure and proprioception. It is a
specific repair hypothesis, not a promise of a positive result.

## What changes (identically for both segment arms)

The anchor is replaced by a spatial one, decoded from the 4 summary tokens only:

- **Visual:** every camera × patch token (3 × 16) of every 4th frame of the window
  (frames t+4, t+8, …), each token layer-normalized, predicted by cross-attention queries
  built from time, camera and patch embeddings.
- **Proprio:** the normalized 16-d proprio of every frame of the window.
- **Loss weights:** 0.25 for visual reconstruction (same as the old anchor), 0.25 for proprio.

Everything else is unchanged from 52634: data, split, DINOv2-S 4×4 features at 20 Hz,
fuser, memory, predictor, composer, prediction and composition losses, 12k steps, batch 48,
seed 20260915.

- Arms retrained: `segment_nocomp_spatial` and `segment_comp_spatial`.
- `frame_rollout` and `map_union` are not retrained. Their 52634 checkpoints remain
  reference points only, never the final comparison, because they differ in version.

## Evaluation split (test is not used)

The 83 test episodes have been read repeatedly during design, so they are excluded. The 76
validation episodes are divided by a fixed hash order:

- **`val_select` (38 episodes):** readout early stopping only.
- **`val_decide` (38 episodes):** every decision metric.

Readouts are trained on train episodes and use the same corrected procedure as 52642:
train-set standardization, S1 (direct 32/64 only) and S2 (calibrated on the same inference
path, including 128). The old 52634 segment checkpoints are re-evaluated on `val_decide`
with the same code, so old and new targets are compared on identical windows.

## Locked metrics

Scale-free primary metric: Pearson r of `count_empty` on `val_decide` contact windows,
S2 readout, paired episode bootstrap (1000 resamples). MAE and `count_increment` are secondary.

1. **Target/readout adequacy.** The true-future-summary readout r (S2), averaged over lengths
   32/64/128, for each segment arm. New target minus old target, same arm.
2. **Composition.** `segment_comp_spatial:composed` minus each of
   `segment_nocomp_spatial:additive` and `segment_nocomp_spatial:direct_whole`, on
   64=24+40 and 128=64+64 (primary), plus 64=32+32 and 128=32+32+64 (secondary).

## Decision rules

| Outcome | Decision |
|---|---|
| Target improves (adequacy Δr ≥ +0.05 for both arms) **and** composition beats both no-composition references on both primary splits, with at least one interval excluding zero and no negative point estimate on the four splits | Composition "clearly better": confirm with 3 seeds on an evaluation set not used for design. No planning before that. |
| Target improves, but composition is not better as defined above | **Close this compositional trajectory-JEPA version on Scrub.** |
| Target does not improve (Δr < +0.05 for either arm), whether or not composition changes | **Stop investing in Scrub for this direction.** No further chain of pooling → encoder → label → arena changes. |

With 38 decision episodes, intervals will be wide. An inconclusive result falls into the
"not better" rows. It is not a reason to add another round.

## Budget

- One GPU job: profile first, then train both arms (12k steps each) and evaluate on `val_decide`.
- 3 h walltime cap, no chained jobs.
