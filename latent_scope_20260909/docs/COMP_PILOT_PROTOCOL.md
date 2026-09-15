# Offline composition pilot protocol (Scrub demonstrations)

Locked 2026-09-15, before any training. This is a new offline protocol, not the old gated
Stage C. It tests whether an action-conditioned segment summary that is trained to compose
predicts accumulated in-segment effects better than matched controls. One training seed:
a pilot that can detect non-learning or large gaps, not a method conclusion.

## Data

- All 504 released ScrubCuttingBoard demonstrations (revision `522e4ffa…`).
- Episode split by stable hash, seed 20260915, 70/15/15 train/val/test, assigned before
  windows exist. Windows never cross splits.
- Inputs: frozen DINOv2-S (pinned), full-frame resize 256→224 without cropping, 16×16
  patches average-pooled to 4×4 per camera, fp16, all frames (20 Hz); proprio (16) and
  actions (12) at 20 Hz. Pool 4×4 is a pilot choice; it is not shown to preserve all
  contact detail.
- Labels: playback monitor from job 52617 (count, contact, grasped, sponge xy, board pose).
  Not called exact native labels. All 504 episodes kept; results also reported separately
  for the 442 label-agree and 62 label-disagree episodes.

## Segment targets used for evaluation (never model inputs)

For a window of frames [t, u):

- `count_increment`: native accepted contacts added given the real history.
- `count_empty`: accepted contacts from an empty history within the window.
- `area_empty`: union area of 1 cm discs around grasped contact positions (cm²).
- `map`: 12×12 occupancy grid of grasped contact positions in the board frame
  (2.5 cm cells, ±15 cm). Composes exactly by OR.

Hard stratum: windows whose parts' `count_empty` sum differs from the whole window's value.

## Arms (identical inputs, context module, data, steps and optimizer)

All world-model arms see the same causal history: fused frames every 4 steps from episode
start, capped at the last 96 tokens, plus the current frame. The future segment's actions
are inputs. Future frames are used only to build training targets.

| Arm | What it learns | Contact labels in training? |
|---|---|---|
| `segment_nocomp` | Predicts a 4-token segment summary and an endpoint latent from context and actions; targets from an EMA trajectory encoder over the future frames | No |
| `segment_comp` | Same, plus a learned composer C: C(left, right) matches the whole-segment target for observed and predicted summaries | No |
| `frame_rollout` | Predicts the ordered fused latent of every future frame; long segments are rolled out sequentially | No |
| `map_union` | Predicts the contact map and scalar targets directly; composes maps by fixed OR | **Yes, directly in the predictor.** This is a supervised check on whether a learned composer is needed, not a matched arm. |

`segment_nocomp` and `segment_comp` share every loss except the two composition terms:

- full-segment and half-segment summary/endpoint prediction;
- rolled right-half prediction from the predicted left endpoint and updated memory;
- target-encoder reconstruction anchor;
- memory-update loss;
- variance regularization.

Training lengths are 32 and 64 steps, which include 16- and 32-step halves.

## Readout (same architecture and procedure, separate weights per arm)

Each representation is a token set: segment arms give 5 tokens (summary plus endpoint);
the frame arm gives one token per predicted frame; `map_union` gives 144 cell tokens.
A one-layer attention-pool readout with an MLP head is trained on frozen train-split
**direct** predictions of lengths 32 and 64, with the same optimizer, steps and
validation early stopping for every arm. It is applied unchanged to composed predictions.
A second readout on the segment arms' true EMA summaries gives the target-readout ceiling.
Test data is never used for selection.

## Evaluation (test episodes)

- Direct: 32, 64, and 128 (length extrapolation).
- Composed:
  - 64 = 32+32 (seen lengths);
  - 64 = 24+40 (unseen lengths and split);
  - 128 = 64+64 (unseen total);
  - 128 = 32+32+64 (three parts).
- How each arm composes:
  - `segment_comp`: its composer, left to right;
  - `segment_nocomp`: (a) additive readouts for count/area plus OR of readout maps, and (b) direct prediction of the whole;
  - `frame_rollout`: sequential rollout;
  - `map_union`: OR of maps plus additive scalars.

Primary metric: `count_empty` and `count_increment` MAE on composed 128 = 64+64 and
64 = 24+40, in the hard stratum. Secondary: area MAE, map F1, easy stratum, all windows.
Uncertainty: paired bootstrap over test episodes (1000 resamples); overlapping windows are
not independent samples.

## Budget and reporting

- Encode: 1 GPU, 2 h cap (job 52631).
- Pilot training and evaluation: 1 GPU, 8 h total cap, including a short profile run.
- An arm that has not converged within its step budget is reported as incomplete, not as a loss.
- One seed: a pilot. Any positive or negative method claim needs 3 seeds.

## Decision rules for the pilot

- `segment_comp` beats `segment_nocomp` on composed hard-stratum targets with an episode
  bootstrap interval excluding zero: first evidence; run 3 seeds.
- Beats only the additive sum, but not `map_union` or `frame_rollout`: learned composer
  not yet justified.
- All arms poor, including the true-summary readout ceiling: target or input problem;
  do not conclude against composition.
- Controls learn well, composition adds nothing: direct negative evidence at this
  operating point (confirm with 3 seeds).
