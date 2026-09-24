# Controlled longer-rollout training — locked before submission, 2026-09-22

Hypothesis: short-trained patch dynamics learn some action effects but become
inaccurate off the observed-state trajectory. Longer open-loop training may improve
long-horizon visual-event prediction, not just average feature error.

## Fixed components

Same cached features, 24 train / 12 reused validation prefixes, patch architecture,
normalization, delta scale, feature-only loss and frozen evaluation metric as 53741.
No future image at open-loop inference. No query/affinity/reward/state supervision
in predictor training. No test split, data collection, encoding or policy/MPC.
All continuation arms start from EXACTLY the 53741 step4 checkpoint; AdamW is reset
identically (lr 3e-4, weight decay .01, clipping 1). Sampler/init seed 20260928.

Reproduce frozen step4 train/val native-horizon feature/query MSE and first-argmax
regret within 1e-4 BEFORE any new training. Stop on reference or finite-value failure.

## Three continuation arms

| Arm | Unroll length | Updates | Batch | Unrolled transition examples |
|---|---:|---:|---:|---:|
| more4_updates | 4 | 600 | 32 | 76,800 |
| more16 | 16 | 600 | 32 | 307,200 |
| more4_transitions | 4 | 2400 | 32 | 307,200 |

First comparison matches updates; second matches transition count, NOT exact runtime
or optimizer updates. Report elapsed training time and parameters. No arm is selected
mid-training by validation. Both train banks H32/48 are retained (different actions).
Every loss includes all predicted steps; no teacher forcing inside training windows.

## Non-deployable refresh diagnostic

Frozen step4 reinitializes from TRUE observed frame every 4/8/16 steps, versus a
matched baseline that simply holds that observed frame until the next refresh.
This reveals how much accuracy depends on regular access to real observations.
It is NOT counterfactual planning, closed-loop control, or an oracle selection gain.

## Evaluation and decision

- Native bank H32/48/64: train/val feature MSE, all 8 query MSEs, event miss/detection/
  false-positive/timing counts; within-prefix ranking and first/uniform-tie regret.
- Compare frozen step4, three continuations, old frozen no-action reference, and the
  more16 model with candidate actions cyclically reassigned within each prefix.
  Old no-action is NOT matched to extra continuation budget; use shuffle for same-model
  action sensitivity and matched action-aware continuation arms for schedule claims.
- Paired bootstrap by prefix, 2,000 samples; positive differences mean improvement.
  Report both primary horizons H48/64. CIs are exploratory on reused validation.
- Query-progress gate: at BOTH H48 and H64, more16 ordered-query MSE at least 20%
  below old step4 and 10% below the better of the two short-training continuations;
  uniform-tie regret must not worsen from old step4 (tolerance 1e-8).
  This gate is NOT a planning gate: also show actual default and all event counts.
- Pass: a candidate training fix worth confirming; not a novelty or control claim.
- Feature gains only: do not invest in another longer rollout sweep. Review whether
  task-relevant predictive targets/data coverage, not unroll length, need changing.
- Short continuation matches/beats long: extra training explains any gain; do not
  credit longer-horizon consistency. No automatic follow-on jobs in any case.

One GPU, 20-minute hard cap, tests first, snapshots/hashes and immutable output dir.
No new abstraction design is launched before reviewing this bounded result.
