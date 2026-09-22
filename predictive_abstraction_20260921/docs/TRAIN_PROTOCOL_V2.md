# Prospective branch pilot v2 — 2026-09-22

Supersedes the old full-training recipe. This is a small engineering/development
pilot on Wall, not a manipulation benchmark and not evidence of paper novelty.
Old code/tests remain reproducible but `train_branches.py` is the new entry point.

## Purpose and decision order

1. Validate spatial reconstruction, prospective query construction, gradients and
   candidate grouping with a 20-step-per-stage profile. No research verdict from it.
2. Only after profile passes, fit a one-seed pilot. Check observed-future codec
   answer accuracy before blaming forecasting. Compare selection regret, not merely
   latent reconstruction. Full run is NOT automatically scheduled after profile.
3. Require evidence of action dependence and better selection than default,
   no-action, query-only, direct query prediction and ordered spatial-frame prediction.
   A low error averaged across mostly unsuccessful candidates is insufficient.
4. Wall alone cannot establish a method paper. No arena expansion until the mechanism
   has a credible result; no claim of improvement from a hand-written proposal/FSM.

## Data and queries

Job 53629 collected 24 train and 12 validation prefixes, disjoint from the previous
48-prefix screen. Split BEFORE branching. All eight candidates at a prefix/horizon
share RGB history and goal images selected BEFORE the candidate outcomes. No
success-conditioned filtering. Three doorway layouts appear in both splits;
therefore this is NOT unseen-layout generalization. Stationary four-frame histories
do not test memory necessity. Train horizons 32/48; development OOD horizon 64.

Candidate IDs [1,2,3,17,53,89,100,120] from a 128-candidate bank fixed before
collection. These cover deterministic A→B, direct B, perturbed A→B and unguided
actions. This eight-candidate subset has its OWN headroom: do not reuse 19/48.
384 train and 288 validation branches. No test data created/read by this pipeline.

Targets at native cadence come only from future RGB: reach A/B, occupancy A/B,
endpoint A/B and strict A-before-B/B-before-A. Exclude the current observation.
The same eight queries apply to all candidates, allowing within-prefix ranking.
The query grammar and fixed color-similarity kernel are hand-designed. Training
uses no simulator coordinates/rewards/contact labels, but is NOT a claim that
the objectives or goal/proposal interface were learned without domain knowledge.

Frozen DINOv3 4×4×1024 features retain patch order. Query anchors use a fixed seeded
1024→32 random projection PER PATCH, then flatten spatial positions; both arms use
the same projection (saved in checkpoint). Forecast input is history plus proposed
actions, never observed future. A query is only supplied to the readout. Auxiliary
anchor/history features are canonicalized across a prefix after checking raw RGB
hash equality, avoiding batch-size-dependent fp16 differences between branches.

## Models

- Query summary: observed-future codec (4×128 tokens) + query reader trained to
  answer queries. Freeze both. Student forecasts tokens with answer loss THROUGH
  frozen reader plus 0.1 latent alignment.
- Generic summary: same bottleneck but reconstruct ALL H×16×1024 future features.
  Freeze codec; fit identical-architecture reader, then matched summary student.
- Frame: direct parallel prediction of H×16 spatial tokens, ordered by time/patch;
  reconstruct ALL feature channels plus the SAME answer loss. It is NOT autoregressive;
  do not attribute an automatic H-fold inference advantage to the summary.
- Endpoint: analogous spatial tokens for last frame and full last-frame feature
  reconstruction; same answer loss (including trajectory queries).
- Direct: query attends history/actions directly, no summary bottleneck. Its context
  IS cacheable and must be cached in any later latency comparison.
- No-action: separately trained direct predictor with all actions replaced by zeros.
- Query-only and train-mean constants: detect query/horizon/layout shortcuts.
- Test-time within-prefix action permutation for query-summary diagnoses dependence;
  not a separately trained matched baseline.

Every predictor sees the same seeded minibatches and has identical update counts.
Architectures/total trainable parameters are reported, not asserted identical.
Codec joint query training versus generic reconstruction/readout training differs;
this pilot is not a final compute-matched comparison. Frame/generic reconstruct
features scaled by a fixed train-only RMS, without spatial pooling/channel slicing.
Reader architecture is common, fitted weights are separate as required per latent space.
No composer, EMA teacher or recurrent-memory benefit is claimed by this implementation.

## Evaluation and limitations

MAE/MSE by query and horizon. Primary diagnostic: ordered-query selection regret
within each shared eight-candidate group (oracle RGB score minus selected RGB score).
Report default and oracle scores, informative prefix count, and paired bootstrap
over independent prefixes (2,000 resamples), NOT individual branches/windows.
Observed-future codec is a diagnostic ceiling, not deployable. All 12 validation
prefixes are development; repeated inspection precludes confirmatory claims.

No physical-success or closed-loop MPC result is produced by this runner. No speed
claim yet. An apparent one-seed gain must later pass an untouched-prefix evaluation,
retraining seeds, proposal-matched physical selection and closed-loop evaluation.
If direct prediction matches summary, the current abstraction claim is unsupported.
If no-action matches forecasting, there is no demonstrated counterfactual learning.

## Full-pilot launch note after profile 53630

Profile passed (14 tests; 20 updates/stage). Full pilot starts fresh at seed 20260922,
batch 16, codec updates 1,200 each, generic-reader updates 600, predictor updates
1,800 each. Reuses the 53630 feature cache. Same optimization/data as profile;
post-training train-fit MSE and readable report added for diagnosis only.
The profile's data-only statistics show oracle-minus-default order-score headroom
of just 0.01853/0.02369/0.00615 at H=32/48/64. This is not physical success and not
the old 128-candidate screen. Current selection evaluation is therefore a small
development diagnostic, not a decisive control gate. No proposal/data/target changes
or hypothesis verdict are made based on the undertrained profile scores.
