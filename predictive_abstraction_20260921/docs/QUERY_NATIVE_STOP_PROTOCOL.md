# Last bounded Wall implementation test and explicit stopping rule

Locked before submission, 2026-09-22. This is an operational research-budget decision,
not a statistical proof that all predictive abstraction is ineffective.

## Why this single remaining test

Observed-future queries can be read. Frozen forecast readout adaptation failed;
spatial short dynamics help; longer rollout training improves features but not ordered
queries. The remaining scoped intervention is to remove latent-coordinate matching
and learn a representation by the questions it must answer. This was identified in
ARCHITECTURE_DECISION_AFTER_FORECAST_FAILURE.md, branch A, but is now a combined
end-to-end query-learning test, NOT a clean attribution to one loss term.

## Fixed data and architecture comparison

- Same 24 train / 12 development-validation prefixes; H32/48 train, H64 extrapolation.
  No dataset expansion, new encoder, physical state/reward/hand annotations or policy.
- Shared history/action context encoder. No query/future input to representation.
- Summary: 16 time-indexed predicted latent tokens, width 128.
- Frame-token control: H time-indexed predicted latent tokens, width 128. Same
  predictor/readout parameters as summary; no latent-coordinate target. This is NOT
  a reproduction of a published frame world model; it is the matched uncompressed arm.
- Direct: queries attend to cached history/action context without future-token
  bottleneck. It uses the same query reader, but fewer predictor parameters; report
  counts and runtime rather than pretending exact parameter equality.
- No-action: independently trained summary with actions zeroed.
- Common reader: anchor embedding + time/duration -> attention to representation ->
  affinity at each native future timestep. Exact shared temporal algebra yields the
  eight endpoint/occupancy/reach/order queries. No query-dependent reforecast required.
- Train all four with identical sampler/init seed 20260929, 2,000 AdamW updates,
  2 prefixes x 8 candidates/batch, lr 3e-4, weight decay .01, clipping 1.
- Common objective: balanced dense affinity MSE (> .05 / <= .05 strata), mean eight
  query MSE, plus .1 within-prefix answer-difference loss (true margin > .05).
  Same targets and losses for every arm. NO feature/latent-coordinate alignment loss.
- Targets are automatically derived from RGB, with task-designed visual affinity and
  query algebra. This satisfies no physical/annotated supervision, but NOT a claim
  of task-agnostic self-supervised JEPA. A generic objective remains unproven.

## Query-independence check

For validation only, swap A/B query anchors with the next prefix of the same split
and layout (cyclic sorted order). Never select donor by outcome. Compute corresponding
RGB labels on compute; never train on these cross-query labels. Same candidate actions,
different questions chosen independently of their proposal's original A/B.
Report positive events/ordered branches: if sparse or absent, the test is inconclusive,
not evidence of generalization; do not resample queries until results look favorable.

## Metrics and fixed operational stop rule

Report train/val/native/cross-query results, all query MSEs, events, pair ranking,
first/uniform-tie selection regret, default, frozen-model action shuffle, and paired
prefix bootstrap (2,000 samples). Validation is reused; CIs are exploratory.

Native feasibility requires at BOTH H48 and H64:

1. Ordered MSE at least 20% below BOTH no-action and constant-zero predictions.
2. Actual first-argmax regret <= default regret + .01 (score units, not success pp).
3. Ordered MSE at least 10% below the same model with actions shuffled within prefix.

- No action-aware arm passes: STOP_CURRENT_WALL_METHOD_IMPLEMENTATION. No automatic
  continuation with a new loss, bigger model, more epochs or another arena.
- Frame/direct passes, summary fails: STOP_COMPACT_VARIANT_CONTROLS_HAVE_SIGNAL.
- Summary passes but is not at least 10% lower ordered MSE than BOTH frame/direct at
  both main horizons: NO_COMPACT_ACCURACY_ADVANTAGE_STOP_TUNING. An unmeasured compute
  advantage is not ruled out, but does not automatically reopen this cycle.
- Summary passes and clears that margin: only CANDIDATE_FOR_FRESH_CONFIRMATION.
  Cross-query support/quality, fresh held-out prefixes, 3 seeds, end-to-end compute
  accounting and actual selection/control still required before a method claim.
  Confirmation must be a separately reviewed protocol, not another development sweep.

One seed cannot prove universal failure or success. The stopping rule answers
"should we keep spending on this implementation?", not "can no future method work?".
A runtime/test error is a blocked measurement, not a scientific negative; only fix
reproduction bugs, without changing targets or thresholds to pass.

One GPU job, 20-minute cap; all tests first, immutable snapshots and hashes. No
dependent job. This closes the current repair cycle on a negative result.
