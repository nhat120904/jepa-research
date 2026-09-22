# Bounded observed-future rescue experiment

Locked before this job; development only, not a WM/policy success claim.
Motivation: full pilot 53642 completed cleanly, but even the observed-future query
codec had poor candidate selection. Need localize this before changing forecasting.
No evidence yet that features, compression or optimization alone caused the failure.

## Fixed scope

Reuse 53630 features/53629 data: 24 train prefixes, 12 validation; no new data,
test access, simulator state, coordinates, reward labels, VLA training or MPC.
All arms observe true future features. One seed 20260923; 20-minute GPU job cap.
All labels derived from native RGB. Query grammar/proposal remain hand-designed;
do not describe this as learning the task specification from unlabeled data.

## 1. Feature recoverability probe

Frame-to-anchor learned metric uses fixed projected per-patch DINO features (16×32
flattened, SAME projection as query anchors). Train-only per-channel centering/scaling,
shared MLP embedding, similarity exp(-mean squared embedding distance). Train 1,000
steps, 256 frame/anchor pairs per batch, half target affinity >0.05 and half <=0.05;
sampling strata from TRAIN only. Targets are native RGB chroma affinities, not positions.
At evaluation apply exact reach/occupancy/endpoint/strict-order aggregation.

This has DENSE intermediate supervision and a fixed known aggregator. It is a
recoverability/engineering reference, not a matched learned-summary baseline. Passing
does not validate the summary or forecasting. Failing does not prove DINO lacks all
information: the projection/probe/optimization may fail. RGB trace aggregation is
asserted equal to stored query labels before training.

## 2. Tiny-fit sanity gate

One TRAIN prefix at H=32 selected for largest within-bank ordered-score variance.
Selection is explicitly a unit-test choice, not a held-out result. Train observed
full-sequence and four-token-summary models for 600 steps each on its eight branches.
Both use MSE plus pair-difference loss. Pass if ordered MSE <10% of that group's
constant-mean MSE. If neither passes, skip the larger factorial ablation; diagnose
the simple learning setup first. No forecast retraining automatically follows.

## 3. Matched 2×2 codec ablation (only if tiny gate passes)

Axes: retain all H temporal tokens versus pool to four tokens; MSE-only versus
MSE + within-prefix candidate-difference regression (weight 1).
Same spatial projector, temporal Transformer, QueryReader, initial seed, 1,200
updates, optimizer, and exact same grouped minibatches (two prefixes × eight
candidates; same horizon). Train only H32/48, validation also H64.

The sequence model still projects each spatial frame to width128; bypassing token
pooling tests temporal compression, not removal of every representation bottleneck.
Pair objective applies to ALL eight query types, only candidate pairs from the SAME
prefix and query, masks target differences <=0.05. No goal/success privileged labels.
MSE-only arm also uses grouped sampling to isolate the objective effect. No candidate
filtering by outcome and no validation tuning/checkpoint selection.

## Report and subsequent decisions

Train/validation per-query MSE, ordered candidate-selection regret, per-prefix predicted
and true spread. Save every prefix, not just positive examples. Validation set is reused
development; one seed and small headroom cannot support significance/confirmation claims.

- Metric probe succeeds but sequence reader fails: investigate readout/temporal objective;
  the failure cannot simply be assigned to missing visual signal.
- Sequence succeeds but summary fails: evidence for this temporal bottleneck/training
  setup; consider structured temporal tokens before changing forecasting.
- Paired objective helps summary beyond the matched sequence effect: candidate change
  for the next WM pilot; must also give that loss to frame/direct forecasting controls.
- Both tiny models fail: stop full expansion; test optimization/query scaling. Do not
  conclude the general idea impossible or launch a bigger dataset automatically.

No favorable branch is assumed. Current job exits with report/checkpoints, no automatic
downstream job. A repaired observed codec is necessary diagnostic progress, not evidence
that an action-conditioned WM will predict it or improve control.
