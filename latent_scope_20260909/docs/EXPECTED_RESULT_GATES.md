# Expected-result gates for the composition direction

These gates separate the existence of useful decisions from evidence for the proposed
compositional trajectory target. A positive early gate is necessary but is not itself a
method result.

## Gate 1: valid candidate headroom

The carrier must match the intended source event exactly in simulator state and native task
history. Restored branches must match each other exactly in all three camera observations,
and candidate proposal/evaluation must use that canonical restored observation. The source
screenshot is retained only as a renderer-state diagnostic, as locked in
`STAGE_B_PROTOCOL.md`. On independent validation prefixes, the oracle
best candidate must improve eventual native completion over candidate 0 by at least 10
percentage points. Report the paired confidence interval. Progress variation is a useful
diagnostic, but progress-only gain does not pass the native-completion gate.

For the 8-versus-16 profile, a promising result is outcome variation on at least one task
and no loss of candidate diversity at 16 steps. This small event-conditioned profile only
authorizes a larger unbiased screen; it does not estimate deployable planner performance.

## Gate 2: permitted-observation readout ceiling

Using a separate training split, a history-aware readout given ground-truth candidate
trajectories must recover at least half of Gate 1's oracle gain and have positive selected
success gain on held-out candidate groups. If it cannot, no learned target using the same
observations has a credible path to exploit the available headroom.

## Gate 3: learned planning improvement

All arms use the same frozen proposal/continuation policy, candidate bank, data split,
parameter budget and supervision. The compositional arm must:

1. improve selected native success over candidate 0 by at least 5 percentage points;
2. beat the strongest matched control by at least 5 percentage points;
3. beat the parameter- and supervision-matched unstructured segment arm with a paired
   confidence interval whose lower bound is above zero; and
4. reproduce the direction across at least three training seeds and both a trajectory-
   accumulation pilot and a contact-rich task.

Prediction loss or progress MSE alone is not evidence of control improvement. If direct
value or full temporal frame rollout matches composition, the broader predictor may still
be useful, but the composition-specific method claim fails.
