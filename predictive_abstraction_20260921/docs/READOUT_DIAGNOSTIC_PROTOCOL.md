# Frozen-predictor readout diagnostic — 2026-09-22

Question: does the existing action-conditioned forecast contain recoverable event
information that the frozen metric kernel cannot read? This is a development
diagnostic, NOT a method win, planning experiment or untouched-test confirmation.

## Frozen and trainable components

- Freeze every parameter of the 53698 metric and predictors. Reuse cached features.
- Sources: observed future; compact16; matched frame_bin_loss; compact16_no_action.
- Original readout: exp(-mean squared embedding distance), followed by exact temporal
  query algebra. Reproduce all three forecast references within 1e-4.
- New readout: identical small frame/anchor pair MLP, same initialization, optimizer,
  sampling and 1,200 updates per source. Inputs include frame, anchor, difference,
  and product. It cannot see true future when evaluating a forecast.
- Supervision remains RGB-derived affinity, not physical state/reward/success.
  It is task-designed visual supervision; do not call it task-agnostic JEPA.
- Fit only 18 of the original 24 training prefixes (index modulo 4 != 3).
  The other 6 are readout-held-out, but WERE seen by the original predictor.
  The 12 validation prefixes were not used for optimization but have repeatedly
  informed development. Fit forecasts are in-sample, not out-of-fold.
- Readers train on H32/48; H64 remains a length extrapolation diagnostic.
- Pair sampling balances affinity > .05 versus <= .05; evaluation is unbalanced.

## Controls and measurements

1. Zero scores; observed future + original kernel; observed future + clean reader.
2. Each frozen forecast + original kernel / clean reader / source-adapted reader.
3. Query MSE, within-prefix ranking on true-score differences > .05, regret with
   actual argmax-first and expected uniform tie resolution (epsilon 1e-8).
   Uniform ties are analytic expectations, not actual new simulator rollouts.
4. Count visual events with true peak >= .5, missed peaks < .1, detected peaks >= .5,
   false positives on true peak <= .05, and timing error conditional on detection.
   Count reversed-order errors only when true A-before-B >= .5 and B-before-A <= .05.
   Thresholds describe visual affinity, not physical success.
5. Paired prefix bootstrap (2,000 resamples) for adapted-vs-kernel regret change;
   candidate branches are not independent samples. Reused-val CIs are exploratory.

## Interpretation fixed before execution

- Readout adaptation improving training alone is not recovery of useful signal.
- If observed clean reader works and adapted forecast readers improve held-out
  queries/ranking, investigate readout domain shift and decision-aligned training.
- If observed reader works but forecast readers remain weak, investigate prediction
  and data generalization next. Failure of one MLP does NOT prove information absent.
- If visit detection works but order/timing fails, investigate temporal localization.
- Improvement over old kernel alone does not establish an action benefit: compare
  adapted no-action and matched frame controls, ties included.
- No automatic architecture training, dataset expansion, composer, VLA or MPC job.
  One GPU job, explicit 15-minute maximum; run tests first, fail closed on errors.
