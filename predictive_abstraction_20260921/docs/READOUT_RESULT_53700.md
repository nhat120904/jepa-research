# 53700: readout adaptation did not rescue the forecast

COMPLETED exit 0, 31 seconds; 29 tests passed. Both squeue and sacct checked.
Frozen forecast/kernel references reproduced. No WM retraining, new rollout or test read.

## Main numbers: validation H48, 12 prefixes / 8 candidates each

| Source/readout | A-before-B MSE | Regret (first / uniform ties) | Missed visual events |
|---|---:|---:|---:|
| Observed future + original kernel | .002839 | .002465 / .002465 | 0/41 |
| Observed future + learned reader | .014624 | .002470 / .002470 | 0/41 |
| Compact forecast + original kernel | .032110 | .137127 / .137127 | 27/41 |
| Compact forecast + adapted reader | .031966 | .123009 / .123009 | 25/41 |
| Matched frame forecast + adapted reader | .031857 | .200195 / .200195 | 25/41 |
| No-action forecast + adapted reader | .031771 | .023694 / .156294 | 26/41 |

Regret lower is better; these are visual query scores, NOT physical success rates.
Missed event means true affinity peak >= .5 but predicted peak < .1. The 41 events
are candidate/anchor instances, not 41 independent episodes.

Compact readout adaptation changes uniform-tie regret by +.01412 at H48 (positive is
improvement), with paired prefix bootstrap CI [0, .04235]; H32 worsens by .01346,
H64 is essentially unchanged. No reliable cross-horizon rescue. Fit query MSE at H48
improves .03984 -> .02345, but validation barely changes .03211 -> .03197.

No-action's strong argmax-first regret is a tie-breaking/default advantage: resolving
ties uniformly changes .02369 to .15629. Compact beats that random-tie expectation at
H48 but still loses badly to the actual default; this does not establish control benefit.
Even no-action adapted reader reduces missed events without discriminating candidates,
so reducing missed-event counts alone is not proof of action-conditioned prediction.

## Decision

Do not keep changing readouts/codecs on these forecasts. Observed-future information
is readable; frozen forecast representations do not yield sufficient held-out event
and ranking quality to this local reader. This is NOT proof that every possible reader
fails, nor that predictive abstraction is impossible.

Proceed to the previously specified branch C: short spatial transition baseline,
teacher-forced vs open-loop evaluation, action/no-action and persistence controls.
No composer, policy/VLA, new arena, MPC or dataset expansion yet.

All validation is reused development, one seed. Readout holdout was not held out from
the OLD WM training. Raw outputs:
`/mnt/data/nhatnc129/jepa/predictive_abstraction/readout_53700/output/`.
