# 53748: stop the longer-rollout rescue branch

Queue/accounting verified: COMPLETED exit 0, 1m38s, 37 tests passed. Frozen 53741
train/val reference reproduced before continuation. Predeclared query-progress gate
FAILED at BOTH H48 and H64.

| Native H48 validation | Feature MSE | Ordered-query MSE | Regret | Missed events |
|---|---:|---:|---:|---:|
| Original step4 | .679778 | .031719 | .194354 | 31/41 |
| Longer rollout (16 x 600) | .503264 | .031383 | .186579 | 22/41 |
| More short training (4 x 2400) | .557285 | .029706 | .194349 | 22/41 |

Longer unroll lowers feature MSE by about 26%, but ordered MSE only about 1% at H48;
at H64 it worsens .029672 -> .030061. Actual default regrets .023694/.006154 at H48/64
remain far better than .186579/.162867 for longer rollout.

Feature-error improvement relative to compute-count matched short continuation has
positive exploratory prefix-bootstrap CIs, but ordered-query differences do not.
Matching unrolled transition count is not exact FLOP/time or update matching.

## What refresh diagnoses, and what it does not

With a TRUE observation refresh every 8 steps, frozen step4 at H48 misses 0/41 events
and regret is .002465. Holding the refreshed image instead misses 7/41 and regret
is .059750. Short transitions add useful prediction between refreshes.

This is not an MPC result: the offline refresh uses future images from each candidate's
own executed trajectory, unavailable while choosing that candidate. Do not turn this
into an oracle-selection/control claim. It supports local predictive signal, not
adequate long-horizon imagined trajectories.

## Decision

Do not launch another unroll-length, optimizer-step or readout sweep. Current
feature-coordinate prediction improvements have not delivered useful ordered queries.
The last bounded implementation test is QUERY_NATIVE_STOP_PROTOCOL.md: train the
representation directly for RGB-derived queries, with matched token/frame/direct
controls and no latent-coordinate loss. If it fails, stop this Wall implementation
cycle; do not keep invoking a missing baseline to postpone the stop indefinitely.

This pilot is still 1 seed, 24 train / 12 reused validation prefixes. It cannot prove
the entire class of predictive abstractions impossible. It DOES justify stopping the
current longer-rollout recipe as a method rescue.

Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/rollout_extension_53748/output/`.
