# 53741: short-dynamics gate passed; long-horizon event prediction still failed

COMPLETED, exit 0, 54 seconds. Queue and accounting checked; 33 tests passed. Tiny
fit passed (14.442 persistence loss -> .510), observed-future reference reproduced.

## Positive but limited finding

Step4 passed the predeclared short-dynamics gate on reused validation:

- Teacher-forced feature MSE 14.5% below one-frame persistence.
- Open-loop H8 feature MSE 51.8% below initial-frame persistence.
- Open-loop H8 feature MSE 46.0% below trained no-action model.

Actions matter: on the H48 bank, H8 MSE .16150 versus .29809 with action sequences
reassigned among candidates. These are feature errors, NOT physical state errors.

## The model is not ready for query selection

| H48 open-loop source | Feature MSE | Ordered query MSE | Uniform-tie regret | Missed visual events |
|---|---:|---:|---:|---:|
| Step1 training | .973212 | .031755 | .200195 | 39/41 |
| Step4 training | .679778 | .031719 | .194354 | 31/41 |
| No-action step4 | 1.057093 | .031754 | .156294 | 41/41 |

Actual default regret is .023694. Step4 remains much worse. At H64, step4 misses
35/43 events, and query MSE .029672 is essentially unchanged from persistence.

Two caveats prevent overstating the positive result:

1. H8 contains ZERO positive A/B events in this bank. Passing its feature gate does
   not test successful prediction of task events or ordered queries.
2. Teacher-forced step4 misses 0/41 events at H48, but teacher-forced persistence also
   misses 0/41. Both see real intermediate futures. This does not prove the learned
   transition can independently predict events. TF feature gains are real but modest.

The large TF/open-loop difference is consistent with accumulated/off-manifold error;
it does not uniquely establish its cause. The model could also have systematic local
bias, limited data, or a feature loss insensitive to precise event occurrence.

## Next bounded experiment

Continue from the SAME step4 checkpoint: 4 steps x 600 updates; 16 steps x 600;
4 steps x 2400. This separates rollout length from optimizer-update count and
approximately from unrolled transition compute. Add matched real-state-refresh and
persistence-refresh diagnostics at intervals 4/8/16. Training remains image-feature
self-supervision; no new reader, physical labels, data, policy, composer or MPC.

See ROLLOUT_EXTENSION_PROTOCOL.md. This is baseline capability repair, not yet the
predictive-abstraction contribution. If longer training improves features but not
queries again, do not automatically launch an even longer training sweep.

Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/local_transition_53741/output/`.
