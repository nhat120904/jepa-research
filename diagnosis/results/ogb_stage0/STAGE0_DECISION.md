# OGBench-Cube Stage 0 decision

> **HISTORICAL ARTIFACT RETRACTED; CONCLUSION CLEANLY REPLICATED 2026-08-11.**
> Do not cite the numbers or candidate endpoints in this file: candidate
> restoration called `CubeEnv.reset`, whose goal-observation construction
> executes random actions and restores only qpos/qvel. Hidden controller/contact
> solver state therefore leaked into rollouts. A fresh two-world audit with full
> `mjData` reset, warm-start disabled, exact renderer-repeat gates, and a
> same-renderer goal independently recovers 21/32 true-endpoint successes and
> residual regret 3.29 cm [1.49, 5.67]. Cite
> `../ogb_true_endpoint_corrected/TRUE_ENDPOINT_DECISION.md` and its fresh
> artifacts instead. The text below is retained only as provenance.

Date: 2026-08-10. Verdict: **STRONG PASS for the locked mechanism gate**.
Stage 1 method status: **HOLD**.

## What was tested

The released `quentinll/lewm-cube` checkpoint and official LeWM CEM settings
were used on 32 deterministically selected OGBench-Cube snapshots. For every
snapshot, the initial and final CEM populations contained the same 300 action
sequences under all three evaluations:

1. LeWM predicted-rollout latent goal cost, which the planner actually used;
2. latent goal cost after executing each sequence and encoding its true rendered
   endpoint, which removes learned dynamics error while retaining the released
   representation and cost;
3. simulator cube-to-target distance, used only after planning as an offline
   physical oracle.

The simulator did not generate candidates, alter CEM, train the model, or choose
an action available to the acting planner.

## Locked result

All three precommitted final-population criteria have lower 95% snapshot-
bootstrap bounds above zero:

| Gate | Estimate | 95% CI | Pass |
|---|---:|---:|:---:|
| true-endpoint-latent physical selection regret | 3.13 cm | [1.43, 5.43] cm | yes |
| paired physical-oracle success advantage over true-endpoint latent selection | 12.5 pp (4/32) | [3.125, 25.0] pp | yes |
| at least one successful candidate is present | 78.125% (25/32) | [62.5, 90.625]% | yes |

Supporting decomposition on the same final populations:

| Selector | Mean physical endpoint distance | Success |
|---|---:|---:|
| learned rollout latent cost | 12.09 cm | 50.0% (16/32) |
| true endpoint latent cost | 7.61 cm | 65.625% (21/32) |
| physical oracle | 4.48 cm | 78.125% (25/32) |

The full learned-cost selection regret is 7.61 cm [4.75, 10.75]. Its mean
distance decomposition is exact up to rounding: 4.48 cm is removed by replacing
the predicted endpoint with the encoded true endpoint, while 3.13 cm remains
because the true-endpoint latent cost still misselects physical outcomes.

The association is much weaker within the post-search population. Mean
learned-cost versus physical-cost Spearman is 0.123 in the initial population
and 0.0016 in the final population. This comparison is descriptive because CEM
changes and narrows the candidate distribution. On the final population,
learned versus true-endpoint latent Spearman is only 0.0315 [0.0108, 0.0526],
whereas true-endpoint latent versus physical Spearman is 0.345 [0.198, 0.481].
This says both dynamics error and residual representation/cost mismatch matter;
it does not justify blaming the representation alone.

## Integrity checks

- Official closed-loop baseline: 68% (34/50), versus the paper's 74% reference
  averaged over three training seeds; no tuning was performed.
- All 32 Slurm array tasks `38013_[0-31]` completed with exit code 0.
- Aggregation job `38016` completed and used 10,000 bootstrap draws clustered by
  snapshot.
- The 32 snapshots come from 32 distinct dataset episodes.
- qpos, qvel, target, dataset-state, and pixel restoration maximum errors: 0.
- action-normalizer round-trip maximum absolute error: `5.960464477539063e-08`.
- Exact checkpoint revision:
  `b0747c5002e86d2ce8f3cd8178004b97524c587d`.
- Exact dataset revision:
  `02a19a67a0dc8c9d6215f89c19e0a597691e152a`.

Output SHA256:

```text
15b2c52ad5788579944b6eca521ce7327cde0caa9078c65f148dbe64b32fe635  summary.json
a17059b7d64a16e71dfe009b79a5a57e9706768dd3f7411905abca3836111d92  snapshot_metrics.csv
bfb3ce1a098e056b2c18b5ca9f6a03f43afb1f494743cc0ed071738cd002a5da  candidate_costs.csv.gz
eff8803af3ce8d00c99b5accc95c820123407a82f0c940cf6bfcdbf7494fe4b4  manifest.json
```

## Decision boundary

This is sufficient evidence to continue studying the mechanism outside
MetaWorld. It is not yet sufficient for a top-tier method claim: the audit uses
one checkpoint, one task, only 32 snapshots, and rows are not claimed held out
from checkpoint training. The success-level pass is also thin: it is supported
by four paired gaps, with a lower bootstrap bound equal to one snapshot out of
32. It must replicate before becoming a headline generality claim.

Do not convert the physical oracle into the default training signal. That would
make the method simulator-assisted and change the comparison class. Stage 1 may
start only after specifying a non-privileged objective whose inputs exist in the
offline training data, plus comparisons against at least dynamics adaptation,
proposal/coverage improvement, and a simple learned cost-calibration baseline.
Until then, the defensible paper direction is a mechanistic diagnosis with
cross-substrate replication, not a weak corrective method attached to it.

## Post-hoc evaluator caveat (2026-08-11)

The PFCG pilot's artifact-replay gate found that this audit restored qpos/qvel
but not full MuJoCo `mjData`. `CubeEnv.initialize_episode` executes random
actions while constructing a goal observation, allowing controller/contact-
solver state to enter later candidate replay. Stored final-population outcomes
were consequently not candidate-level reproducible, and independently rendered
endpoints were also unstable.

A corrected full-state-reset, no-render replay retained the aggregate learned-
cost selection result (16/32 successes, 12.11 cm selected distance) and physical
candidate availability (25/32), essentially matching the historical 16/32,
12.09 cm, and 25/32. The broad cost-selection failure and candidate coverage
conclusions therefore survive. The true-endpoint latent selector was not
reproduced under a controlled renderer, so the 21/32 success and 3.13 cm
representation-regret decomposition above is now **historical/blocked from new
causal use** until a full-state, render-controlled rerun is completed. See
`../ogb_pfcg/PFCG_PILOT_DECISION.md`.

## Corrected resolution (2026-08-11)

The required rerun is complete. Two independently compiled worlds produced
exactly identical endpoint pixels, physical outcomes, encoder costs, and
selected indices over all 32×300 candidates. The evaluator also exactly matched
the separately corrected physical replay, and the worst same-state encoder
domain ratio was `6.5906e-4` against a precommitted 0.25 maximum.

Using a simulator-rendered exact future goal to remove cross-renderer offset,
the corrected true-endpoint selector succeeds on 21/32 snapshots and leaves
3.286 cm physical regret [1.495, 5.672]; physical-oracle availability remains
25/32. Thus the historical artifact remains invalid, but the scoped conclusion
that this frozen encoder plus terminal squared-L2 cost leaves residual
same-population misselection is restored by independent corrected evidence.
See `../ogb_true_endpoint_corrected/TRUE_ENDPOINT_DECISION.md`.
