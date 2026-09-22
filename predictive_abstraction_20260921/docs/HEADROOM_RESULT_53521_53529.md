# Navigation headroom preflight — jobs 53521 and 53529

Date: 2026-09-21. Development result, not confirmation and not a method result.

Review update: job 53568 measured the competent fixed A-then-B candidate at 11/48.
The 19/48 oracle therefore leaves **8/48 = 16.67 pp** over that default, not 32.14 pp
over a competent baseline. See [review](REVIEW_AND_POLICY_GUIDANCE_20260921.md).
The training protocol needs revision; the recommendation below was preliminary.

Both jobs completed with exit code 0. The initial unguided bank (53521) passed all nine
tests and produced the planned 64/16/16 RGB-action split, but contained no ordered-success
candidate in 24 x 64 branches. It did support reach-B on 12/24 prefixes; the realized-RGB
query selected a success on all twelve. The failure was isolated to proposal support.

Protocol amendment v2 used a shared, hand-designed RGB-goal-conditioned proposal. Job
53529 passed ten tests and evaluated 48 prefixes x 128 candidates at horizon 48:

| Task | Mean candidate success | Prefix has oracle candidate | RGB query selects success | Informative prefixes |
|---|---:|---:|---:|---:|
| Reach B | 21.18% | 38/48 (79.17%) | 38/48 (79.17%) | 38/48 |
| A then B | 7.44% | 19/48 (39.58%) | 19/48 (39.58%) | 19/48 |

For ordered prefixes with support, success counts ranged from 1/128 to 49/128. Thus the
bank is neither uniformly impossible nor universally successful. The realized-future RGB
query had zero physical selection regret in this screen, which validates the intended
query semantics in this toy renderer. It does not show that a predictor can forecast that
query, nor that the same observation kernel will work on realistic imagery.

The v2 proposal reads red centroids from start/A/B RGB only. It does not receive simulator
state, future frames, collision geometry or outcomes. Nevertheless it is task-specific
hand engineering and must remain fixed/shared across every learned model and control.

## Decision

Proceed to frozen spatial feature encoding and the predeclared matched codec/forecasting
pilot. Keep 53521's dataset; do not replace it based on headroom outcomes. Treat test
episodes as sealed. First train/evaluate on train/validation only, and preserve distinct
measurements for codec accuracy, counterfactual forecasting, candidate selection and MPC.

No world model was trained in either job. These results therefore cannot support the
predictive-abstraction method claim or any comparison with frame world models.
