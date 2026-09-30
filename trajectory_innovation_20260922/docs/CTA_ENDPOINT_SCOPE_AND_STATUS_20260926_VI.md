# Endpoint sufficiency, codec claim, and execution status

2026-09-26. Read-only assessment of task/model source and small job reports;
no model training or simulator work performed in this assessment.

## Correct the current codec claim

`cov8` is coverage at the branch endpoint (or early success termination).
It depends on block/goal geometry there, not the preceding path. The FULL
scorer is `Scorer('future')`, whose `FutureTokens` explicitly has `path=False`:
it sees the endpoint frame and endpoint agent proprioception, together with
current context and goal. It is not an uncompressed whole-trajectory reader.

The CTA source encoder currently receives intermediate frames too. Thus
CODE .706 versus FULL .785 establishes retention of an endpoint-coverage
ranking signal relative to a learned endpoint reader. It does not establish
trajectory-query sufficiency, useful temporal order retention, or superiority
over an endpoint-only code. FULL is an empirical comparator, not a universal
mathematical upper bound.

For current coverage ranking, equal endpoint geometry means equal target.
For continuation planning, equal *complete Markov state*, remaining time and
termination/controller context imply equal continuation value under the same
controller; one endpoint image need not reveal all those variables. Intermediate
frames may expose motion/contact or aid training, but that possible advantage
must be compared to fair endpoint/short-history baselines. The environment
terminates on success, so early success is retained in the branch endpoint/status;
there is no independent order-of-subgoals objective in this PushT setup.

## Missing decisive controls

Before claiming whole-trajectory abstraction, compare a retrained endpoint-only
codec (`path=False`) against the trajectory codec with matched context, token/bit
budget, supervision and compute. Simply hiding middle frames at inference is an
out-of-distribution corruption test and cannot replace this control. Separately
compare conditional per-frame codes under the same *total* bit budget.

To test temporal information, define held-out, order-sensitive queries over the
segment and demonstrate actual-future readability before evaluating source and
predicted code. Reconstruction alone does not establish task relevance. If the
queries are auxiliary on PushT, report them as representation tests rather than
as necessities for native success. Task/metric selection must not manufacture
a planning claim from irrelevant temporal labels.

PushT remains useful for code prediction, control and latency diagnostics, but
cannot by itself identify the value of whole-trajectory coding under the current
endpoint-only task reader. Do not claim the codec is generally 'good enough' from
89–90% retained endpoint gap. The endpoint/per-frame controls deserve priority
before scaling the final WM-debug direction solely for a trajectory-method claim.

## Job status verified with squeue and sacct

No active user jobs at inspection.

| Job | Actual outcome |
|---|---|
| 54980 | COMPLETED 0:0 in 7s; CPU suites/precision regression and metadata audit |
| 54982 | COMPLETED 0:0 in 1m06s; frozen WM diagnostic |
| 54981_0 and 54981_1 | Both TIMEOUT at 20m29s; no completed preflight verdict |
| 54983 | CANCELLED before any closed-loop rollout because dependencies failed |
| 54984 | CANCELLED before aggregation |

The static precision mismatch is fixed and its regression test passed; real
checkpoint preflight is still unverified. A timeout is not evidence that the
corrected scores disagree, nor evidence that they agree. The next runtime work
must profile/cache redundant feature extraction or provision a justified bound;
do not relax agreement thresholds or treat the cancelled closed loop as a result.

## New WM evidence (600 decisions / 50 dev roots)

Source: `cta_r2_wm_audit_54982/report.json` in shared results.

| Readout | Coverage retained gap |
|---|---:|
| Actual source code | .694 |
| WM free running | .301 |
| Context-only prior free running | .000 |
| WM with true prefixes | .695 |
| Context-only prior with true prefixes | .596 |
| WM with wrong sibling actions and true prefixes | .522 |

WM true-prefix minus wrong-action true-prefix: +.173 [.081,.294]. WM versus
prior with true prefixes: +.099 [.028,.174]. Action information is useful, but
true prefixes themselves supply considerable future information even without
actions. These diagnostics do not prove that replacing autoregression will
recover the source-code gap. Free-running token error is around .85 throughout;
with true prefixes it is .85 at token 0 and much lower later. Token accuracy
alone remains a poor indicator of useful within-bank selection.
